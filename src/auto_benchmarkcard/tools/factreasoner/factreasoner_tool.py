import copy
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import asyncio
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import fact_reasoner.core.base as _fr_base
from fact_reasoner.assessor import FactReasoner
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.reviser import Reviser
from mellea.backends import ModelOption
from mellea.backends.openai import OpenAIBackend

from auto_benchmarkcard.card_utils import (
    _evidence_grounded_in_contexts,
    _is_structured_source,
    detect_absence_placeholders,
    verify_against_artifacts,
)

logger = logging.getLogger(__name__)


# FactReasoner v0.5.9 build_relations has a bug in the contexts_per_atom_only (FR1)
# path: it iterates `atom.get_contexts()`, which returns a dict, so it yields context
# ids (strings) instead of Context objects. NLI then scores the id string against the
# atom and every relation comes back neutral. We make get_contexts return a dict view
# that iterates as values while preserving the dict API (.keys()/[]/in/len) that the
# serialization and baseline call sites rely on.
class _ContextsView(dict):
    def __iter__(self):
        return iter(self.values())


_fr_base.Atom.get_contexts = lambda self: _ContextsView(self.contexts)


# v0.5.9 build() calls the serial Reviser.run (one LLM call per atom, sequential),
# which dominates wall-clock when revise_atoms is on (~6s/atom -> minutes per card).
# The Reviser already ships an async run_batch; route run() through it so atoms are
# revised concurrently (identical prompts/output, ~10x faster). This also matters for
# scaling decontextualization to many cards.
def _parallel_reviser_run(self, units, response):
    import concurrent.futures

    if not units:
        return []
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run_batch(units, response)).result()
    except RuntimeError:
        return asyncio.run(self.run_batch(units, response))


Reviser.run = _parallel_reviser_run


# redirect_stdout/redirect_stderr swap the *global* sys.stdout/stderr, which is not
# thread-safe: when several cards are scored concurrently (a multi-threaded batch) two
# threads can interleave their save/restore so sys.stdout is left pointing at another
# thread's already-closed devnull, and the next print() dies with "I/O operation on
# closed file". Only the lock holder suppresses; concurrent callers run without
# redirection (verbose but correct) and never touch the global streams. Single-threaded
# production keeps the original quiet behavior.
_SUPPRESS_LOCK = threading.Lock()


@contextmanager
def _maybe_suppress_streams():
    if _SUPPRESS_LOCK.acquire(blocking=False):
        try:
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                yield
        finally:
            _SUPPRESS_LOCK.release()
    else:
        yield


class _LogprobsOpenAIBackend(OpenAIBackend):
    """OpenAI-compatible Mellea backend that surfaces choice logprobs where
    FactReasoner's NLI extractor expects them.

    Mellea's OpenAIBackend stores the full chat response under "oai_chat_response"
    (logprobs nested under choices[0]); FactReasoner's extract_logprobs_from_output
    looks for a top-level "logprobs". We lift the choice logprobs so grounded-NLI
    scoring works against RITS / HF OpenAI-compatible endpoints (no mellea-ibm).
    """

    async def post_processing(self, mot, tools, conversation, thinking, seed, _format):
        await super().post_processing(mot, tools, conversation, thinking, seed, _format)
        choice = mot._meta.get("oai_chat_response_choice")
        if isinstance(choice, dict) and choice.get("logprobs") is not None:
            mot._meta["logprobs"] = choice["logprobs"]


# RITS routes each model under its own url slug; map the FactReasoner model to its
# (url-slug, served-model-name). Falls back to a derived slug for unmapped models.
_RITS_ROUTES = {
    "meta-llama/llama-3.3-70b-instruct": (
        "llama-3-3-70b-instruct",
        "meta-llama/llama-3-3-70b-instruct",
    ),
}


def _rits_route(model_id: str) -> tuple[str, str]:
    """Return the (url-slug, served-model-name) for a model on RITS."""
    key = model_id.lower()
    if key in _RITS_ROUTES:
        return _RITS_ROUTES[key]
    slug = model_id.split("/")[-1].lower().replace(".", "-")
    return slug, key


# Referent-explicit hypotheses (SATORI pattern): NLI is structurally blind to
# wrong-referent errors when the hypothesis never names what the claim is about,
# so each atom is wrapped into a full claim naming the benchmark (and, when the
# composer recorded one, the field's non-target referent) before scoring. The
# wrap changes only the hypothesis text; atom decomposition, ids, fields and
# contexts stay untouched. The phrases are a local static map on purpose: the
# composer's field caps serve prompting and must stay independent of scoring.
FIELD_DESCRIPTION_PHRASES = {
    "benchmark_details.name": "the benchmark's name",
    "benchmark_details.overview": "an overview of the benchmark",
    "benchmark_details.data_type": "the benchmark's data type",
    "benchmark_details.domains": "the benchmark's domain(s)",
    "benchmark_details.languages": "the language(s) the benchmark covers",
    "benchmark_details.similar_benchmarks": "benchmarks similar to this benchmark",
    "benchmark_details.resources": "resources available for the benchmark",
    "purpose_and_intended_users.goal": "the benchmark's goal",
    "purpose_and_intended_users.audience_evaluators": "who runs the benchmark",
    "purpose_and_intended_users.audience_consumers": "who uses the benchmark's results",
    "purpose_and_intended_users.tasks": "the task(s) the benchmark evaluates",
    "purpose_and_intended_users.limitations": "the benchmark's limitations",
    "purpose_and_intended_users.out_of_scope_uses": "uses out of scope for the benchmark",
    "data.source": "the source of the benchmark's data",
    "data.size": "the size of the benchmark's dataset",
    "data.format": "the format of the benchmark's data",
    "data.annotation": "how the benchmark's data was annotated",
    "data.collection_date": "when the benchmark's data was collected",
    "data.contamination_controls": "the benchmark's contamination controls",
    "methodology.methods": "the benchmark's evaluation method(s)",
    "methodology.metrics": "the benchmark's evaluation metric(s)",
    "methodology.calculation": "how the benchmark's scores are calculated",
    "methodology.interpretation": "how the benchmark's scores are interpreted",
    "methodology.baseline_results": "the benchmark's baseline results",
    "methodology.validation": "how the benchmark's annotations were validated",
    "methodology.human_baseline": "the benchmark's human baseline",
    "methodology.validity_justification": "why the benchmark's metrics are valid",
    "methodology.judge_score_consolidation": "how the benchmark's judge scores are consolidated",
    "ethical_and_legal_considerations.privacy_and_anonymity": "the benchmark's privacy and anonymity handling",
    "ethical_and_legal_considerations.data_licensing": "the benchmark's data license",
    "ethical_and_legal_considerations.consent_procedures": "the benchmark's consent procedures",
    "ethical_and_legal_considerations.compliance_with_regulations": "the benchmark's regulatory compliance",
}


def _referent_prefix(benchmark_name: str, referent: Optional[Dict[str, str]]) -> str:
    """The 'For ...' clause naming what the claim is about.

    referent is {"id", "name", "type"} for a non-target frame entity, or None for
    the benchmark itself. Sibling gets an explicit not-the-target contrast so NLI
    sees the referent mismatch it is otherwise blind to."""
    if not referent:
        return f'For the benchmark "{benchmark_name}"'
    rname = referent.get("name") or referent.get("id") or ""
    rtype = referent.get("type") or ""
    if rtype in ("subset", "split", "setting"):
        return f'For the "{rname}" {rtype} of benchmark "{benchmark_name}"'
    if rtype == "version":
        return f'For version "{rname}" of benchmark "{benchmark_name}"'
    if rtype == "metric":
        return f'For the "{rname}" metric of benchmark "{benchmark_name}"'
    if rtype == "sibling":
        return f'For the related benchmark "{rname}" (not "{benchmark_name}" itself)'
    return f'For "{rname}", related to benchmark "{benchmark_name}"'


def verbalize_hypothesis(
    text: str,
    field: Optional[str],
    benchmark_name: str,
    referent: Optional[Dict[str, str]] = None,
) -> str:
    """Wrap one atom text into a referent-explicit NLI hypothesis."""
    if field:
        phrase = FIELD_DESCRIPTION_PHRASES.get(field, f'the field "{field}"')
    else:
        phrase = "the card's content"
    return f"{_referent_prefix(benchmark_name, referent)}, {phrase}: {text}"


def field_referents_from_provenance(
    provenance: Optional[Dict[str, Any]], frame: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, str]]:
    """Map dotted field path -> non-target referent recorded in its provenance entry.

    Reads entry["referent"] (a frame node id) where the composer provides it and
    resolves display name/type against the card's referent frame. Fields with no
    referent key, an empty one, or the frame target are omitted, so their
    hypotheses carry target-only context. Unknown node ids degrade to the id's
    own name/type parts rather than being dropped: a recorded non-target referent
    must reach the hypothesis even if the frame sidecar is missing."""
    if not isinstance(provenance, dict):
        return {}
    frame = frame if isinstance(frame, dict) else {}
    nodes = {
        n.get("id"): n for n in frame.get("nodes", []) if isinstance(n, dict) and n.get("id")
    }
    target_id = frame.get("target_id") or "target"
    referents: Dict[str, Dict[str, str]] = {}
    for sec, fields in provenance.items():
        if not isinstance(fields, dict):
            continue
        for fld, entry in fields.items():
            if not isinstance(entry, dict):
                continue
            ref = str(entry.get("referent", "") or "").strip()
            if not ref or ref == target_id:
                continue
            node = nodes.get(ref) or {}
            head, _, tail = ref.partition(":")
            referents[f"{sec}.{fld}"] = {
                "id": ref,
                "name": str(node.get("name") or tail or ref),
                "type": str(node.get("type") or (head if tail else "")),
            }
    return referents


def verbalize_rag_hypotheses(
    formatted_rag_results: Dict[str, Any],
    benchmark_name: str,
    field_referents: Optional[Dict[str, Dict[str, str]]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Wrap every atom's text into a referent-explicit hypothesis.

    Returns (new_rag, telemetry). Granularity is unchanged: same atoms in the
    same order with the same ids, fields and context assignments; only "text" is
    wrapped (the original survives as "text_unverbalized"). The input dict is
    not mutated. Telemetry counts wrapped hypotheses and how many carried
    non-target referent context."""
    field_referents = field_referents or {}
    atoms = (
        formatted_rag_results.get("atoms", [])
        if isinstance(formatted_rag_results, dict)
        else []
    )
    telemetry: Dict[str, Any] = {
        "atoms_total": len(atoms),
        "verbalized": 0,
        "with_referent_context": 0,
        "referent_fields": [],
    }
    referent_fields = set()
    new_atoms = []
    for atom in atoms:
        text = atom.get("text", "") if isinstance(atom, dict) else ""
        if not text:
            new_atoms.append(atom)
            continue
        field = atom.get("field")
        referent = field_referents.get(field) if field else None
        wrapped_atom = dict(atom)
        wrapped_atom["text_unverbalized"] = text
        wrapped_atom["text"] = verbalize_hypothesis(
            text, field, benchmark_name, referent=referent
        )
        new_atoms.append(wrapped_atom)
        telemetry["verbalized"] += 1
        if referent:
            telemetry["with_referent_context"] += 1
            referent_fields.add(field)
    telemetry["referent_fields"] = sorted(referent_fields)
    out = dict(formatted_rag_results) if isinstance(formatted_rag_results, dict) else {}
    out["atoms"] = new_atoms
    return out, telemetry


def _create_atom_marginal_mappings(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Map atom IDs to atom data and variable names to marginal data."""
    atoms_by_id = {atom["id"]: atom for atom in formatted_rag_results.get("atoms", [])}
    marginals_by_var = {m.get("variable", ""): m for m in marginals}
    return atoms_by_id, marginals_by_var


def _find_marginal_for_atom(
    atom_id: str, atom_text: str, marginals_by_var: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Find the marginal probability data corresponding to an atom."""
    exact = marginals_by_var.get(atom_id)
    if exact is not None:
        return exact
    # Legacy fallback: variable names may embed the atom ID or a text prefix.
    for var_name, marginal in marginals_by_var.items():
        if atom_id in var_name or (atom_text and atom_text[:50] in var_name):
            return marginal
    return None


def _determine_flag_reason(field_stats: Dict[str, Any], threshold: float) -> tuple[bool, str, str]:
    """Determine if a field should be flagged and why, based on probability stats."""
    avg_probability = field_stats.get("avg_probability", 1.0)
    all_neutral = field_stats.get("all_neutral", False)
    neutral_count = field_stats.get("neutral_count", 0)

    should_flag = avg_probability < threshold or all_neutral

    if not should_flag:
        return False, "", ""

    if all_neutral:
        if field_stats.get("escalated_still_neutral"):
            reason = "all_atoms_neutral_escalated"
            reason_desc = f"All {neutral_count} atoms neutral after full-source escalation"
        else:
            reason = "all_atoms_neutral"
            reason_desc = f"All {neutral_count} atoms have neutral scores (no evidence found)"
    else:
        reason = "low_factuality_score"
        reason_desc = f"Average factuality score {avg_probability:.3f} below threshold {threshold}"

    return True, reason, reason_desc


def analyze_factuality_by_field(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate factuality scores per benchmark card field and compute accuracy metrics."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    field_stats = {}
    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            if field not in field_stats:
                field_stats[field] = {
                    "total_atoms": 0,
                    "high_confidence_correct": 0,  # p_true > 0.8
                    "likely_correct": 0,  # p_true > 0.6
                    "uncertain": 0,  # 0.4 <= p_true <= 0.6
                    "likely_incorrect": 0,  # p_true < 0.4
                    "high_confidence_incorrect": 0,  # p_true < 0.2
                    "avg_probability": 0,
                    "probabilities": [],
                    "atoms": [],
                }

            field_stats[field]["total_atoms"] += 1
            field_stats[field]["probabilities"].append(p_true)
            field_stats[field]["atoms"].append(
                {
                    "id": atom_id,
                    "text": atom_text,
                    "p_true": p_true,
                    "variable": marginal_data.get("variable", ""),
                }
            )

            if p_true > 0.8:
                field_stats[field]["high_confidence_correct"] += 1
            elif p_true > 0.6:
                field_stats[field]["likely_correct"] += 1
            elif p_true >= 0.4:
                field_stats[field]["uncertain"] += 1
            elif p_true >= 0.2:
                field_stats[field]["likely_incorrect"] += 1
            else:
                field_stats[field]["high_confidence_incorrect"] += 1

    summary = {
        "total_fields": len(field_stats),
        "fields_with_errors": 0,
        "most_problematic_field": None,
        "most_accurate_field": None,
        "overall_field_accuracy": {},
    }

    for field, stats in field_stats.items():
        if stats["total_atoms"] > 0:
            # Exclude neutral scores (0.5) from averaging -- they indicate no evidence
            non_neutral_probabilities = [p for p in stats["probabilities"] if p != 0.5]

            if len(non_neutral_probabilities) == 0:
                stats["avg_probability"] = 0.5
                stats["all_neutral"] = True
            else:
                stats["avg_probability"] = sum(non_neutral_probabilities) / len(
                    non_neutral_probabilities
                )
                stats["all_neutral"] = False

            stats["non_neutral_count"] = len(non_neutral_probabilities)
            stats["neutral_count"] = len(stats["probabilities"]) - len(non_neutral_probabilities)

            accurate_atoms = stats["high_confidence_correct"] + stats["likely_correct"]
            stats["accuracy_percentage"] = (accurate_atoms / stats["total_atoms"]) * 100

            error_atoms = stats["likely_incorrect"] + stats["high_confidence_incorrect"]
            stats["error_percentage"] = (error_atoms / stats["total_atoms"]) * 100

            if error_atoms > 0:
                summary["fields_with_errors"] += 1

            summary["overall_field_accuracy"][field] = stats["accuracy_percentage"]

    if summary["overall_field_accuracy"]:
        most_accurate = max(summary["overall_field_accuracy"].items(), key=lambda x: x[1])
        least_accurate = min(summary["overall_field_accuracy"].items(), key=lambda x: x[1])

        summary["most_accurate_field"] = {
            "field": most_accurate[0],
            "accuracy": most_accurate[1],
        }
        summary["most_problematic_field"] = {
            "field": least_accurate[0],
            "accuracy": least_accurate[1],
        }

    return {"summary": summary, "field_details": field_stats}


def print_clean_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> None:
    """Print each atom with its factuality probability score."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_counter = 1
    for atom_id, atom in atoms_by_id.items():
        atom_text = atom.get("text", "")
        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])
            logger.debug("Atom %d: %s, Probability=%.3f", atom_counter, atom_text, p_true)
        else:
            logger.debug("Atom %d: %s, Probability=N/A", atom_counter, atom_text)
        atom_counter += 1


def create_atom_summary(
    formatted_rag_results: Dict[str, Any], marginals: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build a per-atom summary with field labels and scores, sorted lowest first."""
    atoms_by_id, marginals_by_var = _create_atom_marginal_mappings(formatted_rag_results, marginals)

    atom_summary = []

    for atom_id, atom in atoms_by_id.items():
        field = atom.get("field", "unknown")
        atom_text = atom.get("text", "")

        marginal_data = _find_marginal_for_atom(atom_id, atom_text, marginals_by_var)

        if marginal_data:
            p_true = marginal_data.get("p_true", marginal_data.get("probabilities", [0, 0])[1])

            if p_true > 0.8:
                confidence = "HIGH_CONFIDENCE_CORRECT"
            elif p_true > 0.6:
                confidence = "LIKELY_CORRECT"
            elif p_true >= 0.4:
                confidence = "UNCERTAIN"
            elif p_true >= 0.2:
                confidence = "LIKELY_INCORRECT"
            else:
                confidence = "HIGH_CONFIDENCE_INCORRECT"

            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": round(p_true, 4),
                    "confidence_level": confidence,
                    "variable_name": marginal_data.get("variable", ""),
                }
            )
        else:
            atom_summary.append(
                {
                    "atom_id": atom_id,
                    "field": field,
                    "text": atom_text,
                    "factuality_score": None,
                    "confidence_level": "NO_SCORE",
                    "variable_name": None,
                }
            )

    # Sort by factuality score (lowest first to highlight problems)
    atom_summary.sort(
        key=lambda x: x["factuality_score"] if x["factuality_score"] is not None else -1
    )

    return atom_summary


def _active_factreasoner_engine() -> str:
    """Return the engine type FactReasoner will run on (rits/hf/openai)."""
    from auto_benchmarkcard.config import Config

    # FactReasoner may need a separate provider that supports token logprobs,
    # independently of the composer's serving route.
    override = os.environ.get("FACTREASONER_ENGINE_TYPE", "").strip().lower()
    return override or Config.LLM_ENGINE_TYPE.lower()


def _provider_routing() -> Optional[Dict[str, Any]]:
    """OpenRouter provider preferences for the FactReasoner route, or None.

    Most OpenRouter providers silently drop `logprobs`; left to itself the router picks
    the cheapest endpoint and the NLI extractor's assert fires on the first atom. The
    provider is pinned by name with no fallback, which also keeps the route
    reproducible. `require_parameters` is deliberately NOT set: Mellea adds request
    fields the pinned provider does not advertise, and with that flag OpenRouter
    answered 404 "Filter by Parameters" on the first NLI call (2026-09-06) even though
    the same provider returns logprobs for the same request without it.
    """
    order = [x.strip() for x in os.environ.get("FACTREASONER_PROVIDER", "").split(",") if x.strip()]
    if not order:
        return None
    return {"provider": {"order": order, "allow_fallbacks": False}}


def _create_factreasoner_backend(model_id: str):
    """Create a Mellea backend for FactReasoner from the configured engine.

    RITS and HuggingFace both expose OpenAI-compatible chat endpoints, so a single
    logprobs-surfacing OpenAI backend serves both (no mellea-ibm needed). Any other
    engine type uses a generic OpenAI-compatible endpoint (FACTREASONER_API_BASE).
    """
    try:
        engine_type = _active_factreasoner_engine()
        max_new = ModelOption.MAX_NEW_TOKENS

        if engine_type == "rits":
            api_key = os.environ.get("RITS_API_KEY")
            api_url = os.environ.get("RITS_API_URL")
            if not (api_key and api_url):
                return None
            slug, rits_model = _rits_route(model_id)
            base_url = f"{api_url.rstrip('/')}/{slug}/v1"
            return _LogprobsOpenAIBackend(
                model_id=rits_model,
                base_url=base_url,
                api_key=api_key,
                default_headers={"RITS_API_KEY": api_key},
                model_options={max_new: 1024},
            )

        if engine_type == "hf":
            api_key = os.environ.get("HF_TOKEN")
            api_base = os.environ.get("HF_API_URL", "https://router.huggingface.co/v1")
            if not api_key:
                logger.warning("HF_TOKEN not set; cannot create FactReasoner backend")
                return None
            headers = {}
            hf_org = os.environ.get("HF_ORG")
            if hf_org:
                headers["X-HF-Bill-To"] = hf_org
            return _LogprobsOpenAIBackend(
                model_id=model_id,
                base_url=api_base,
                api_key=api_key,
                default_headers=headers or None,
                model_options={max_new: 1024},
            )

        # Generic OpenAI-compatible fallback
        api_base = os.environ.get("FACTREASONER_API_BASE") or os.environ.get("LLM_API_BASE")
        api_key = os.environ.get("FACTREASONER_API_KEY") or os.environ.get("LLM_API_KEY")
        if api_base and api_key:
            model_options: Dict[str, Any] = {max_new: 1024}
            routing = _provider_routing()
            if routing:
                model_options["extra_body"] = routing
            return _LogprobsOpenAIBackend(
                model_id=model_id,
                base_url=api_base,
                api_key=api_key,
                model_options=model_options,
            )
    except Exception:
        logger.debug("Could not create FactReasoner backend", exc_info=True)
    return None


def evaluate_factuality(
    formatted_rag_results: Dict[str, Any],
    model: str = "llama-3.3-70b-instruct",
    nli_prompt_version: str = "v1",
    cache_dir: str = "factreasoner_cache",
    merlin_path: str = "merlin/bin/merlin",
    debug_mode: bool = False,
    revise_atoms: bool = False,
) -> Dict[str, Any]:
    """Run NLI-based factuality evaluation on benchmark card claims.

    Uses FactReasoner (v0.5.9, Mellea backend) to check whether source evidence
    supports, contradicts, or is neutral about each extracted claim, then runs
    probabilistic reasoning to produce per-claim confidence scores. Runs FR1
    (each atom scored against its own retrieved contexts); ``revise_atoms`` turns
    on decontextualization of atoms before scoring.
    """
    os.makedirs(cache_dir, exist_ok=True)
    backend = _create_factreasoner_backend(model)
    if backend is None:
        raise RuntimeError(
            "FactReasoner backend unavailable; check RITS_API_KEY/RITS_API_URL "
            "(LLM_ENGINE_TYPE=rits) or HF_TOKEN (LLM_ENGINE_TYPE=hf)."
        )

    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    nli_extractor = NLIExtractor(backend)

    pipeline = FactReasoner(
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        context_retriever=None,
        context_summarizer=None,
        merlin_path=merlin_path,
        # v0.5.9 needs context priors to anchor evidence as "supported"; without
        # them grounded relations cannot propagate and every atom stays at 0.5.
        use_priors=True,
    )

    pipeline.from_dict_with_contexts(data=formatted_rag_results)

    # Suppress FactReasoner's verbose stdout/stderr during processing (thread-safe;
    # see _maybe_suppress_streams).
    with _maybe_suppress_streams():
        asyncio.run(
            pipeline.build(
                has_atoms=True,
                has_contexts=True,
                revise_atoms=revise_atoms,
                rel_atom_context=True,
                rel_context_context=False,
                contexts_per_atom_only=True,
                remove_duplicates=False,
                summarize_contexts=False,
            )
        )
        results, marginals = pipeline.score()

    # score() returns marginals for atom AND context variables; keep atom variables
    # only and backfill orphan atoms (no NLI relation -> no marginal) as neutral.
    marginals_by_var = {m.get("variable"): m for m in marginals}
    entropy = 0.0
    eps = 1e-7
    valid_marginals = []
    for atom in formatted_rag_results.get("atoms", []):
        aid = atom.get("id")
        info = marginals_by_var.get(aid)
        probs = info.get("probabilities") if info else None
        if not (probs and len(probs) > 1):
            probs = [0.5, 0.5]
        p_true = max(probs[1], eps)
        p_false = max(1.0 - p_true, eps)
        entropy += -p_true * math.log10(p_true) - p_false * math.log10(p_false)
        valid_marginals.append({"variable": aid, "probabilities": probs, "p_true": probs[1]})

    n = len(valid_marginals)
    normalized_entropy = entropy / n if n > 0 else 0.0
    scaled_entropy = entropy / math.log10(n) if n > 1 else 0.0

    field_analysis = analyze_factuality_by_field(formatted_rag_results, valid_marginals)
    atom_summary = create_atom_summary(formatted_rag_results, valid_marginals)

    flagged_count = len([m for m in valid_marginals if m.get("p_true", 1) < 0.3])
    logger.debug(
        "FactReasoner: %d claims evaluated, %d/%d flagged",
        len(valid_marginals), flagged_count, len(valid_marginals),
    )

    return {
        "results": {
            "num_atoms": results.get("num_atoms", len(pipeline.atoms)),
            "num_contexts": results.get("num_contexts", len(pipeline.contexts)),
            "factuality_score": results.get("factuality_score"),
        },
        "marginals": valid_marginals,
        "entropy_metrics": {
            "total_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "scaled_entropy": scaled_entropy,
            "num_variables": n,
        },
        "fact_graph_info": {
            "num_atoms": len(pipeline.atoms),
            "num_contexts": len(pipeline.contexts),
        },
        "field_analysis": field_analysis,
        "atom_summary": atom_summary,
    }


def _window_text(text: str, window_chars: int = 12000, overlap_chars: int = 500) -> List[str]:
    """Deterministic overlapping character windows covering the full text."""
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]
    step = window_chars - overlap_chars
    return [text[i:i + window_chars] for i in range(0, max(len(text) - overlap_chars, 1), step)]


def evaluate_factuality_two_tier(
    formatted_rag_results: Dict[str, Any],
    source_text: str,
    model: str = "llama-3.3-70b-instruct",
    cache_dir: str = "factreasoner_cache",
    merlin_path: str = "merlin/bin/merlin",
    revise_atoms: bool = False,
    window_chars: int = 12000,
    overlap_chars: int = 500,
) -> Dict[str, Any]:
    """Two-tier factuality: top-k pass, then escalate only the all-neutral atoms
    against large windows of the full source text.

    First pass is evaluate_factuality unchanged. Atoms whose marginal stays at
    exactly 0.5 (= no informative NLI relation at all) get a second pass where
    their contexts are overlapping windows covering the whole source. After
    that, "still neutral" genuinely means "not in the source": affected fields
    are marked escalated_still_neutral so the flag reason is distinguishable.
    Second-pass marginals win for escalated atoms (the first-pass 0.5 carries
    zero information by construction); non-escalated atoms are untouched, so
    their scores stay bit-identical to a plain run.
    """
    first = evaluate_factuality(
        formatted_rag_results, model=model, cache_dir=cache_dir,
        merlin_path=merlin_path, revise_atoms=revise_atoms,
    )
    if revise_atoms:
        logger.warning("two-tier escalation scores the unrevised atom texts")
    neutral_ids = [m["variable"] for m in first["marginals"] if m.get("p_true") == 0.5]
    if not neutral_ids:
        first["escalation"] = {"skipped": "no_neutral_atoms"}
        return first
    if not (source_text or "").strip():
        first["escalation"] = {"skipped": "no_source"}
        return first

    windows = _window_text(source_text, window_chars, overlap_chars)
    contexts = [
        {"id": f"esc_w{i}", "title": f"source window {i + 1}/{len(windows)}", "text": w}
        for i, w in enumerate(windows)
    ]
    atoms_by_id = {a["id"]: a for a in formatted_rag_results.get("atoms", [])}
    esc_atoms = [
        {**atoms_by_id[aid], "contexts": [c["id"] for c in contexts]}
        for aid in neutral_ids if aid in atoms_by_id
    ]
    # Escalate in small groups with one retry each: atoms x windows in a single
    # asyncio.gather is hundreds of long-premise NLI calls, and one request
    # timeout would otherwise void the whole card's escalation (observed on
    # RITS). A group that fails twice just leaves its atoms neutral.
    second_by_var: Dict[str, Any] = {}
    failed_groups = 0
    group_size = 4
    for gi in range(0, len(esc_atoms), group_size):
        group = esc_atoms[gi:gi + group_size]
        # from_dict_with_contexts requires input/output keys on the dict.
        esc_rag = {
            "input": formatted_rag_results.get("input", ""),
            "output": formatted_rag_results.get("output", ""),
            "topic": formatted_rag_results.get("topic"),
            "atoms": group,
            "contexts": contexts,
        }
        for attempt in (1, 2):
            try:
                second = evaluate_factuality(
                    esc_rag, model=model, cache_dir=cache_dir,
                    merlin_path=merlin_path, revise_atoms=False,
                )
                second_by_var.update({m["variable"]: m for m in second["marginals"]})
                break
            except Exception as e:  # noqa: BLE001 - degrade this group to neutral
                if attempt == 2:
                    failed_groups += 1
                    logger.warning(
                        "escalation group %d failed twice (%s); its atoms stay neutral",
                        gi // group_size, e,
                    )

    if not second_by_var:
        logger.warning("escalation produced no marginals; keeping first-pass results")
        first["escalation"] = {
            "failed": f"all {failed_groups} groups failed",
            "escalated_atoms": len(neutral_ids), "num_windows": len(windows),
        }
        return first
    neutral_set = set(neutral_ids)
    merged = [
        second_by_var.get(m["variable"], m) if m["variable"] in neutral_set else m
        for m in first["marginals"]
    ]

    entropy = 0.0
    eps = 1e-7
    for m in merged:
        p_true = max(m.get("p_true", 0.5), eps)
        p_false = max(1.0 - p_true, eps)
        entropy += -p_true * math.log10(p_true) - p_false * math.log10(p_false)
    n = len(merged)

    field_analysis = analyze_factuality_by_field(formatted_rag_results, merged)
    # Post-merge, any atom still at 0.5 was escalated and stayed neutral, so an
    # all_neutral field here survived a full-source check -- but only claim that
    # when every escalation group actually ran; atoms in failed groups were
    # never checked against the full source.
    if failed_groups == 0:
        for stats in field_analysis.get("field_details", {}).values():
            if stats.get("all_neutral"):
                stats["escalated_still_neutral"] = True

    still_neutral = [
        aid for aid in neutral_ids
        if second_by_var.get(aid, {}).get("p_true", 0.5) == 0.5
    ]
    out = dict(first)
    out["marginals"] = merged
    out["field_analysis"] = field_analysis
    out["atom_summary"] = create_atom_summary(formatted_rag_results, merged)
    out["entropy_metrics"] = {
        "total_entropy": entropy,
        "normalized_entropy": entropy / n if n else 0.0,
        "scaled_entropy": entropy / math.log10(n) if n > 1 else 0.0,
        "num_variables": n,
    }
    out["escalation"] = {
        "escalated_atoms": len(neutral_ids),
        "num_windows": len(windows),
        "window_chars": window_chars,
        "source_chars": len(source_text),
        "failed_groups": failed_groups,
        "resolved": {
            aid: second_by_var[aid]["p_true"]
            for aid in neutral_ids
            if aid in second_by_var and second_by_var[aid].get("p_true") != 0.5
        },
        "still_neutral": still_neutral,
    }
    return out


def flag_benchmark_card_fields(
    benchmark_card: Dict[str, Any],
    field_analysis: Dict[str, Any],
    threshold: float = 0.8,
    provenance: Optional[Dict[str, Any]] = None,
    retrieved_contexts: Optional[List[str]] = None,
    structured_artifacts: Optional[Dict[str, Any]] = None,
    high_signal_grounding: bool = False,
) -> Dict[str, Any]:
    """Add a flagged_fields section to the benchmark card for low-confidence fields.

    A flagged field is only un-flagged when its provenance evidence is actually
    grounded in the retrieved source text (not merely asserted by the composer):
    the NLI model may fail to match wording, but the claim must still appear in a
    real source for the flag to be lifted.

    When structured_artifacts is given ({"eee": <json>, "hf": <json>}), a claimed
    structured source is verified against the actual artifact instead of being
    trusted on the composer's say-so; with no artifact on disk the claim falls
    back to prose grounding. structured_artifacts=None trusts the composer's
    structured-source label as given.
    """

    flagged_card = copy.deepcopy(benchmark_card)
    field_details = field_analysis.get("field_details", {})

    flagged_fields = {}

    for field_name, field_stats in field_details.items():
        if field_name == "benchmark_details.name" or field_name.endswith(".name"):
            continue

        should_flag, reason, reason_desc = _determine_flag_reason(field_stats, threshold)

        if should_flag:
            parts = field_name.split(".")
            section_key = parts[0] if len(parts) > 0 else ""
            field_key = parts[1] if len(parts) > 1 else ""

            section_prov = (provenance or {}).get(section_key, {})
            field_prov = section_prov.get(field_key, {})
            evidence = field_prov.get("evidence") or ""

            if structured_artifacts is None:
                # No artifacts given: trust the composer's structured-source label.
                unflag = bool(
                    field_prov.get("source")
                    and evidence
                    and (
                        _is_structured_source(field_prov.get("source"))
                        or _evidence_grounded_in_contexts(evidence, retrieved_contexts)
                    )
                )
            else:
                # Artifact verification is a RESCUE, not a veto, and runs on the
                # field VALUE itself (plus evidence) regardless of what source the
                # composer claims -- a claim whose numbers/names/config keys are
                # really in the eee/hf artifact is grounded even if the composer
                # forgot or mislabeled its provenance. Independently, evidence
                # genuinely present in the retrieved text also lifts the flag.
                section_val = benchmark_card.get(section_key)
                value = section_val.get(field_key) if isinstance(section_val, dict) else None
                bench_name = (benchmark_card.get("benchmark_details") or {}).get("name")
                verified = verify_against_artifacts(
                    value, evidence, structured_artifacts, benchmark_name=bench_name
                )
                grounded = bool(evidence) and _evidence_grounded_in_contexts(
                    evidence, retrieved_contexts, high_signal=high_signal_grounding
                )
                unflag = (verified is True) or grounded
            if unflag:
                should_flag = False
                logger.debug(
                    "Skipping flag for %s: claim grounded (source=%s)",
                    field_name,
                    field_prov.get("source"),
                )

        if should_flag:
            if reason == "all_atoms_neutral_escalated":
                flag_reason = (
                    "[Probable Hallucination], no supporting evidence found in full source material"
                )
            elif reason == "all_atoms_neutral":
                flag_reason = (
                    "[Possible Hallucination], no supporting evidence found in source material"
                )
            else:
                avg_prob = field_stats.get("avg_probability", 1.0)
                flag_reason = f"[Factuality Score: {avg_prob:.2f}], low factual alignment with source material"

            flagged_fields[field_name] = flag_reason

    # Deterministic absence-placeholder pass over the card values (catches fields
    # the composer abstained on but that slipped past the canonical sentinel, so
    # they were never atomized). Never overwrites an NLI reason.
    for path, reason in detect_absence_placeholders(benchmark_card).items():
        flagged_fields.setdefault(path, reason)

    if flagged_fields:
        flagged_card["flagged_fields"] = flagged_fields

    return flagged_card


def save_factuality_results(results: Dict[str, Any], output_path: str) -> None:
    """Save factuality evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_formatted_rag_results(file_path: str) -> Dict[str, Any]:
    """Load formatted RAG results from a JSON or JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RAG results file not found: {file_path}")

    if file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            line = f.readline().strip()
            if line:
                return json.loads(line)
            else:
                raise ValueError("Empty JSONL file")
    else:
        with open(file_path, "r") as f:
            return json.load(f)


def main():
    """CLI interface for factuality evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate factuality of RAG results and flag benchmark cards"
    )
    parser.add_argument("input_file", help="Path to formatted RAG results file (.json or .jsonl)")
    parser.add_argument("--benchmark-card", help="Path to original benchmark card (for flagging)")
    parser.add_argument(
        "--output-dir",
        default="factuality_results",
        help="Output directory for results",
    )
    parser.add_argument("--model", default="llama-3.3-70b-instruct", help="LLM model name")
    parser.add_argument("--cache-dir", default="factreasoner_cache", help="Cache directory")
    parser.add_argument("--merlin-path", default="merlin/bin/merlin", help="Path to merlin binary")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Factuality threshold for flagging fields",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    parser.add_argument("--revise-atoms", action="store_true", help="Decontextualize atoms before scoring")

    args = parser.parse_args()

    benchmark_name = os.path.basename(args.input_file)
    benchmark_name = (
        benchmark_name.replace("formatted_rag_results_", "")
        .replace(".jsonl", "")
        .replace(".json", "")
    )

    logger.debug("Starting FactReasoner evaluation...")

    try:
        logger.debug("Loading evidence and claims...")
        rag_results = load_formatted_rag_results(args.input_file)

        logger.debug(
            "Found %d atoms and %d contexts",
            len(rag_results.get("atoms", [])),
            len(rag_results.get("contexts", [])),
        )

        logger.debug("Running probabilistic reasoning...")
        factuality_results = evaluate_factuality(
            formatted_rag_results=rag_results,
            model=args.model,
            cache_dir=args.cache_dir,
            merlin_path=args.merlin_path,
            debug_mode=False,
            revise_atoms=args.revise_atoms,
        )

        output_path = os.path.join(args.output_dir, f"factuality_results_{benchmark_name}.json")
        save_factuality_results(factuality_results, output_path)

        if args.benchmark_card:
            logger.debug("Flagging low-confidence fields (threshold: %.2f)", args.threshold)

            with open(args.benchmark_card, "r") as f:
                benchmark_card_data = json.load(f)

            if isinstance(benchmark_card_data, dict) and "benchmark_card" in benchmark_card_data:
                benchmark_card = benchmark_card_data["benchmark_card"]
            else:
                benchmark_card = benchmark_card_data

            field_analysis = factuality_results.get("field_analysis", {})
            flagged_card = flag_benchmark_card_fields(
                benchmark_card=benchmark_card,
                field_analysis=field_analysis,
                threshold=args.threshold,
            )

            flagged_output_path = os.path.join(
                args.output_dir, f"benchmark_card_{benchmark_name}_flagged.json"
            )
            with open(flagged_output_path, "w") as f:
                json.dump({"benchmark_card": flagged_card}, f, indent=2)

            field_analysis_data = factuality_results.get("field_analysis", {})
            field_details = field_analysis_data.get("field_details", {})
            total_fields = len(field_details)

            flagged_count = 0
            for field_name, field_stats in field_details.items():
                avg_probability = field_stats.get("avg_probability", 1.0)
                all_neutral = field_stats.get("all_neutral", False)
                if avg_probability < args.threshold or all_neutral:
                    flagged_count += 1

            logger.debug("Flagged benchmark card created")
            logger.debug("Fields flagged for review: %d/%d", flagged_count, total_fields)

            if flagged_count > 0:
                logger.debug("Fields requiring review:")
                for field_name, field_stats in field_details.items():
                    avg_probability = field_stats.get("avg_probability", 1.0)
                    all_neutral = field_stats.get("all_neutral", False)
                    if avg_probability < args.threshold or all_neutral:
                        atoms_to_review = len(
                            [
                                atom
                                for atom in field_stats.get("atoms", [])
                                if atom.get("p_true", 1.0) < args.threshold
                                or atom.get("p_true", 1.0) == 0.5
                            ]
                        )
                        logger.debug(
                            "  • %s (score: %.2f, %d atoms)",
                            field_name.replace("_", " ").title(),
                            avg_probability,
                            atoms_to_review,
                        )
            else:
                logger.debug("All fields passed factuality review")

        logger.debug("FactReasoner evaluation completed")
        logger.debug("Full results: %s", json.dumps(factuality_results, indent=2))

    except Exception as e:
        logger.error("Error during factuality evaluation: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
