"""LangGraph worker nodes for the benchmark processing pipeline."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import requests

from auto_benchmarkcard.card_utils import (
    apply_deterministic_overrides,
    apply_single_judge_bias_rule,
    backfill_from_provenance,
    extract_card,
    extract_hf_tags,
    extract_missing_fields,
    normalize_not_specified,
)
from auto_benchmarkcard.config import Config
from auto_benchmarkcard.output import sanitize_benchmark_name, sanitize_chroma_collection_name
from auto_benchmarkcard.tools.ai_atlas_nexus.ai_atlas_nexus_tool import (
    identify_and_integrate_risks, omit_empty_risk_urls)
from auto_benchmarkcard.tools.composer.composer_tool import (
    compose_benchmark_card, drop_inapplicable_judge_missing, enforce_card_shape,
    _eee_identity_subject, _get_hf_readme, _hf_subject_conflicts, _readme_lead, _subject_overlap)
from auto_benchmarkcard.tools.docling.docling_tool import extract_paper_with_docling
from auto_benchmarkcard.tools.html.html_tool import extract_html_content
from auto_benchmarkcard.tools.github.github_tool import derive_github_repo, fetch_github_readme
from auto_benchmarkcard.tools.extractor.extractor_tool import extract_ids
from auto_benchmarkcard.tools.factreasoner.factreasoner_tool import (
    evaluate_factuality_two_tier,
    flag_benchmark_card_fields,
)
from auto_benchmarkcard.tools.eee.paper_resolver import (
    resolve_paper, _repo_corroboration, _token_is_namelike, verify_hf_match,
    verify_paper_binding, _fetch_paper_meta_for_verify, _introduces_other_benchmark,
    _parent_only_binding, _distinctive_base, _title_bears_name, NAME_TOKEN_FLOOR_LEN)
from auto_benchmarkcard.tools.hf.hf_tool import hf_dataset_metadata
from auto_benchmarkcard.tools.rag.atomizer import atomize_benchmark_card
from auto_benchmarkcard.tools.rag.format_converter import (
    convert_rag_to_required_format,
    save_formatted_results,
)
from auto_benchmarkcard.tools.rag.indexer import MetadataIndexer
from auto_benchmarkcard.tools.rag.rag_retriever import RAGRetriever
from auto_benchmarkcard.tools.rag.source_retrieval import (
    assemble_source_text,
    source_scoped_rag,
)
from auto_benchmarkcard.tools.unitxt import unitxt_tool

from auto_benchmarkcard.validation_policy import ANALYTICAL_FIELDS, IDENTITY_FIELDS

logger = logging.getLogger(__name__)


def handle_error(error: Exception, operation: str, state) -> Dict[str, Any]:
    """Log an exception and return updated state with the failure recorded."""
    error_msg = f"{operation} failed: {error}"
    logger.error(error_msg, exc_info=True)

    if hasattr(error, "original_error") and error.original_error:
        logger.error("Original error: %s", error.original_error)

    errors = state.get("errors", [])
    errors.append(error_msg)

    return {"errors": errors, "completed": [f"{operation.lower()} failed"]}


def record_skip(message: str, operation: str, state) -> Dict[str, Any]:
    """Record a skipped step when preconditions are not met (no traceback)."""
    logger.warning("%s: %s", operation, message)
    errors = state.get("errors", [])
    errors.append(f"{operation}: {message}")
    return {"errors": errors, "completed": [f"{operation.lower()} failed"]}


def run_unitxt(state) -> Dict[str, Any]:
    """Retrieve UnitXT metadata for the benchmark."""
    try:
        result = unitxt_tool.unitxt_benchmark_lookup(
            state["query"], catalog_path=state.get("catalog_path")
        )
        unitxt_data = result.model_dump(mode="json")

        logger.info("UnitXT metadata retrieved")
        name = unitxt_data.get("name", "N/A")
        description = unitxt_data.get("description", "")
        if description:
            desc_preview = description[:60] + "..." if len(description) > 60 else description
            logger.info(f"Found: {name} - {desc_preview}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(unitxt_data, "unitxt", filename)
        logger.info("UnitXT output saved to: %s", output_file)

        return {
            "unitxt_json": unitxt_data,
            "completed": ["unitxt done"],
        }
    except Exception as e:
        return handle_error(e, "UnitXT lookup", state)


def run_extractor(state) -> Dict[str, Any]:
    """Extract HF repo ID, paper URL, and risk tags from UnitXT data."""
    logger.info("Starting ID and URL extraction")
    try:
        extracted = extract_ids.func(
            source=state["unitxt_json"], want=["hf_repo", "paper_url", "risk_tags"]
        )

        hf_repo = extracted.get("hf_repo")
        paper_url = extracted.get("paper_url")

        logger.info("ID extraction completed")
        hf_status = hf_repo if hf_repo else "None"
        paper_status = "Found" if paper_url else "None"
        logger.info(f"Extracted: HF={hf_status}, Paper={paper_status}")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(extracted, "extractor", filename)
        logger.info("Extractor output saved to: %s", output_file)

        return {
            "extracted_ids": extracted,
            "hf_repo": hf_repo,
            "completed": [f"extract hf_repo={hf_repo}, paper_url={paper_url}"],
        }
    except Exception as e:
        return handle_error(e, "ID extraction", state)


def run_hf_extractor(state):
    """Extract paper URL from HF metadata when UnitXT didn't provide one."""
    logger.info("Starting HuggingFace extraction")
    try:
        current_extracted = state.get("extracted_ids", {})
        hf_data = state["hf_json"]
        paper_url = None

        hf_extracted = extract_ids.func(source=hf_data, want=["paper_url"])
        paper_url = hf_extracted.get("paper_url")
        if paper_url:
            logger.info("Found paper_url in HF top-level metadata: %s", paper_url)

        if not paper_url:
            for dataset_id, dataset_metadata in hf_data.items():
                if isinstance(dataset_metadata, dict):
                    hf_extracted = extract_ids.func(source=dataset_metadata, want=["paper_url"])
                    extracted_paper_url = hf_extracted.get("paper_url")
                    if extracted_paper_url:
                        paper_url = extracted_paper_url
                        logger.info("Found paper_url in HF dataset %s: %s", dataset_id, paper_url)
                        break

        if paper_url:
            updated_extracted = current_extracted.copy()
            updated_extracted["paper_url"] = paper_url
            updated_extracted["paper_url_from_hf"] = paper_url

            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                updated_extracted, "extractor", filename
            )
            logger.info("HF extractor output saved to: %s", output_file)

            return {
                "extracted_ids": updated_extracted,
                "hf_extraction_attempted": True,
                "completed": [f"hf_extract paper_url={paper_url}"],
            }
        else:
            logger.info("No paper_url found in HF metadata")
            return {
                "hf_extraction_attempted": True,
                "completed": ["hf_extract no paper_url found"],
            }

    except Exception as e:
        result = handle_error(e, "HF extraction", state)
        result["hf_extraction_attempted"] = True
        return result


_BIBTEX_ARXIV_RE = re.compile(r"arxiv[:\s.]*(\d{4}\.\d{4,5})", re.IGNORECASE)
_PAPER_LINK_RE = re.compile(
    r"\*{0,2}Paper\*{0,2}\s*[:：]\s*(https?://\S+)", re.IGNORECASE
)
_ARXIV_URL_RE = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def _extract_paper_from_hf(hf_json: Dict[str, Any]) -> str | None:
    """Try to extract a paper URL from HF README markdown (BibTeX, Paper: link, tags)."""
    if not hf_json:
        return None

    # Check tags for arxiv: entries (structured, most reliable)
    card_data = hf_json.get("card_data") or {}
    tags = card_data.get("tags") or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("arxiv:"):
                arxiv_id = tag.split(":", 1)[1].strip()
                if arxiv_id:
                    return f"https://arxiv.org/abs/{arxiv_id}"

    readme = hf_json.get("readme_markdown") or ""
    if not readme:
        return None

    # Parse **Paper:** link
    m = _PAPER_LINK_RE.search(readme)
    if m:
        url = m.group(1).rstrip(")")
        return url

    # Parse BibTeX blocks for arxiv IDs
    bibtex_blocks = re.findall(r"```bibtex(.*?)```", readme, re.DOTALL | re.IGNORECASE)
    for block in bibtex_blocks:
        m = _BIBTEX_ARXIV_RE.search(block)
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"

    # Scan entire README for arxiv URLs
    m = _ARXIV_URL_RE.search(readme)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"

    return None


# Resolve-time paper-binding verifier thresholds (named, not magic), mirroring the HF gate's floors.
PAPER_LLM_ACCEPT_CONF = 0.7        # LLM confidence floor for a confident accept AND a confident reject;
                                   # below it the verdict is hesitant -> deterministic backstop decides.
PAPER_SAMENAME_OVERLAP_FLOOR = 0.10  # same-name backstop: a bound paper that declares the SAME name but
                                     # whose abstract is near-disjoint from the identity subject is a
                                     # same-name collision (the actionbench class). Mirrors the HF
                                     # _SUBJECT_OVERLAP_FLOOR; undecidable (thin) -> no reject.
PAPER_TIER1_ACCEPT_OVERLAP = 0.30  # 0-LLM deterministic-ACCEPT requires POSITIVE subject evidence
                                   # (overlap is not None AND >= this), exactly like HF Tier-1
                                   # (HF_TIER1_ACCEPT_OVERLAP). A name-bearing pre-set with thin/None
                                   # overlap (e.g. an abstract-less OpenAlex DOI record) is NOT auto-
                                   # accepted -- it routes to the LLM, so a same-name wrong-domain paper
                                   # cannot slip through with 0 LLM.


def _paper_identity_context(state):
    """Assemble the benchmark identity + search context once, shared by the binding gate and the
    fall-through search call. Mirrors the field derivation run_paper_resolver already used for
    resolve_paper, so the gate verifies against the SAME identity the search path would."""
    extracted_ids = state.get("extracted_ids") or {}
    eee_metadata = state.get("eee_metadata") or {}
    hf_json = state.get("hf_json") or {}
    card_data = hf_json.get("card_data") or {}
    dataset_info = hf_json.get("dataset_info") or {}
    full_name = (card_data.get("pretty_name") or dataset_info.get("dataset_name")
                 or extracted_ids.get("full_name"))
    overview = (card_data.get("description") or dataset_info.get("description")
                or _eee_identity_subject(eee_metadata) or extracted_ids.get("overview"))
    domains = card_data.get("task_categories") or extracted_ids.get("domains")
    domain = ", ".join(domains) if isinstance(domains, list) else domains
    return {
        "suite_name": state["query"],
        "full_name": full_name,
        "overview": overview,
        "domain": domain,
        "domains": domains,
        "sub_benchmarks": eee_metadata.get("contains", []),
        "metrics": (list(eee_metadata.get("metrics", {}).keys())
                    if eee_metadata.get("metrics") else []),
        "eval_library": eee_metadata.get("eval_library"),
        "hf_repo_id": hf_json.get("id"),
        "identity_subject": _eee_identity_subject(eee_metadata) or (overview or ""),
    }


def _verdict_has_decision(v: Dict[str, Any]) -> bool:
    """True when a non-errored verify_paper_binding verdict carries a REAL match decision (so a
    not-a-match may be trusted as a confident reject). The normalized verdict authoritatively derives
    is_match from a valid match_index, so when no `match_index` key is present the boolean is_match IS
    the decision. When a `match_index` key IS present (a raw/passed-through verdict), it must be a valid
    0-based integer OR the explicit string "none" to count as a decision; a missing/null/garbage value
    is a degenerate no-decision result (treated as ambiguous -> keep-degraded, never a reject)."""
    if "match_index" not in v:
        return True
    mi = v.get("match_index")
    if isinstance(mi, bool):
        return False
    if isinstance(mi, int):
        return True
    return isinstance(mi, str) and mi.strip().lower() == "none"


def _verify_paper_binding_gate(state, paper_url, binding_source, ctx):
    """Verify a paper_url bound by a NON-resolver path (EEE/UnitXT pre-set or HF arxiv-tag extraction).

    Returns (action, record) where action is "keep" or "reject" and record is the sidecar binding
    block. The deterministic 0-LLM fast path keeps a self-referential pre-set (the bound paper's title
    bears the benchmark's distinctive name AND has positive subject overlap, mirroring HF Tier-1) at
    zero LLM cost; otherwise the LLM verifier decides, with a deterministic same-name-domain backstop
    (subject-overlap, in the worker per the no-circular-import rule). Precision-preserving outage policy
    (mirror of the HF gate): a paper-metadata FETCH failure or an LLM ERROR keeps the binding degraded
    (never drop a possibly-correct binding under an outage, and never verify a blank candidate); only a
    confident reject or the deterministic backstop drops. Keyed on structure/semantics, never a
    benchmark-name list."""
    suite = ctx["suite_name"]
    full_name = ctx["full_name"]
    base = {"binding_source": binding_source, "paper_url": paper_url}

    meta = _fetch_paper_meta_for_verify(paper_url)
    title = meta.get("title", "")
    abstract = meta.get("abstract", "")
    fetch_error = bool(meta.get("fetch_error"))

    # Fetch failure (outage / unfetchable URL): KEEP degraded, do NOT verify a BLANK candidate. A
    # transient arXiv/OpenAlex outage must never drop a correct pre-set via an LLM "none" on empty
    # content. Mirror of the HF gate, which reads an already-fetched README and never hits a
    # verify-time fetch failure. (A SUCCESSFUL fetch with a genuinely empty abstract is fetch_error
    # False -> still routed to the verifier below, not masked as an outage.)
    if fetch_error:
        return "keep", {**base, "tier": "deterministic", "llm_verdict": None,
                        "final_action": "kept", "verdict": "unverified_degraded",
                        "reason": "paper metadata unfetchable (outage/unfetchable URL); kept degraded"}

    # Deterministic name guard (0 LLM): a plainly DIFFERENT declared name is a reject.
    declared_other = _introduces_other_benchmark(title, suite, full_name)
    if declared_other:
        return "reject", {**base, "tier": "deterministic", "llm_verdict": None,
                          "final_action": "rejected_to_none", "verdict": "rejected",
                          "declared_other": declared_other,
                          "reason": f"bound paper declares a different benchmark {declared_other!r}"}

    # Derivative-binds-parent guard (0 LLM): a variant-qualified identity pre-set to the PARENT's
    # paper is rejected unless the paper mentions the variant. Sits BEFORE the Tier-1 accept: a
    # variant repo whose pretty_name collapses to the parent name would otherwise be name-bearing
    # and could keep the parent paper at Tier-1. Shares _parent_only_binding with resolve_paper.
    parent_only = _parent_only_binding(suite, full_name, title, abstract)
    if parent_only:
        return "reject", {**base, "tier": "deterministic", "llm_verdict": None,
                          "final_action": "rejected_to_none", "verdict": "rejected",
                          "derivative_guard": parent_only,
                          "reason": f"bound paper is the parent {parent_only['parent']!r} without "
                                    f"the variant qualifier {parent_only['qualifier']!r}"}

    # Deterministic ACCEPT (0 LLM): the bound paper's title bears the benchmark's distinctive name as a
    # whole-token run AND there is POSITIVE subject evidence (overlap >= PAPER_TIER1_ACCEPT_OVERLAP) --
    # a self-referential binding (the benchmark's own paper). Cost guard: correct, content-corroborated
    # pre-sets skip the LLM. Exactly mirrors HF Tier-1 (requires overlap is not None AND >= 0.30): a
    # name-bearing pre-set with THIN/None overlap (e.g. an abstract-less OpenAlex DOI record, or a
    # same-name wrong-domain paper) is NOT auto-accepted -- it routes to the LLM verifier, which keeps a
    # correct same-name paper and rejects a wrong-domain one. mgsm never qualifies (GSM8K's title bears
    # no "mgsm").
    db = _distinctive_base(full_name or suite)
    name_bearing = len(db) >= NAME_TOKEN_FLOOR_LEN and _title_bears_name(title, db)
    fast_overlap = _subject_overlap(abstract, ctx["identity_subject"])
    if name_bearing and fast_overlap is not None and fast_overlap >= PAPER_TIER1_ACCEPT_OVERLAP:
        return "keep", {**base, "tier": "deterministic", "llm_verdict": None,
                        "final_action": "kept", "verdict": "confirmed",
                        "reason": f"title bears the benchmark name + subject overlap "
                                  f"{fast_overlap:.3f} (self-referential)"}

    # LLM verify (1 call): pass the fetched title/abstract so verify_paper_binding does not re-fetch.
    v = verify_paper_binding(
        suite, paper_url, full_name=full_name, overview=ctx["overview"], domain=ctx["domain"],
        sub_benchmarks=ctx["sub_benchmarks"], metrics=ctx["metrics"],
        eval_library=ctx["eval_library"], paper_title=title, paper_abstract=abstract)
    is_match = bool(v.get("is_match"))
    conf = float(v.get("confidence", 0.0) or 0.0)
    errored = bool(v.get("error"))

    if (not errored) and is_match and conf >= PAPER_LLM_ACCEPT_CONF:
        return "keep", {**base, "tier": "llm", "llm_verdict": v, "final_action": "kept",
                        "verdict": "confirmed", "reason": f"LLM match (conf={conf})"}
    # CONFIDENT REJECT: a non-errored not-a-match verdict, regardless of the numeric match-confidence.
    # `confidence` is the LLM's confidence that the candidate IS a match, so a genuine no-match returns
    # conf=0.0 (verify_paper_binding maps a "none" match_index -> is_match=False, conf=0.0, error=False).
    # Gating the reject on conf>=PAPER_LLM_ACCEPT_CONF made this branch dead for the conf=0.0 no-match,
    # so a correct, well-reasoned rejection fell through to keep-degraded. error=True (an LLM outage or
    # a repeated parse failure) is the ONLY outage signal that reaches here -- a metadata FETCH failure
    # already returned keep-degraded above -- so a non-errored verdict is a real, content-grounded
    # judgement. Drop + fall through to the verified search path, which re-resolves from scratch and
    # recovers a wrongly-dropped correct paper (and re-sources authors from the found paper).
    #
    # GUARD against a malformed verdict: only a REAL not-a-match decision rejects. A verdict that
    # carries a `match_index` key whose value is missing/null/non-integer-and-not-"none" is ambiguous
    # (a degenerate LLM/parse result that slipped past error=True), not a decision -- treat it as
    # keep-degraded, never a reject, so a correct pre-set is never dropped on a no-decision verdict.
    if (not errored) and (not is_match):
        if _verdict_has_decision(v):
            return "reject", {**base, "tier": "llm", "llm_verdict": v,
                              "final_action": "rejected_to_none", "verdict": "rejected",
                              "reason": f"LLM not-a-match (conf={conf})"}
        return "keep", {**base, "tier": "llm", "llm_verdict": v, "final_action": "kept",
                        "verdict": "unverified_degraded",
                        "reason": "LLM verdict carried no usable match decision; kept (degraded)"}

    # Hesitant same-name-domain backstop: now reached only for a non-errored is_match=True verdict
    # (the not-a-match case is rejected above). A near-disjoint abstract under a (possibly low-conf)
    # match is a same-name collision whose domain differs (the actionbench class) -> reject. The name
    # guard above already cleared a different declared name.
    if not errored:
        overlap = _subject_overlap(abstract, ctx["identity_subject"])
        if overlap is not None and overlap < PAPER_SAMENAME_OVERLAP_FLOOR:
            return "reject", {**base, "tier": "llm", "llm_verdict": v,
                              "final_action": "rejected_to_none", "verdict": "rejected",
                              "reason": f"hesitant LLM + same-name subject overlap {overlap:.3f} "
                                        f"below floor (domain collision)"}

    # LLM error, or a low-confidence non-errored match with no negative deterministic signal: keep
    # degraded (never drop a possibly-correct binding under an outage / ambiguity -- mirror of the HF
    # unverified_degraded keep).
    detail = "LLM unavailable" if errored else f"LLM hesitant (conf={conf})"
    return "keep", {**base, "tier": "llm", "llm_verdict": v, "final_action": "kept",
                    "verdict": "unverified_degraded",
                    "reason": f"{detail}; pre-set binding kept (degraded)"}


def _write_paper_binding_sidecar(state, ctx, record):
    """Write a paper-verification.json carrying the binding decision, so a pre-set/extracted paper
    binding is never silent (mirror of the HF _write_hf_verification_sidecar). Used on the KEEP paths;
    the search fall-through writes its own sidecar with the binding carried via prior_binding. Writes
    to tool_output/paper_resolver/ where the gate reader globs."""
    om = state.get("output_manager")
    if om is None:
        return
    try:
        om.save_tool_output({
            "suite": ctx["suite_name"], "full_name": ctx["full_name"], "domain": ctx["domain"],
            "binding": record,
            "resolved_url": record.get("paper_url") if record.get("final_action") == "kept" else None,
            "resolved_from": (f"binding_verified:{record.get('binding_source')}"
                              if record.get("final_action") == "kept" else None),
        }, "paper_resolver", "paper-verification.json")
    except Exception as e:  # a telemetry write must never break the worker
        logger.warning("Could not write paper-binding sidecar: %s", e)


def run_paper_resolver(state) -> Dict[str, Any]:
    """Resolve paper URL using 3-tier approach: HF README > API search > fallback.

    A paper_url bound by a NON-resolver path (an EEE/UnitXT pre-set or an HF arxiv-tag extraction) is
    VERIFIED here before it is trusted (subject-coherence verifier, parity with the paper search path
    and the HF binding verifier). On a confident reject the pre-set is dropped and the verified search
    path runs as the fall-through (or honest-thin). Behind Config.PAPER_VERIFICATION_ENABLED (default
    ON); flag OFF reproduces the pre-feature behavior byte-for-byte. Every binding writes a
    paper-verification.json sidecar so a bypass can never again be silent.
    """
    logger.info("Starting paper resolution")

    try:
        extracted_ids = state.get("extracted_ids") or {}
        hf_json = state.get("hf_json") or {}
        ctx = _paper_identity_context(state)
        output_manager = state.get("output_manager")
        output_dir = output_manager.get_tool_output_path("paper_resolver") if output_manager else None

        # The binding record from a rejected pre-set/Tier-1 paper, carried into the search fall-through
        # so the sidecar still records the dropped binding ("sidecar for every binding").
        prior_binding = None

        # --- Pre-set paper (bind points 1 & 2): EEE/UnitXT pre-set or hf_extractor arxiv tag --------
        preset_url = extracted_ids.get("paper_url")
        if preset_url:
            if not Config.PAPER_VERIFICATION_ENABLED:
                # Flag OFF: reproduce the pre-feature skip exactly.
                return {
                    "paper_resolver_attempted": True,
                    "completed": ["paper_resolver skipped (paper_url already set)"],
                }
            binding_source = "hf_extractor" if extracted_ids.get("paper_url_from_hf") == preset_url \
                else "eee_unitxt"
            action, record = _verify_paper_binding_gate(state, preset_url, binding_source, ctx)
            if action == "keep":
                _write_paper_binding_sidecar(state, ctx, record)
                return {
                    "extracted_ids": extracted_ids,
                    "paper_resolver_attempted": True,
                    "paper_verified": True,
                    "completed": [f"paper_resolver binding {record['verdict']} url={preset_url}"],
                }
            # Reject: drop the pre-set paper (and its HF-tag marker) and fall through to search.
            logger.warning("Paper binding REJECTED for %r (%s): dropping and falling through to search",
                           preset_url, record.get("reason"))
            extracted_ids = {k: v for k, v in extracted_ids.items()
                             if k not in ("paper_url", "paper_url_from_hf")}
            prior_binding = record

        # --- Tier 1: extract from HF README (no API calls) -- also verified before trusting ---------
        if Config.PAPER_VERIFICATION_ENABLED:
            readme_url = _extract_paper_from_hf(hf_json)
            if readme_url:
                action, record = _verify_paper_binding_gate(state, readme_url, "hf_readme", ctx)
                if action == "keep":
                    updated = extracted_ids.copy()
                    updated["paper_url"] = readme_url
                    _write_paper_binding_sidecar(state, ctx, record)
                    logger.info("Paper URL from HF README (verified): %s", readme_url)
                    return {
                        "extracted_ids": updated,
                        "paper_resolver_attempted": True,
                        "paper_verified": True,
                        "completed": [f"paper_resolver hf_readme {record['verdict']} url={readme_url}"],
                    }
                logger.warning("HF-README paper binding REJECTED for %r (%s): falling through to search",
                               readme_url, record.get("reason"))
                prior_binding = record
        else:
            # Flag OFF: today's behavior -- trust the HF-README url without verification.
            readme_url = _extract_paper_from_hf(hf_json)
            if readme_url:
                updated = extracted_ids.copy()
                updated["paper_url"] = readme_url
                logger.info("Paper URL from HF README: %s", readme_url)
                return {
                    "extracted_ids": updated,
                    "paper_resolver_attempted": True,
                    "completed": [f"paper_resolver hf_readme url={readme_url}"],
                }

        # --- Tier 2/3: full verified resolution (resolve_paper verifies internally) -----------------
        resolved = resolve_paper(
            suite_name=ctx["suite_name"],
            sub_benchmarks=ctx["sub_benchmarks"],
            metrics=ctx["metrics"],
            eval_library=ctx["eval_library"],
            full_name=ctx["full_name"],
            overview=ctx["overview"],
            domains=ctx["domains"],
            output_dir=output_dir,
            hf_repo_id=ctx["hf_repo_id"],
            prior_binding=prior_binding,
        )

        if resolved:
            url = resolved["url"]
            updated = extracted_ids.copy()
            updated["paper_url"] = url
            if resolved.get("abstract"):
                updated["paper_abstract"] = resolved["abstract"]
            if resolved.get("title"):
                updated["paper_title"] = resolved["title"]
            if resolved.get("year"):
                updated["paper_year"] = resolved["year"]
            if resolved.get("authors"):
                updated["paper_authors"] = resolved["authors"]
            logger.info("Paper URL resolved: %s", url)
            return {
                "extracted_ids": updated,
                "paper_resolver_attempted": True,
                "paper_verified": True,
                "completed": [f"paper_resolver resolved url={url}"],
            }

        logger.info("Paper resolution found no match for '%s'", state["query"])
        return {
            "extracted_ids": extracted_ids,
            "paper_resolver_attempted": True,
            "paper_verified": True,
            "completed": ["paper_resolver no match"],
        }

    except Exception as e:
        result = handle_error(e, "Paper resolution", state)
        result["paper_resolver_attempted"] = True
        return result


def _biorxiv_full_pdf(url: str) -> str:
    """Map a bioRxiv/medRxiv content URL to its full-text PDF (<landing>.full.pdf).

    Handles a version suffix (vN), a trailing .full, and drops any query/fragment
    that is not meaningful for the PDF endpoint.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    low = path.lower()
    if low.endswith(".pdf"):
        return url
    if low.endswith(".full"):
        path = path + ".pdf"
    else:
        path = path + ".full.pdf"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _normalize_paper_url(url: str) -> str | None:
    """Transform paper URLs into Docling-friendly forms. Returns None if not extractable."""
    lower = url.lower()
    # S2 landing pages are HTML, never pass to Docling
    if "semanticscholar.org" in lower:
        return None
    # ACL Anthology: append .pdf
    if "aclanthology.org" in lower and not lower.endswith(".pdf"):
        return url.rstrip("/") + ".pdf"
    # bioRxiv / medRxiv content pages serve their full text at <landing>.full.pdf
    if "biorxiv.org/content/" in lower or "medrxiv.org/content/" in lower:
        return _biorxiv_full_pdf(url)
    return url


def _resolve_landing_url(url: str) -> str:
    """Follow redirects (e.g. doi.org -> publisher landing) to discover the final URL.

    Bounded by a timeout: an unbounded fetch is a known batch-hang risk. Returns the
    original URL on timeout/connection error/any failure (optimistic).
    """
    try:
        resp = requests.get(url, allow_redirects=True, stream=True, timeout=10)
        final = resp.url or url
        resp.close()
        return final
    except Exception:
        return url


def _check_paper_accessible(url: str):
    """HEAD request to detect paywalls / bot-blocks before Docling.

    Returns (accessible, reason, http_status). A .pdf URL is trusted regardless of
    the HEAD content-type, since some hosts (bioRxiv/Cloudflare) serve HTML or 405
    on HEAD for a PDF that GET would deliver. reason is one of: ok, blocked, html_only.
    """
    lower = url.lower()
    is_pdf = lower.endswith(".pdf")
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10)
        status = resp.status_code
        if status in (401, 403):
            return False, "blocked", status
        if is_pdf:
            return True, "ok", status
        content_type = resp.headers.get("content-type", "")
        # HTML pages from publishers are usually paywalled or not PDF
        if "text/html" in content_type and "arxiv.org" not in lower and "openreview.net" not in lower:
            return False, "html_only", status
        return True, "ok", status
    except Exception:
        return True, "ok", None  # Optimistic fallback


def _docling_telemetry(paper_url_resolved, full_text, reason, http_status=None, url=None):
    """Build the fail-loud docling marker recorded on state and written by run_composer."""
    return {
        "paper_url_resolved": bool(paper_url_resolved),
        "full_text": bool(full_text),
        "degraded_to_abstract_only": bool(paper_url_resolved) and not full_text,
        "reason": reason,
        "http_status": http_status,
        "url": url,
    }


def _github_telemetry(repo_url_resolved, full_text, reason, http_status=None, url=None):
    """Build the fail-loud github-readme marker recorded on state and written by run_composer."""
    return {
        "repo_url_resolved": bool(repo_url_resolved),
        "full_text": bool(full_text),
        "degraded_to_no_content": bool(repo_url_resolved) and not full_text,
        "reason": reason,
        "http_status": http_status,
        "url": url,
    }


def run_docling(state):
    """Extract paper content using Docling."""
    paper_url = state.get("extracted_ids", {}).get("paper_url")
    if paper_url:
        logger.info("Starting paper extraction")

    if not paper_url:
        return {
            "docling_output": None,
            "docling_telemetry": _docling_telemetry(False, False, "no_paper_url"),
            "completed": ["docling skipped (no paper_url)"],
        }

    # Normalize URL and check accessibility before Docling
    normalized = _normalize_paper_url(paper_url)
    if not normalized:
        logger.info("Skipping Docling for non-extractable URL: %s", paper_url)
        # Emit "docling extraction failed" so the orchestrator's _failed() check
        # marks docling as done — otherwise the conditional re-routes here forever.
        return {
            "docling_output": None,
            "docling_telemetry": _docling_telemetry(True, False, "not_extractable", url=paper_url),
            "completed": ["docling extraction failed"],
        }

    # DOI / publisher landing URLs don't reveal the host until the redirect is
    # followed (doi.org/10.1101/... -> biorxiv landing). Resolve once, then
    # re-normalize so the bioRxiv/medRxiv .full.pdf rule can fire on the host.
    low = normalized.lower()
    if "doi.org" in low and not low.endswith(".pdf"):
        resolved = _resolve_landing_url(normalized)
        if resolved and resolved != normalized:
            normalized = _normalize_paper_url(resolved) or resolved

    accessible, access_reason, http_status = _check_paper_accessible(normalized)
    if not accessible:
        logger.info("Paper URL not accessible (%s): %s", access_reason, normalized)
        # Promote HTML-only paper URLs (e.g. bioRxiv DOI landing pages) to
        # extracted_ids.website_url so the html_worker picks them up — otherwise
        # we lose the only known benchmark source for these cards.
        ids = dict(state.get("extracted_ids") or {})
        if not ids.get("website_url"):
            ids["website_url"] = normalized
        reason = "fetch_blocked" if access_reason == "blocked" else "html_only"
        return {
            "docling_output": None,
            "extracted_ids": ids,
            "docling_telemetry": _docling_telemetry(True, False, reason, http_status, normalized),
            "completed": ["docling extraction failed"],
        }

    paper_url = normalized

    try:
        logger.info("Extracting paper from: %s", paper_url)

        docling_result = extract_paper_with_docling.func(paper_url=paper_url)

        if docling_result.get("success"):
            logger.info("Docling extraction completed successfully")
            metadata = docling_result.get("metadata", {})
            title = metadata.get("title", "Unknown Paper")
            char_count = len(docling_result.get("filtered_text", ""))
            logger.info(f"Paper: {title} ({char_count:,} chars)")

            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            output_file = state["output_manager"].save_tool_output(
                docling_result, "docling", filename
            )
            logger.info("Docling output saved to: %s", output_file)

            full_text = bool((docling_result.get("filtered_text") or "").strip())
            return {
                "docling_output": docling_result,
                "docling_telemetry": _docling_telemetry(
                    True, full_text, "ok" if full_text else "empty_body", http_status, paper_url
                ),
                "completed": ["docling done"],
            }
        else:
            warning_msg = docling_result.get("warning")
            if warning_msg:
                logger.warning("Docling warning: %s", warning_msg)
                return {
                    "docling_output": None,
                    "docling_telemetry": _docling_telemetry(
                        True, False, "extraction_failed", http_status, paper_url
                    ),
                    "completed": ["docling extraction failed"],
                }
            else:
                error_msg = (
                    f"Docling extraction failed: {docling_result.get('error', 'Unknown error')}"
                )
                result = record_skip(error_msg, "Docling extraction", state)
                result["docling_output"] = None
                result["docling_telemetry"] = _docling_telemetry(
                    True, False, "extraction_failed", http_status, paper_url
                )
                return result

    except Exception as e:
        result = handle_error(e, "Docling extraction", state)
        result["docling_output"] = None
        result["docling_telemetry"] = _docling_telemetry(
            True, False, "extraction_failed", url=paper_url
        )
        return result


def _is_html_url(url: str) -> bool:
    """Check if a URL is likely a web page (not PDF/arxiv)."""
    lower = url.lower()
    if "arxiv.org" in lower or lower.endswith(".pdf"):
        return False
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return False
    return True


def _rank_html_url(url: str) -> int:
    """Lower rank = try first. Benchmark-specific pages beat model-specific or API URLs."""
    lower = url.lower()
    # Strongly demote llm-stats model pages — they describe the model, not the benchmark
    if "llm-stats" in lower and "/models/" in lower:
        return 100
    # Demote bare API endpoints (no human-readable content)
    if "api." in lower or "/api/" in lower:
        return 50
    # Prefer llm-stats benchmark pages, project sites, blog posts
    if "/benchmarks/" in lower or "/benchmark/" in lower:
        return 0
    return 10


def run_html_extractor(state):
    """Extract content from web pages using trafilatura."""
    try:
        urls = []

        # Collect candidate URLs from EEE metadata
        eee_meta = state.get("eee_metadata") or {}
        for url in eee_meta.get("source_urls", []):
            if _is_html_url(url):
                urls.append(url)

        # Check extracted_ids for website_url
        ids = state.get("extracted_ids") or {}
        website_url = ids.get("website_url")
        if website_url and _is_html_url(website_url):
            urls.insert(0, website_url)

        # Rank URLs: benchmark pages first, model pages last.
        # Stable sort by rank preserves original order within each tier.
        urls = sorted(urls, key=_rank_html_url)

        if not urls:
            logger.info("No HTML URLs to extract")
            return {
                "html_content": {"success": False, "text": "", "url": "", "title": ""},
                "completed": ["html skipped (no web URLs)"],
            }

        # Try the first viable URL
        for url in urls[:3]:
            try:
                logger.info("Extracting HTML from: %s", url)
                result = extract_html_content.func(url=url)

                if result.get("success"):
                    char_count = len(result.get("text", ""))
                    logger.info("HTML extraction OK: %d chars from %s", char_count, url)

                    filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
                    state["output_manager"].save_tool_output(result, "html", filename)

                    return {
                        "html_content": result,
                        "completed": ["html done"],
                    }
                else:
                    logger.debug("HTML extraction failed for %s: %s", url, result.get("error"))

            except Exception as e:
                logger.debug("HTML extraction error for %s: %s", url, e)

        return {
            "html_content": {"success": False, "text": "", "url": urls[0], "title": ""},
            "completed": ["html extraction failed"],
        }

    except Exception as e:
        return handle_error(e, "HTML extraction", state)


def run_github_readme(state):
    """Fetch a repo README so source-starved cards have real text to ground on.

    Derives the repo from structured links and free text (the resolved paper abstract,
    the html page), since some benchmarks name their repo only in prose. Fail-loud
    telemetry mirrors docling; the fetch is bounded by a hard timeout.
    """
    try:
        ids = state.get("extracted_ids") or {}
        eee_meta = state.get("eee_metadata") or {}
        card_data = (state.get("hf_json") or {}).get("card_data") or {}
        html_content = state.get("html_content") or {}

        # Structured links first, then free-text carriers (abstract, page text)
        candidates = [
            ids.get("paper_url"),
            ids.get("website_url"),
            card_data.get("repository"),
            card_data.get("homepage"),
            ids.get("paper_abstract"),
            html_content.get("text"),
        ]
        candidates.extend(eee_meta.get("source_urls") or [])

        repo_url = derive_github_repo(*[c for c in candidates if c])
        if not repo_url:
            logger.info("No GitHub repo derivable for '%s'", state["query"])
            return {
                "github_readme": {"success": False, "text": "", "url": ""},
                "github_telemetry": _github_telemetry(False, False, "no_github_repo"),
                "completed": ["github skipped (no repo)"],
            }

        logger.info("Fetching GitHub README from: %s", repo_url)
        result = fetch_github_readme(repo_url)
        status = result.get("http_status")
        err = result.get("error") or ""

        if result.get("success"):
            char_count = len(result.get("text", ""))
            logger.info("GitHub README OK: %d chars from %s", char_count, repo_url)
            filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
            state["output_manager"].save_tool_output(result, "github", filename)
            return {
                "github_readme": result,
                "github_telemetry": _github_telemetry(True, True, "ok", status, repo_url),
                "completed": ["github done"],
            }

        if status in (401, 403):
            reason = "fetch_blocked"
        elif status == 404 or err == "empty_readme":
            reason = "no_readme"
        elif err == "timeout":
            reason = "timeout"
        else:
            reason = "extraction_failed"
        logger.info("GitHub README not fetched (%s): %s", reason, repo_url)
        return {
            "github_readme": {"success": False, "text": "", "url": repo_url},
            "github_telemetry": _github_telemetry(True, False, reason, status, repo_url),
            "completed": ["github extraction failed"],
        }

    except Exception as e:
        return handle_error(e, "github extraction", state)


# Resolve-time HF verifier thresholds (named, not magic). The ACCEPT floor gates only the
# zero-LLM Tier-1 accept; a correct-but-low-overlap repo (e.g. activitynet ~0.095) is not rejected
# by it, it routes to the LLM. The LLM floors mirror the paper verifier's ~0.7.
HF_TIER1_ACCEPT_OVERLAP = 0.30   # Tier-1 deterministic-accept subject-overlap floor
HF_LLM_ACCEPT_CONF = 0.7         # Tier-2 LLM confidence floor for BOTH a confident accept and a
                                 # confident reject; below it the verdict is "weak/hesitant" and the
                                 # authority policy (corroborated/EEE keep-degraded vs generic reject)
                                 # decides. (A single floor replaces the old separate reject floor.)

# LLM JSON is parsed with json.loads, which preserves the model's raw types. A model that emits a
# QUOTED boolean ("is_match": "false"), a PROSE verdict ("is_match": "no match"), or a string
# confidence ("confidence": "0.9") would otherwise (a) read as truthy -> silently KEEP a wrong match,
# or (b) raise on the >= comparison -> get masked into a keep by run_hf's outer guard. Coerce both,
# mirroring how resolve_paper type-guards its match_index. A string is a match ONLY if it is an
# explicit truthy token: a POSITIVE whitelist, so any other string (including prose like "no match"
# or "unsure") reads as NOT a match (route to the precision policy, never a silent keep).
_TRUTHY_MATCH_STRINGS = {"true", "yes", "y", "t", "1", "match"}


def _coerce_match(value) -> bool:
    """True only for an explicit match signal. A real bool is itself; a string matches ONLY if it is
    a known truthy token (case-insensitive whitelist) -- every other string (prose, "no match",
    "unsure", "false") is False; a number is True only if nonzero; anything else (None, list) is
    False. Conservative direction is always False so an ambiguous verdict never silently keeps."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_MATCH_STRINGS
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _coerce_conf(value) -> float:
    """LLM confidence as a float in [0, 1]-ish; 0.0 when missing or unparseable (a string like
    'high' must not raise on the threshold comparison)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# A leading YAML frontmatter block, tolerant of the closing fence ending at EOF with NO trailing
# newline (`---\nlicense: mit\n---`). The composer's _readme_lead frontmatter strip requires a
# trailing `\n` after the closing fence, so a fence-at-EOF README would otherwise leave "license: mit"
# as apparent content; this predicate strips it itself so a frontmatter-only placeholder is contentless.
# Bounded to a SINGLE block -- the body may not contain a `---` line of its own -- so a content-bearing
# README that merely ENDS with a `---` horizontal rule is not swallowed to EOF (re.S non-greedy would
# otherwise expand the body to the LAST `---`). The `(?:(?!^---$).)*?` body excludes a lone-`---` line.
_FRONTMATTER_EOF_RE = re.compile(r"^\s*---\n(?:(?!^---$).)*?\n---\s*$", re.S | re.M)


def _hf_repo_is_contentless(hf_data) -> bool:
    """True when a bound HF repo carries no usable descriptive content to corroborate the binding:
    its README is empty or only YAML frontmatter (e.g. '---\\nlicense: mit\\n---', with or without a
    trailing newline) AND its card_data has neither a description nor a pretty_name. Such a placeholder
    repo can corroborate nothing, so the authority-keep policy must not rescue it (its license/tags
    would leak). Reuses the same README extractor the verifier uses; keyed on empty CONTENT, never on a
    name or on subject-overlap (a thin EEE identity also yields None overlap on a content-bearing repo)."""
    if not isinstance(hf_data, dict):
        return True
    readme = _get_hf_readme(hf_data)
    if isinstance(readme, str):
        # Strip a frontmatter block whose closing fence may sit at EOF (no trailing newline) before
        # the lead check, so such a placeholder is not mistaken for content.
        readme = _FRONTMATTER_EOF_RE.sub("", readme)
    if _readme_lead(readme).strip():
        return False
    card_data = hf_data.get("card_data") or {}
    if isinstance(card_data, dict):
        for _k in ("description", "pretty_name"):
            val = card_data.get(_k)
            if isinstance(val, str) and val.strip():
                return False
    return True


def _write_hf_verification_sidecar(state, payload):
    """Persist the per-card HF match-verification record (never silent). Mirrors the
    paper-verification.json sidecar; descriptive telemetry, read by the gate."""
    om = state.get("output_manager")
    if om is None:
        return
    try:
        om.save_tool_output(payload, "hf_verifier", f"hf-verification{Config.JSON_EXTENSION}")
    except Exception as e:  # a telemetry write must never break the worker
        logger.warning("Could not write hf-verification sidecar: %s", e)


def _verify_hf_binding(state, hf_data):
    """3-tier resolve-time verification of the mechanically-bound HF repo (run_hf helper).

    Returns (action, payload) where action is "keep" or "reject" and payload is the sidecar record.
    Tier 0 deterministic reject (aggregate_repo), Tier 1 deterministic accept (corroborated + coined
    + subject-overlap >= floor), Tier 2 LLM verify for everything else. Reuses the SAME deterministic
    helpers the downstream backstops use (no reimplementation). Keyed on structure/semantics, never a
    benchmark-name list.
    """
    eee_metadata = state.get("eee_metadata") or {}
    extracted_ids = state.get("extracted_ids") or {}
    repo_id = hf_data.get("id") or state.get("hf_repo")
    name = eee_metadata.get("benchmark_name") or state.get("query") or ""

    # repo_source: eee-provided vs search -> gate. Telemetry-only, cannot change the verify decision.
    existing_hf_repo = eee_metadata.get("existing_hf_repo")
    if existing_hf_repo and repo_id == existing_hf_repo:
        repo_source = "eee"
    else:
        repo_source = "search"

    base = {"benchmark": name, "repo_id": repo_id, "repo_source": repo_source}

    # Deterministic signals (shared with the downstream corroboration / subject backstops).
    corrob = _repo_corroboration(repo_id, name)            # name-only (no arxiv args here)
    namelike = _token_is_namelike(name)                    # ORIGINAL-case name (Correction 2)
    overlap = _subject_overlap(
        _get_hf_readme(hf_data), _eee_identity_subject(eee_metadata))
    deterministic = {
        "name_corroboration": corrob,
        "subject_overlap": overlap,
        "name_class": "coined" if namelike else "generic",
    }

    # Tier 0 -- deterministic REJECT (0 LLM): the basename bears no whole-token run of the name.
    # EXEMPT repo_source=='eee': an EEE-provided repo is the authoritative dataset the eval actually
    # ran on, and the SEARCH path can never produce aggregate_repo (its matcher guarantees an exact-
    # norm or contiguous-token-run basename), while true aggregates (OpenEval) are already filtered
    # upstream by _is_aggregate_source_repo. So Tier-0's aggregate_repo reject only ever bites a
    # CORRECT EEE abbreviation (BIG-Bench Hard -> bbh, GSM8K -> grade-school-math, MMLU -> cais/mmlu).
    # Route those to Tier-2 so the LLM verifies the README against the full name instead.
    if corrob == "aggregate_repo" and repo_source != "eee":
        return "reject", {**base, "deterministic": deterministic, "tier": "reject",
                          "llm_verdict": None, "final_action": "rejected_to_none",
                          "verdict": "rejected", "reason": "aggregate_repo (name not a token run)"}

    # Tier 1 -- deterministic ACCEPT (0 LLM): corroborated AND coined AND positive subject evidence.
    # All three required (corroborated + coined alone would auto-accept the actionbench class).
    if (corrob is None and namelike
            and overlap is not None and overlap >= HF_TIER1_ACCEPT_OVERLAP):
        return "keep", {**base, "deterministic": deterministic, "tier": "accept",
                        "llm_verdict": None, "final_action": "kept", "verdict": "confirmed",
                        "reason": f"corroborated coined name, subject overlap {overlap:.3f}"}

    # Tier 2 -- LLM verify (1 call). Everything else: derivative, generic name, thin/low overlap.
    identity = {
        "suite_name": name,
        "full_name": extracted_ids.get("full_name"),
        "subject_text": _eee_identity_subject(eee_metadata) or extracted_ids.get("overview"),
        "metrics": list((eee_metadata.get("metrics") or {}).keys()),
        "eval_library": eee_metadata.get("eval_library"),
    }
    card_data = hf_data.get("card_data") or {}
    candidate = {
        "repo_id": repo_id,
        "readme_lead": _readme_lead(_get_hf_readme(hf_data), n=1500),
        "pretty_name": card_data.get("pretty_name"),
        "description": card_data.get("description"),
        "task_categories": card_data.get("task_categories") or [],
        "tags": hf_data.get("tags") or card_data.get("tags") or [],
    }
    v = verify_hf_match(identity, candidate)
    is_match = _coerce_match(v.get("is_match"))
    conf = _coerce_conf(v.get("confidence"))
    errored = bool(v.get("error"))
    # Authority/corroboration raises the bar to OVERRIDE the mechanical binding: a corroborated coined
    # repo OR an EEE-provided repo (the authoritative dataset the eval ran on) is only dropped on a
    # CONFIDENT not-a-match; a hesitant/low-confidence/errored verdict keeps it degraded. A generic,
    # uncorroborated, non-EEE repo requires positive evidence and is dropped on anything weak.
    authoritative = (corrob is None and namelike) or repo_source == "eee"

    # Confident accept: explicit match at/above the accept floor.
    if (not errored) and is_match and conf >= HF_LLM_ACCEPT_CONF:
        return "keep", {**base, "deterministic": deterministic, "tier": "llm", "llm_verdict": v,
                        "final_action": "kept", "verdict": "confirmed",
                        "reason": f"LLM match (conf={conf})"}

    # Confident not-a-match: explicit reject at/above the same floor (actionbench: conf 0.95 -> reject
    # even though it is corroborated+coined; a confident domain rejection overrides authority).
    if (not errored) and (not is_match) and conf >= HF_LLM_ACCEPT_CONF:
        return "reject", {**base, "deterministic": deterministic, "tier": "llm", "llm_verdict": v,
                          "final_action": "rejected_to_none", "verdict": "rejected",
                          "reason": f"LLM not-a-match (conf={conf})"}

    # Otherwise: hesitant not-a-match (<0.7), low-confidence accept, or LLM error.
    detail = "LLM unavailable" if errored else f"LLM weak/hesitant verdict (conf={conf})"
    if authoritative:
        # An empty/contentless repo can corroborate nothing, so the authority-keep policy must not
        # rescue it even when it is corroborated+coined or EEE-provided: a placeholder repo with no
        # README and no card description (e.g. Ross12/Acadreason, README == only YAML frontmatter)
        # only leaks its license/tags into the card. Reject -> honest-thin. Keyed on empty CONTENT
        # (the README strips to nothing AND card_data carries no description/pretty_name), never on
        # subject-overlap being None (which a thin EEE identity also produces) and never on a name.
        if _hf_repo_is_contentless(hf_data):
            return "reject", {**base, "deterministic": deterministic, "tier": "llm", "llm_verdict": v,
                              "final_action": "rejected_to_none", "verdict": "rejected",
                              "reason": f"{detail}; empty/contentless repo cannot corroborate"}
        return "keep", {**base, "deterministic": deterministic, "tier": "llm", "llm_verdict": v,
                        "final_action": "kept", "verdict": "unverified_degraded",
                        "reason": f"{detail}; corroborated/EEE-provided repo kept (degraded)"}
    return "reject", {**base, "deterministic": deterministic, "tier": "llm", "llm_verdict": v,
                      "final_action": "rejected_to_none", "verdict": "rejected",
                      "reason": f"{detail}; uncorroborated/generic repo rejected"}


def run_hf(state):
    """Fetch HuggingFace dataset metadata, then verify the mechanically-bound repo is this
    benchmark's dataset (resolve-time HF-match verification, parity with the paper verifier).

    A confidently-wrong match is rejected to None (honest-thin): hf_repo is cleared and hf_rejected
    set so the orchestrator advances instead of looping, and the wrong repo's metadata never reaches
    the composer or _extract_paper_from_hf. Behind Config.HF_VERIFICATION_ENABLED (default ON);
    flag OFF reproduces the pre-feature behavior byte-for-byte.
    """
    if not state["hf_repo"]:
        return record_skip("No hf_repo available", "HuggingFace lookup", state)

    try:
        hf_data = hf_dataset_metadata.func(repo_id=state["hf_repo"])

        logger.info("HuggingFace metadata retrieved successfully")
        card_data = hf_data.get("card_data") or {}
        dataset_info = hf_data.get("dataset_info") or {}
        name = dataset_info.get("dataset_name", card_data.get("pretty_name", "Unknown"))
        task_cats = card_data.get("task_categories", [])
        if task_cats:
            task_preview = ", ".join(task_cats[:2])
            logger.info(f"Dataset: {name} ({task_preview})")

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(hf_data, "hf", filename)
        logger.info("HuggingFace output saved to: %s", output_file)

        if not Config.HF_VERIFICATION_ENABLED:
            return {
                "hf_json": hf_data,
                "completed": ["hf done"],
            }

        try:
            action, payload = _verify_hf_binding(state, hf_data)
        except Exception as e:
            # A verifier bug must never crash the worker or lose a card: keep today's behavior.
            logger.warning("HF verification errored; keeping binding %r: %s", state["hf_repo"], e)
            _write_hf_verification_sidecar(state, {
                "benchmark": (state.get("eee_metadata") or {}).get("benchmark_name") or state.get("query"),
                "repo_id": hf_data.get("id") or state.get("hf_repo"),
                "tier": "error", "final_action": "kept", "verdict": "verifier_error",
                "reason": f"verifier exception: {e}"})
            return {"hf_json": hf_data, "completed": ["hf done"]}

        _write_hf_verification_sidecar(state, payload)

        if action == "reject":
            logger.warning("HF match REJECTED for %r (%s): clearing binding to honest-thin",
                           state["hf_repo"], payload.get("reason"))
            # Also drop the rejected repo from extracted_ids so the composer cannot derive any
            # display/identity field from it: extracted_ids["hf_repo"] (set before run_hf) otherwise
            # still feeds benchmark_details.hf_url (a resources[hf] link) and the org_url / fallback
            # org-logo in apply_display_fields. Clear ONLY hf_repo; every other extracted id
            # (paper_url, paper_url_from_hf, risk_tags, abstract/title/year/authors, ...) is preserved
            # so the paper path -- which never reads extracted_ids["hf_repo"] -- is unaffected.
            cleared_ids = dict(state.get("extracted_ids") or {})
            cleared_ids["hf_repo"] = None
            return {
                "hf_repo": None,
                "hf_rejected": True,
                "extracted_ids": cleared_ids,
                "completed": ["hf done"],
            }

        return {
            "hf_json": hf_data,
            "completed": ["hf done"],
        }
    except Exception as e:
        return handle_error(e, "HuggingFace lookup", state)


def _write_docling_telemetry(state):
    """Persist the fail-loud docling marker to its own sidecar and warn when degraded.

    Uses the marker run_docling left on state; if docling never ran (no paper_url),
    records the honest no-paper case. A separate file avoids colliding with the
    provenance sidecar writes.
    """
    om = state.get("output_manager")
    if om is None:
        return
    tel = state.get("docling_telemetry")
    if not tel:
        paper_url = (state.get("extracted_ids") or {}).get("paper_url")
        docling_output = state.get("docling_output")
        full_text = bool(
            docling_output
            and docling_output.get("success")
            and (docling_output.get("filtered_text") or "").strip()
        )
        reason = "ok" if full_text else ("no_paper_url" if not paper_url else "extraction_failed")
        tel = _docling_telemetry(bool(paper_url), full_text, reason, url=paper_url)
    filename = f"docling_telemetry_{sanitize_benchmark_name(state['query'])}.json"
    om.save_tool_output(tel, "composer", filename)
    if tel.get("degraded_to_abstract_only"):
        logger.warning(
            "DOCLING_DEGRADED query=%s reason=%s status=%s url=%s",
            state["query"], tel.get("reason"), tel.get("http_status"), tel.get("url"),
        )


def _write_github_telemetry(state):
    """Persist the fail-loud github-readme marker to its own sidecar and warn when degraded.

    Uses the marker run_github_readme left on state; if it never ran, records the honest
    no-repo case. A separate file mirrors the docling telemetry sidecar.
    """
    om = state.get("output_manager")
    if om is None:
        return
    tel = state.get("github_telemetry")
    if not tel:
        gh = state.get("github_readme")
        full_text = bool(gh and gh.get("success") and (gh.get("text") or "").strip())
        tel = _github_telemetry(False, full_text, "ok" if full_text else "no_github_repo")
    filename = f"github_telemetry_{sanitize_benchmark_name(state['query'])}.json"
    om.save_tool_output(tel, "composer", filename)
    if tel.get("degraded_to_no_content"):
        logger.warning(
            "GITHUB_DEGRADED query=%s reason=%s status=%s url=%s",
            state["query"], tel.get("reason"), tel.get("http_status"), tel.get("url"),
        )


def run_composer(state):
    """Compose a benchmark card from all collected metadata using an LLM."""
    logger.info("Starting benchmark card composition")

    try:
        # Extract just the benchmark name from catalog queries (after the last dot)
        query_for_composer = state["query"]
        if state.get("catalog_path") and "." in state["query"]:
            query_for_composer = state["query"].split(".")[-1]

        unitxt_for_composer = state.get("unitxt_json") if not state.get("eee_metadata") else None

        result = compose_benchmark_card.func(
            unitxt_metadata=unitxt_for_composer,
            hf_metadata=state.get("hf_json"),
            extracted_ids=state.get("extracted_ids", {}),
            docling_output=state.get("docling_output"),
            query=query_for_composer,
            eee_metadata=state.get("eee_metadata"),
            html_content=state.get("html_content"),
            github_readme=state.get("github_readme"),
        )

        logger.info("Successfully composed benchmark card")

        if result.get("provenance"):
            provenance_filename = f"provenance_{sanitize_benchmark_name(state['query'])}.json"
            provenance_output = state["output_manager"].save_tool_output(
                result["provenance"], "composer", provenance_filename
            )
            logger.info(f"Provenance tracking saved to: {provenance_output}")

        # Persist the composer's run telemetry (quote-verify counters, per-group validator
        # telemetry, aboutness). Additive sidecar only, never part of the card.
        if result.get("composition_metadata"):
            metadata_filename = (
                f"composition_metadata_{sanitize_benchmark_name(state['query'])}.json")
            state["output_manager"].save_tool_output(
                result["composition_metadata"], "composer", metadata_filename)
            logger.info("Composition metadata saved to: composer/%s", metadata_filename)

        _write_docling_telemetry(state)
        _write_github_telemetry(state)

        benchmark_card = result.get("benchmark_card", {})
        details = benchmark_card.get("benchmark_details", {})
        name = details.get("name", "N/A")
        domains = details.get("domains", [])
        languages = details.get("languages", [])

        domain_str = ", ".join(domains[:2]) if domains else "General"
        lang_str = ", ".join(languages[:2]) if languages else "Unknown"
        logger.info(f"Card: {name} | {domain_str} | {lang_str}")

        return {
            "composed_card": result,
            "completed": ["composer done"],
        }
    except Exception as e:
        return handle_error(e, "Composer", state)


def run_risk_identification(state):
    """Identify risks using AI Atlas Nexus and integrate them into the card."""
    logger.info("Starting risk identification")

    if not state.get("composed_card"):
        return record_skip("No composed card available", "Risk identification", state)

    try:
        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        risk_enhanced_card = identify_and_integrate_risks(benchmark_card)
        risk_enhanced_card = apply_single_judge_bias_rule(risk_enhanced_card)
        # A5 presentation hygiene: drop empty url keys (e.g. the synthetic single-judge risk).
        risk_enhanced_card = omit_empty_risk_urls(risk_enhanced_card)

        filename = f"{sanitize_benchmark_name(state['query'])}{Config.JSON_EXTENSION}"
        output_file = state["output_manager"].save_tool_output(
            {"benchmark_card": risk_enhanced_card}, "risk_enhanced", filename
        )
        logger.info("Risk-enhanced card saved to: %s", output_file)

        possible_risks = risk_enhanced_card.get("possible_risks", [])
        if possible_risks:
            risk_output = {
                "benchmark": state["query"],
                "risks_identified": len(possible_risks),
                "taxonomy": "ibm-risk-atlas",
                "risks": possible_risks,
            }
            risk_filename = f"risks_{sanitize_benchmark_name(state['query'])}.json"
            risk_output_file = state["output_manager"].save_tool_output(
                risk_output, "ai_atlas_nexus", risk_filename
            )
            logger.info("Risk identification results saved to: %s", risk_output_file)

        logger.info("Risk identification completed")
        possible_risks = risk_enhanced_card.get("possible_risks", [])

        if possible_risks:
            risk_count = len(possible_risks)
            risk_names = [
                risk.get("category", risk.get("name", "Unknown")) for risk in possible_risks[:2]
            ]
            risk_preview = ", ".join(risk_names)
            if risk_count > 2:
                risk_preview += f" (+{risk_count-2} more)"
            logger.info(f"Risks: {risk_preview}")
        else:
            logger.info("Risks: None detected")

        return {
            "risk_enhanced_card": {"benchmark_card": risk_enhanced_card},
            "completed": ["risk identification done"],
        }

    except Exception as e:
        return handle_error(e, "Risk identification", state)


def run_rag(state) -> Dict[str, Any]:
    """Atomize the benchmark card and retrieve supporting evidence via RAG."""
    logger.info("Starting RAG processing")

    if not state.get("composed_card"):
        return record_skip("No composed card available", "RAG processing", state)

    retriever = None
    try:
        benchmark_name = sanitize_benchmark_name(state["query"])
        # Unique per-benchmark collection so a later benchmark can't retrieve an
        # earlier one's chunks via chromadb's process-shared default collection.
        # Use the chromadb-name sanitizer so any benchmark name yields a valid name.
        collection_name = sanitize_chroma_collection_name(state["query"], suffix=uuid4().hex[:8])

        unitxt_data = state.get("unitxt_json") or {}
        hf_data = state.get("hf_json") or {}
        docling_data = state.get("docling_output")
        html_data = state.get("html_content")
        github_data = state.get("github_readme")

        indexer = MetadataIndexer()
        documents = indexer.create_documents(
            unitxt_data, hf_data, state["query"], docling_data, html_data, github_data
        )

        from auto_benchmarkcard.config import get_llm_handler

        try:
            llm = get_llm_handler(Config.COMPOSER_MODEL)
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=Config.ENABLE_LLM_RERANKING,
                enable_hybrid_search=Config.ENABLE_HYBRID_SEARCH,
                enable_query_expansion=Config.ENABLE_QUERY_EXPANSION,
                llm_handler=llm,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.warning("Enhanced retriever failed, using basic fallback: %s", e)
            retriever = RAGRetriever(
                embedding_model=Config.DEFAULT_EMBEDDING_MODEL,
                enable_llm_reranking=False,
                enable_hybrid_search=False,
                enable_query_expansion=False,
                collection_name=collection_name,
            )

        if not documents:
            return record_skip("No source documents available for evidence retrieval", "RAG processing", state)

        retriever.index_documents(documents)

        benchmark_card = state["composed_card"]
        if "benchmark_card" in benchmark_card:
            benchmark_card = benchmark_card["benchmark_card"]

        statements = atomize_benchmark_card(
            benchmark_card, "all",
            engine_type=Config.LLM_ENGINE_TYPE,
            model_name=Config.COMPOSER_MODEL,
        )

        statement_texts = []
        for statement_obj in statements:
            if isinstance(statement_obj, str):
                statement_texts.append(statement_obj)
            else:
                statement_texts.append(statement_obj.get("text", ""))

        logger.debug("Processing %d statements for evidence retrieval", len(statement_texts))

        if retriever.enable_llm_reranking and retriever.llm_handler:
            import asyncio

            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                logger.warning("nest_asyncio not installed, async reranking may fail in nested event loops")

            try:
                batch_chunks = asyncio.run(
                    retriever.retrieve_for_statements_batch_parallel(statement_texts)
                )
            except RuntimeError:
                logger.warning("Async reranking failed (event loop conflict), falling back to sync")
                batch_chunks = retriever.retrieve_for_statements_batch(statement_texts)
        else:
            batch_chunks = retriever.retrieve_for_statements_batch(statement_texts)

        results = []
        for statement_obj, chunks in zip(statements, batch_chunks):
            results.append({"statement": statement_obj, "retrieved_chunks": chunks})

        raw_results = {
            "benchmark": state["query"],
            "num_statements": len(statements),
            "num_documents_indexed": len(documents),
            "results": results,
        }

        formatted_results = convert_rag_to_required_format(raw_results, "all", benchmark_card)

        rag_filename = f"formatted_rag_results_{benchmark_name}.jsonl"
        rag_tool_dir = state["output_manager"].get_tool_output_path("rag")
        output_path = os.path.join(rag_tool_dir, rag_filename)
        save_formatted_results(formatted_results, output_path)

        logger.info("RAG processing completed")
        atom_count = len(formatted_results.get("atoms", []))
        context_count = len(formatted_results.get("contexts", []))
        logger.info(f"RAG: {atom_count} claims, {context_count} evidence sources")
        logger.info("RAG results saved to: %s", output_path)

        return {
            "rag_results": formatted_results,
            "completed": ["rag done"],
        }

    except Exception as e:
        return handle_error(e, "RAG processing", state)
    finally:
        if retriever is not None:
            retriever.cleanup()


# Presentation-layer flag policy, applied downstream of the frozen FactReasoner flagging
# and the gold-set flag-F1 instrument. Production cards only: the reapply self-test runs
# its own flag reimplementation (scripts/revalidate_gold_set), so these never affect
# byte-identity. General and benchmark-agnostic.
PRESENTATION_EXCLUDED_FLAGS = frozenset({
    "benchmark_details.resources",  # deterministically assembled URLs (extraction artifact)
    "data.annotation",              # verbatim-extracted, not composed
})
SOURCE_BLOCKED_FLAG_REASON = "[Unverified — limited/blocked source]"
_SOURCE_BLOCKED_MIN_CHARS = 500


def _source_coverage_blocked(state, source_text) -> bool:
    """True when the primary source was too thin or was blocked, so a neutral-escalated
    'no supporting evidence' flag cannot tell a fabrication from an unverifiable-but-real
    value. Signals: escalation source under _SOURCE_BLOCKED_MIN_CHARS chars, or a degraded
    docling fetch (paywall / Cloudflare 403 -> abstract-only or no content)."""
    if len(source_text) < _SOURCE_BLOCKED_MIN_CHARS:
        return True
    tel = state.get("docling_telemetry")
    if isinstance(tel, dict) and (
        tel.get("degraded_to_abstract_only") or tel.get("degraded_to_no_content")
    ):
        return True
    return False


def _apply_presentation_flag_policy(flagged_fields, source_blocked):
    """Display-layer flag policy on a card's flagged_fields (mutates in place).

    C2: drop extraction-artifact / verbatim-extracted fields (resources, annotation) from
    the displayed flags, preserving any schema_invalid residual so the gate's parse-failure
    metric still sees it. C1: on a source-blocked card relabel neutral "no supporting
    evidence" hallucination flags as unverified, leaving contradiction (low-factuality) and
    schema_invalid reasons intact. Production-only and benchmark-agnostic: the gold-set
    instruments run a separate flag reimplementation, so byte-identity is untouched."""
    for path in PRESENTATION_EXCLUDED_FLAGS:
        reason = flagged_fields.get(path)
        if reason is not None and "schema_invalid" not in reason.lower():
            del flagged_fields[path]
    if source_blocked:
        for path, reason in list(flagged_fields.items()):
            if "Hallucination" in reason:
                flagged_fields[path] = SOURCE_BLOCKED_FLAG_REASON


def _first_arxiv_tag(hf_json: Dict[str, Any]) -> str | None:
    """First arxiv id in the HF sidecar's TOP-LEVEL tags, for repo corroboration only.

    The arxiv tag lives in the top-level `tags` list on these sidecars, not `card_data.tags`.
    Read here ONLY to corroborate the bound repo against the resolved paper -- the
    paper-from-HF anchor deliberately keeps preferring the README Paper: link
    (_extract_paper_from_hf is unchanged), so a wrong top-level tag never becomes the paper."""
    for tag in (hf_json.get("tags") or []):
        if isinstance(tag, str) and tag.startswith("arxiv:"):
            arxiv_id = tag.split(":", 1)[1].strip()
            if arxiv_id:
                return arxiv_id
    return None


def _drop_uncorroborated_hf(card, hf_overrides, reason):
    """Neutralize HF-derived identity when the bound repo failed corroboration.

    Clears the override set in place (so apply_deterministic_overrides and its flag-clearing
    are skipped) and resets the repo-org display fields logo/org_url to the NS sentinel. A
    wrong/derivative/aggregate repo poisons every tag-derived field, so prefer NS over shipping
    the wrong entity's metadata. No-op when reason is falsy (repo corroborated); never fabricates."""
    if not reason:
        return card
    hf_overrides.clear()
    bd = card.get("benchmark_details")
    if isinstance(bd, dict):
        for field in ("logo", "org_url"):
            val = bd.get(field)
            if isinstance(val, str) and val.strip() and val != "Not specified":
                bd[field] = "Not specified"
    return card


# M1: fields that an incoherent HF repo can splice into the card. The overview comes straight from
# the HF README (source=hf); the methodology/data fields can come from the WRONG paper when the bound
# repo's own arxiv tag became the resolved paper (paper_url_from_hf). Each is blanked only when it
# actually came from a suspect source, so a paper/deterministic-grounded field is never touched.
_SUBJECT_INCOHERENT_FIELDS = (
    ("benchmark_details", "overview"),
    ("methodology", "metrics"),
    ("methodology", "calculation"),
    ("methodology", "methods"),
    ("data", "size"),
    ("data", "source"),
)
_NS_LIST_FIELDS = {("methodology", "metrics"), ("methodology", "methods")}


def _prov_source(provenance, sec, field):
    """The provenance source token for a card field, or '' when absent."""
    entry = ((provenance or {}).get(sec) or {}).get(field) if isinstance(provenance, dict) else None
    return str(entry.get("source", "")).strip().lower() if isinstance(entry, dict) else ""


def _drop_subject_incoherent_hf(card, provenance, wrong_paper):
    """Blank fields spliced from an HF repo whose subject does not match the card identity.

    A field is blanked only when it came from a suspect source: provenance source==hf (the README,
    e.g. overview), OR -- for the paper-grounded methodology/data fields -- the wrong-paper indicator
    (the bound repo's own arxiv tag became the resolved paper, so source==paper points at the WRONG
    paper). A paper-grounded field on a real paper, or a deterministic/hf-tag field, is left intact.
    Mutates card in place; never fabricates -- it only abstains to the NS sentinel."""
    for sec, field in _SUBJECT_INCOHERENT_FIELDS:
        section = card.get(sec)
        if not isinstance(section, dict) or field not in section:
            continue
        src = _prov_source(provenance, sec, field)
        suspect = src == "hf" or (wrong_paper and src == "paper")
        if not suspect:
            continue
        section[field] = ["Not specified"] if (sec, field) in _NS_LIST_FIELDS else "Not specified"


def run_factreasoner(state):
    """Evaluate factuality of card claims and produce the final flagged card."""
    logger.info("Starting factuality evaluation")

    if not state.get("rag_results"):
        return record_skip("No RAG results available", "FactReasoner evaluation", state)

    try:
        benchmark_name = sanitize_benchmark_name(state["query"])
        rag_results = state["rag_results"]

        # Ship config: source-scoped NLI contexts + neutral escalation against
        # the primary source without RAG fill (measured on the gold set).
        generation_contexts = (
            [c.get("text", "") for c in rag_results.get("contexts", [])]
            if isinstance(rag_results, dict)
            else []
        )
        scoped_rag, scope_info = source_scoped_rag(
            rag_results,
            benchmark_name,
            docling_data=state.get("docling_output"),
            html_data=state.get("html_content"),
            hf_json=state.get("hf_json"),
        )
        if not scope_info.get("source_scoped"):
            logger.warning(
                "Source-scoped retrieval unavailable (%s); falling back to generation contexts",
                scope_info.get("fallback"),
            )
        # Empty source on fallback keeps escalation structurally skipped: the
        # fallback path must equal the pre-ship behavior (plain FR1 on
        # generation contexts), not a hybrid that was never measured.
        source_text = (
            assemble_source_text(
                docling_data=state.get("docling_output"),
                html_data=state.get("html_content"),
                hf_json=state.get("hf_json"),
            )
            if scope_info.get("source_scoped")
            else ""
        )
        factuality_results = evaluate_factuality_two_tier(
            formatted_rag_results=scoped_rag,
            source_text=source_text,
            model=Config.FACTREASONER_MODEL,
            cache_dir=Config.FACTREASONER_CACHE_DIR,
            merlin_path=str(Config.MERLIN_BIN),
        )
        factuality_results["retrieval"] = scope_info

        factuality_filename = f"factuality_results_{benchmark_name}.json"
        factuality_output = state["output_manager"].save_tool_output(
            factuality_results, "factreasoner", factuality_filename
        )
        if scope_info.get("source_scoped"):
            # Audit copy. Must not land in tool_output/rag/: the generation
            # formatted_rag_results there is the frozen input for revalidation
            # and the judge bundles every file under rag/.
            save_formatted_results(
                scoped_rag,
                os.path.join(
                    state["output_manager"].get_tool_output_path("factreasoner"),
                    f"source_scoped_rag_{benchmark_name}.jsonl",
                ),
            )

        risk_card_src = state.get("risk_enhanced_card") or state.get("composed_card") or {}
        clean_card = extract_card(risk_card_src)

        composed_card_data = state.get("composed_card", {})
        provenance_data = composed_card_data.get("provenance") if isinstance(composed_card_data, dict) else None

        field_analysis = factuality_results.get("field_analysis", {})

        retrieved_contexts = (
            [c.get("text", "") for c in scoped_rag.get("contexts", [])]
            if isinstance(scoped_rag, dict)
            else []
        )

        flagged_card = flag_benchmark_card_fields(
            benchmark_card=clean_card,
            field_analysis=field_analysis,
            threshold=Config.DEFAULT_FACTUALITY_THRESHOLD,
            provenance=provenance_data,
            retrieved_contexts=retrieved_contexts,
            structured_artifacts={"eee": state.get("eee_metadata"), "hf": state.get("hf_json")},
            high_signal_grounding=scope_info.get("source_scoped", False),
        )

        if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
            field_details = field_analysis.get("field_details", {})
            fields_to_unflag = []
            for field_name, flag_reason in flagged_card["flagged_fields"].items():
                if field_name in ANALYTICAL_FIELDS:
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging analytical field %s (reasoned, not extracted)", field_name)
                elif field_name in IDENTITY_FIELDS and field_details.get(field_name, {}).get("all_neutral") is True:
                    fields_to_unflag.append(field_name)
                    logger.debug("Unflagging identity field %s (neutral NLI expected for descriptive content)", field_name)
            for field_name in fields_to_unflag:
                del flagged_card["flagged_fields"][field_name]
            if fields_to_unflag:
                logger.info("Field-type-aware scoring: unflagged %d identity/analytical fields", len(fields_to_unflag))

            # C1/C2: display-layer flag policy (resources/annotation exclusion + source-
            # blocked relabel). Production only; the gold-set flag instruments are untouched.
            _apply_presentation_flag_policy(
                flagged_card["flagged_fields"],
                _source_coverage_blocked(state, source_text),
            )

        if provenance_data:
            # Card values backfill from the generation-time contexts; only the
            # unflag decisions see the source-scoped contexts. prose_only gates this to
            # free-text fields (v2 scoping): this path stamps schema_version="v2", so a
            # verbatim quote never lands as an enum/list/deterministic field value (S2).
            flagged_card = backfill_from_provenance(
                flagged_card, provenance_data, generation_contexts, prose_only=True
            )

        flagged_card = normalize_not_specified(flagged_card)

        hf_overrides = extract_hf_tags(state.get("hf_json"))
        if hf_overrides:
            # Corroborate the bound HF repo against the benchmark identity before trusting its
            # tag-derived fields. A wrong/derivative/aggregate repo (arxiv-tag disagreement, or a
            # basename that doesn't bear the distinctive name) gets its overrides + repo-org
            # display fields dropped instead of shipped.
            hf_json = state.get("hf_json") or {}
            repo_id = hf_json.get("id")
            reason = None
            if repo_id:
                paper_url = (state.get("extracted_ids") or {}).get("paper_url") or ""
                pm = _ARXIV_URL_RE.search(paper_url)
                reason = _repo_corroboration(
                    repo_id, benchmark_name,
                    repo_arxiv_id=_first_arxiv_tag(hf_json),
                    paper_arxiv_id=pm.group(1) if pm else None,
                )
            if reason:
                flagged_card = _drop_uncorroborated_hf(flagged_card, hf_overrides, reason)
                state["repo_corroboration"] = {"repo_id": repo_id, "reason": reason}
                logger.warning(
                    "HF repo %r does not corroborate benchmark %r (%s); dropped HF-derived "
                    "identity fields", repo_id, benchmark_name, reason,
                )
            elif repo_id:
                # M1: the name/arxiv corroboration above cannot catch a same-name repo whose
                # arxiv tag became the paper (actionbench-the-slug vs facebook/actionbench =
                # ActionMesh). Compare the README's leading description against the EEE-identity
                # subject; near-disjoint content means the repo describes a different benchmark, so
                # blank the fields it spliced (overview from the README, and the methodology/data
                # fields if the wrong paper -- the repo's own arxiv tag -- supplied them).
                if _hf_subject_conflicts(
                    _get_hf_readme(hf_json), _eee_identity_subject(state.get("eee_metadata"))
                ):
                    extracted_ids = state.get("extracted_ids") or {}
                    paper_url = extracted_ids.get("paper_url") or ""
                    repo_arxiv = _first_arxiv_tag(hf_json)
                    pm = _ARXIV_URL_RE.search(paper_url)
                    wrong_paper = bool(
                        extracted_ids.get("paper_url_from_hf") == paper_url and paper_url
                        and repo_arxiv and pm and pm.group(1) == repo_arxiv
                    )
                    _drop_subject_incoherent_hf(flagged_card, provenance_data, wrong_paper)
                    state["repo_corroboration"] = {
                        "repo_id": repo_id, "reason": "subject_incoherent"}
                    logger.warning(
                        "HF repo %r subject does not match benchmark %r (subject_incoherent); "
                        "blanked spliced overview/methodology/data fields", repo_id, benchmark_name,
                    )

        if hf_overrides:
            logger.info("Applying %d deterministic HF overrides: %s",
                        len(hf_overrides), list(hf_overrides.keys()))
            flagged_card = apply_deterministic_overrides(flagged_card, hf_overrides)

            if "flagged_fields" in flagged_card and isinstance(flagged_card["flagged_fields"], dict):
                for dotted_key in hf_overrides:
                    flag_key = dotted_key.split(".")[-1]
                    for k in (flag_key, dotted_key):
                        if k in flagged_card["flagged_fields"]:
                            del flagged_card["flagged_fields"][k]
                            logger.debug("Cleared stale flag for %s (deterministic override)", k)

        enforce_card_shape(flagged_card)
        flagged_card["missing_fields"] = drop_inapplicable_judge_missing(
            flagged_card, extract_missing_fields(flagged_card))

        flagged_card["card_info"] = {
            "created_at": datetime.now().isoformat(),
            "llm": Config.COMPOSER_MODEL,
            "schema_version": "v2",
        }

        card_filename = f"benchmark_card_{benchmark_name}.json"
        output_path = state["output_manager"].save_benchmark_card(
            {"benchmark_card": flagged_card},
            card_filename,
        )

        logger.info("FactReasoner evaluation complete")
        marginals = factuality_results.get("marginals", [])
        claims_evaluated = len(marginals)
        flagged_fields = len(
            [m for m in marginals if m.get("p_true", 1.0) < 0.3 or m.get("p_true", 1.0) == 0.5]
        )
        logger.info(
            f"Factuality: {claims_evaluated} claims evaluated, {flagged_fields}/{claims_evaluated} fields flagged"
        )

        logger.info("Factuality results saved to: %s", factuality_output)
        logger.info("Final benchmark card saved to: %s", output_path)

        return {
            "factuality_results": factuality_results,
            "final_card": {"benchmark_card": flagged_card},
            "completed": ["factreasoner done"],
        }

    except Exception as e:
        return handle_error(e, "FactReasoner evaluation", state)
