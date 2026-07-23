"""EEE-to-BenchmarkCard workflow.

Alternative pipeline entry point that starts from Every Eval Ever (EEE)
evaluation data instead of UnitXT. Scans EEE evaluation JSONs, resolves
HuggingFace repos, then feeds into the standard composition pipeline.

Flow:
  EEE data → scan & aggregate → resolve HF repos
    → [HF Worker] → [Docling Worker] → [Composer Worker]
    → [Risk Worker] → [RAG Worker] → [FactReasoner Worker]
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from auto_benchmarkcard.card_utils import extract_missing_fields, map_benchmark_type
from auto_benchmarkcard.config import Config
from auto_benchmarkcard.tools.composer import aboutness
from auto_benchmarkcard.tools.composer.composer_tool import (
    drop_inapplicable_judge_missing, enforce_card_shape)
from collections import defaultdict

from auto_benchmarkcard.tools.eee.eee_tool import (
    scan_eee_folder,
    eee_to_pipeline_inputs,
    composite_to_pipeline_inputs,
    lookup_unitxt_paper,
)
from auto_benchmarkcard.workflow import (
    GraphState,
    OutputManager,
    build_workflow,
    sanitize_benchmark_name,
    setup_logging_suppression,
)

logger = logging.getLogger(__name__)


def build_eee_initial_state(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    output_manager: OutputManager,
) -> Dict[str, Any]:
    """Build initial workflow state from EEE pipeline inputs.

    Pre-populates extracted_ids, hf_repo, and eee_metadata so the orchestrator
    skips UnitXT + extractor steps automatically.
    """
    return {
        "query": benchmark_name,
        "catalog_path": None,
        "output_manager": output_manager,
        "unitxt_json": None,
        "extracted_ids": pipeline_inputs["extracted_ids"],
        "hf_repo": pipeline_inputs["hf_repo"],
        "hf_json": None,
        "docling_output": None,
        "docling_telemetry": None,
        "composed_card": None,
        "risk_enhanced_card": None,
        "completed": ["eee_scan done", "eee_resolve done"],
        "errors": [],
        "hf_extraction_attempted": False,
        "paper_resolver_attempted": False,
        "rag_results": None,
        "factuality_results": None,
        "final_card": None,
        "eee_metadata": pipeline_inputs["eee_metadata"],
        "html_content": None,
    }


_CARD_FIELD_ORDER = [
    "benchmark_details",
    "purpose_and_intended_users",
    "data",
    "methodology",
    "ethical_and_legal_considerations",
    "possible_risks",
    "flagged_fields",
    "missing_fields",
    "card_info",
]


def _reorder_card_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder card fields to match canonical schema order."""
    ordered = {}
    for key in _CARD_FIELD_ORDER:
        if key in card:
            ordered[key] = card[key]
    for key in card:
        if key not in ordered:
            ordered[key] = card[key]
    return ordered


def _stamp_card_info(card: Dict[str, Any]) -> None:
    """Stamp card_info on the EEE path when absent. The risk_enhanced_card fallback (used
    when the FactReasoner worker did not run) carries no card_info; setdefault keeps a real
    FactReasoner stamp intact when that path ran. Mirrors the workers.py stamp shape."""
    card.setdefault("card_info", {
        "created_at": datetime.now().isoformat(),
        "llm": Config.COMPOSER_MODEL,
        "schema_version": "v2",
    })


def _enrich_baseline_results(
    final_card: Dict[str, Any], eee_metadata: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Fill methodology.baseline_results from EEE evaluation data when Stage-B left it blank.

    Mutates the card in place (card is a reference into final_card). Mirrors
    _inject_eee_interpretation: returns the EEE-derived provenance entry (source=eee, derived)
    when it actually fills the field, else None so a genuine Stage-B baseline keeps its own
    provenance (entry-iff-value). The grounding gate treats source=eee as grounded.
    """
    eval_summary = eee_metadata.get("evaluation_summary", {})
    if not eval_summary:
        return None

    card = final_card.get("benchmark_card", final_card)
    methodology = card.get("methodology", {})
    if not _is_not_specified(methodology.get("baseline_results")):
        return None

    top = eval_summary.get("top_performers", [])
    stats = eval_summary.get("score_statistics", {})
    # Display the metric's human name, not the raw EEE id, so a baseline like "mean
    # llm_stats.<slug>.score = ..." reads as "mean <Benchmark> score = ..." (F4 prose hygiene).
    metric = eval_summary.get("primary_metric", "score")
    _mcfg = (eee_metadata.get("metrics") or {}).get(metric)
    if isinstance(_mcfg, dict) and _mcfg.get("metric_name"):
        metric = _mcfg["metric_name"]
    n_models = eval_summary.get("total_models_evaluated", 0)
    if not (top and stats):
        return None

    top_str = ", ".join(f"{p['model']} ({p['score']:.4f})" for p in top[:3])
    methodology["baseline_results"] = (
        f"Based on {n_models} model evaluations from Every Eval Ever: "
        f"mean {metric} = {stats['mean']:.4f} (std = {stats['std_dev']:.4f}). "
        f"Top performers: {top_str}."
    )
    card["methodology"] = methodology
    return {
        "source": "eee",
        "evidence": "",
        "status": "derived",
        "verified": True,
        "evidence_ids": [],
    }


def _is_not_specified(value: Any) -> bool:
    """True when a field value is empty or the canonical 'Not specified' sentinel."""
    if not value:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("not specified", "not specified.")
    return False


_HUMAN_BASELINE_RE = re.compile(
    r"\bhuman[\s-]+(performance|evaluators?|experts?|baseline|annotators?|raters?|subjects?|participants?)\b",
    re.I,
)
_HUMAN_NUMERIC_RE = re.compile(r"\d|%|\baccuracy\b|\bcorrect", re.I)
# Model-baseline markers. Their presence means the prose is (or includes) a model baseline,
# so it must NOT be relocated as a human baseline even if it mentions humans for comparison.
_MODEL_MARKER_RE = re.compile(
    r"\b(?:LLMs?|language\s+models?|GPT[\w.-]*|o1|o3|claude|gemini|llama|mistral|qwen|deepseek|palm)\b"
    r"|\bmodels?\s+(?:scored|achieved?|perform\w*|demonstrat\w*|outperform\w*)\b"
    r"|\b\d+\s+(?:LLMs?|models?)\b"
    r"|\bevaluation\s+of\s+\d+\b",
    re.I,
)


def _is_human_baseline_text(text: Any) -> bool:
    """True when baseline_results prose is human-performance text, not a model baseline.

    Requires positive human phrasing AND a number/percent/accuracy, AND no model-baseline
    markers. The negative guard is what protects a genuine model baseline that merely
    references humans for comparison (e.g. "60 LLMs ... vs human performance of 97%").
    """
    if not isinstance(text, str):
        return False
    if not _HUMAN_BASELINE_RE.search(text):
        return False
    if not _HUMAN_NUMERIC_RE.search(text):
        return False
    if _MODEL_MARKER_RE.search(text):
        return False
    return True


# A verbatim, sentence-bounded "human <performance noun> ... NN%" clause. Anchored on the same
# human-performance phrasing as _HUMAN_BASELINE_RE and requiring a number in the same clause, so
# "human-written problems" (a data-construction mention, not a baseline) never matches.
_HUMAN_CLAUSE_RE = re.compile(
    r"\bhuman[\s-]+(?:performance|evaluators?|experts?|baseline|annotators?|raters?|subjects?|participants?)\b"
    r"[^.!?;]*?\d[\d.,]*\s*%?[^.!?;]*",
    re.I,
)


def _extract_human_clause(text: Any) -> Optional[str]:
    """Return a verbatim human-performance clause carrying a number from MIXED prose, or None.

    Used to lift a human baseline buried in a baseline_results / overview that ALSO reports model
    scores (so the whole-field relocation below does not apply). The returned clause is a verbatim
    substring of the source field, so the refresh merge can ground it against the card's own field
    without paper evidence_ids. Conservative: requires the human-performance phrasing and a number
    in one clause, and a substantive length."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = _HUMAN_CLAUSE_RE.search(text)
    if not m:
        return None
    clause = m.group(0).strip(" ,;:-")
    if len(clause) < 12:
        return None
    return clause


def _relocate_human_baseline(
    final_card: Dict[str, Any], output_manager: OutputManager, safe_name: str
) -> bool:
    """Move human-performance text mis-filed in methodology.baseline_results into
    methodology.human_baseline, then reset baseline_results to "Not specified" so
    _enrich_baseline_results can inject the EEE model scores authoritatively.

    Deterministic and conservative (see _is_human_baseline_text): only fires on human prose
    with no model-baseline markers, so a genuine model baseline is never clobbered. The move
    into human_baseline happens only when human_baseline is empty (a real one is preserved),
    carrying over its source provenance entry. Returns True when it relocated.
    """
    card = final_card.get("benchmark_card", final_card)
    methodology = card.get("methodology")
    if not isinstance(methodology, dict):
        return False
    baseline = methodology.get("baseline_results")
    if not _is_human_baseline_text(baseline):
        # Whole-field relocation does not apply. Try lifting a verbatim human-performance clause
        # out of a MIXED baseline_results / overview (model scores plus a buried human number)
        # into an empty human_baseline, leaving baseline_results untouched.
        return _lift_mixed_human_clause(card, methodology, output_manager, safe_name)

    if _is_not_specified(methodology.get("human_baseline")):
        methodology["human_baseline"] = baseline
        existing = _load_provenance(output_manager, safe_name).get("methodology", {})
        if not (isinstance(existing, dict) and existing.get("human_baseline")):
            moved = existing.get("baseline_results") if isinstance(existing, dict) else None
            entry = moved if isinstance(moved, dict) else {
                "source": "derived",
                "evidence": baseline,
                "status": "derived",
                "verified": True,
                "evidence_ids": [],
            }
            _record_derived_provenance(
                output_manager, safe_name, "methodology", "human_baseline", entry
            )

    methodology["baseline_results"] = "Not specified"
    card["methodology"] = methodology
    return True


def _lift_mixed_human_clause(
    card: Dict[str, Any],
    methodology: Dict[str, Any],
    output_manager: OutputManager,
    safe_name: str,
) -> bool:
    """Lift a verbatim human-performance clause from MIXED baseline_results / overview into an
    empty human_baseline, without disturbing baseline_results.

    Fires only when human_baseline is empty and a conservative human-performance clause with a
    number is found (see _extract_human_clause). The clause is a verbatim substring of its source
    field, so the refresh merge grounds it cross-field; the live pipeline also records a derived
    provenance entry so its grounding gate scores the field grounded. Returns True when it lifted.
    """
    if not _is_not_specified(methodology.get("human_baseline")):
        return False
    details = card.get("benchmark_details")
    overview = details.get("overview") if isinstance(details, dict) else None
    for src_text in (methodology.get("baseline_results"), overview):
        clause = _extract_human_clause(src_text)
        if clause:
            methodology["human_baseline"] = clause
            card["methodology"] = methodology
            _record_derived_provenance(
                output_manager, safe_name, "methodology", "human_baseline",
                {"source": "derived", "evidence": clause, "status": "derived",
                 "verified": True, "evidence_ids": []},
            )
            return True
    return False


def _derive_interpretation(eee_metadata: Dict[str, Any]) -> Optional[str]:
    """Map the primary metric's EEE lower_is_better signal onto the interpretation enum.

    Reads evaluation_summary.metric_config.lower_is_better (the primary metric), falling
    back to that primary metric's per-metric lower_is_better. Returns None when EEE carries
    no signal at all, so the caller leaves the field untouched. A False signal is valid
    (higher_is_better) and must not be read as missing.
    """
    eval_summary = eee_metadata.get("evaluation_summary") or {}
    lib = (eval_summary.get("metric_config") or {}).get("lower_is_better")
    if lib is None:
        primary = eval_summary.get("primary_metric")
        metric_cfg = (eee_metadata.get("metrics") or {}).get(primary) if primary else None
        if isinstance(metric_cfg, dict):
            lib = metric_cfg.get("lower_is_better")
    if lib is None:
        return None
    return "lower_is_better" if lib else "higher_is_better"


def _primary_metric_label(eee_metadata: Dict[str, Any]) -> Optional[str]:
    """A clean, human-readable name of the EEE primary metric, or None.

    Mirrors the metric-name lookup in _enrich_baseline_results, then keeps only a tidy metric word
    (e.g. 'accuracy', 'F1', 'pass@1'). Raw ids, benchmark-name-derived labels ('AutoLogi score',
    'average_across_subjects') and anything underscore-heavy or long read poorly in prose, so they
    collapse to None and the caller omits the metric phrase rather than print an awkward label."""
    eval_summary = eee_metadata.get("evaluation_summary", {}) or {}
    metric = eval_summary.get("primary_metric")
    cfg = (eee_metadata.get("metrics") or {}).get(metric) if metric else None
    label = cfg["metric_name"] if (isinstance(cfg, dict) and cfg.get("metric_name")) else metric
    if not isinstance(label, str):
        return None
    label = label.strip()
    low = label.lower()
    if (not label or "." in label or "_" in label or len(label) > 24
            or low == "score" or low.endswith(" score")):
        return None
    return label


def compose_interp_prose(
    direction: Optional[str],
    baseline_results: Any,
    human_baseline: Any,
    metric: Optional[str] = None,
) -> Optional[str]:
    """Build a grounded interpretation sentence from the card's own direction signal plus its
    reported baseline / human numbers.

    Restates the direction and the already-grounded numbers only; it invents no thresholds. Returns
    None when there is nothing grounded to say (no direction signal, or neither a baseline nor a
    human reference point is present), so the caller abstains and keeps whatever was there. Shared
    by the live finalize (_inject_eee_interpretation) and the refresh registry's recompute so both
    paths produce byte-identical prose.
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        return None
    has_baseline = isinstance(baseline_results, str) and not _is_not_specified(baseline_results)
    has_human = isinstance(human_baseline, str) and not _is_not_specified(human_baseline)
    if not (has_baseline or has_human):
        return None
    better = "Higher" if direction == "higher_is_better" else "Lower"
    metric_phrase = f" on {metric}" if metric else ""
    parts = [f"{better} scores{metric_phrase} indicate stronger performance."]
    if has_baseline:
        parts.append(f"For reference, reported baselines: {_clip_for_prose(baseline_results)}.")
    if has_human:
        parts.append(f"Human reference: {_clip_for_prose(human_baseline)}.")
    return " ".join(parts)


def _clip_for_prose(text: str, limit: int = 220) -> str:
    """Trim a restated field to one or two sentences without a mid-word cut. Backs off to the last
    sentence end within the limit, else the last word boundary, appending an ellipsis when cut."""
    text = text.strip().rstrip(".")
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind("; "))
    if end >= limit // 2:
        return head[:end]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip(" ,;:-") + "..."


def _inject_eee_interpretation(
    card: Dict[str, Any], eee_metadata: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Set methodology.interpretation from EEE, post-composition (contract S4.6).

    interpretation is an enum derived from structured EEE data, but Stage-B sees no text
    evidence for it and forces "Not specified". This mirrors the benchmark_type/appears_in
    injection: [0] writes it after composition, not via Stage-B. Only overwrites a
    Not-specified/empty value so a genuine Stage-B interpretation is preserved. Returns the
    provenance entry to record (source=eee, derived) when it injects, else None.
    """
    interp = _derive_interpretation(eee_metadata)
    if interp is None:
        return None
    methodology = card.get("methodology", {})
    if not _is_not_specified(methodology.get("interpretation")):
        return None
    # Prefer grounded prose that restates the direction and the card's own already-grounded
    # baseline / human numbers over the bare enum; fall back to the enum when there is no grounded
    # anchor to restate (compose_interp_prose returns None then).
    prose = compose_interp_prose(
        interp,
        methodology.get("baseline_results"),
        methodology.get("human_baseline"),
        _primary_metric_label(eee_metadata),
    )
    methodology["interpretation"] = prose if prose else interp
    card["methodology"] = methodology
    # missing_fields was computed during composition, when interpretation was still
    # "Not specified"; drop the now-stale entry so the filled value is not also reported
    # missing (same stale-entry cleanup the deterministic-override path does for flags).
    missing = card.get("missing_fields")
    if isinstance(missing, list) and "methodology.interpretation" in missing:
        card["missing_fields"] = [m for m in missing if m != "methodology.interpretation"]
    return {
        "source": "eee",
        "evidence": "",
        "status": "derived",
        "verified": True,
        "evidence_ids": [],
    }


def _derive_judge_uses_llm(
    eee_metadata: Dict[str, Any], hf_json: Optional[Dict[str, Any]] = None
) -> Optional[bool]:
    """Return True when the EEE metric descriptions OR the HF readme signal model-graded /
    LLM-as-judge / equality-checker scoring (shared detector in aboutness).

    Returns None when no such signal is present -- absence of a signal is NOT evidence that
    scoring is programmatic, so we never derive False here.
    """
    text = aboutness.collect_judge_scan_text(eee_metadata, hf_json)
    return True if aboutness.describes_llm_judge(text) else None


def _is_text_grounded(entry: Any) -> bool:
    """True when a provenance entry is grounded in source text (paper/html/hf/docling, stated)."""
    if not isinstance(entry, dict):
        return False
    src = str(entry.get("source", "")).lower()
    text_src = any(t in src for t in ("paper", "html", "hf_readme", "hf", "docling"))
    return text_src and entry.get("status") == "stated"


def _is_aboutness_exec_demotion(entry: Any) -> bool:
    """True when a judge_uses_llm provenance entry is the aboutness exec-scoring demotion: a
    derived entry citing real execution-scoring evidence (non-empty evidence_ids). That
    demotion is more specific than an EEE description, so an EEE signal must not override it.
    """
    if not isinstance(entry, dict):
        return False
    is_derived = entry.get("source") == "derived" or entry.get("status") == "derived"
    return is_derived and bool(entry.get("evidence_ids"))


def _eee_derived_entry() -> Dict[str, Any]:
    """A fresh EEE-derived provenance entry (structured source -> grounded by the gate)."""
    return {"source": "eee", "evidence": "", "status": "derived",
            "verified": True, "evidence_ids": []}


def _clear_stale_bookkeeping(card: Dict[str, Any], dotted_field: str) -> None:
    """Drop a now-filled field from missing_fields / flagged_fields; those were recorded during
    composition when the value was still Not specified."""
    missing = card.get("missing_fields")
    if isinstance(missing, list) and dotted_field in missing:
        card["missing_fields"] = [m for m in missing if m != dotted_field]
    flagged = card.get("flagged_fields")
    if isinstance(flagged, dict):
        flagged.pop(dotted_field, None)


def _inject_eee_judge_llm(
    card: Dict[str, Any],
    eee_metadata: Dict[str, Any],
    output_manager: OutputManager,
    safe_name: str,
) -> Optional[Dict[str, Any]]:
    """Set methodology.judge_uses_llm=True from an EEE/HF LLM-judge signal, post-composition.

    Mirrors _inject_eee_interpretation. Scans BOTH the EEE metric descriptions AND the HF readme
    via the shared aboutness detector. Conservative guards (return None = no injection):
    judge_uses_llm already True; a genuine Stage-B judge cluster (real judge_models/judge_num);
    an aboutness exec-scoring demotion (derived entry with evidence_ids, e.g. Pass@1); or a
    text-grounded FALSE provenance entry UNLESS the source unambiguously NAMES a judge model
    (the misread-denial override). When a single judge model is named it also fills
    judge_models + judge_num=1 and records their eee-derived provenance. Returns the
    judge_uses_llm eee/derived entry when it injects, else None.
    """
    hf_json = _load_hf_json(output_manager, safe_name)
    scan_text = aboutness.collect_judge_scan_text(eee_metadata, hf_json)
    if not aboutness.describes_llm_judge(scan_text):
        return None
    methodology = card.get("methodology")
    if not isinstance(methodology, dict):
        return None
    if methodology.get("judge_uses_llm") is True:
        return None
    models = methodology.get("judge_models")
    if isinstance(models, list) and any(
        isinstance(m, str) and not _is_not_specified(m) for m in models
    ):
        return None
    num = methodology.get("judge_num")
    if isinstance(num, int) and not isinstance(num, bool):
        return None
    named_model = aboutness.extract_judge_model(scan_text)
    existing = _load_provenance(output_manager, safe_name).get("methodology", {})
    entry = existing.get("judge_uses_llm") if isinstance(existing, dict) else None
    if _is_aboutness_exec_demotion(entry):
        return None  # real execution-scoring evidence -- never override
    if _is_text_grounded(entry) and named_model is None:
        return None  # an unnamed signal must not override a text-grounded denial

    methodology["judge_uses_llm"] = True
    _clear_stale_bookkeeping(card, "methodology.judge_uses_llm")
    # J2: a single named judge fills judge_models + judge_num (left Not specified before).
    if named_model is not None:
        methodology["judge_models"] = [named_model]
        methodology["judge_num"] = 1
        for f in ("judge_models", "judge_num"):
            _clear_stale_bookkeeping(card, f"methodology.{f}")
            _record_derived_provenance(
                output_manager, safe_name, "methodology", f, _eee_derived_entry()
            )
    card["methodology"] = methodology
    return _eee_derived_entry()


def _load_provenance(output_manager: OutputManager, safe_name: str) -> Dict[str, Any]:
    """Read the composer provenance sidecar for one benchmark; {} if absent/unreadable."""
    prov_path = os.path.join(
        output_manager.get_tool_output_path("composer"), f"provenance_{safe_name}.json"
    )
    if os.path.exists(prov_path):
        try:
            with open(prov_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _load_hf_json(output_manager: OutputManager, safe_name: str) -> Dict[str, Any]:
    """Read the HF tool-output sidecar for one benchmark; {} if absent/unreadable. Mirrors
    _load_provenance; used to scan readme_markdown for an LLM-judge signal Stage-B missed."""
    hf_path = os.path.join(output_manager.get_tool_output_path("hf"), f"{safe_name}.json")
    if os.path.exists(hf_path):
        try:
            with open(hf_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _record_derived_provenance(
    output_manager: OutputManager,
    safe_name: str,
    section: str,
    field: str,
    entry: Dict[str, Any],
) -> None:
    """Merge an EEE-derived provenance entry for one field into the composer sidecar.

    The sidecar is written during composition, before these post-composition injections, so
    merge into the existing file rather than overwrite it. EEE-derived fields (interpretation,
    benchmark_type, appears_in) cite structured EEE data; the grounding gate treats source=eee
    as grounded, not a hallucination.
    """
    filename = f"provenance_{safe_name}.json"
    provenance = _load_provenance(output_manager, safe_name)
    provenance.setdefault(section, {})[field] = entry
    output_manager.save_tool_output(provenance, "composer", filename)


def _record_identity_name_provenance(
    output_manager: OutputManager,
    safe_name: str,
    card: Dict[str, Any],
    benchmark_name: str,
    eee_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Record a deterministic identity provenance entry for benchmark_details.name.

    The name is a deterministic pipeline input (the query / EEE benchmark_name), not an LLM
    claim. Stage-B emits the name value but, because benchmark_details.name is evidence-exempt
    (field_spec.EVIDENCE_EXEMPT), it attaches no provenance when it cites no document -- so the
    grounding gate sees no sidecar entry and scores the name UNGROUNDED. Record a deterministic
    entry (verified=true -> classifier buckets it GROUNDED-DERIVED) only when (a) the card name
    equals the known identity, so an LLM-rephrased or confused name stays ungrounded, and (b)
    the sidecar has no name entry yet, so a genuine Stage-B docling citation is preserved.
    Returns the recorded entry, else None.
    """
    details = card.get("benchmark_details", {})
    name_val = details.get("name") if isinstance(details, dict) else None
    if not isinstance(name_val, str) or _is_not_specified(name_val):
        return None
    identities = {benchmark_name, eee_metadata.get("benchmark_name")}
    identities = {i.strip().lower() for i in identities if isinstance(i, str) and i.strip()}
    if name_val.strip().lower() not in identities:
        return None
    existing = _load_provenance(output_manager, safe_name).get("benchmark_details", {})
    if isinstance(existing, dict) and existing.get("name"):
        return None
    entry = {
        "source": "deterministic",
        "evidence": name_val,
        "quote": name_val,
        "quote_loc": {"doc": "identity", "char_start": None},
        "status": "stated",
        "verified": True,
        "evidence_ids": [],
    }
    _record_derived_provenance(output_manager, safe_name, "benchmark_details", "name", entry)
    return entry


def finalize_eee_card(
    final_card: Dict[str, Any],
    eee_metadata: Dict[str, Any],
    output_manager: OutputManager,
    benchmark_name: str,
) -> Dict[str, Any]:
    """Apply the EEE post-composition finalize steps to a composed/risk-enhanced card.

    Runs the human_baseline relocation, EEE baseline_results / benchmark_type / interpretation /
    judge injections, card_info stamp, shape + missing_fields recompute, and the field reorder,
    recording the matching provenance sidecar entries. Mutates final_card in place and returns
    the reordered card (re-wrapped if it carried a benchmark_card key).

    Extracted verbatim from process_single_benchmark so the refresh-from-cache harness can run
    the identical finalize offline; the live pipeline calls it on the same path. safe_name is
    derived from output_manager (idempotent re-sanitize of the name it was built with), so it
    equals the value the caller passed inline before this extraction.
    """
    safe_name = output_manager.benchmark_name
    # Move human-performance text mis-filed in baseline_results into human_baseline so
    # the EEE model scores inject below. Deterministic; leaves a model baseline intact.
    _relocate_human_baseline(final_card, output_manager, safe_name)
    baseline_prov = _enrich_baseline_results(final_card, eee_metadata)

    card = final_card.get("benchmark_card", final_card)

    # Inject composite/single metadata into benchmark_details.
    # Write back to card in case get() created a detached empty dict.
    details = card.get("benchmark_details", {})
    raw_btype = eee_metadata.get("benchmark_type", "single")
    # Map onto the Tier-1 enum (S4.2); the raw value still drives the composite
    # contains-injection below. Fixes the all-"single" bug (composite was injected
    # raw instead of as composite_suite).
    details["benchmark_type"] = map_benchmark_type(raw_btype)
    # benchmark_type and appears_in are EEE-derived and injected here, not by Stage-B,
    # so they carry no text evidence. Record them as structured (eee) derivations in
    # the provenance sidecar so the grounding gate scores them grounded, not as
    # hallucinations (same coupling as methodology.interpretation below).
    eee_prov_entry = {
        "source": "eee",
        "evidence": "",
        "status": "derived",
        "verified": True,
        "evidence_ids": [],
    }
    _record_derived_provenance(
        output_manager, safe_name, "benchmark_details", "benchmark_type", eee_prov_entry
    )
    # baseline_results: when _enrich filled it from EEE (Stage-B left it Not-specified),
    # record the matching eee-derived entry so the grounding gate scores it grounded.
    if baseline_prov is not None:
        _record_derived_provenance(
            output_manager, safe_name, "methodology", "baseline_results", eee_prov_entry
        )
    # name: deterministic identity fill (see _record_identity_name_provenance) -- only
    # when Stage-B cited no document for it and the value matches the known identity.
    _record_identity_name_provenance(
        output_manager, safe_name, card, benchmark_name, eee_metadata
    )
    if raw_btype == "composite" and eee_metadata.get("contains"):
        details["contains"] = eee_metadata["contains"]
    if eee_metadata.get("appears_in"):
        details["appears_in"] = eee_metadata["appears_in"]
        _record_derived_provenance(
            output_manager, safe_name, "benchmark_details", "appears_in", eee_prov_entry
        )
    card["benchmark_details"] = details

    # interpretation is an EEE-derived enum (S4.6) injected post-composition like
    # benchmark_type/appears_in above; Stage-B leaves it "Not specified" with no text
    # evidence. Record it as a structured (eee) derivation in the provenance sidecar
    # so the grounding gate does not score it as a hallucination.
    interp_prov = _inject_eee_interpretation(card, eee_metadata)
    if interp_prov is not None:
        _record_derived_provenance(
            output_manager, safe_name, "methodology", "interpretation", interp_prov
        )

    # judge_uses_llm: EEE metric descriptions can signal model-graded / LLM-as-judge
    # scoring that Stage-B missed. Mirror the interpretation injection; conservative
    # guards leave genuine Stage-B / aboutness judge decisions untouched.
    judge_prov = _inject_eee_judge_llm(card, eee_metadata, output_manager, safe_name)
    if judge_prov is not None:
        _record_derived_provenance(
            output_manager, safe_name, "methodology", "judge_uses_llm", judge_prov
        )

    # card_info: the EEE fallback path (risk_enhanced_card) is not stamped by the
    # FactReasoner worker, so stamp it here before save.
    _stamp_card_info(card)

    # Recompute shape and missing_fields after all EEE injections, so a field EEE
    # filled (e.g. methodology.baseline_results) is not left listed as missing.
    enforce_card_shape(card)
    card["missing_fields"] = drop_inapplicable_judge_missing(
        card, extract_missing_fields(card))

    ordered_card = _reorder_card_fields(card)
    if "benchmark_card" in final_card:
        final_card["benchmark_card"] = ordered_card
    else:
        final_card = ordered_card
    return final_card


def process_single_benchmark(
    benchmark_name: str,
    pipeline_inputs: Dict[str, Any],
    base_output_path: Optional[str] = None,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full pipeline for a single EEE benchmark."""
    safe_name = sanitize_benchmark_name(benchmark_name)
    output_manager = OutputManager(safe_name, base_output_path)

    from auto_benchmarkcard.llm_handler import set_usage_log_path
    set_usage_log_path(os.path.join(
        output_manager.get_tool_output_path("llm_usage"),
        f"llm_usage_{safe_name}.jsonl",
    ))

    eee_metadata = pipeline_inputs.get("eee_metadata", {})
    output_manager.save_tool_output(eee_metadata, "eee", f"{safe_name}.json")

    hf_repo = pipeline_inputs.get("hf_repo")
    extracted_ids = pipeline_inputs.get("extracted_ids", {})
    if not extracted_ids.get("paper_url") and hf_repo:
        unitxt_paper = lookup_unitxt_paper(hf_repo)
        if unitxt_paper:
            extracted_ids["paper_url"] = unitxt_paper

    # Paper resolution is now handled inside the workflow (paper_resolver_worker)
    # after HF metadata is available, giving it richer context.

    initial_state = build_eee_initial_state(benchmark_name, pipeline_inputs, output_manager)
    workflow = build_workflow()

    logger.info("Processing benchmark: %s (hf_repo=%s)", benchmark_name, pipeline_inputs.get("hf_repo"))

    try:
        # Default LangGraph recursion_limit (25) gets hit on long pipelines:
        # orchestrator + 11 workers + retries can easily exceed 25 super-steps.
        # 75 gives headroom for retries without masking real loops.
        final_state = workflow.invoke(initial_state, config={"recursion_limit": 75})

        # Use explicit None checks — {} is falsy and would skip to wrong fallback
        final_card = final_state.get("final_card")
        if final_card is None:
            final_card = final_state.get("risk_enhanced_card")
        if final_card is None:
            final_card = final_state.get("composed_card")
        card_source = (
            "factreasoner" if final_state.get("final_card") is not None
            else "risk" if final_state.get("risk_enhanced_card") is not None
            else "composer"
        )
        logger.info("Card source: %s", card_source)
        if final_card and eee_metadata:
            final_card = finalize_eee_card(
                final_card, eee_metadata, output_manager, benchmark_name
            )

            card_filename = f"benchmark_card_{safe_name}.json"
            output_manager.save_benchmark_card(final_card, card_filename)
            logger.info("Saved benchmark card with evaluation summary: %s", card_filename)

        completed = final_state.get("completed", [])
        errors = final_state.get("errors", [])
        logger.info("Completed steps: %s", completed)
        if errors:
            logger.warning("Errors: %s", errors)

        return final_card

    except Exception as e:
        logger.error("Failed to process %s: %s", benchmark_name, e, exc_info=debug)
        return None


def _apply_benchmark_filter(
    benchmarks: Dict[str, Any],
    composites: Dict[str, Any],
    benchmarks_filter: Optional[List[str]],
) -> tuple:
    """Apply user filter, auto-expanding composite names to sub-benchmarks.

    Returns (filtered_benchmarks, filter_set) where filter_set is None if no filter.
    """
    if not benchmarks_filter:
        return benchmarks, None

    filter_set = {b.lower() for b in benchmarks_filter}
    for b in list(filter_set):
        for folder, comp in composites.items():
            if folder.lower() == b:
                sub_names = {n.lower() for n in comp.sub_benchmarks}
                logger.info(
                    "Expanding composite '%s' -> also generating: %s",
                    folder, comp.sub_benchmarks,
                )
                filter_set.update(sub_names)

    filtered = {k: v for k, v in benchmarks.items() if k.lower() in filter_set}
    logger.info("Filtered to %d benchmarks: %s", len(filtered), list(filtered.keys()))
    return filtered, filter_set


def _resolve_hf_repos(
    benchmarks: Dict[str, Any],
    appears_in_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve HF repos and build pipeline inputs for each benchmark."""
    logger.info("Resolving HuggingFace repos...")
    pipeline_inputs_map: Dict[str, Dict[str, Any]] = {}
    for name, bench in sorted(benchmarks.items()):
        inputs = eee_to_pipeline_inputs(
            bench,
            benchmark_type="single",
            appears_in=appears_in_map.get(name, []),
        )
        pipeline_inputs_map[name] = inputs
        hf = inputs.get("hf_repo", "None")
        logger.info("  %s -> hf_repo=%s (%d models)", name, hf, bench.num_models_evaluated)
    return pipeline_inputs_map


def run_eee_pipeline(
    eee_path: str,
    output_path: Optional[str] = None,
    max_files_per_benchmark: int = 50,
    benchmarks_filter: Optional[List[str]] = None,
    debug: bool = False,
    no_collapse: bool = False,
    on_card_generated: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Scan EEE evaluation data, discover benchmarks, and generate cards for each.

    Args:
        on_card_generated: Optional callback called after each successful card generation.
            Receives (benchmark_name, card_dict). Used by the webhook worker to upload cards.
    """
    setup_logging_suppression(debug_mode=debug)

    logger.info("Models — composer: %s | factreasoner: %s",
                Config.COMPOSER_MODEL, Config.FACTREASONER_MODEL)

    logger.info("Scanning EEE data at: %s", eee_path)
    scan_result = scan_eee_folder(eee_path, max_files_per_benchmark, no_collapse=no_collapse)

    if scan_result.errors:
        for err in scan_result.errors:
            logger.warning("Scan error: %s", err)

    logger.info(
        "Found %d unique benchmarks, %d composite suites in %d files",
        len(scan_result.benchmarks), len(scan_result.composites), scan_result.total_eval_files,
    )

    # Build reverse map: benchmark_name -> composite folders it appears in
    appears_in_map: Dict[str, List[str]] = defaultdict(list)
    for folder, comp in scan_result.composites.items():
        for sub in comp.sub_benchmarks:
            appears_in_map[sub].append(folder)

    benchmarks, filter_set = _apply_benchmark_filter(
        scan_result.benchmarks, scan_result.composites, benchmarks_filter,
    )

    pipeline_inputs_map = _resolve_hf_repos(benchmarks, appears_in_map)

    summary: Dict[str, Any] = {
        "total_benchmarks": len(pipeline_inputs_map) + len(scan_result.composites),
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    # Process individual benchmarks
    for i, (name, inputs) in enumerate(sorted(pipeline_inputs_map.items()), 1):
        logger.info("[%d/%d] Processing: %s", i, len(pipeline_inputs_map), name)

        try:
            if not inputs.get("hf_repo"):
                logger.info("No HF repo for '%s', proceeding with EEE data + paper resolver", name)

            card = process_single_benchmark(
                benchmark_name=name,
                pipeline_inputs=inputs,
                base_output_path=output_path,
                debug=debug,
            )

            if card:
                summary["successful"].append(name)
                if on_card_generated:
                    on_card_generated(name, card)
            else:
                summary["failed"].append(name)
        except Exception:
            logger.exception("Unexpected error processing benchmark '%s'", name)
            summary["failed"].append(name)

    # Process composite suites
    for folder, composite in sorted(scan_result.composites.items()):
        if filter_set is not None and folder.lower() not in filter_set:
            continue

        logger.info("Processing composite suite: %s (%d sub-benchmarks)",
                     folder, len(composite.sub_benchmarks))

        try:
            inputs = composite_to_pipeline_inputs(composite, scan_result)
            card = process_single_benchmark(
                benchmark_name=folder,
                pipeline_inputs=inputs,
                base_output_path=output_path,
                debug=debug,
            )

            if card:
                summary["successful"].append(f"{folder} (composite)")
                if on_card_generated:
                    on_card_generated(folder, card)
            else:
                summary["failed"].append(f"{folder} (composite)")
        except Exception:
            logger.exception("Unexpected error processing composite '%s'", folder)
            summary["failed"].append(f"{folder} (composite)")

    logger.info("EEE pipeline complete: success=%d, failed=%d, skipped=%d",
                len(summary["successful"]), len(summary["failed"]), len(summary["skipped"]))

    return summary
