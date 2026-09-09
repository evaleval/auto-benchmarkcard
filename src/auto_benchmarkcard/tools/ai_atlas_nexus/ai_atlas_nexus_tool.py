"""AI Atlas Nexus integration for AI risk identification.

Retrieve-then-rerank over the IBM AI Risk Atlas: the BenchmarkRiskDetector
over-fetches candidate risks from the taxonomy, then one LLM call selects 0-5
risks conditioned on the composed card plus the referent frame and writes a
benchmark-specific justification for each. LLM failure falls back to the
plain detector top-N (telemetry fallback=true).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ai_atlas_nexus.blocks.risk_detector import BenchmarkRiskDetector
from ai_atlas_nexus.library import AIAtlasNexus

logger = logging.getLogger(__name__)

# over-fetch size for the retrieval pass and the hard cap on shipped risks
RISK_CANDIDATE_COUNT = 12
MAX_SELECTED_RISKS = 5
_JUSTIFICATION_MAX_CHARS = 240
_CANDIDATE_DESC_MAX_CHARS = 300

_nexus_instance: Optional[AIAtlasNexus] = None


def _get_nexus() -> AIAtlasNexus:
    """Return a cached AIAtlasNexus instance (lazy singleton)."""
    global _nexus_instance
    if _nexus_instance is None:
        _nexus_instance = AIAtlasNexus()
    return _nexus_instance


def identify_risks_with_benchmark_detector(
    ai_atlas_nexus: AIAtlasNexus,
    usecases: List[str],
    inference_engine,
    taxonomy: str = "ibm-ai-risk-atlas",
    max_risk: Optional[int] = None,
) -> List[List]:
    """Match benchmark use cases to known AI risks using the BenchmarkRiskDetector."""
    try:
        all_risks = ai_atlas_nexus.get_all_risks(taxonomy)

        benchmark_detector = BenchmarkRiskDetector(
            risks=all_risks,
            inference_engine=inference_engine,
            max_risk=max_risk,
        )

        return benchmark_detector.detect(usecases)

    except Exception as e:
        logger.error("Error in benchmark risk detection: %s", e)
        return []


def create_inference_engine():
    """Create an inference engine for risk classification using the configured LLM backend."""
    try:
        from auto_benchmarkcard.config import get_llm_handler, Config

        handler = get_llm_handler(Config.COMPOSER_MODEL)
        return handler.engine
    except Exception as e:
        logger.warning("Failed to create inference engine for risk identification: %s", e)
        logger.warning("Risk identification will be skipped.")
        return None


def _format_risk(risk_obj) -> Dict[str, Any]:
    """Flatten an atlas Risk object into the card-facing dict."""
    return {
        "id": getattr(risk_obj, "id", "unknown"),
        "category": getattr(risk_obj, "name", "Unknown Risk"),
        "description": [getattr(risk_obj, "description", "")],
        "tag": getattr(risk_obj, "tag", None),
        "type": getattr(risk_obj, "type", None),
        "concern": getattr(risk_obj, "concern", None),
        "url": getattr(risk_obj, "url", None) or None,
        "taxonomy": getattr(risk_obj, "isDefinedByTaxonomy", None),
    }


def _get_rerank_handler():
    """LLM handler for the selection call; same composer model the pipeline already uses."""
    from auto_benchmarkcard.config import Config, get_llm_handler

    return get_llm_handler(Config.COMPOSER_MODEL)


RISK_SELECTION_PROMPT_TEMPLATE = """You are reviewing the documentation card of an AI benchmark. From a fixed candidate list of AI risks, select the risks that this specific benchmark plausibly raises.

BENCHMARK CARD SUMMARY:
{card_summary}

CANDIDATE RISKS (one per line, format: id | name | description):
{candidate_block}

Select between 0 and {max_risk} risks from the candidate list.
Rules:
- Select a risk only if you can justify it from concrete content of the card summary (its data source, data type, tasks, domains, audience, or methodology).
- For each selected risk write a one-line benchmark-specific justification that references that concrete content, for example: "collects real user health conversations -> privacy exposure".
- Selecting nothing is a valid and expected answer. Padding the list with generic risks that would apply to any benchmark is worse than returning fewer risks or none.
- Never select a risk that is not in the candidate list.

Answer with ONLY a JSON object of this form:
{{"selected": [{{"id": "<candidate id>", "justification": "<one line>"}}]}}
Use {{"selected": []}} if no candidate risk is specifically supported by the card."""


def build_risk_selection_prompt(
    card_summary: str, candidates: List[Dict[str, Any]], max_risk: int = MAX_SELECTED_RISKS
) -> str:
    """Render the selection prompt from the card summary and candidate risks."""
    lines = []
    for cand in candidates:
        desc = _normalize_risk_description(cand.get("description"))
        desc = " ".join(desc.split())[:_CANDIDATE_DESC_MAX_CHARS]
        lines.append(f"{cand.get('id', '')} | {cand.get('category', '')} | {desc}")
    return RISK_SELECTION_PROMPT_TEMPLATE.format(
        card_summary=card_summary,
        candidate_block="\n".join(lines),
        max_risk=max_risk,
    )


def _extract_json_block(raw: str) -> Optional[str]:
    """First balanced {...} or [...] block in raw (string-aware), or None."""
    if not raw:
        return None
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    open_c = raw[start]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def select_risks_with_justifications(
    llm_handler,
    card_summary: str,
    candidates: List[Dict[str, Any]],
    max_risk: int = MAX_SELECTED_RISKS,
) -> Optional[Dict[str, Any]]:
    """One LLM call that picks 0..max_risk candidate risks with per-risk justifications.

    Returns {"selected": [{"candidate": <candidate dict>, "justification": str}],
    "invalid_items": int} or None when the response is unparseable (caller falls
    back to the plain detector order). Unknown ids are dropped, duplicates
    collapse, the selection is capped at max_risk."""
    prompt = build_risk_selection_prompt(card_summary, candidates, max_risk)
    raw = llm_handler.generate(prompt)
    if not isinstance(raw, str):
        return None

    parsed = None
    for block in (raw.strip(), _extract_json_block(raw)):
        if not block:
            continue
        try:
            parsed = json.loads(block)
            break
        except (ValueError, TypeError):
            continue
    if isinstance(parsed, dict):
        items = parsed.get("selected")
    else:
        items = parsed
    if not isinstance(items, list):
        return None

    by_id = {c["id"]: c for c in candidates}
    by_name = {str(c.get("category", "")).strip().lower(): c for c in candidates}
    selected = []
    seen = set()
    invalid = 0
    for item in items:
        if len(selected) >= max_risk:
            break
        if not isinstance(item, dict):
            invalid += 1
            continue
        rid = item.get("id")
        cand = by_id.get(rid) if isinstance(rid, str) else None
        if cand is None and isinstance(rid, str):
            cand = by_name.get(rid.strip().lower())
        if cand is None:
            invalid += 1
            continue
        if cand["id"] in seen:
            continue
        seen.add(cand["id"])
        justification = item.get("justification")
        if isinstance(justification, str):
            justification = " ".join(justification.split())[:_JUSTIFICATION_MAX_CHARS]
        else:
            justification = ""
        selected.append({"candidate": cand, "justification": justification})
    return {"selected": selected, "invalid_items": invalid}


def identify_risks_from_benchmark_metadata(
    benchmark_card: Dict[str, Any],
    taxonomy: str = "ibm-risk-atlas",
    max_risk: int = MAX_SELECTED_RISKS,
    frame: Optional[Dict[str, Any]] = None,
    llm_handler=None,
    candidate_count: int = RISK_CANDIDATE_COUNT,
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Identify AI risks for a composed card: over-fetch candidates from the
    detector, then one LLM call selects 0..max_risk with justifications.

    Returns (risks, telemetry). risks is None when the stage could not run at
    all (old skip semantics), [] when it ran and selected nothing."""
    telemetry: Dict[str, Any] = {
        "candidate_count": 0,
        "candidates": [],
        "selected_count": 0,
        "selections": [],
        "fallback": False,
        "fallback_reason": None,
        "frame_node_count": len(frame.get("nodes", [])) if isinstance(frame, dict) else 0,
    }
    try:
        inference_engine = create_inference_engine()
        if not inference_engine:
            logger.warning("No inference engine available - skipping risk identification")
            telemetry["skipped"] = "no_inference_engine"
            return None, telemetry

        ai_atlas_nexus = _get_nexus()
        usecase = create_usecase_from_benchmark_card(benchmark_card, frame=frame)
        if not usecase:
            logger.warning("Could not create usecase description from benchmark card")
            telemetry["skipped"] = "no_usecase"
            return None, telemetry

        logger.debug("Retrieving candidate AI risks...")
        risk_lists = identify_risks_with_benchmark_detector(
            ai_atlas_nexus=ai_atlas_nexus,
            usecases=[usecase],
            inference_engine=inference_engine,
            taxonomy=taxonomy,
            max_risk=candidate_count,
        )
        risk_objects = risk_lists[0] if risk_lists else []
        candidates = [_format_risk(r) for r in risk_objects[:candidate_count]]
        telemetry["candidate_count"] = len(candidates)
        telemetry["candidates"] = [
            {"id": c["id"], "category": c["category"]} for c in candidates
        ]
        if not candidates:
            logger.debug("No candidate risks retrieved")
            return [], telemetry

        selection = None
        try:
            handler = llm_handler if llm_handler is not None else _get_rerank_handler()
            selection = select_risks_with_justifications(
                handler, usecase, candidates, max_risk=max_risk
            )
            if selection is None:
                telemetry["fallback_reason"] = "unparseable_response"
        except Exception as e:
            logger.warning("Risk selection LLM call failed, using detector order: %s", e)
            telemetry["fallback_reason"] = f"llm_error: {e}"[:200]

        if selection is None:
            telemetry["fallback"] = True
            fallback_risks = [dict(c) for c in candidates[:max_risk]]
            telemetry["selected_count"] = len(fallback_risks)
            return fallback_risks, telemetry

        risks = []
        for item in selection["selected"]:
            risk = dict(item["candidate"])
            if item["justification"]:
                risk["justification"] = item["justification"]
            risks.append(risk)
        telemetry["selected_count"] = len(risks)
        telemetry["selections"] = [
            {"id": r["id"], "category": r["category"], "justification": r.get("justification", "")}
            for r in risks
        ]
        telemetry["invalid_selection_items"] = selection["invalid_items"]
        logger.debug("Selected %d of %d candidate risks", len(risks), len(candidates))
        return risks, telemetry

    except Exception as e:
        logger.error("Error identifying risks: %s", e)
        telemetry["error"] = str(e)[:200]
        return None, telemetry


def _frame_names_line(frame: Optional[Dict[str, Any]]) -> str:
    """Frame node names grouped by type ("target: X; sibling: A, B"), or ""."""
    if not isinstance(frame, dict):
        return ""
    by_type: Dict[str, List[str]] = {}
    for node in frame.get("nodes", []):
        if isinstance(node, dict) and node.get("name"):
            by_type.setdefault(str(node.get("type", "other")), []).append(str(node["name"]))
    if not by_type:
        return ""
    return "; ".join(f"{ntype}: {', '.join(names)}" for ntype, names in by_type.items())


def create_usecase_from_benchmark_card(
    benchmark_card: Dict[str, Any], frame: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Build a textual use-case description from benchmark card metadata for risk detection."""
    _EMPTY = {"not specified", "not specified.", "no information found", ""}

    def _is_specified(val) -> bool:
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip().lower() not in _EMPTY
        if isinstance(val, list):
            return bool(val) and not (
                len(val) == 1 and isinstance(val[0], str) and val[0].strip().lower() in _EMPTY
            )
        return bool(val)

    def _join_list(val) -> str:
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        return str(val)

    try:
        details = benchmark_card.get("benchmark_details", {})
        purpose = benchmark_card.get("purpose_and_intended_users", {})
        data = benchmark_card.get("data", {})
        methodology = benchmark_card.get("methodology", {})
        ethical = benchmark_card.get("ethical_and_legal_considerations", {})

        name = details.get("name", "")
        overview = details.get("overview", "")
        data_type = details.get("data_type", "")
        domains = details.get("domains", [])
        languages = details.get("languages", [])
        tasks = purpose.get("tasks", [])
        goal = purpose.get("goal", "")
        audience = purpose.get("audience", [])
        limitations = purpose.get("limitations", "")

        data_source = data.get("source", "")
        data_size = data.get("size", "")
        annotation = data.get("annotation", "")

        methods = methodology.get("methods", [])
        metrics = methodology.get("metrics", [])
        license_info = ethical.get("data_licensing", "")

        parts = []

        if name:
            parts.append(f"{name} is a benchmark")

        if _is_specified(overview):
            parts.append(overview)

        if _is_specified(goal):
            parts.append(f"The goal is {goal}")

        if _is_specified(data_type):
            parts.append(f"Data type: {_join_list(data_type)}")

        if _is_specified(audience):
            parts.append(f"Intended audience: {_join_list(audience)}")

        if _is_specified(data_source):
            parts.append(f"The data was sourced from: {data_source}")

        if _is_specified(data_size):
            parts.append(f"Dataset size: {data_size}")

        if _is_specified(annotation):
            parts.append(f"Annotation: {annotation}")

        if _is_specified(methods):
            parts.append(f"Evaluation methods: {_join_list(methods)}")

        if _is_specified(metrics):
            parts.append(f"Metrics: {_join_list(metrics)}")

        if _is_specified(languages):
            parts.append(f"Languages: {_join_list(languages)}")

        if _is_specified(domains):
            parts.append(f"Domains: {_join_list(domains)}")

        if _is_specified(tasks):
            parts.append(f"Tasks: {_join_list(tasks)}")

        if _is_specified(limitations):
            parts.append(f"Known limitations: {limitations}")

        if _is_specified(license_info):
            parts.append(f"Data license: {license_info}")

        frame_line = _frame_names_line(frame)
        if frame_line:
            parts.append(f"Named entities around the benchmark ({frame_line})")

        if parts:
            usecase = ". ".join(parts).strip()
            if not usecase.endswith("."):
                usecase += "."
            return usecase
        else:
            logger.warning("Insufficient information to create usecase description")
            return None

    except Exception as e:
        logger.error("Error creating usecase from benchmark card: %s", e)
        return None


def _normalize_risk_description(desc: Any) -> str:
    """Collapse a risk description to a single string. The detector path wraps it in a list
    while the deterministic single-judge rule writes a plain string -- normalize both so every
    possible_risks item has a string description."""
    if isinstance(desc, list):
        return " ".join(str(d).strip() for d in desc if str(d).strip())
    return str(desc).strip() if desc is not None else ""


def integrate_risks_into_benchmark_card(
    benchmark_card: Dict[str, Any], risks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Add identified risks to the benchmark card under 'possible_risks'."""
    try:
        updated_card = benchmark_card.copy()

        cleaned_risks = []
        for risk in risks:
            cleaned_risk = {
                "category": risk["category"],
                "description": _normalize_risk_description(risk.get("description")),
                "url": risk.get("url") or None,
            }
            justification = risk.get("justification")
            if isinstance(justification, str) and justification.strip():
                cleaned_risk["justification"] = justification.strip()
            cleaned_risks.append(cleaned_risk)

        updated_card["possible_risks"] = cleaned_risks

        logger.debug("Successfully integrated %d risks into benchmark card", len(cleaned_risks))
        return updated_card

    except Exception as e:
        logger.error("Error integrating risks into benchmark card: %s", e)
        return benchmark_card


def omit_empty_risk_urls(card: Dict[str, Any]) -> Dict[str, Any]:
    """Presentation hygiene: drop the 'url' key from any possible_risks entry whose url is
    empty/None, so the frontend never renders a broken link (e.g. the synthetic single-judge
    bias risk, whose url is None by the frozen rule). Real atlas risks keep their urls.
    Mutates in place; returns the card."""
    risks = card.get("possible_risks")
    if isinstance(risks, list):
        for risk in risks:
            if isinstance(risk, dict) and not risk.get("url"):
                risk.pop("url", None)
    return card


def identify_and_integrate_risks(
    benchmark_card: Dict[str, Any],
    frame: Optional[Dict[str, Any]] = None,
    telemetry_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Identify AI risks and integrate them into the benchmark card.

    frame is the composer's referent-frame sidecar (conditioning input);
    telemetry_out, when a dict, is filled in place with the stage telemetry
    (candidates, selections, fallback flag) for the caller's sidecar."""
    try:
        logger.debug("Running AI Atlas Nexus analysis...")

        risks, telemetry = identify_risks_from_benchmark_metadata(benchmark_card, frame=frame)
        if isinstance(telemetry_out, dict):
            telemetry_out.update(telemetry)

        if risks is None:
            logger.warning("Risk identification failed - returning original benchmark card")
            return benchmark_card

        if not risks:
            logger.debug("No risks identified - returning original benchmark card")
            return benchmark_card

        updated_card = integrate_risks_into_benchmark_card(benchmark_card, risks)

        logger.debug("Risk analysis completed")
        return updated_card

    except Exception as e:
        logger.error("Error in risk identification and integration: %s", e)
        return benchmark_card
