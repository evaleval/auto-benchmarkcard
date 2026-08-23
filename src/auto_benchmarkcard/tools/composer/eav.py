"""Entity-Attribution Verification (EAV): a model-based assignment audit.

Runs between evidence finalization and Stage B. Stage A proves every quote is
verbatim from a source; this pass audits what a verified quote is ASSIGNED to:
whether it genuinely describes its claimed referent (frame entity) and whether
its content is what its card field is about. Wrong-section splices survive
verbatim verification precisely because the quote is real; only the attribution
is wrong, and that is the axis checked here.

Generative, not embedding-based: one grouped LLM call per batch renders the
referent frame plus one compact line per item and asks for per-item verdicts.
Actions are conservative and fail-open per item: a field mismatch demotes the
item out of the digest (Stage B never sees it), a referent mismatch with a
valid corrected frame id updates the item's referent, a referent mismatch with
no valid correction demotes. Items are never added and quotes never edited.
Demotion reasons land in telemetry for the composer's later repair loop; the
module stays pure (digest in, digest out; the input digest is never mutated).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from auto_benchmarkcard.tools.composer import evidence

logger = logging.getLogger(__name__)

try:
    from auto_benchmarkcard.tools.composer.validator import robust_json_loads
except Exception:  # the balanced-brace fallback in _parse_verdicts covers this
    robust_json_loads = None

# Fields where a mis-assigned quote does the most damage: numbers and results
# splice silently (a sibling's size or score reads exactly like the target's).
HIGH_STAKES_FIELDS = frozenset({
    "benchmark_details.overview",
    "data.size",
    "data.size_breakdown",
    "data.source",
    "methodology.metrics",
    "methodology.calculation",
    "methodology.baseline_results",
    "methodology.human_baseline",
})
_JUDGE_PREFIX = "methodology.judge_"

# Document regions where quotes about OTHER systems and ablated variants cluster.
RISKY_REGIONS = frozenset({"results", "ablation", "appendix"})

REASON_CAP = 300  # max characters kept from a model-written verdict reason

EAV_SYSTEM = (
    "You are a careful auditor of evidence assignments for a benchmark documentation "
    "card. Every quote below was copied verbatim from a real source document; your job "
    "is to judge what each quote is assigned to, not whether it is real. You use only "
    "the quote text shown, never outside knowledge, and you never invent entity ids."
)

EAV_INSTRUCTIONS = (
    "Each evidence item below is filed under a card FIELD and claimed to be about one "
    "entity (its REFERENT). For every item return one verdict object:\n"
    '- "about_ok": true if the quote genuinely describes the claimed referent; false if '
    "it describes a different entity from the list above (a sibling benchmark, a "
    "subset, a version, a split, a metric).\n"
    '- "field_ok": true if the quote\'s content is what the FIELD is about (a metrics '
    "quote is about how performance is measured, a size quote about dataset size, a "
    "source quote about where the data came from, and so on); false if the content "
    "belongs to a different topic.\n"
    '- "referent": when about_ok is false and you can tell which listed entity the '
    "quote is really about, that entity's id; otherwise repeat the claimed id.\n"
    '- "reason": a short justification.\n'
    "\n"
    'Return ONLY a JSON object of this form: {"verdicts": [{"id": "<item id>", '
    '"about_ok": true|false, "field_ok": true|false, "referent": "<entity id>", '
    '"reason": "<short>"}]}\n'
    "Include exactly one verdict per item. Use only entity ids from the list above. "
    "Judge only from the quote text given."
)


def _is_high_stakes(field: str) -> bool:
    return field in HIGH_STAKES_FIELDS or field.startswith(_JUDGE_PREFIX)


def _select_items(evidence_digest: Dict[str, List[dict]], target_id: str) -> List[dict]:
    """Items worth checking, priority-ordered: high-stakes field first, then
    non-target referent, then risky region; digest order within each bucket.
    An item is taken once, at its highest-priority reason. An empty referent is
    not treated as non-target (it means no frame was present at extraction, not
    a claim about another entity)."""
    buckets: Tuple[List[dict], List[dict], List[dict]] = ([], [], [])
    for field, items in evidence_digest.items():
        high = _is_high_stakes(field)
        for it in items or []:
            if not it.get("evidence_id"):
                continue  # verdicts are keyed by id; an id-less item cannot be audited
            referent = it.get("referent", "")
            if high:
                buckets[0].append(it)
            elif referent and referent != target_id:
                buckets[1].append(it)
            elif it.get("region", "") in RISKY_REGIONS:
                buckets[2].append(it)
    return [*buckets[0], *buckets[1], *buckets[2]]


def _item_line(it: dict) -> str:
    path = it.get("section_path", "") or ""
    section = path.split(" > ")[-1] if path else ""
    section = section or it.get("region", "") or "unknown"
    referent = it.get("referent", "") or "unknown"
    return (f'- id={it["evidence_id"]} | field={it.get("field", "")} '
            f'| referent={referent} | section={section} | quote="{it.get("quote", "")}"')


def build_eav_prompt(batch: List[dict], frame: Dict) -> str:
    """Assemble the grouped verdict prompt for one batch of evidence items."""
    return "\n\n".join([
        EAV_SYSTEM,
        evidence.render_frame_block(frame),
        EAV_INSTRUCTIONS,
        "Items:\n" + "\n".join(_item_line(it) for it in batch),
    ])


def _balanced_block(raw: str, open_ch: str, close_ch: str) -> Optional[str]:
    """First balanced open..close block in raw (string-aware), or None."""
    start = raw.find(open_ch)
    if start < 0:
        return None
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
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def _parse_list(raw: str) -> Optional[List]:
    block = _balanced_block(raw, "[", "]")
    if not block:
        return None
    try:
        parsed = json.loads(block)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_verdicts(raw) -> List[dict]:
    """Tolerant verdict parse. Handles the {"verdicts": [...]} envelope (via
    robust_json_loads when importable, else a balanced-brace scan), a bare
    top-level list, and a single bare verdict object; fences and surrounding
    prose are tolerated. Anything unparseable yields [] (batch fail-open)."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    first_obj = raw.find("{")
    first_list = raw.find("[")
    if first_list != -1 and (first_obj == -1 or first_list < first_obj):
        verdicts = _parse_list(raw)
        return [v for v in verdicts or [] if isinstance(v, dict)]

    obj = None
    if robust_json_loads is not None:
        obj, _ = robust_json_loads(raw)
    else:
        block = _balanced_block(raw, "{", "}")
        if block:
            try:
                parsed = json.loads(block)
            except ValueError:
                parsed = None
            obj = parsed if isinstance(parsed, dict) else None
    verdicts = None
    if isinstance(obj, dict):
        if isinstance(obj.get("verdicts"), list):
            verdicts = obj["verdicts"]
        elif "id" in obj:
            verdicts = [obj]
    if verdicts is None:
        verdicts = _parse_list(raw)
    return [v for v in verdicts or [] if isinstance(v, dict)]


def run_eav(evidence_digest: Dict[str, List[dict]], frame: Optional[Dict], llm_handler,
            max_items: int = 60, batch_size: int = 25) -> Tuple[Dict[str, List[dict]], Dict]:
    """Audit evidence assignments; return (updated_digest, telemetry).

    Selects the items worth checking (see _select_items), caps at max_items
    (overflow lands in telemetry["skipped"]), asks the LLM for grouped verdicts
    in batches of batch_size, and applies them: field_ok false demotes the item
    out of the digest, about_ok false re-points the referent when the model
    names a valid frame id and demotes otherwise. Unknown ids, malformed
    entries, and non-bool flags are ignored per item; a batch that raises is
    left entirely untouched and counted in telemetry["batch_errors"]. The input
    digest is never mutated; a field whose last item is demoted is dropped from
    the output (a digest key always implies at least one item)."""
    telemetry: Dict = {"checked": 0, "skipped": 0, "demoted": [], "corrected": 0,
                       "batch_errors": 0, "calls": 0}
    if not evidence_digest or llm_handler is None or not frame or not frame.get("nodes"):
        return evidence_digest, telemetry

    target_id = frame.get("target_id", "target")
    frame_ids = {n["id"] for n in frame.get("nodes", [])}

    selected = _select_items(evidence_digest, target_id)
    if len(selected) > max_items:
        telemetry["skipped"] = len(selected) - max_items
        selected = selected[:max_items]
    telemetry["checked"] = len(selected)
    if not selected:
        return evidence_digest, telemetry

    items_by_id: Dict[str, dict] = {}
    for items in evidence_digest.values():
        for it in items or []:
            items_by_id.setdefault(it.get("evidence_id", ""), it)

    demote: Dict[str, str] = {}   # evidence_id -> reason
    correct: Dict[str, str] = {}  # evidence_id -> corrected referent
    step = max(1, batch_size)
    for start in range(0, len(selected), step):
        batch = selected[start:start + step]
        batch_ids = {it["evidence_id"] for it in batch}
        try:
            prompt = build_eav_prompt(batch, frame)
            telemetry["calls"] += 1
            verdicts = _parse_verdicts(llm_handler.generate(prompt))
        except Exception as exc:
            telemetry["batch_errors"] += 1
            logger.warning("EAV batch failed, %d items left untouched: %s", len(batch), exc)
            continue
        for v in verdicts:
            vid = str(v.get("id", "") or "").strip()
            if vid not in batch_ids:
                continue  # unknown or out-of-batch id: fail-open
            reason = str(v.get("reason", "") or "").strip()[:REASON_CAP]
            if v.get("field_ok") is False:
                demote[vid] = "field_mismatch" + (f": {reason}" if reason else "")
            elif v.get("about_ok") is False:
                new_ref = str(v.get("referent", "") or "").strip()
                if new_ref in frame_ids and new_ref != items_by_id[vid].get("referent", ""):
                    correct[vid] = new_ref
                else:
                    demote[vid] = "referent_unresolved" + (f": {reason}" if reason else "")

    updated: Dict[str, List[dict]] = {}
    for field, items in evidence_digest.items():
        kept: List[dict] = []
        for it in items or []:
            vid = it.get("evidence_id", "")
            if vid in demote:
                telemetry["demoted"].append({
                    "id": vid, "field": field,
                    "referent": it.get("referent", ""), "reason": demote[vid],
                })
                continue
            if vid in correct:
                it = {**it, "referent": correct[vid]}
                telemetry["corrected"] += 1
            kept.append(it)
        if kept:
            updated[field] = kept
    return updated, telemetry
