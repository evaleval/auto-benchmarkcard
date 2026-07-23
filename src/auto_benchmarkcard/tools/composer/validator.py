"""Stage-B validator-loop.

Replaces the old blind 3x retry. Each Stage-B group is generated, validated, and
repaired with a source-free error->fix table for at most a few rounds; the best
attempt is kept and any still-invalid field is recorded in flagged_fields rather
than raising, so the card always ships.

Validators: structural (Pydantic), enum-membership (registry-driven), canonical
Not-specified, evidence-linkage (cited evidence_id must exist in the digest),
list-hygiene, and empty-structured collapse. Enum vocab + field classification
come from field_spec (the single source of truth shared with the prompt builder).
"""

import json
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ValidationError

from auto_benchmarkcard.card_utils import (_ABSENCE_PLACEHOLDER_RE,
                                           is_not_specified)
from auto_benchmarkcard.tools.composer import field_spec as fs

_NS_STRINGS = {"not specified", "not specified.", "no information found", "no information found."}
_MAX_ROUNDS = 3
_SEVERITY = {"structural": 100, "evidence_missing": 10, "leaked_evidence_id": 8, "enum_not_member": 5}

# Truncation handling. When an output is cut off (stop_reason == "length"), re-run the BASE
# prompt with a larger token budget rather than appending repair text (which only re-truncates).
# A single escalation; if it still truncates we stop and keep the best attempt.
_TRUNC_LADDER = (32768,)
_MAX_TRUNC_ESCALATIONS = 1

# Evidence ids are citations, never field values. A leak is a value whose *substantive*
# content is just evidence-id citation(s): a bare id ("E09"), the bracket form ("[E09]"),
# or a labelled citation ("Provenance: [E07]"). _EID_TOKEN_RE matches one id token (bracketed
# or bare); _EID_BOILERPLATE_RE matches the citation-label/connector words that surround an id
# but carry no field content. Strip both: if nothing substantive remains, the value is a leak.
_EID_TOKEN_RE = re.compile(r"\[\s*E\d+\s*\]|E\d+")
_EID_BOILERPLATE_RE = re.compile(
    r"\b(provenance|evidence|sources?|see|refs?|references?|reference|cite[ds]?|cites|"
    r"from|ids?|and|or)\b",
    re.IGNORECASE,
)


@dataclass
class VErr:
    path: str
    type: str
    detail: str = ""
    allowed: Optional[List[str]] = None
    observed: Any = None
    cap: Optional[int] = None
    n_miss: int = 0  # out-of-vocab token count, for keep-best (enum_check collapses all misses into one VErr)


@dataclass
class Attempt:
    rnd: int
    parsed: Optional[dict]
    norm: Optional[Dict[str, dict]]
    errors: List[VErr]
    stats: Dict[str, int]


@dataclass
class GroupOutcome:
    sections: Dict[str, dict]
    flagged: Dict[str, str]
    telemetry: dict = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# robust JSON parsing (self-contained balanced-brace extractor)
# ---------------------------------------------------------------------------

def robust_json_loads(raw: Any) -> Tuple[Optional[dict], Optional[str]]:
    """Parse a model response into a dict. Tolerates code fences and trailing
    prose by extracting the first balanced {...} object. Returns (obj, None) or
    (None, error_message)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty output"
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s).strip()
    try:
        obj = json.loads(s)
        return (obj, None) if isinstance(obj, dict) else (None, "top-level JSON is not an object")
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        return None, "no JSON object found"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                cand = s[start:i + 1]
                try:
                    obj = json.loads(cand)
                    return (obj, None) if isinstance(obj, dict) else (None, "extracted JSON is not an object")
                except Exception as e:
                    return None, f"unparseable: {e}"
    return None, "unbalanced braces"


# ---------------------------------------------------------------------------
# per-field validation helpers
# ---------------------------------------------------------------------------

def _canonical_ns(field: str):
    return ["Not specified"] if field in fs.LIST_FIELDS else "Not specified"


def _is_ns_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() in _NS_STRINGS


def _is_eid_leak_value(val: Any) -> bool:
    """True when a scalar value's substantive content is just an evidence-id citation.

    Catches bare ids ("E09"), bracketed ids ("[E09]"), and labelled citations
    ("Provenance: [E07]"). Strips id tokens and citation-label/connector boilerplate; if no
    alphanumeric content remains the value is a leaked citation. Prose that merely mentions an
    id keeps its other words and is not flagged. Non-strings (bool/int scalars) are never leaks.
    """
    if not isinstance(val, str) or not _EID_TOKEN_RE.search(val):
        return False
    residue = _EID_TOKEN_RE.sub(" ", val)
    residue = _EID_BOILERPLATE_RE.sub(" ", residue)
    residue = re.sub(r"[^0-9A-Za-z]+", " ", residue).strip()
    return residue == ""


def collapse_empty(val: Any):
    """Empty/placeholder structured value -> the NS sentinel (fixes the {} poison).

    A dict counts as placeholder when it has no keys OR all its values are empty/zero
    (e.g. size_breakdown={'all': 0}). Scoped to dict/list/None so a scalar judge_uses_llm
    =False or judge_num=0 (also COLLAPSE_FIELDS) keeps its real value."""
    if val is None:
        return "Not specified", True
    if isinstance(val, dict) and (not val or not any(val.values())):
        return "Not specified", True
    if isinstance(val, list) and not val:
        return "Not specified", True
    return val, False


def ns_normalize(field: str, val: Any):
    """Canonicalize NS and leak-phrasing absences. Returns (value, was_leak_normalized)."""
    if is_not_specified(val):
        return _canonical_ns(field), False
    if isinstance(val, str):
        text = val
    elif isinstance(val, list):
        text = " ".join(str(v) for v in val)
    else:
        text = None
    if text is not None and _ABSENCE_PLACEHOLDER_RE.search(text):
        return _canonical_ns(field), True
    return val, False


def list_hygiene(path: str, val: Any):
    """Bare-string coerce, drop NS mixed with reals, dedup, truncate to cap. In-code fix only."""
    if isinstance(val, str):
        val = [val]
    if not isinstance(val, list):
        return val
    reals = [x for x in val if not _is_ns_str(x)]
    if not reals:
        return ["Not specified"]
    seen = set()
    out = []
    for x in reals:
        key = x.lower() if isinstance(x, str) else json.dumps(x, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    cap = fs.LIST_CAPS.get(path)
    if cap and len(out) > cap:
        out = out[:cap]
    return out


def enum_tokens(path: str, val: Any) -> List[str]:
    if path in fs.ENUM_COMMA_JOINED:
        return [t.strip().lower() for t in str(val).split(",") if t.strip()]
    if path in fs.ENUM_MULTI:
        items = val if isinstance(val, list) else [val]
        return [str(t).strip() for t in items]
    return [str(val).strip()]


def enum_check(path: str, val: Any) -> Tuple[List[str], bool]:
    """Return (out-of-vocab tokens, used_other_escape)."""
    vocab = fs.enum_vocab(path)
    if vocab is None:
        return [], False
    misses = []
    used_other = False
    for t in enum_tokens(path, val):
        if t.lower().startswith("other:"):
            used_other = True
            continue
        if t not in vocab:
            misses.append(t)
    return misses, used_other


_DATA_TYPE_COMPOUND_RE = re.compile(r"\s+and\s+|\s*&\s*|\s+or\s+|/|,")

# Answer-format / metric phrasings: they describe how a task is answered or scored, not
# the data modality. A data_type other: suffix built only from these carries no modality
# signal and is dropped. Benchmark-agnostic; never reference a specific benchmark.
_DATA_TYPE_OTHER_NOISE = frozenset({
    "multiple-choice", "multiple choice", "mcq", "single-choice",
    "true-false", "true/false", "yes-no", "yes/no", "binary",
    "open-ended", "open ended", "free-form", "free-response",
    "fill-in-the-blank", "cloze", "short-answer", "long-answer", "essay",
    "accuracy", "f1", "bleu", "rouge", "exact-match", "pass@k", "elo",
})

# Obvious modality aliases -> canonical DATA_TYPE_VOCAB member, so a junk other: suffix
# naming a real modality is coerced instead of kept as other:.
_DATA_TYPE_SYNONYMS = {
    "free-text": "text", "textual": "text", "natural-language": "text", "nl": "text",
    "images": "image", "visual": "image", "vision": "image",
    "videos": "video", "speech": "audio",
    "tables": "tabular", "table": "tabular",
    "source-code": "code", "programs": "code", "program": "code",
}


def _clean_data_type_other(val: Any) -> Any:
    """Backstop for a data_type that abuses the other: escape with a compound or a
    task-format/metric phrasing (e.g. 'other:multiple-choice and error-detection').
    Coerce to the in-vocab modalities named in the value, else to a single clean
    other:<token> with answer-format/metric noise dropped, else NS. Idempotent and a
    no-op on clean values (a single in-vocab modality, a clean multi-modality, or a clean
    single other:<modality>). General content rule; benchmark-agnostic."""
    path = "benchmark_details.data_type"
    vocab = fs.enum_vocab(path) or set()
    out: List[str] = []
    for tok in enum_tokens(path, val):
        if not tok.startswith("other:"):
            out.append(tok)
            continue
        suffix = tok[len("other:"):].strip()
        parts = [p.strip() for p in _DATA_TYPE_COMPOUND_RE.split(suffix) if p.strip()]
        mods = [_DATA_TYPE_SYNONYMS.get(p, p) for p in parts
                if p in vocab or p in _DATA_TYPE_SYNONYMS]
        if mods:
            out.extend(mods)
            continue
        kept = [p for p in parts if p not in _DATA_TYPE_OTHER_NOISE]
        if kept:
            out.append("other:" + kept[0])
    seen = set()
    cleaned = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            cleaned.append(t)
    if not cleaned:
        return _canonical_ns("data_type")
    return ", ".join(cleaned)


def evidence_check(path: str, field_prov: Any, digest_ids: set) -> Optional[VErr]:
    ids = field_prov.get("evidence_ids") if isinstance(field_prov, dict) else None
    if not ids or not isinstance(ids, list):
        return VErr(path, "evidence_missing", "value present but no evidence_ids cited")
    bad = [i for i in ids if i not in digest_ids]
    if bad:
        return VErr(path, "evidence_missing", f"cites unknown evidence_ids {bad}")
    return None


# Contract v1 §2: provenance source/evidence stay STRINGS. The Dict[..., Any]
# widening (needed so quote_loc/evidence_ids survive) dropped Pydantic's type wall,
# so a dict/list-valued source or evidence (the LLM confusing it with the adjacent
# evidence_ids) would now pass through verbatim and later crash the downstream
# consumers (_is_structured_source does (source or "").lower(); backfill_from_provenance
# does evidence.lower()). A non-string source/evidence carries no usable grounding,
# so we drop it (treated as absent); quote_loc/evidence_ids keep their structured types.
_PROV_STR_KEYS = ("source", "evidence")


def _sanitize_provenance(prov: Any) -> Any:
    if not isinstance(prov, dict):
        return {}
    out = {}
    for field, entry in prov.items():
        if isinstance(entry, dict):
            entry = {k: v for k, v in entry.items()
                     if not (k in _PROV_STR_KEYS and not isinstance(v, str))}
        out[field] = entry
    return out


# ---------------------------------------------------------------------------
# section + group validation
# ---------------------------------------------------------------------------

def _required_fields(model_cls) -> List[str]:
    return [n for n, f in model_cls.model_fields.items() if f.is_required()]


def _typecheck(model_cls, secdict: dict) -> Tuple[Optional[dict], List[VErr]]:
    try:
        obj = model_cls(**secdict)
        return obj.model_dump(exclude_none=True), []
    except ValidationError as e:
        errs = []
        for er in e.errors():
            loc = er["loc"][0] if er.get("loc") else "?"
            errs.append(VErr(f"_SEC_.{loc}", "structural", er.get("msg", "invalid")))
        return None, errs


def validate_sections(parsed: dict, section_names: List[str],
                      section_models: Dict[str, Any], digest_ids: set,
                      exempt_extra: set) -> Tuple[Dict[str, dict], List[VErr], Dict[str, int]]:
    errors: List[VErr] = []
    stats = {"ns_normalized": 0, "n_other": 0, "n_enum_fields": 0,
             "established_mismatch": 0, "evidence_demoted": 0}
    norm: Dict[str, dict] = {}
    evidence_exempt = fs.EVIDENCE_EXEMPT | exempt_extra

    for sec in section_names:
        raw_sec = parsed.get(sec)
        if not isinstance(raw_sec, dict):
            errors.append(VErr(f"{sec}", "structural", "section missing or not an object"))
            raw_sec = {}
        secdict = dict(raw_sec)
        prov = _sanitize_provenance(secdict.get("provenance"))
        secdict["provenance"] = prov

        for fld in list(secdict):
            if fld == "provenance":
                continue
            path = f"{sec}.{fld}"
            val = secdict[fld]

            if path in fs.COLLAPSE_FIELDS:
                val, _ = collapse_empty(val)
            val, leaked = ns_normalize(fld, val)
            if leaked:
                stats["ns_normalized"] += 1
            secdict[fld] = val
            if is_not_specified(val):
                continue

            # Evidence-id leakage. A field whose substantive content is an evidence-id citation
            # is a citation, not a value. For list fields drop only the offending element(s),
            # keeping valid items (-> NS if that empties the list); for scalar fields flag the
            # whole field so the repair loop / finalize replaces it with NS.
            if fld in fs.LIST_FIELDS:
                if isinstance(val, list):
                    kept = [x for x in val if not _is_eid_leak_value(x)]
                    if len(kept) != len(val):
                        val = kept or _canonical_ns(fld)
                        secdict[fld] = val
                        if is_not_specified(val):
                            continue
                elif _is_eid_leak_value(val):
                    secdict[fld] = _canonical_ns(fld)
                    continue
            elif _is_eid_leak_value(val):
                errors.append(VErr(path, "leaked_evidence_id",
                                   "field value is an evidence-id citation, not a value"))
                continue

            if fld in fs.LIST_FIELDS:
                val = list_hygiene(path, val)
                secdict[fld] = val
                if is_not_specified(val):
                    continue

            if path in fs.ENUM_REGISTRY:
                stats["n_enum_fields"] += 1
                misses, used_other = enum_check(path, val)
                if used_other:
                    stats["n_other"] += 1
                if misses:
                    if path in fs.B_AUTHORED_ENUMS:
                        errors.append(VErr(path, "enum_not_member",
                                           f"{misses} not allowed",
                                           allowed=sorted(fs.enum_vocab(path))[:25],
                                           observed=val, n_miss=len(misses)))
                    else:
                        stats["established_mismatch"] += 1
                        # Established-enum backstop (defense-in-depth with the Wave-1 evidence drop):
                        # an out-of-vocab established value must never ship. Drop only the bad
                        # tokens and keep the valid ones; coerce to NS only when nothing valid
                        # remains -- never guess. data_type + data_licensing are scalars (a
                        # comma-joined string / a single token, joined back the same way);
                        # languages is a list, so filter element-wise.
                        if path in ("benchmark_details.data_type",
                                    "ethical_and_legal_considerations.data_licensing"):
                            vocab = fs.enum_vocab(path)
                            kept = [t for t in enum_tokens(path, val)
                                    if t in vocab or t.lower().startswith("other:")]
                            secdict[fld] = ", ".join(kept) if kept else _canonical_ns(fld)
                        elif path == "benchmark_details.languages":
                            vocab = fs.enum_vocab(path)
                            kept = [t for t in enum_tokens(path, val)
                                    if t in vocab or t.lower().startswith("other:")]
                            secdict[fld] = kept if kept else _canonical_ns(fld)
                elif used_other and path == "benchmark_details.data_type":
                    # C4: the misses path above is skipped when only other: tokens are
                    # present, so a compound / task-format other: in the modality field
                    # (e.g. 'other:multiple-choice and error-detection') would ship as-is.
                    # Clean it to real modalities or a single clean other:<modality>.
                    cleaned = _clean_data_type_other(val)
                    if cleaned != val:
                        secdict[fld] = cleaned

            if path not in evidence_exempt:
                ev = evidence_check(path, prov.get(fld), digest_ids)
                if ev:
                    errors.append(ev)

        norm[sec] = secdict

    # structural typecheck per section against the real Pydantic model
    for sec in section_names:
        typed, serrs = _typecheck(section_models[sec], norm[sec])
        for e in serrs:
            e.path = e.path.replace("_SEC_", sec, 1)
        errors.extend(serrs)
        if typed is not None:
            typed["provenance"] = norm[sec].get("provenance")
            norm[sec] = typed
    return norm, errors, stats


# ---------------------------------------------------------------------------
# error -> fix table (SOURCE-FREE) + repair prompt
# ---------------------------------------------------------------------------

FIX_TABLE = {
    "unparseable": ("Your output was not valid JSON. Return ONE JSON object only - no prose, no "
                    "markdown fences, no trailing commas; every string double-quoted and every brace balanced."),
    "structural": ("Field {F} is required and must have the correct type. If you have no grounded value, "
                   "set it to \"Not specified\" (or [\"Not specified\"] for a list field). Do not omit it."),
    "enum_not_member": ("Value {X} for field {F} is not in the allowed set {S}. Pick the single closest "
                        "allowed value, or if none fits use \"other:<short text>\". Do not invent a value outside the set."),
    "evidence_missing": ("Field {F} has a substantive value but cites no valid evidence id from the digest. "
                         "Either add an evidence_ids entry referencing the digest id(s) that support it, or set "
                         "the field to \"Not specified\". Only ids that already appear in the digest are valid - do not invent ids."),
    "leaked_evidence_id": ("Field {F} is a bare evidence id (e.g. E01). Evidence ids are citations, not values - "
                           "put the real text or \"Not specified\", and cite the id under provenance.evidence_ids instead."),
}

_FIX_RULES_BLOCK = (
    "FIX RULES:\n"
    "- Return the FULL corrected JSON for this group. Do not change fields that are not listed below.\n"
    "- Do not add new fields. Keep every evidence_ids you already had for unlisted fields.\n"
    "- \"Not specified\" is always an acceptable value when you have no grounded evidence."
)


def _fmt_allowed(allowed: Optional[List[str]]) -> str:
    if not allowed:
        return "(the allowed set)"
    shown = ", ".join(allowed[:25])
    return "{" + shown + ("" if len(allowed) <= 25 else ", ...") + "}"


def build_repair_prompt(base_prompt: str, errors: List[VErr]) -> str:
    lines = []
    for e in errors:
        tmpl = FIX_TABLE.get(e.type, "Fix field {F}.")
        observed = str(e.observed)[:80] if e.observed is not None else ""
        lines.append("- " + tmpl.format(F=e.path, X=repr(observed), S=_fmt_allowed(e.allowed)))
    return (
        base_prompt
        + "\n\n=== YOUR PREVIOUS OUTPUT FAILED VALIDATION - FIX ONLY THESE FIELDS ===\n"
        + "\n".join(lines)
        + "\n\n" + _FIX_RULES_BLOCK
    )


# ---------------------------------------------------------------------------
# keep-best + finalize
# ---------------------------------------------------------------------------

def _attempt_key(a: Attempt):
    parse_ok = a.norm is not None
    weight = sum(_SEVERITY.get(e.type, 1) for e in a.errors)
    # enum_check collapses every out-of-vocab token of a field into ONE enum_not_member
    # VErr, so len(errors)/weight alone can't tell a 1-bad-token attempt from a 3-bad-token
    # one. Fold the total miss-token count in so the cleaner attempt wins, and break any
    # remaining tie toward the EARLIER round (a later round that didn't measurably improve
    # is never preferred). lower is better; a parsed attempt always beats an unparseable one.
    miss_tokens = sum(e.n_miss for e in a.errors)
    return (0 if parse_ok else 1, len(a.errors), weight, miss_tokens, a.rnd)


def keep_best(attempts: List[Attempt]) -> Attempt:
    return min(attempts, key=_attempt_key)


def _set_ns(sections: Dict[str, dict], path: str):
    sec, _, fld = path.partition(".")
    if sec in sections and isinstance(sections[sec], dict):
        sections[sec][fld] = _canonical_ns(fld)


def _ensure_required(model_cls, secdict: dict) -> dict:
    out = dict(secdict or {})
    for name in _required_fields(model_cls):
        if name not in out or out[name] is None:
            out[name] = _canonical_ns(name)
    return out


def _skeleton(model_cls) -> dict:
    return {name: _canonical_ns(name) for name in _required_fields(model_cls)}


def finalize(best: Attempt, section_names: List[str], section_models: Dict[str, Any],
             telemetry: dict) -> GroupOutcome:
    flagged: Dict[str, str] = {}
    evidence_demoted = 0

    if best.norm is None:
        sections = {sec: _skeleton(section_models[sec]) for sec in section_names}
        for sec in section_names:
            for fld in sections[sec]:
                flagged[f"{sec}.{fld}"] = "[schema_invalid] unparseable group output"
    else:
        sections = best.norm
        for e in best.errors:
            if e.type == "enum_not_member":
                flagged[e.path] = f"[schema_invalid] enum: {e.detail}"  # keep best-effort value
            elif e.type == "evidence_missing":
                _set_ns(sections, e.path)
                flagged[e.path] = f"[schema_invalid] evidence: {e.detail}"
                evidence_demoted += 1
            elif e.type == "leaked_evidence_id":
                _set_ns(sections, e.path)
                flagged[e.path] = f"[schema_invalid] leaked_evidence_id: {e.detail}"
            elif e.type == "structural":
                _set_ns(sections, e.path)
                flagged[e.path] = f"[schema_invalid] structural: {e.detail}"

    for sec in section_names:
        sections[sec] = _ensure_required(section_models[sec], sections.get(sec) or {})

    telemetry = dict(telemetry)
    telemetry["evidence_demoted"] = telemetry.get("evidence_demoted", 0) + evidence_demoted
    telemetry["schema_invalid"] = sorted(flagged)
    return GroupOutcome(sections=sections, flagged=flagged, telemetry=telemetry)


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------

def _call_gen(generate_fn, prompt, max_tokens):
    """Call generate_fn, optionally raising the per-call token budget on truncation.

    Default path (max_tokens is None) calls generate_fn(prompt) so legacy/test fakes that
    accept only a prompt keep working; the max_tokens kwarg is passed only when escalating,
    with a TypeError fallback for 1-arg fakes."""
    if max_tokens is None:
        return generate_fn(prompt)
    try:
        return generate_fn(prompt, max_tokens=max_tokens)
    except TypeError:
        return generate_fn(prompt)


def _unwrap(raw):
    """generate_fn may return raw text (str) or (text, stop_reason). Normalize to a pair."""
    return (raw[0], raw[1]) if isinstance(raw, tuple) else (raw, None)


def run_group_with_repair(group_name: str, section_names: List[str],
                          section_models: Dict[str, Any], base_prompt: str,
                          generate_fn: Callable[[str], str], digest_ids,
                          det_filled_paths=None, max_rounds: int = _MAX_ROUNDS) -> GroupOutcome:
    """Generate -> validate -> source-free repair (capped) -> keep best -> ship.

    generate_fn(prompt) -> raw model text. digest_ids: iterable of valid evidence ids
    from [A]'s digest. det_filled_paths: extra evidence-exempt dotted paths filled by [0]."""
    digest_ids = set(digest_ids or ())
    exempt_extra = set(det_filled_paths or ())
    attempts: List[Attempt] = []
    prompt = base_prompt
    cur_max_tokens = None
    trunc_escalations = 0
    truncation_retries = 0

    for rnd in range(max_rounds):
        try:
            raw = _call_gen(generate_fn, prompt, cur_max_tokens)
        except Exception as e:  # generation failed -> treat as an unparseable attempt
            text, stop_reason = "", None
            parsed, perr = None, f"generation error: {e}"
        else:
            text, stop_reason = _unwrap(raw)
            parsed, perr = robust_json_loads(text)

        if parsed is None:
            att = Attempt(rnd, None, None, [VErr(group_name, "unparseable", perr or "unparseable")], {})
        else:
            norm, errors, stats = validate_sections(
                parsed, section_names, section_models, digest_ids, exempt_extra)
            att = Attempt(rnd, parsed, norm, errors, stats)
        attempts.append(att)

        if att.norm is not None and not att.errors:
            return _finalize_with_telemetry(att, attempts, section_names, section_models,
                                            converged=True, truncation_retries=truncation_retries)

        truncated = stop_reason == "length"
        if truncated and trunc_escalations >= _MAX_TRUNC_ESCALATIONS:
            # Out of escalations and still overflowing the cap: a longer repair prompt would
            # only re-truncate. Stop and keep the best attempt.
            break

        if rnd < max_rounds - 1:
            if truncated:
                # Re-run the BASE prompt with a bigger budget; do NOT append repair text.
                cur_max_tokens = _TRUNC_LADDER[min(trunc_escalations, len(_TRUNC_LADDER) - 1)]
                trunc_escalations += 1
                truncation_retries += 1
                prompt = base_prompt
            else:
                err_list = att.errors or [VErr(group_name, "unparseable", perr or "unparseable")]
                prompt = build_repair_prompt(base_prompt, err_list)

    best = keep_best(attempts)
    return _finalize_with_telemetry(best, attempts, section_names, section_models,
                                    converged=False, truncation_retries=truncation_retries)


def _finalize_with_telemetry(best: Attempt, attempts: List[Attempt], section_names,
                             section_models, converged: bool, truncation_retries: int = 0) -> GroupOutcome:
    stats = best.stats or {}
    n_enum = stats.get("n_enum_fields", 0)
    telemetry = {
        "first_pass_valid": (attempts[0].norm is not None and not attempts[0].errors),
        "iterations_to_valid": next((a.rnd + 1 for a in attempts if a.norm is not None and not a.errors), len(attempts)),
        "converged": converged,
        "round_error_counts": [len(a.errors) for a in attempts],
        "n_enum_fields": n_enum,
        "n_other": stats.get("n_other", 0),
        "other_rate": (stats.get("n_other", 0) / n_enum) if n_enum else 0.0,
        "ns_normalized": stats.get("ns_normalized", 0),
        "established_mismatch": stats.get("established_mismatch", 0),
        "evidence_demoted": stats.get("evidence_demoted", 0),
        "truncation_retries": truncation_retries,
    }
    return finalize(best, section_names, section_models, telemetry)
