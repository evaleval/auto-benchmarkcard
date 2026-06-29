"""Card processing utilities: field extraction, normalization, and HF tag overrides."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_card(obj: dict) -> dict:
    """Unwrap a 'benchmark_card' key if present, otherwise return as-is."""
    return obj.get("benchmark_card", obj) if isinstance(obj, dict) else obj


def is_not_specified(value: Any) -> bool:
    """Return True if value is a sentinel 'Not specified' placeholder."""
    _EMPTY_VALUES = {"not specified", "not specified.", "no information found", "no information found."}
    if isinstance(value, str) and value.strip().lower() in _EMPTY_VALUES:
        return True
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str) and value[0].strip().lower() in _EMPTY_VALUES:
        return True
    return False


_SKIP_SECTIONS = {"possible_risks", "flagged_fields", "missing_fields", "card_info", "evaluation_summary", "provenance"}


def extract_missing_fields(data: Any, prefix: str = "") -> List[str]:
    """Recursively collect dotted paths for fields with 'Not specified' values."""
    missing_fields = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key in _SKIP_SECTIONS:
                continue
            current_path = f"{prefix}.{key}" if prefix else key

            if is_not_specified(value):
                missing_fields.append(current_path)
            elif isinstance(value, (dict, list)):
                missing_fields.extend(extract_missing_fields(value, current_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{prefix}[{i}]" if prefix else f"[{i}]"

            # Only flag individual list items if the list has multiple elements
            if isinstance(item, str) and is_not_specified(item) and len(data) > 1:
                missing_fields.append(current_path)
            elif isinstance(item, (dict, list)):
                missing_fields.extend(extract_missing_fields(item, current_path))

    return missing_fields


# The five scored card sections (shared by the gold-set metric tooling).
GOLD_SECTIONS = (
    "benchmark_details", "purpose_and_intended_users",
    "data", "methodology", "ethical_and_legal_considerations",
)

# Fields with a structured external source (HF tags / EEE metadata) that can be
# filled deterministically. Everything else in the gold sections counts as prose.
# This split drives the prose-vs-deterministic Not-specified breakdown; tune here.
DETERMINISTIC_FIELDS = {
    "benchmark_details.languages",
    "benchmark_details.data_type",
    "ethical_and_legal_considerations.data_licensing",
    "data.size",
    "data.format",
    "methodology.metrics",
}

_FIELD_SKIP = {"provenance", "flagged_fields", "missing_fields"}


# Composer abstentions that leak through as a field VALUE instead of being
# normalized to the canonical "Not specified" sentinel. These reference the
# composer's own input ("facts provided", "the source facts") rather than making
# a substantive claim about the benchmark, so they masquerade as filled content
# and a faithfulness judge marks them unsupported. Anchored on PROVIDED / source
# facts only: phrases like "no information was found regarding IRB" are honest,
# judge-supported statements about the source and must NOT match.
_ABSENCE_PLACEHOLDER_RE = re.compile(
    r"no facts provided"
    r"|provided in the source facts"
    r"|no specific .{0,40}?(?:was|were) provided",
    re.I,
)


def detect_absence_placeholders(card: Dict[str, Any]) -> Dict[str, str]:
    """Find gold-section fields whose value is an explicit no-data placeholder
    that slipped past the canonical 'Not specified' sentinel. Returns a
    {dotted_path: reason} map for flagging. Deterministic, no inference."""
    found: Dict[str, str] = {}
    for section in GOLD_SECTIONS:
        sec = card.get(section)
        if not isinstance(sec, dict):
            continue
        for field, value in sec.items():
            if is_not_specified(value):
                continue
            if isinstance(value, list):
                text = " ".join(str(v) for v in value)
            elif isinstance(value, str):
                text = value
            else:
                continue
            if _ABSENCE_PLACEHOLDER_RE.search(text):
                found[f"{section}.{field}"] = "[Absence] explicit no-data placeholder"
    return found


def card_field_stats(card: Dict[str, Any]) -> Dict[str, int]:
    """Count fields and 'Not specified' values across the five gold sections,
    split into prose vs deterministic (structured) fields.

    `card` must be the unwrapped card dict (see extract_card). Returns counts:
    n_fields, n_not_specified, n_prose, n_prose_ns, n_det, n_det_ns, n_flagged.
    """
    n_fields = n_ns = n_prose = n_prose_ns = n_det = n_det_ns = 0
    for sec in GOLD_SECTIONS:
        fields = card.get(sec, {})
        if not isinstance(fields, dict):
            continue
        for k, v in fields.items():
            if k in _FIELD_SKIP:
                continue
            n_fields += 1
            ns = 1 if is_not_specified(v) else 0
            n_ns += ns
            if f"{sec}.{k}" in DETERMINISTIC_FIELDS:
                n_det += 1
                n_det_ns += ns
            else:
                n_prose += 1
                n_prose_ns += ns
    flagged = card.get("flagged_fields")
    n_flagged = len(flagged) if isinstance(flagged, dict) else 0
    return {
        "n_fields": n_fields, "n_not_specified": n_ns,
        "n_prose": n_prose, "n_prose_ns": n_prose_ns,
        "n_det": n_det, "n_det_ns": n_det_ns,
        "n_flagged": n_flagged,
    }


_PROSE_SOURCES = ("paper", "html")
_STRUCTURED_SOURCES = ("deterministic", "eee", "unitxt", "extracted_id")


def _content_words(text: str) -> set:
    """Content-token set (words >3 chars, numbers >=2 digits) for grounding overlap."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if (t.isdigit() and len(t) >= 2) or len(t) > 3}


# Number literals as they appear in cards/evidence: 1,319 / 0.712 / 88% / 2024.
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?%?")
# Multi-word capitalized spans ("Terminal Bench", "Claude 3.5 Sonnet"),
# digit/dash-bearing model slugs ("GPT-4o", "Llama-3.3-70B-Instruct"), and
# snake_case/dotted config identifiers ("lower_is_better", "openeval.bold.logprob")
# as they appear in EEE metric_configs.
_NAME_SPAN_RE = re.compile(r"\b[A-Z][\w.+-]*(?:\s+[A-Z][\w.+-]*)+\b")
_SLUG_RE = re.compile(r"\b[A-Za-z]+\w*\d[\w.-]*\b|\b[A-Za-z]\w*(?:-[\w.]+)+\b")
_IDENT_RE = re.compile(r"\b[A-Za-z]\w*(?:[._]\w+)+\b")


def _norm_number(tok: str) -> Optional[str]:
    """Canonical form for numeric comparison: '1,319'->'1319', '0.700'->'0.7', '70%'->'70'."""
    t = tok.replace(",", "").rstrip("%")
    try:
        return format(float(t), "g")
    except ValueError:
        return None


def _number_tokens(text: str, min_signal: bool = False) -> list:
    """(normalized, had_percent) number tokens in text. min_signal drops bare
    single digits (keeps >=2-digit integers, decimals, percents) so '3 tasks'
    can't veto."""
    out = []
    for tok in _NUMBER_RE.findall(text or ""):
        if min_signal and "." not in tok and "%" not in tok and "," not in tok and len(tok) < 2:
            continue
        n = _norm_number(tok)
        if n is not None:
            out.append((n, tok.endswith("%")))
    return out


def _number_literals(text: str, min_signal: bool = False) -> set:
    """Normalized number literals in text (see _number_tokens)."""
    return {n for n, _ in _number_tokens(text, min_signal)}


def _number_pool(text: str) -> set:
    """Lookup pool of normalized numbers; percent tokens also contribute their
    fractional form ('70%' -> {'70', '0.7'}) so decimal claims can match them."""
    pool = set()
    for n, had_pct in _number_tokens(text):
        pool.add(n)
        if had_pct:
            try:
                pool.add(format(float(n) / 100, "g"))
            except ValueError:
                pass
    return pool


def _num_matches(token, pool: set) -> bool:
    """Match a (normalized, had_percent) claim token against a pool. Scaling is
    only applied when an explicit percent marker licenses it -- a bare 0.05 must
    not match a benign 5 somewhere in the artifacts."""
    n, had_pct = token
    if n in pool:
        return True
    if had_pct:
        try:
            return format(float(n) / 100, "g") in pool
        except ValueError:
            return False
    return False


def _name_literals(text: str) -> set:
    """Lowercased proper-name spans, model slugs and config identifiers in text."""
    names = set()
    for rx in (_NAME_SPAN_RE, _SLUG_RE, _IDENT_RE):
        for m in rx.findall(text or ""):
            if len(m) >= 4:
                names.add(m.lower())
    return names


def verify_against_artifacts(
    field_value: Any,
    evidence: str,
    artifacts: Optional[Dict[str, Any]],
    benchmark_name: Optional[str] = None,
) -> Optional[bool]:
    """Check a structured-source claim against the actual artifact JSON on disk.

    `artifacts` maps source kind to loaded JSON, e.g. {"eee": ..., "hf": ...}.
    Returns None when no artifact is available (caller must fall back to prose
    grounding -- never blind-trust the composer's source label). True/False =
    the claim's number/name literals are present/absent in the pooled artifacts.
    The benchmark's own name is excluded from the name units -- it appears in
    every artifact, so matching on it would verify any claim vacuously.
    """
    if not artifacts:
        return None
    blobs = [json.dumps(a) for a in artifacts.values() if a]
    if not blobs:
        return None
    blob = " ".join(blobs)
    pool_nums = _number_pool(blob)
    pool_text = blob.lower()

    if isinstance(field_value, str):
        value_text = field_value
    elif field_value is None:
        value_text = ""
    else:
        value_text = json.dumps(field_value)
    claim = f"{evidence or ''} {value_text}"

    def _norm_span(s):
        return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

    num_tokens = _number_tokens(claim, min_signal=True)
    nums = {n for n, _ in num_tokens}
    names = _name_literals(claim)
    bn = _norm_span(benchmark_name or "")
    if bn:
        # Drop units that ARE the benchmark name (not units merely containing it,
        # like a metric key 'openeval.bold.logprob' for benchmark 'BOLD').
        names = {n for n in names if _norm_span(n) not in (bn, f"the {bn}")}
    if nums:
        if not names and len(nums) < 2:
            # A single bare number with no identifying name is too weak to
            # verify (it could match anything in the artifacts) -- abstain.
            return None
        return all(_num_matches(t, pool_nums) for t in num_tokens) and (
            not names or any(n in pool_text for n in names)
        )
    if names:
        return any(n in pool_text for n in names)
    # No verifiable units (generic prose, e.g. an interpretation sentence):
    # abstain rather than guess -- common words overlapping the artifact JSON
    # is no evidence the claim came from it. Caller falls back to prose grounding.
    return None


def _evidence_grounded_in_contexts(
    evidence: str,
    retrieved_contexts: Optional[List[str]],
    threshold: float = 0.6,
    high_signal: bool = False,
) -> bool:
    """Whether provenance evidence is actually supported by the retrieved source text.

    Returns True only when enough of the evidence's content tokens appear in the
    retrieved chunks. This replaces trusting the composer's self-asserted provenance,
    which is circular: the model that wrote the field also vouches for it.

    With high_signal=True an additional veto applies on top of the base overlap
    check (strictly stricter, never looser): if the evidence contains number
    literals or proper-name spans, ALL numbers and at least one name must appear
    in the contexts -- common words overlapping is not enough to vouch for a
    fabricated score or model name.
    """
    if not evidence or not retrieved_contexts:
        return False
    ev_words = _content_words(evidence)
    if not ev_words:
        return False
    ctx_words: set = set()
    for chunk in retrieved_contexts:
        if chunk:
            ctx_words |= _content_words(chunk)
    base = bool(ctx_words) and len(ev_words & ctx_words) / len(ev_words) >= threshold
    if not high_signal or not base:
        return base
    num_tokens = _number_tokens(evidence, min_signal=True)
    names = _name_literals(evidence)
    if not (num_tokens or names):
        return base
    ctx_text = " ".join(c for c in retrieved_contexts if c)
    if num_tokens:
        ctx_nums = _number_pool(ctx_text)
        if not all(_num_matches(t, ctx_nums) for t in num_tokens):
            return False
    if names and not any(n in ctx_text.lower() for n in names):
        return False
    return True


def _is_structured_source(source: Optional[str]) -> bool:
    """True when provenance cites a structured/verified origin and no free-prose source.

    Structured facts (HF tags, EEE results, registry ids) are verified upstream and
    legitimately absent from the retrieved prose, so they need no prose grounding; a
    free-prose source (paper/html) is where the self-vouching circularity bites.
    """
    s = (source or "").lower()
    if any(p in s for p in _PROSE_SOURCES):
        return False
    return any(t in s for t in _STRUCTURED_SOURCES)


def backfill_from_provenance(
    card: Dict[str, Any],
    provenance: Dict[str, Any],
    retrieved_contexts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fill 'Not specified' fields from provenance evidence, but only when that evidence
    is grounded in the retrieved source or is a structured fact -- never on the composer's
    unverified say-so alone."""
    for section_key, section_val in card.items():
        if not isinstance(section_val, dict):
            continue
        section_prov = provenance.get(section_key, {})
        if not section_prov:
            continue

        for field_key, field_val in section_val.items():
            field_prov = section_prov.get(field_key, {})
            if not field_prov or not field_prov.get("evidence"):
                continue

            evidence = field_prov["evidence"]
            if isinstance(evidence, str) and any(
                neg in evidence.lower() for neg in ["no information", "not specified", "not available", "none found"]
            ):
                continue
            if not (
                _is_structured_source(field_prov.get("source"))
                or _evidence_grounded_in_contexts(evidence, retrieved_contexts)
            ):
                continue
            if is_not_specified(field_val) and len(evidence) > 10:
                logger.debug(
                    "Backfilling %s.%s from provenance (%s)",
                    section_key, field_key, field_prov.get("source", "?"),
                )
                card[section_key][field_key] = evidence

    return card


def normalize_not_specified(card: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize inconsistent 'Not specified' representations."""
    list_fields = {
        "domains", "languages", "similar_benchmarks", "resources",
        "audience", "tasks", "out_of_scope_uses", "methods", "metrics",
    }

    for section_key, section_val in card.items():
        if not isinstance(section_val, dict):
            continue
        for field_key, field_val in section_val.items():
            if field_key in ("provenance", "flagged_fields", "missing_fields", "card_info", "possible_risks"):
                continue

            is_list_field = field_key in list_fields

            if field_val is None or field_val == "" or field_val == []:
                card[section_key][field_key] = ["Not specified"] if is_list_field else "Not specified"
            elif isinstance(field_val, list) and all(
                isinstance(item, str) and is_not_specified(item) for item in field_val
            ):
                card[section_key][field_key] = ["Not specified"]

    return card


_LANG_CODE_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "pt": "Portuguese", "ru": "Russian", "it": "Italian", "nl": "Dutch",
    "pl": "Polish", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "tr": "Turkish", "hi": "Hindi", "bn": "Bengali",
    "vi": "Vietnamese", "th": "Thai", "he": "Hebrew", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "uk": "Ukrainian", "id": "Indonesian",
    "ms": "Malay", "el": "Greek", "bg": "Bulgarian", "ca": "Catalan",
    "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "fa": "Persian", "ur": "Urdu",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "mr": "Marathi",
    "sw": "Swahili", "af": "Afrikaans", "eu": "Basque", "gl": "Galician",
    "ka": "Georgian", "hy": "Armenian", "az": "Azerbaijani", "be": "Belarusian",
    "is": "Icelandic", "mk": "Macedonian", "sq": "Albanian", "sr": "Serbian",
    "cy": "Welsh", "ga": "Irish", "mt": "Maltese", "la": "Latin",
}

_LICENSE_MAP = {
    "mit": "MIT License",
    "apache-2.0": "Apache License 2.0",
    "cc-by-4.0": "Creative Commons Attribution 4.0",
    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
    "cc-by-nc-nd-4.0": "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
    "cc0-1.0": "Creative Commons Zero v1.0 Universal",
    "openrail": "Open RAIL License",
    "gpl-3.0": "GNU General Public License v3.0",
    "lgpl-3.0": "GNU Lesser General Public License v3.0",
    "bsd-3-clause": "BSD 3-Clause License",
    "bsd-2-clause": "BSD 2-Clause License",
    "odc-by": "Open Data Commons Attribution License",
    "odbl": "Open Data Commons Open Database License",
    "pddl": "Open Data Commons Public Domain Dedication and License",
}

_SIZE_CATEGORY_MAP = {
    "n<1K": "Less than 1K examples",
    "1K<n<10K": "1K to 10K examples",
    "10K<n<100K": "10K to 100K examples",
    "100K<n<1M": "100K to 1M examples",
    "1M<n<10M": "1M to 10M examples",
    "10M<n<100M": "10M to 100M examples",
    "100M<n<1B": "100M to 1B examples",
    "1B<n<10B": "1B to 10B examples",
    "10B<n<100B": "10B to 100B examples",
    "100B<n<1T": "100B to 1T examples",
}


def extract_hf_tags(hf_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract structured field values from HuggingFace dataset tags."""
    if not hf_metadata:
        return {}

    tags = hf_metadata.get("tags")
    if tags is None:
        for v in hf_metadata.values():
            if isinstance(v, dict) and "tags" in v:
                tags = v["tags"]
                break
    if not tags or not isinstance(tags, list):
        return {}

    result: Dict[str, Any] = {}

    languages = []
    modalities = []
    task_labels = []
    fmt = None
    size = None
    license_val = None

    for tag in tags:
        if not isinstance(tag, str) or ":" not in tag:
            continue
        prefix, _, value = tag.partition(":")
        value = value.strip()
        if not value:
            continue

        if prefix == "language":
            lang_name = _LANG_CODE_MAP.get(value, value)
            languages.append(lang_name)

        elif prefix == "modality":
            modalities.append(value.capitalize())

        elif prefix in ("task_ids", "task_categories"):
            label = value.replace("-", " ").replace("_", " ")
            label = label[0].upper() + label[1:]
            task_labels.append(label)

        elif prefix == "format":
            fmt = value

        elif prefix == "size_categories":
            size = _SIZE_CATEGORY_MAP.get(value, value)

        elif prefix == "license":
            if value not in ("other", "unknown"):
                license_val = _LICENSE_MAP.get(value, value)

    if languages:
        result["benchmark_details.languages"] = sorted(set(languages))
    if modalities:
        result["benchmark_details.data_type"] = ", ".join(sorted(set(modalities))).lower()
    if task_labels:
        result["purpose_and_intended_users.tasks"] = sorted(set(task_labels))
    if fmt:
        result["data.format"] = fmt
    if size:
        result["data.size"] = size
    if license_val:
        result["ethical_and_legal_considerations.data_licensing"] = license_val

    return result


# Fields where deterministic HF tags always win over LLM output
_ALWAYS_OVERRIDE = {
    "benchmark_details.languages",
    "benchmark_details.data_type",
    "ethical_and_legal_considerations.data_licensing",
}

_EMPTY = {"not specified", "not specified.", "no information found", ""}


def apply_deterministic_overrides(
    card: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply HF-extracted tag values to the card, preferring factual tags over LLM output."""
    for dotted_key, override_val in overrides.items():
        section, _, field = dotted_key.partition(".")
        if not field or section not in card or not isinstance(card[section], dict):
            continue

        old = card[section].get(field)

        if dotted_key in _ALWAYS_OVERRIDE:
            card[section][field] = override_val
            if old != override_val:
                logger.debug("HF override (always) %s: %r -> %r", dotted_key, old, override_val)
            continue

        # For other fields, only override if LLM value is empty/generic
        llm_is_empty = False
        if old is None:
            llm_is_empty = True
        elif isinstance(old, str) and old.strip().lower() in _EMPTY:
            llm_is_empty = True
        elif isinstance(old, list) and (
            not old
            or (len(old) == 1 and isinstance(old[0], str) and old[0].strip().lower() in _EMPTY)
        ):
            llm_is_empty = True

        if llm_is_empty:
            card[section][field] = override_val
            logger.debug("HF override (fill) %s: %r -> %r", dotted_key, old, override_val)
        else:
            logger.debug("HF override skipped %s: LLM has specific value %r", dotted_key, str(old)[:80])

    return card
