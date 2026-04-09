"""Card processing utilities: field extraction, normalization, and HF tag overrides."""

import logging
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
            if isinstance(item, str) and item == "Not specified" and len(data) > 1:
                missing_fields.append(current_path)
            elif isinstance(item, (dict, list)):
                missing_fields.extend(extract_missing_fields(item, current_path))

    return missing_fields


def backfill_from_provenance(
    card: Dict[str, Any], provenance: Dict[str, Any]
) -> Dict[str, Any]:
    """Fill 'Not specified' fields using provenance evidence when available."""
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
                isinstance(item, str) and item == "Not specified" for item in field_val
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
