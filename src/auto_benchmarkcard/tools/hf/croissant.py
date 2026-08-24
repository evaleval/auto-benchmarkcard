"""Deterministic structural facts from the Hugging Face dataset-viewer hosted API.

Three endpoints describe a bound dataset repo with no LLM in the loop:

  croissant  https://huggingface.co/api/datasets/{repo_id}/croissant
             JSON-LD with license and distribution (hosting format).
  size       https://datasets-server.huggingface.co/size?dataset={repo_id}
             Exact per-config and per-split row counts.
  info       https://datasets-server.huggingface.co/info?dataset={repo_id}
             Configs, splits, features. Used only as the size fallback.

Values are formatted to match the pipeline's deterministic channel (extract_hf_tags,
extract_deterministic_facts) so downstream override and comparison logic treats them as
native: data.size is a leading-count string ("17,584 examples"), size_breakdown is a
{name: int} dict, license names come from _LICENSE_MAP with unmappable strings passed
through raw, and hosting formats go through _label_hosting_format. Everything is
fail-soft: an endpoint that errors, times out, or returns a malformed body contributes
nothing, and fetch_croissant_facts never raises.

The croissant body may additionally carry Croissant-RAI extension properties
(rai: namespace, docs.mlcommons.org/croissant/docs/croissant-rai-spec.html). These are
author-written ethics prose, not structural facts, so they are returned separately under
a top-level "rai" key ({card_field_path: text}) plus "_rai_meta" telemetry, for the
wiring layer to hand to the composer as evidence-like material rather than overrides.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

from auto_benchmarkcard.card_utils import (
    _AMBIGUOUS_LICENSE_TAGS,
    _LICENSE_MAP,
    _label_hosting_format,
)

logger = logging.getLogger(__name__)

_CROISSANT_URL = "https://huggingface.co/api/datasets/{repo_id}/croissant"
_VIEWER_URL = "https://datasets-server.huggingface.co/{endpoint}"

# Croissant distribution encodingFormat MIME types mapped to the bare storage tokens
# that _label_hosting_format knows (card_utils.STORAGE_FORMAT_TOKENS space). The git
# repo FileObject carries "git+https", which is deliberately absent.
_MIME_FORMAT = {
    "application/x-parquet": "parquet",
    "application/parquet": "parquet",
    "application/vnd.apache.parquet": "parquet",
    "text/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/json": "json",
    "application/jsonlines": "jsonl",
    "application/x-ndjson": "jsonl",
    "text/plain": "text",
    "application/zip": "zip",
    "application/x-arrow": "arrow",
    "application/vnd.apache.arrow.file": "arrow",
}


# Croissant-RAI extension vocabulary. Canonical property names come from the formal
# vocabulary (docs/croissant_rai.ttl in mlcommons/croissant); the namespace is
# http://mlcommons.org/croissant/RAI/. Card field -> the RAI properties that feed it,
# in join order. Values are author-written prose offered to the composer as evidence,
# never overrides, so the conservative subset below is deliberate.
_RAI_FIELD_SOURCES = {
    "ethical_and_legal_considerations.privacy_and_anonymity": ("personalSensitiveInformation",),
    "data.source": ("dataCollection", "dataCollectionType"),
    "purpose_and_intended_users.limitations": ("dataBiases", "dataLimitations"),
    "purpose_and_intended_users.goal": ("dataUseCases",),
    "data.annotation": (
        "dataAnnotationProtocol",
        "dataAnnotationPlatform",
        "dataAnnotationAnalysis",
        "annotationsPerItem",
        "annotatorDemographics",
        "machineAnnotationTools",
    ),
}

# The rest of the 20-property vocabulary: recognized for telemetry, mapped to nothing.
_RAI_UNMAPPED = (
    "dataCollectionMissingData",
    "dataCollectionRawData",
    "dataCollectionTimeframe",
    "dataImputationProtocol",
    "dataPreprocessingProtocol",
    "dataManipulationProtocol",
    "dataReleaseMaintenancePlan",
    "dataSocialImpact",
)

# Spellings the spec's own prose/examples and the mlcroissant serializer use instead of
# the TTL names (rai:useCases in the overview table, RAI.dataDataManipulationProtocol
# and RAI.dataCollectionTimeFrame in mlcroissant constants.py, etc.).
_RAI_ALIASES = {
    "useCases": "dataUseCases",
    "annotationProtocol": "dataAnnotationProtocol",
    "annotationPlatform": "dataAnnotationPlatform",
    "annotatorsDemographics": "annotatorDemographics",
    "dataReleaseMaintenance": "dataReleaseMaintenancePlan",
    "dataCollectionTimeFrame": "dataCollectionTimeframe",
    "dataDataManipulationProtocol": "dataManipulationProtocol",
}

# Properties that predate the RAI split out of croissant core. The HF croissant
# @context still aliases these three as bare keys into the cr: namespace
# (http://mlcommons.org/croissant/), so accept the bare, cr:-prefixed, and core-URL
# forms for exactly them.
_RAI_LEGACY_CORE = frozenset({"dataCollection", "dataBiases", "personalSensitiveInformation"})

_RAI_NS_MARKER = "mlcommons.org/croissant/RAI/"
_CR_NS_MARKER = "mlcommons.org/croissant/"

# Deterministic order for _rai_meta.properties_found: mapped properties in table order,
# then the unmapped tail.
_RAI_ORDER = tuple(
    p for props in _RAI_FIELD_SOURCES.values() for p in props
) + _RAI_UNMAPPED

# Segment labels used when several RAI properties join into one card field.
_RAI_LABELS = {
    "dataCollection": "Collection",
    "dataCollectionType": "Collection type",
    "dataBiases": "Biases",
    "dataLimitations": "Limitations",
    "dataAnnotationProtocol": "Annotation protocol",
    "dataAnnotationPlatform": "Annotation platform",
    "dataAnnotationAnalysis": "Annotation analysis",
    "annotationsPerItem": "Annotations per item",
    "annotatorDemographics": "Annotator demographics",
    "machineAnnotationTools": "Machine annotation tools",
}

_RAI_MAX_CHARS = 1000


def _get_json(
    url: str,
    params: Optional[Dict[str, str]],
    timeout: int,
    endpoints: Dict[str, Any],
    name: str,
) -> Optional[Dict[str, Any]]:
    """One GET, no retries. Records the HTTP status (or an error tag) under endpoints[name]
    and returns the parsed dict body, or None when anything went wrong."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:
        endpoints[name] = f"error: {type(exc).__name__}"
        return None
    endpoints[name] = resp.status_code
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except Exception:
        endpoints[name] = "malformed"
        return None
    if not isinstance(body, dict):
        endpoints[name] = "malformed"
        return None
    return body


def _int(value: Any) -> Optional[int]:
    """The value as an int count, or None (bool is not a count)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _size_facts_from_size(body: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, int]]]:
    """(total_rows, breakdown) from a /size body. Per-config totals when the repo has
    multiple configs, else the per-split rows of its single config. Exactness only: a
    partial response yields nothing (its counts are lower bounds, not ground truth)."""
    if body.get("partial"):
        return None, None
    size = body.get("size")
    if not isinstance(size, dict):
        return None, None
    dataset = size.get("dataset")
    total = _int(dataset.get("num_rows")) if isinstance(dataset, dict) else None

    configs = size.get("configs")
    rows_by_config: Dict[str, int] = {}
    for c in configs if isinstance(configs, list) else []:
        if not isinstance(c, dict):
            continue
        name, n = c.get("config"), _int(c.get("num_rows"))
        if isinstance(name, str) and name and n is not None:
            rows_by_config[name] = n
    if total is None and rows_by_config:
        total = sum(rows_by_config.values())
    if len(rows_by_config) > 1:
        return total, rows_by_config

    # Zero or one config: the splits list belongs to that single config.
    splits = size.get("splits")
    breakdown: Dict[str, int] = {}
    for s in splits if isinstance(splits, list) else []:
        if not isinstance(s, dict):
            continue
        name, n = s.get("split"), _int(s.get("num_rows"))
        if isinstance(name, str) and name and n is not None:
            breakdown[name] = n
    if total is None and breakdown:
        total = sum(breakdown.values())
    return total, breakdown or None


def _size_facts_from_info(body: Dict[str, Any]) -> Tuple[Optional[int], Optional[Dict[str, int]]]:
    """(total_rows, breakdown) from an /info body, the fallback when /size gave nothing.
    Same preference as _size_facts_from_size: per-config totals, else per-split."""
    if body.get("partial"):
        return None, None
    dataset_info = body.get("dataset_info")
    if not isinstance(dataset_info, dict):
        return None, None
    per_config: Dict[str, int] = {}
    last_splits: Optional[Dict[str, int]] = None
    for cfg, v in dataset_info.items():
        splits = v.get("splits") if isinstance(v, dict) else None
        if not isinstance(splits, dict):
            continue
        counts: Dict[str, int] = {}
        for sname, s in splits.items():
            n = _int(s.get("num_examples")) if isinstance(s, dict) else None
            if isinstance(sname, str) and sname and n is not None:
                counts[sname] = n
        if counts and isinstance(cfg, str) and cfg:
            per_config[cfg] = sum(counts.values())
            last_splits = counts
    if not per_config:
        return None, None
    total = sum(per_config.values())
    if len(per_config) > 1:
        return total, per_config
    return total, last_splits


def _license_from_croissant(body: Dict[str, Any]) -> Optional[str]:
    """The croissant JSON-LD license, mapped into the extract_hf_tags license-name space.
    A choosealicense/SPDX slug that _LICENSE_MAP knows becomes its display name; an
    unmappable one passes through raw for the conflict-surfacing layer. Placeholder tags
    (other/unknown/cc) name no license and yield nothing."""
    raw = body.get("license")
    if isinstance(raw, list):
        raw = next((x for x in raw if isinstance(x, (str, dict))), None)
    if isinstance(raw, dict):
        raw = raw.get("@id") or raw.get("url") or raw.get("name")
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    slug = raw
    is_url = "://" in raw
    if is_url:
        segments = [seg for seg in urlparse(raw).path.split("/") if seg]
        if not segments:
            return None
        slug = segments[-1]
    key = slug.lower()
    if key in _AMBIGUOUS_LICENSE_TAGS:
        return None
    if key in _LICENSE_MAP:
        return _LICENSE_MAP[key]
    return slug if is_url else raw


def _format_from_croissant(body: Dict[str, Any]) -> Optional[str]:
    """The dominant hosting format among the croissant distribution entries, labelled the
    way the deterministic channel labels storage tokens ("parquet (HuggingFace hosting
    format)"). None when no known MIME type appears."""
    dist = body.get("distribution")
    if not isinstance(dist, list):
        return None
    counts: Counter = Counter()
    for entry in dist:
        if isinstance(entry, dict):
            token = _MIME_FORMAT.get(str(entry.get("encodingFormat", "")).lower())
            if token:
                counts[token] += 1
    return _label_hosting_format(counts.most_common(1)[0][0]) if counts else None


def _rai_property_name(key: Any) -> Optional[str]:
    """The canonical RAI property name for a JSON-LD key, or None. Accepts the compact
    rai: prefix and full RAI-namespace URL keys for the whole vocabulary, and the bare,
    cr:-prefixed, and core-namespace-URL forms for the three legacy properties. Variant
    spellings normalize through _RAI_ALIASES."""
    if not isinstance(key, str):
        return None
    if key.startswith("rai:"):
        name = key[4:]
    elif _RAI_NS_MARKER in key:
        name = key.rsplit("/", 1)[-1]
    else:
        bare = key[3:] if key.startswith("cr:") else key
        if _CR_NS_MARKER in bare:
            bare = bare.rsplit("/", 1)[-1]
        if bare not in _RAI_LEGACY_CORE:
            return None
        name = bare
    name = _RAI_ALIASES.get(name, name)
    return name if name in _RAI_ORDER else None


def _rai_text(value: Any) -> Optional[str]:
    """Author-written text from a JSON-LD value: a string, a number, a {"@value": ...}
    node, or a list of those joined with "; ". None when nothing usable (bools are
    flags, not prose)."""
    if isinstance(value, list):
        parts = [t for t in (_rai_text(v) for v in value) if t]
        return "; ".join(parts) or None
    if isinstance(value, dict):
        return _rai_text(value.get("@value"))
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _rai_truncate(text: str) -> str:
    return text if len(text) <= _RAI_MAX_CHARS else text[: _RAI_MAX_CHARS - 3] + "..."


def _rai_facts_from_croissant(body: Dict[str, Any]) -> Tuple[Dict[str, str], list]:
    """({card_field_path: text}, [canonical properties found]) from the RAI extension
    keys of a croissant JSON-LD body. A field fed by one property carries its text as
    is; one fed by several gets labelled newline-joined segments so the composer can
    attribute them. Every value is capped at _RAI_MAX_CHARS."""
    values: Dict[str, str] = {}
    for key, raw in body.items():
        name = _rai_property_name(key)
        if name and name not in values:
            text = _rai_text(raw)
            if text:
                values[name] = text
    if not values:
        return {}, []
    found = [name for name in _RAI_ORDER if name in values]
    fields: Dict[str, str] = {}
    for path, props in _RAI_FIELD_SOURCES.items():
        present = [p for p in props if p in values]
        if not present:
            continue
        if len(present) == 1:
            text = values[present[0]]
        else:
            text = "\n".join(f"{_RAI_LABELS[p]}: {values[p]}" for p in present)
        fields[path] = _rai_truncate(text)
    return fields, found


def fetch_croissant_facts(repo_id: str, timeout: int = 20) -> Dict[str, Any]:
    """Deterministic structural facts for a bound HF dataset repo from the hosted
    dataset-viewer API.

    Returns a dict with any of data.size, data.size_breakdown, data.format,
    ethical_and_legal_considerations.data_licensing, plus a _croissant_meta entry
    recording which endpoints answered (name -> HTTP status or error tag) and when.
    When the croissant body carries Croissant-RAI extension properties, adds "rai"
    ({card_field_path: author-written text}, distinct from the structural keys so the
    wiring layer never treats it as an override) and "_rai_meta" ({"properties_found":
    [canonical names]}). Fail-soft: never raises; an entirely failed fetch returns the
    meta alone."""
    endpoints: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoints": endpoints,
    }
    facts: Dict[str, Any] = {"_croissant_meta": meta}

    croissant = _get_json(
        _CROISSANT_URL.format(repo_id=repo_id), None, timeout, endpoints, "croissant"
    )
    if croissant:
        try:
            license_val = _license_from_croissant(croissant)
            if license_val:
                facts["ethical_and_legal_considerations.data_licensing"] = license_val
            fmt = _format_from_croissant(croissant)
            if fmt:
                facts["data.format"] = fmt
        except Exception:
            logger.debug("croissant body parse failed for %s", repo_id, exc_info=True)
        try:
            rai_fields, rai_found = _rai_facts_from_croissant(croissant)
            if rai_fields:
                facts["rai"] = rai_fields
            if rai_found:
                facts["_rai_meta"] = {"properties_found": rai_found}
        except Exception:
            logger.debug("croissant RAI parse failed for %s", repo_id, exc_info=True)

    total: Optional[int] = None
    breakdown: Optional[Dict[str, int]] = None
    size_body = _get_json(
        _VIEWER_URL.format(endpoint="size"), {"dataset": repo_id}, timeout, endpoints, "size"
    )
    if size_body:
        try:
            if size_body.get("partial"):
                meta["partial"] = True
            total, breakdown = _size_facts_from_size(size_body)
        except Exception:
            logger.debug("size body parse failed for %s", repo_id, exc_info=True)
    if total is None and breakdown is None:
        info_body = _get_json(
            _VIEWER_URL.format(endpoint="info"), {"dataset": repo_id}, timeout, endpoints, "info"
        )
        if info_body:
            try:
                if info_body.get("partial"):
                    meta["partial"] = True
                total, breakdown = _size_facts_from_info(info_body)
            except Exception:
                logger.debug("info body parse failed for %s", repo_id, exc_info=True)

    if total is not None and total > 0:
        facts["data.size"] = f"{total:,} examples"
    # A one-entry breakdown restates the total; the deterministic README parser also
    # requires >= 2 rows, so mirror that floor here.
    if isinstance(breakdown, dict) and len(breakdown) >= 2:
        facts["data.size_breakdown"] = breakdown
    return facts
