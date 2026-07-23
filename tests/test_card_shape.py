"""Tests for the fixed card schema shape and accurate missing_fields.

enforce_card_shape re-adds every section-model field that model_dump(exclude_none=True)
dropped (in practice the Optional display fields authors/logo/org_url), so every card
carries the identical model-field key-set. extract_missing_fields skips display-only
fields. Network-free.
"""

import glob
import json
from pathlib import Path

import pytest

from auto_benchmarkcard.card_utils import (
    _DISPLAY_ONLY_FIELDS, extract_card, extract_missing_fields, is_not_specified)
from auto_benchmarkcard.tools.composer import field_spec as fs
from auto_benchmarkcard.tools.composer.composer_tool import SECTION_MODELS, enforce_card_shape

# Non-model keys EEE injects into benchmark_details (deliberately outside the schema).
BD_EXTRAS_ALLOWED = {"benchmark_type", "appears_in", "contains"}
BD_EXTRAS_REQUIRED = {"benchmark_type", "appears_in"}


def _canonical_ns(field):
    return ["Not specified"] if field in fs.LIST_FIELDS else "Not specified"


def _model_keys(section):
    return set(SECTION_MODELS[section].model_fields) - {"provenance"}


def _complete_card():
    """A fully-filled, valid card (every required field present, no display fields)."""
    return {
        "benchmark_details": {
            "name": "X", "overview": "ov", "data_type": "text", "domains": ["d"],
            "languages": ["English"], "similar_benchmarks": ["s"], "resources": ["r"],
            "org_url": "https://huggingface.co/org"},
        "purpose_and_intended_users": {
            "goal": "g", "audience": ["a"],
            "tasks": ["t"], "limitations": "l", "out_of_scope_uses": ["o"]},
        "data": {"source": "src", "size": "10", "format": "json", "annotation": "manual"},
        "methodology": {
            "methods": ["m"], "metrics": ["Accuracy"], "calculation": "c",
            "interpretation": "i", "baseline_results": "filled result", "validation": "v"},
        "ethical_and_legal_considerations": {
            "privacy_and_anonymity": "p", "data_licensing": "MIT",
            "consent_procedures": "c", "compliance_with_regulations": "r"},
    }


# ---------------------------------------------------------------- enforce_card_shape ----

def test_enforce_fills_all_fields_with_canonical_absence():
    # Empty sections: enforce must insert every model field with its canonical NS value.
    card = {sec: {} for sec in SECTION_MODELS}
    enforce_card_shape(card)
    for sec in SECTION_MODELS:
        assert set(card[sec]) == _model_keys(sec)
        for field in card[sec]:
            assert card[sec][field] == _canonical_ns(field)


def test_enforce_display_fields_list_vs_scalar():
    card = _complete_card()
    enforce_card_shape(card)
    bd = card["benchmark_details"]
    assert bd["authors"] == ["Not specified"]  # list-typed (in LIST_FIELDS)
    assert bd["logo"] == "Not specified"        # scalar
    assert bd["org_url"] == "https://huggingface.co/org"  # already present, untouched


def test_enforce_union_absent_branch_is_string_sentinel():
    # size_breakdown / judge_num / judge_uses_llm are Union[..., str]; absent -> "Not specified".
    card = _complete_card()
    enforce_card_shape(card)
    assert card["data"]["size_breakdown"] == "Not specified"
    assert card["methodology"]["judge_num"] == "Not specified"
    assert card["methodology"]["judge_uses_llm"] == "Not specified"


def test_enforce_yields_identical_keyset_per_section():
    card = _complete_card()
    enforce_card_shape(card)
    for sec in SECTION_MODELS:
        assert set(card[sec]) == _model_keys(sec)


def test_enforce_idempotent_and_no_clobber():
    card = _complete_card()
    enforce_card_shape(card)
    snapshot = json.loads(json.dumps(card))
    enforce_card_shape(card)
    assert card == snapshot  # second pass changes nothing
    assert card["methodology"]["baseline_results"] == "filled result"  # filled value kept
    assert card["benchmark_details"]["org_url"] == "https://huggingface.co/org"


def test_enforce_skips_provenance_and_nondict_sections():
    card = {"benchmark_details": "not a dict", "data": {}}
    enforce_card_shape(card)
    assert card["benchmark_details"] == "not a dict"   # non-dict left alone
    assert "provenance" not in card["data"]            # provenance never inserted


# ---------------------------------------------------------------- extract_missing_fields ----

def test_missing_fields_excludes_display_fields():
    card = _complete_card()
    enforce_card_shape(card)  # adds authors=["Not specified"], logo="Not specified"
    missing = extract_missing_fields(card)
    assert not (set(missing) & _DISPLAY_ONLY_FIELDS)


def test_missing_fields_filled_baseline_absent_genuine_ns_stays():
    card = _complete_card()
    card["data"]["collection_date"] = "Not specified"  # genuine NS content field
    enforce_card_shape(card)
    missing = extract_missing_fields(card)
    assert "methodology.baseline_results" not in missing  # filled
    assert "data.collection_date" in missing              # genuine NS stays


# ---------------------------------------------------------------- re-pilot cards ----

_REPILOT_GLOB = "output/d6_repilot_wave1/output/*/benchmarkcard/benchmark_card_*.json"


def _repilot_card_paths():
    root = Path(__file__).resolve().parents[1]
    return sorted(glob.glob(str(root / _REPILOT_GLOB)))


@pytest.mark.parametrize("path", _repilot_card_paths() or [None])
def test_repilot_cards_shape_and_missing_fields(path):
    if path is None:
        pytest.skip("re-pilot cards not present (output/ is gitignored)")
    card = extract_card(json.load(open(path)))

    enforce_card_shape(card)

    for sec in SECTION_MODELS:
        keys = set(card[sec])
        assert "provenance" not in keys
        if sec == "benchmark_details":
            assert _model_keys(sec) <= keys
            assert BD_EXTRAS_REQUIRED <= keys
            assert (keys - _model_keys(sec)) <= BD_EXTRAS_ALLOWED
        else:
            assert keys == _model_keys(sec)

    # missing_fields recomputed after enforce: no display fields, filled baseline absent,
    # every genuine NS content field present.
    missing = set(extract_missing_fields(card))
    assert not (missing & _DISPLAY_ONLY_FIELDS)
    assert "methodology.baseline_results" not in missing  # EEE-filled in every re-pilot card
    genuine_ns = {
        f"{sec}.{field}"
        for sec in SECTION_MODELS for field, val in card[sec].items()
        if field != "provenance"
        and f"{sec}.{field}" not in _DISPLAY_ONLY_FIELDS
        and is_not_specified(val)
    }
    assert genuine_ns <= missing
