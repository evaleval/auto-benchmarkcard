"""Unit checks for composer-redesign phase [0]: Tag-0 plumbing, the frozen
measurement instrument, the set-driven display registry, deterministic byte-identity
(both override sites), benchmark_type enum mapping, the EEE->Stage-B prefill seam, the
single-LLM-judge bias rule, and the v2-scoped prose backfill gate.

Pure unit tests -- no inference, no gold data. Run: pytest scripts/test_composer_phase0.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from auto_benchmarkcard.card_utils import (  # noqa: E402
    DETERMINISTIC_FIELDS,
    _DISPLAY_ONLY_FIELDS,
    _PROSE_BACKFILL_FIELDS,
    apply_deterministic_overrides as workers_apply_overrides,
    apply_single_judge_bias_rule,
    backfill_from_provenance,
    card_field_stats,
    extract_hf_tags,
    map_benchmark_type,
    normalize_not_specified,
)
from auto_benchmarkcard.llm_handler import _coerce_to_text, _extract_json_object  # noqa: E402
from auto_benchmarkcard.tools.composer.composer_tool import (  # noqa: E402
    apply_deterministic_overrides,
    apply_display_fields,
    build_eee_prefill_block,
    extract_deterministic_facts,
)
from auto_benchmarkcard.tools.rag.atomizer import (  # noqa: E402
    _map_to_valid_field,
    exclude_risk_sections,
    text_to_statements,
)
from auto_benchmarkcard.validation_policy import IDENTITY_FIELDS  # noqa: E402

# Live HF tags shaped like the dataset metadata the pipeline ingests.
HF = {
    "tags": [
        "language:fr", "language:en", "license:mit",
        "size_categories:10K<n<100K", "format:parquet",
        "modality:text", "task_categories:text-classification",
    ],
    "card_data": {"authors": ["Jane Doe", "John Roe"]},
}
HF_KEYS = (
    "benchmark_details.languages",
    "benchmark_details.data_type",
    "data.format",
    "data.size",
    "ethical_and_legal_considerations.data_licensing",
)


# --- Tag-0 plumbing ---------------------------------------------------------

def test_extract_json_object_handles_nested():
    raw = 'prefix {"a": {"b": 1}, "c": [2, 3]} trailing junk'
    assert _extract_json_object(raw) == '{"a": {"b": 1}, "c": [2, 3]}'


def test_extract_json_object_ignores_braces_in_strings():
    raw = '{"k": "a } b { c"}'
    assert _extract_json_object(raw) == raw


def test_extract_json_object_none_when_absent():
    assert _extract_json_object("no object here") is None


def test_coerce_to_text_variants():
    class FakeMessage:
        content = "hello"

    class FakePromptValue:
        def to_string(self):
            return "from to_string"

    assert _coerce_to_text("plain") == "plain"
    assert _coerce_to_text(FakePromptValue()) == "from to_string"
    assert _coerce_to_text(FakeMessage()) == "hello"
    assert _coerce_to_text([{"content": "last"}]) == "last"
    assert _coerce_to_text([FakeMessage()]) == "hello"


# --- [M] frozen measurement instrument --------------------------------------

def test_deterministic_fields_frozen_and_exact():
    assert isinstance(DETERMINISTIC_FIELDS, frozenset)
    assert DETERMINISTIC_FIELDS == frozenset({
        "benchmark_details.languages",
        "benchmark_details.data_type",
        "ethical_and_legal_considerations.data_licensing",
        "data.size",
        "data.format",
        "methodology.metrics",
    })


def test_judge_skip_is_explicit_literal_decoupled():
    # Importing here (not at top) keeps the script-module import side effects local.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import judge_gold_set  # noqa: E402

    expected_base = {
        "benchmark_details.languages",
        "benchmark_details.data_type",
        "ethical_and_legal_considerations.data_licensing",
        "data.size", "data.format", "methodology.metrics",
        "benchmark_details.resources", "benchmark_details.appears_in",
        # additive structured/scalar v2 fields (bv-to-0 item 7)
        "data.size_breakdown", "data.collection_date",
        "methodology.judge_uses_llm", "methodology.judge_num", "methodology.judge_models",
    }
    assert judge_gold_set.JUDGE_SKIP == frozenset(expected_base) | _DISPLAY_ONLY_FIELDS
    # Decoupled: JUDGE_SKIP must not be built from DETERMINISTIC_FIELDS at runtime.
    assert "JUDGE_SKIP = frozenset({" in open(
        os.path.join(os.path.dirname(__file__), "judge_gold_set.py")
    ).read()


# --- Display registry (set-driven) ------------------------------------------

def test_display_registry_contents():
    assert _DISPLAY_ONLY_FIELDS == frozenset({
        "benchmark_details.authors",
        "benchmark_details.logo",
        "benchmark_details.org_url",
    })


def test_card_field_stats_excludes_display():
    card = {
        "benchmark_details": {
            "name": "X", "overview": "Not specified",
            "authors": ["A"], "logo": "http://l", "org_url": "http://o",
        },
        "purpose_and_intended_users": {}, "data": {},
        "methodology": {}, "ethical_and_legal_considerations": {},
    }
    stats = card_field_stats(card)
    # name + overview counted; authors/logo/org_url excluded entirely.
    assert stats["n_fields"] == 2


def test_card_field_stats_excludes_structured_and_eee_injected():
    # Structured-typed v2 fields and the EEE-injected appears_in/benchmark_type must NOT
    # count as prose (bv-to-0 item 1 / trap #7a). Only name + overview remain.
    card = {
        "benchmark_details": {
            "name": "X", "overview": "an overview",
            "appears_in": ["http://x"], "benchmark_type": "single",
        },
        "data": {"size_breakdown": {"a": 10}},
        "methodology": {
            "judge_uses_llm": True, "judge_num": 1, "judge_models": ["m"],
        },
        "purpose_and_intended_users": {}, "ethical_and_legal_considerations": {},
    }
    stats = card_field_stats(card)
    assert stats["n_fields"] == 2  # only name + overview
    for f in ("appears_in", "benchmark_type", "size_breakdown",
              "judge_uses_llm", "judge_num", "judge_models"):
        assert f in __import__("auto_benchmarkcard.card_utils", fromlist=["_FIELD_SKIP"])._FIELD_SKIP


def test_atomizer_strips_display_fields():
    card = {"benchmark_details": {"name": "X", "authors": ["A"], "logo": "u", "org_url": "o"}}
    out = exclude_risk_sections(card)
    assert "authors" not in out["benchmark_details"]
    assert "logo" not in out["benchmark_details"]
    assert "org_url" not in out["benchmark_details"]
    assert out["benchmark_details"]["name"] == "X"


# --- Byte-identity / no-op for the two override sites ------------------------

def test_det_facts_hf_keys_byte_identical_to_extract_hf_tags():
    det = extract_deterministic_facts(None, HF, None)
    hft = extract_hf_tags(HF)
    for k in HF_KEYS:
        assert det.get(k) == hft.get(k), k


def test_composer_override_is_noop_on_prefilled_card():
    # A card already pre-filled with extract_hf_tags values; re-applying the composition
    # override (site #1906) with det_facts must not change those fields (S1 no-op).
    hft = extract_hf_tags(HF)
    det = extract_deterministic_facts(None, HF, None)
    card = {
        "benchmark_details": {
            "languages": hft["benchmark_details.languages"],
            "data_type": hft["benchmark_details.data_type"],
        },
        "data": {"format": hft["data.format"], "size": hft["data.size"]},
        "methodology": {"metrics": ["Accuracy"]},
        "ethical_and_legal_considerations": {
            "data_licensing": hft["ethical_and_legal_considerations.data_licensing"],
        },
    }
    import copy
    before = copy.deepcopy(card)
    after = apply_deterministic_overrides(card, det)
    assert after["benchmark_details"]["languages"] == before["benchmark_details"]["languages"]
    assert after["benchmark_details"]["data_type"] == before["benchmark_details"]["data_type"]
    assert after["ethical_and_legal_considerations"]["data_licensing"] == \
        before["ethical_and_legal_considerations"]["data_licensing"]
    assert after["data"]["format"] == before["data"]["format"]
    assert after["data"]["size"] == before["data"]["size"]
    assert after["methodology"]["metrics"] == before["methodology"]["metrics"]


def test_both_override_sites_agree_on_all_six_det_fields():
    # The contract requires BOTH override sites byte-identical (S10). Start from an
    # LLM-shaped card (languages/data_type/license differ from HF; size/format empty)
    # and run each site's override on its own source; the six det fields must match.
    import copy
    base = {
        "benchmark_details": {"languages": ["German"], "data_type": "text"},
        "data": {"size": "Not specified", "format": "Not specified"},
        "methodology": {"metrics": ["Not specified"]},
        "ethical_and_legal_considerations": {"data_licensing": "Not specified"},
    }
    det = extract_deterministic_facts(None, HF, None)   # composition-site source
    hft = extract_hf_tags(HF)                            # workers-site source
    comp = apply_deterministic_overrides(copy.deepcopy(base), det)
    work = workers_apply_overrides(copy.deepcopy(base), hft)
    for sec, fld in [
        ("benchmark_details", "languages"),
        ("benchmark_details", "data_type"),
        ("data", "size"),
        ("data", "format"),
        ("ethical_and_legal_considerations", "data_licensing"),
        ("methodology", "metrics"),
    ]:
        assert comp[sec][fld] == work[sec][fld], \
            f"{sec}.{fld}: composition={comp[sec][fld]!r} workers={work[sec][fld]!r}"


def test_license_has_no_raw_meta_fallback():
    # extract_hf_tags is the single license source: no raw meta/card_data fallback that
    # would emit a non-Tier-1 value diverging from the workers site (S1/S4.4).
    det_other = extract_deterministic_facts(None, {"tags": ["license:other", "language:en"], "license": "mit"}, None)
    assert "ethical_and_legal_considerations.data_licensing" not in det_other
    det_cardonly = extract_deterministic_facts(None, {"tags": ["language:en"], "card_data": {"license": "mit"}}, None)
    assert "ethical_and_legal_considerations.data_licensing" not in det_cardonly


# --- benchmark_type enum mapping --------------------------------------------

def test_benchmark_type_mapping():
    assert map_benchmark_type("single") == "single"
    assert map_benchmark_type("composite") == "composite_suite"
    assert map_benchmark_type("sub-benchmark") == "single"
    assert map_benchmark_type("Composite") == "composite_suite"
    assert map_benchmark_type(None) == "single"
    assert map_benchmark_type("leaderboard_aggregate") == "leaderboard_aggregate"


# --- EEE -> Stage B prefill seam --------------------------------------------

def test_eee_prefill_block_reconstructs_blocks():
    det = {
        "benchmark_type": "composite",
        "composite_context": "Suite X contains A, B.",
        "evaluation_summary": {
            "primary_metric": "acc",
            "top_performers": [{"model": "m1", "score": 0.812}],
            "score_statistics": {"mean": 0.7, "std_dev": 0.1, "min": 0.5, "max": 0.9},
            "total_models_evaluated": 12,
        },
    }
    block = build_eee_prefill_block(det)
    assert "[DETERMINISTIC] Suite X contains A, B." in block
    assert "[EEE] Evaluation results from Every Eval Ever:" in block
    assert "m1: 0.8120 (acc)" in block
    assert "Total models evaluated: 12" in block


def test_eee_prefill_block_empty_without_eee():
    assert build_eee_prefill_block({}) == ""


# --- single-LLM-judge bias rule (flat [B+V] judge_* fields) -----------------

def test_bias_rule_noop_without_judge_fields():
    card = {"methodology": {}, "possible_risks": []}
    apply_single_judge_bias_rule(card)
    assert card["possible_risks"] == []


def test_bias_rule_appends_for_single_judge_and_is_idempotent():
    card = {"methodology": {"judge_uses_llm": True, "judge_num": 1},
            "possible_risks": []}
    apply_single_judge_bias_rule(card)
    apply_single_judge_bias_rule(card)
    cats = [r["category"] for r in card["possible_risks"]]
    assert cats == ["Evaluation bias (single LLM judge)"]


def test_bias_rule_noop_for_multiple_judges():
    card = {"methodology": {"judge_uses_llm": True, "judge_num": 3},
            "possible_risks": []}
    apply_single_judge_bias_rule(card)
    assert card["possible_risks"] == []


def test_bias_rule_guards_sentinel_and_type():
    # Both fields default to the string sentinel; the rule must read neither as a
    # judge count nor as a truthy flag, and must not raise on "Not specified" <= 1.
    sentinel = {"methodology": {"judge_uses_llm": "Not specified",
                                "judge_num": "Not specified"}, "possible_risks": []}
    apply_single_judge_bias_rule(sentinel)
    assert sentinel["possible_risks"] == []

    # uses_llm True but count unknown (sentinel) + one named model: still no flag
    # (the rule keys on a real int count, not on len(judge_models)).
    no_count = {"methodology": {"judge_uses_llm": True, "judge_num": "Not specified",
                                "judge_models": ["GPT-4o"]}, "possible_risks": []}
    apply_single_judge_bias_rule(no_count)
    assert no_count["possible_risks"] == []

    # judge_uses_llm False must not flag even when judge_num == 1.
    not_llm = {"methodology": {"judge_uses_llm": False, "judge_num": 1}, "possible_risks": []}
    apply_single_judge_bias_rule(not_llm)
    assert not_llm["possible_risks"] == []


# --- display befuellung -----------------------------------------------------

def test_apply_display_fields_authors_and_org_url(monkeypatch):
    import auto_benchmarkcard.tools.composer.composer_tool as _ct
    monkeypatch.setattr(_ct, "_fetch_org_logo", lambda org: None)  # no network in tests
    card = {"benchmark_details": {}}
    apply_display_fields(card, HF, {"hf_repo": "evaleval/bigcodebench"})
    bd = card["benchmark_details"]
    assert bd["authors"] == ["Jane Doe", "John Roe"]
    assert bd["org_url"] == "https://huggingface.co/evaleval"
    assert "logo" not in bd  # no fabricated URL; org overview yielded nothing


def test_apply_display_fields_authors_from_paper_and_no_fabricated_logo(monkeypatch):
    # No HF authors and no HF image: authors fall back to the resolved paper's authors,
    # and logo stays unset (never fabricated) when the org avatar fetch yields nothing.
    import auto_benchmarkcard.tools.composer.composer_tool as _ct
    monkeypatch.setattr(_ct, "_fetch_org_logo", lambda org: None)
    card = {"benchmark_details": {}}
    hf = {"tags": [], "card_data": {}}
    apply_display_fields(card, hf, {"hf_repo": "evaleval/biolp",
                                    "paper_authors": ["A. One", "B. Two"]})
    bd = card["benchmark_details"]
    assert bd["authors"] == ["A. One", "B. Two"]
    assert "logo" not in bd


def test_apply_display_fields_logo_from_hf_card_image():
    card = {"benchmark_details": {}}
    hf = {"tags": [], "card_data": {"thumbnail": "https://hf.co/img/logo.png"}}
    apply_display_fields(card, hf, {})
    assert card["benchmark_details"]["logo"] == "https://hf.co/img/logo.png"


# --- v2-scoped prose backfill gate ------------------------------------------

def test_prose_gate_blocks_non_prose_fields():
    card = {"methodology": {"metrics": ["Not specified"], "calculation": "Not specified"}}
    prov = {"methodology": {
        "metrics": {"source": "deterministic", "evidence": "raw quote that is long enough"},
        "calculation": {"source": "deterministic", "evidence": "a grounded prose sentence here"},
    }}
    backfill_from_provenance(card, prov, prose_only=True)
    # metrics (non-prose) stays NS; calculation (prose) gets backfilled.
    assert card["methodology"]["metrics"] == ["Not specified"]
    assert card["methodology"]["calculation"] == "a grounded prose sentence here"


def test_prose_allowlist_excludes_deterministic_and_enum_fields():
    assert not (_PROSE_BACKFILL_FIELDS & DETERMINISTIC_FIELDS)
    assert "methodology.validity_justification" in _PROSE_BACKFILL_FIELDS


# --- bv-to-0 reconciliation: flat-field normalize / atomizer / policy --------

def test_normalize_guards_bool_int_and_new_list_fields():
    card = {
        "methodology": {"judge_uses_llm": False, "judge_num": 0, "judge_models": []},
        "benchmark_details": {"authors": []},
    }
    out = normalize_not_specified(card)
    # Real bool/int answers are filled cells, never coerced to the sentinel.
    assert out["methodology"]["judge_uses_llm"] is False
    assert out["methodology"]["judge_num"] == 0
    # Empty list fields collapse to the list-form sentinel (authors/judge_models added).
    assert out["methodology"]["judge_models"] == ["Not specified"]
    assert out["benchmark_details"]["authors"] == ["Not specified"]


def test_atomizer_strips_structured_v2_fields():
    card = {
        "data": {"source": "s", "size_breakdown": {"train": 5}},
        "methodology": {"calculation": "c", "judge_uses_llm": True,
                        "judge_num": 1, "judge_models": ["m"],
                        "judge_score_consolidation": "majority vote"},
    }
    out = exclude_risk_sections(card)
    assert "size_breakdown" not in out["data"]
    for f in ("judge_uses_llm", "judge_num", "judge_models"):
        assert f not in out["methodology"]
    # prose fields survive; judge_score_consolidation is whitelisted, not stripped.
    assert out["data"]["source"] == "s"
    assert out["methodology"]["judge_score_consolidation"] == "majority vote"


def test_atomizer_whitelist_endstate_paths():
    text = "\n".join([
        "- runs the suite [purpose_and_intended_users.audience_evaluators]",
        "- reads the results [purpose_and_intended_users.audience_consumers]",
        "- scores are combined by mean [methodology.judge_score_consolidation]",
        "- validity claim [methodology.validity_justification]",
    ])
    fields = {s["field"] for s in text_to_statements(text)}
    assert "purpose_and_intended_users.audience_evaluators" in fields
    assert "purpose_and_intended_users.audience_consumers" in fields
    assert "methodology.judge_score_consolidation" in fields
    assert "methodology.validity_justification" in fields


def test_map_to_valid_field_audience_split_and_validity_guard():
    # audience-attributed atoms route to the split, not back to goal.
    assert _map_to_valid_field(
        "purpose_and_intended_users.audience", "evaluated by AI researchers"
    ) == "purpose_and_intended_users.audience_evaluators"
    assert _map_to_valid_field(
        "purpose_and_intended_users.audience", "results inform policymaker decisions"
    ) == "purpose_and_intended_users.audience_consumers"
    # validity_justification must not collapse onto methodology.validation.
    assert _map_to_valid_field(
        "methodology.validity", "the construct validity is justified by the authors"
    ) == "methodology.validity_justification"
    # a genuine validation atom is unaffected.
    assert _map_to_valid_field(
        "methodology.qa", "annotator agreement validation procedure"
    ) == "methodology.validation"


def test_identity_fields_audience_split_keeps_legacy():
    # v31 back-compat (self-test byte-identity) + v2 split, both identity.
    assert "purpose_and_intended_users.audience" in IDENTITY_FIELDS
    assert "purpose_and_intended_users.audience_evaluators" in IDENTITY_FIELDS
    assert "purpose_and_intended_users.audience_consumers" in IDENTITY_FIELDS
