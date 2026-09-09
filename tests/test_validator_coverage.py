"""Coverage-completeness flag for benchmark_details.domains (v1 scope).

The coverage-truncation class from the S=150 findings: a list field ships one
true element as if it were the whole set (the math card's domains=['Geometry']
though its own size_breakdown enumerates seven subjects). The check is
deterministic and flag-only: the shipped value is never rewritten, and
non-topical subset names (difficulty tiers, splits, digits) never count toward
the enumerated total, so tier-partitioned benchmarks stay unflagged.
"""

import json

from auto_benchmarkcard.tools.composer import validator as V
from tests.test_composer_stage_b import SECTION_MODELS, _gen

MATH_BREAKDOWN = {
    "Prealgebra": 871, "Algebra": 1187, "Number Theory": 540,
    "Counting & Probability": 474, "Geometry": 479,
    "Intermediate Algebra": 903, "Precalculus": 546,
}


def _frame(names):
    return {"nodes": [{"id": "subset:" + n.lower(), "type": "subset", "name": n}
                      for n in names]}


# --------------------------------------------------- domain_coverage_flag ----

def test_math_like_breakdown_flags_single_domain():
    reason = V.domain_coverage_flag(["Geometry"], size_breakdown=MATH_BREAKDOWN)
    assert reason == "coverage: lists 1 of 7 enumerated subsets/subjects"


def test_difficulty_tiers_never_flag():
    # easy/medium/hard are subsets but not domains: zero topical names, no flag
    assert V.domain_coverage_flag(
        ["Mathematics"], size_breakdown={"easy": 100, "medium": 100, "hard": 100}) is None


def test_no_breakdown_no_frame_is_noop():
    assert V.domain_coverage_flag(["Geometry"]) is None


def test_frame_subset_nodes_flag():
    reason = V.domain_coverage_flag(
        ["Law"], card_frame=_frame(["Law", "Medicine", "History", "Physics", "Chemistry"]))
    assert reason == "coverage: lists 1 of 5 enumerated subsets/subjects"


def test_half_covered_list_not_flagged():
    doms = ["Algebra", "Geometry", "Number Theory", "Precalculus"]
    assert V.domain_coverage_flag(doms, size_breakdown=MATH_BREAKDOWN) is None


def test_all_names_token_present_not_flagged():
    # every topical name appears as whole tokens across the entries -> covered
    assert V.domain_coverage_flag(
        ["Algebra, Geometry and Precalculus problems"],
        size_breakdown={"Algebra": 1, "Geometry": 2, "Precalculus": 3}) is None


def test_whole_token_match_no_substring_credit():
    # 'Art' is not covered by 'state-of-the-art': still 1 of 4 -> flag
    reason = V.domain_coverage_flag(
        ["State-of-the-art reasoning"],
        size_breakdown={"Art": 1, "Music": 2, "Dance": 3, "Theater": 4})
    assert reason == "coverage: lists 1 of 4 enumerated subsets/subjects"


def test_digit_and_split_keys_do_not_count():
    # only Algebra/Geometry are topical -> 1 of 2 is not under half -> no flag
    breakdown = {"train": 7500, "test": 5000, "2021": 1, "Algebra": 1, "Geometry": 2}
    assert V.domain_coverage_flag(["Algebra"], size_breakdown=breakdown) is None


def test_ns_or_nonlist_domains_is_noop():
    assert V.domain_coverage_flag(["Not specified"], size_breakdown=MATH_BREAKDOWN) is None
    assert V.domain_coverage_flag("Not specified", size_breakdown=MATH_BREAKDOWN) is None
    assert V.domain_coverage_flag(None, size_breakdown=MATH_BREAKDOWN) is None


def test_small_breakdown_below_min_names_is_noop():
    # a 2-key breakdown never qualifies as an enumeration
    assert V.domain_coverage_flag(
        ["History"], size_breakdown={"Algebra": 1, "Geometry": 2}) is None


# ----------------------------------------------- run_group_with_repair wiring ----

def _bd_response(domains):
    return json.dumps({"benchmark_details": {
        "name": "MATH", "overview": "Not specified", "data_type": "text",
        "domains": domains, "languages": ["English"],
        "similar_benchmarks": ["Not specified"], "resources": [],
        "provenance": {"domains": {"evidence_ids": ["E1"]}}}})


def test_run_group_flags_and_counts():
    out = V.run_group_with_repair(
        "identity", ["benchmark_details"], SECTION_MODELS, "BASE",
        _gen([_bd_response(["Geometry"])]), {"E1"},
        size_breakdown=MATH_BREAKDOWN)
    assert out.flagged.get("domains") == "coverage: lists 1 of 7 enumerated subsets/subjects"
    assert out.telemetry["coverage_checks"] == 1
    assert out.telemetry["coverage_flags"] == 1
    # flag-only: the shipped value is untouched and the flag is not schema_invalid
    assert out.sections["benchmark_details"]["domains"] == ["Geometry"]
    assert "domains" not in out.telemetry["schema_invalid"]


def test_run_group_difficulty_breakdown_checked_not_flagged():
    out = V.run_group_with_repair(
        "identity", ["benchmark_details"], SECTION_MODELS, "BASE",
        _gen([_bd_response(["Mathematics"])]), {"E1"},
        size_breakdown={"easy": 1, "medium": 2, "hard": 3})
    assert out.flagged == {}
    assert out.telemetry["coverage_checks"] == 1
    assert out.telemetry["coverage_flags"] == 0


def test_run_group_frame_only_flags():
    out = V.run_group_with_repair(
        "identity", ["benchmark_details"], SECTION_MODELS, "BASE",
        _gen([_bd_response(["Geometry"])]), {"E1"},
        card_frame=_frame(["Algebra", "Geometry", "Precalculus", "Number Theory",
                           "Prealgebra"]))
    assert out.flagged.get("domains") == "coverage: lists 1 of 5 enumerated subsets/subjects"


def test_legacy_call_without_kwargs_unchanged():
    out = V.run_group_with_repair(
        "identity", ["benchmark_details"], SECTION_MODELS, "BASE",
        _gen([_bd_response(["Geometry"])]), {"E1"})
    assert out.flagged == {}
    assert "coverage_checks" not in out.telemetry
    assert "coverage_flags" not in out.telemetry
    assert out.sections["benchmark_details"]["domains"] == ["Geometry"]


def test_group_without_benchmark_details_untouched():
    ethics = {"ethical_and_legal_considerations": {
        "privacy_and_anonymity": "p", "data_licensing": "MIT License",
        "consent_procedures": "c", "compliance_with_regulations": "r",
        "provenance": {"privacy_and_anonymity": {"evidence_ids": ["E1"]},
                       "consent_procedures": {"evidence_ids": ["E2"]},
                       "compliance_with_regulations": {"evidence_ids": ["E3"]}}}}
    out = V.run_group_with_repair(
        "ethics", ["ethical_and_legal_considerations"], SECTION_MODELS, "BASE",
        _gen([json.dumps(ethics)]), {"E1", "E2", "E3"},
        size_breakdown=MATH_BREAKDOWN)
    assert out.flagged == {}
    assert "coverage_checks" not in out.telemetry
