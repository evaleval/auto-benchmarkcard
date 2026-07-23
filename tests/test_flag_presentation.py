"""Tests for the display-layer flag policy in workers.py (composer-redesign v2, brief C).

Network-free: exercises the pure presentation helpers directly. The policy is applied
downstream of the frozen FactReasoner flagging and never touches the gold-set instruments
(reapply_flags --self-test runs a separate flag reimplementation). Synthetic field paths,
so the checks are general, not pilot-specific.
"""

from auto_benchmarkcard.workers import (
    PRESENTATION_EXCLUDED_FLAGS,
    SOURCE_BLOCKED_FLAG_REASON,
    _apply_presentation_flag_policy,
    _source_coverage_blocked,
)

PH = "[Probable Hallucination], no supporting evidence found in full source material"
LOWSCORE = "[Factuality Score: 0.20], low factual alignment with source material"
SCHEMA = "[schema_invalid] enum: foo not allowed"


def test_excluded_set_is_dotted_paths():
    assert PRESENTATION_EXCLUDED_FLAGS == {"benchmark_details.resources", "data.annotation"}


def test_source_coverage_blocked_signals():
    # thin escalation source (< 500 chars) -> blocked
    assert _source_coverage_blocked({}, "x" * 120) is True
    # degraded docling fetch (paywall / 403 -> abstract-only or no content) -> blocked
    # even when the abstract text is long.
    assert _source_coverage_blocked(
        {"docling_telemetry": {"degraded_to_abstract_only": True}}, "y" * 2000) is True
    assert _source_coverage_blocked(
        {"docling_telemetry": {"degraded_to_no_content": True}}, "y" * 2000) is True
    # full, non-degraded source -> not blocked
    assert _source_coverage_blocked(
        {"docling_telemetry": {"reason": "html_only"}}, "z" * 2000) is False


def test_c2_excludes_resources_and_annotation():
    ff = {
        "benchmark_details.resources": PH,
        "data.annotation": PH,
        "methodology.methods": PH,
    }
    _apply_presentation_flag_policy(ff, source_blocked=False)
    assert "benchmark_details.resources" not in ff
    assert "data.annotation" not in ff
    assert ff["methodology.methods"] == PH  # unrelated flag untouched


def test_c2_preserves_schema_invalid_residual():
    # a parse-failure residual on an excluded field must survive for the gate's metric.
    ff = {"data.annotation": SCHEMA, "benchmark_details.resources": PH}
    _apply_presentation_flag_policy(ff, source_blocked=False)
    assert ff["data.annotation"] == SCHEMA
    assert "benchmark_details.resources" not in ff


def test_c1_relabels_only_hallucination_on_blocked_card():
    ff = {
        "methodology.methods": PH,
        "benchmark_details.languages": PH,
        "data.size": LOWSCORE,   # contradiction signal -> kept
        "data.format": SCHEMA,   # parse failure -> kept
    }
    _apply_presentation_flag_policy(ff, source_blocked=True)
    assert ff["methodology.methods"] == SOURCE_BLOCKED_FLAG_REASON
    assert ff["benchmark_details.languages"] == SOURCE_BLOCKED_FLAG_REASON
    assert ff["data.size"] == LOWSCORE
    assert ff["data.format"] == SCHEMA


def test_c1_no_relabel_on_normally_sourced_card():
    ff = {"methodology.methods": PH}
    _apply_presentation_flag_policy(ff, source_blocked=False)
    assert ff["methodology.methods"] == PH  # full-source hallucination flag stays


def test_biolp_like_card_combined_c1_c2():
    # biolp regression shape: 4 PH flags incl resources+annotation, source-blocked (403).
    ff = {
        "benchmark_details.resources": PH,
        "data.annotation": PH,
        "methodology.methods": PH,
        "methodology.human_baseline": PH,
    }
    _apply_presentation_flag_policy(ff, source_blocked=True)
    assert set(ff) == {"methodology.methods", "methodology.human_baseline"}
    assert all(v == SOURCE_BLOCKED_FLAG_REASON for v in ff.values())
