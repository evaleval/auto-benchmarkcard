"""F5: suppress the degenerate llm_stats benchmark_score wrapper metric, and scrub a leaked
raw metric id out of prose. Synthetic metric dicts only (no network, no real cards)."""

from auto_benchmarkcard.tools.composer import composer_tool as C


def _eee(metrics):
    return {"benchmark_name": "TestBench", "metrics": metrics}


# An llm_stats aggregate wrapper (metric_kind == benchmark_score) masks the real scorer.
_WRAPPER = {
    "evaluation_description": "scraped proxy", "metric_kind": "benchmark_score",
    "metric_name": "TestBench score", "score_type": "continuous",
}
# A real scorer.
_REAL = {
    "evaluation_description": "first-attempt pass rate", "metric_kind": "score",
    "metric_name": "TestBench Pass@1", "score_type": "continuous",
}


def test_is_degenerate_metric_predicate():
    assert C._is_degenerate_metric("llm_stats.testbench.score", _WRAPPER) is True
    assert C._is_degenerate_metric("testbench_pass_1", _REAL) is False
    assert C._is_degenerate_metric("anything", {}) is False


def test_wrapper_dropped_real_scorer_kept():
    facts = C.extract_deterministic_facts(
        _eee({"testbench_pass_1": _REAL, "llm_stats.testbench.score": _WRAPPER}))
    # the wrapper's other:testbench.score is gone; only the real scorer survives
    assert facts["methodology.metrics"] == ["pass@1"]
    assert "llm_stats.testbench.score" not in facts.get("methodology.metric_configs", {})


def test_only_wrapper_not_blanked():
    # suppression would empty the field -> keep the wrapper rather than ship empty metrics
    facts = C.extract_deterministic_facts(_eee({"llm_stats.testbench.score": _WRAPPER}))
    assert facts["methodology.metrics"] == ["other:testbench.score"]


def test_scrubber_replaces_degenerate_raw_id_with_metric_name():
    card = {"methodology": {"calculation": "Scores use llm_stats.testbench.score on the held-out set."},
            "benchmark_details": {"overview": "ov"}}
    C._scrub_leaked_metric_ids(card, _eee({"llm_stats.testbench.score": _WRAPPER}))
    calc = card["methodology"]["calculation"]
    assert "llm_stats.testbench.score" not in calc      # raw id gone
    assert "other:testbench.score" not in calc          # NOT re-injected as the suppressed canonical
    assert "TestBench score" in calc                    # human metric_name used instead


def test_scrubber_legit_scorer_to_canonical_and_idempotent():
    card = {"methodology": {"methods": "Reported as testbench_pass_1 over the hard set."}}
    eee = _eee({"testbench_pass_1": _REAL})
    C._scrub_leaked_metric_ids(card, eee)
    methods = card["methodology"]["methods"]
    assert "testbench_pass_1" not in methods
    assert "pass@1" in methods                          # a real scorer's id -> canonical display
    snapshot = methods
    C._scrub_leaked_metric_ids(card, eee)               # second pass changes nothing
    assert card["methodology"]["methods"] == snapshot
