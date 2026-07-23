"""F4 (Tier-B): overview fill from EEE description, and the pipeline-jargon prose backstop.
Synthetic inputs only (no network, no real cards)."""

from auto_benchmarkcard.tools.composer import composer_tool as C


# --------------------------------------------------------------- overview fill
_WRAPPER = {"metric_kind": "benchmark_score", "metric_name": "X score",
            "score_type": "continuous", "metric_unit": "proportion"}


def test_overview_from_single_wrapper_blurb():
    eee = {"benchmark_type": "single",
           "metrics": {"llm_stats.x.score": dict(_WRAPPER,
                       evaluation_description="X is a benchmark for grid reasoning. It has 800 tasks.")}}
    ov = C._eee_overview_text(eee)
    assert ov.startswith("X is a benchmark for grid reasoning")


def test_overview_from_composite_contains():
    eee = {"benchmark_type": "composite", "contains": ["BBH", "GPQA", "MUSR"], "metrics": {}}
    assert C._eee_overview_text(eee) == "A composite benchmark suite comprising: BBH, GPQA, MUSR."


def test_overview_meta_no_docs_text_returns_none():
    # aa-index shape: the "description" is a meta-statement about missing documentation
    eee = {"benchmark_type": "single",
           "metrics": {"llm_stats.aa.score": dict(_WRAPPER,
                       evaluation_description="No official academic documentation found for this benchmark. "
                                              "Extensive research yielded no peer-reviewed sources.")}}
    assert C._eee_overview_text(eee) is None


def test_overview_none_when_no_usable_text():
    assert C._eee_overview_text({"metrics": {}}) is None
    assert C._eee_overview_text({"metrics": {"x": dict(_WRAPPER, evaluation_description="too short")}}) is None
    assert C._eee_overview_text(None) is None


def test_overview_reads_unmutated_metrics_even_with_real_scorer():
    # the wrapper blurb must be reachable even when a real scorer would drop the wrapper downstream
    eee = {"benchmark_type": "single", "metrics": {
        "accuracy": {"metric_kind": "accuracy", "evaluation_description": "Accuracy on the task"},
        "llm_stats.x.score": dict(_WRAPPER,
            evaluation_description="X is a multi-domain reasoning benchmark of 1000 expert questions."),
    }}
    assert C._eee_overview_text(eee).startswith("X is a multi-domain reasoning benchmark")


def test_overview_override_is_fill_only():
    facts = {"benchmark_details.overview": "Deterministic EEE overview."}
    husk = {"benchmark_details": {"overview": "Not specified"}}
    C.apply_deterministic_overrides(husk, facts)
    assert husk["benchmark_details"]["overview"] == "Deterministic EEE overview."
    real = {"benchmark_details": {"overview": "A real Stage-B overview."}}
    C.apply_deterministic_overrides(real, facts)
    assert real["benchmark_details"]["overview"] == "A real Stage-B overview."  # not clobbered


def test_extract_facts_emits_overview_for_husk():
    eee = {"benchmark_name": "X", "benchmark_type": "single",
           "metrics": {"llm_stats.x.score": dict(_WRAPPER,
                       evaluation_description="X evaluates models on 500 challenging reasoning problems.")}}
    facts = C.extract_deterministic_facts(eee_metadata=eee)
    assert facts["benchmark_details.overview"].startswith("X evaluates models on 500")


# --------------------------------------------------------- pipeline-jargon backstop
def test_jargon_backstop_drops_furniture_sentence():
    card = {"benchmark_details": {"overview":
            "This is a real benchmark for X. The dataset contains Ergon-native sharded rollout-card "
            "exports for the paper artifact."},
            "methodology": {}}
    C._scrub_leaked_metric_ids(card, {"metrics": {}})
    ov = card["benchmark_details"]["overview"]
    assert "Ergon-native" not in ov and "rollout-card" not in ov
    assert ov == "This is a real benchmark for X."


def test_jargon_backstop_strips_stray_namespace_prefix():
    card = {"methodology": {"calculation": "Scores derived from llm_stats.foo aggregates."}}
    C._scrub_leaked_metric_ids(card, {"metrics": {}})
    assert "llm_stats." not in card["methodology"]["calculation"]


def test_jargon_backstop_clean_prose_untouched_and_idempotent():
    clean = "A standard accuracy benchmark with 1000 questions."
    card = {"benchmark_details": {"overview": clean}, "methodology": {}}
    C._scrub_leaked_metric_ids(card, {"metrics": {}})
    assert card["benchmark_details"]["overview"] == clean


def test_scrub_preserves_composite_names_in_overview():
    # a composite overview naming sub-benchmarks (which double as metric ids with other: canonicals)
    # must NOT be mangled into other:GPQA / other:IFEval by the metric-id scrub
    eee = {"benchmark_type": "composite",
           "metrics": {"GPQA": {"evaluation_description": "COT correct on GPQA"},
                       "IFEval": {"evaluation_description": "score on IFEval"}}}
    card = {"benchmark_details": {"overview": "A composite benchmark suite comprising: GPQA, IFEval."},
            "methodology": {}}
    C._scrub_leaked_metric_ids(card, eee)
    assert card["benchmark_details"]["overview"] == "A composite benchmark suite comprising: GPQA, IFEval."


def test_scrub_leaves_common_word_metric_id_in_prose():
    # "accuracy" is a metric id but also an ordinary word: the scrub must not rewrite "65% accuracy"
    eee = {"metrics": {"accuracy": {"metric_kind": "accuracy", "evaluation_description": "accuracy"}}}
    card = {"benchmark_details": {"overview": "PhD experts reach 65% accuracy on these questions."},
            "methodology": {}}
    C._scrub_leaked_metric_ids(card, eee)
    assert card["benchmark_details"]["overview"] == "PhD experts reach 65% accuracy on these questions."


def test_scrub_runs_with_no_eee_metrics():
    # the jargon backstop must run even when EEE carries no metrics
    card = {"benchmark_details": {"overview": "Real benchmark. Ergon-native rollout-card export."},
            "methodology": {}}
    C._scrub_leaked_metric_ids(card, None)
    assert "Ergon-native" not in card["benchmark_details"]["overview"]
    assert card["benchmark_details"]["overview"] == "Real benchmark."
