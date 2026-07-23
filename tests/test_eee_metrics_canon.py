"""Tests for EEE metric canonicalization (#3, contract S4.5) and the S10 deterministic
provenance no-op invariant (canonical names in the value, raw EEE ids in provenance).
Network-free.
"""

from auto_benchmarkcard.tools.composer.composer_tool import (
    _canonicalize_metric,
    _deterministic_field_provenance,
    canonicalize_metrics,
)


def test_pass_at_k_from_id():
    assert _canonicalize_metric(
        "bigcodebench_hard_set_pass_1", {"score_type": "continuous"}) == "pass@1"
    assert _canonicalize_metric("pass@10", {}) == "pass@10"


def test_pass_at_k_with_at_separator():
    # forward-looking for the 500-batch: the "pass_at_1" spelling folds to pass@1, not other:
    assert _canonicalize_metric("pass_at_1", {}) == "pass@1"
    assert _canonicalize_metric("pass_1", {}) == "pass@1"


def test_continuous_score_id_stays_other_with_prefix_stripped():
    # the batchA reality: .score ids are continuous -> not Accuracy, kept as other:
    assert _canonicalize_metric(
        "llm_stats.aa-lcr.score", {"score_type": "continuous"}) == "other:aa-lcr.score"


def test_proportion_or_binary_score_maps_to_accuracy():
    assert _canonicalize_metric("llm_stats.x.score", {"score_type": "proportion"}) == "Accuracy"
    assert _canonicalize_metric("foo.score", {"score_type": "binary"}) == "Accuracy"


def test_explicit_canonical_names():
    assert _canonicalize_metric("exact_match", {}) == "Exact Match"
    assert _canonicalize_metric("macro_f1", {}) == "F1"
    assert _canonicalize_metric("bleu_4", {}) == "BLEU"
    assert _canonicalize_metric("rouge_l", {}) == "ROUGE"
    assert _canonicalize_metric("win_rate", {}) == "win-rate"
    assert _canonicalize_metric("ndcg_at_10", {}) == "NDCG@k"


def test_unmapped_id_other_fallback_no_guess():
    assert _canonicalize_metric("weird_custom_id", {}) == "other:weird_custom_id"


def test_metric_from_kind_pass_rate():
    # F4: pass_rate kind + "Pass@1" in description -> pass@1 (livecodebenchpro/apex-agents)
    cfg = {"metric_kind": "pass_rate", "evaluation_description": "Pass@1 on Medium Problems",
           "score_type": "continuous", "metric_unit": "proportion"}
    assert _canonicalize_metric("pass_at_k", cfg) == "pass@1"


def test_metric_from_desc_win_rate():
    # F4: a degenerate wrapper whose description states a length-controlled win-rate (alpacaeval)
    cfg = {"metric_kind": "benchmark_score", "metric_unit": "proportion", "score_type": "continuous",
           "evaluation_description": "A length-controlled automatic evaluator using length-controlled win-rates."}
    assert _canonicalize_metric("llm_stats.alpacaeval-2.0.score", cfg) == "win-rate"


def test_proportion_answer_correctness_maps_to_accuracy():
    # F4: proportion unit + answer-correctness signal -> Accuracy (arc "exactly correct", blink MCQ)
    arc = {"metric_unit": "proportion", "score_type": "continuous",
           "evaluation_description": "produce exactly correct output grids for all test inputs to solve it"}
    assert _canonicalize_metric("llm_stats.arc.score", arc) == "Accuracy"
    blink = {"metric_unit": "proportion",
             "evaluation_description": "reformats vision tasks into 3807 multiple-choice questions"}
    assert _canonicalize_metric("llm_stats.blink.score", blink) == "Accuracy"


def test_coding_benchmark_never_accuracy():
    # F4 coding guard: proportion + correctness words BUT code/execution markers -> stays other:
    # (aider-polyglot's true metric is a pass-rate; mislabelling it Accuracy is a silent error)
    aider = {"metric_kind": "benchmark_score", "metric_unit": "proportion", "score_type": "continuous",
             "evaluation_description": ("A coding benchmark evaluating LLMs on 225 Exercism programming "
                                        "exercises. Models receive two attempts to solve each problem, "
                                        "with test error feedback after the first attempt.")}
    assert _canonicalize_metric("llm_stats.aider-polyglot.score", aider) == "other:aider-polyglot.score"
    bcb = {"metric_unit": "proportion",
           "evaluation_description": "solve complex programming tasks via code generation"}
    assert _canonicalize_metric("llm_stats.bigcodebench-full.score", bcb) == "other:bigcodebench-full.score"


def test_proportion_without_signal_stays_other():
    # F4: proportion unit but no metric word and no answer-correctness signal -> honest other:
    # (chexpert: AUC/F1 is not stated in the source, so we never guess it)
    chex = {"metric_unit": "proportion", "score_type": "continuous",
            "evaluation_description": "a large dataset of chest radiographs for automated interpretation"}
    assert _canonicalize_metric("llm_stats.chexpert-cxr.score", chex) == "other:chexpert-cxr.score"


def test_cfg_signals_are_additive_no_break_on_bare_cfg():
    # the no-signal wrapper (only score_type=continuous) is unchanged by the cfg consultation
    assert _canonicalize_metric("llm_stats.aa-lcr.score", {"score_type": "continuous"}) == "other:aa-lcr.score"


def test_no_dedup_distinct_ids_order_preserved():
    out = canonicalize_metrics({
        "bigcodebench_hard_set_pass_1": {"score_type": "continuous"},
        "llm_stats.bigcodebench.score": {"score_type": "continuous"},
    })
    assert out == ["pass@1", "other:bigcodebench.score"]


def test_provenance_keeps_full_raw_ids_when_override_won():
    card = {"methodology": {"metrics": ["pass@1", "other:bigcodebench.score"]}}
    det = {"methodology.metrics": ["pass@1", "other:bigcodebench.score"]}
    eee = {"metrics": {"bigcodebench_hard_set_pass_1": {}, "llm_stats.bigcodebench.score": {}}}
    entry = _deterministic_field_provenance("methodology.metrics", "eee", card, det, eee)
    assert entry is not None
    assert entry["evidence"] == "bigcodebench_hard_set_pass_1, llm_stats.bigcodebench.score"
    assert entry["quote"] == entry["evidence"]
    assert entry["source"] == "deterministic"
    assert entry["status"] == "stated"


def test_provenance_no_op_invariant_suppresses_when_value_differs():
    # S10: emission stays gated on card_value == det_facts value (LLM kept its own metrics);
    # only the stored evidence changed, never the emit/suppress decision.
    card = {"methodology": {"metrics": ["something the LLM wrote"]}}
    det = {"methodology.metrics": ["pass@1"]}
    eee = {"metrics": {"bigcodebench_hard_set_pass_1": {}}}
    assert _deterministic_field_provenance("methodology.metrics", "eee", card, det, eee) is None


def test_provenance_non_metrics_field_quote_unchanged():
    card = {"data": {"size": "1319"}}
    det = {"data.size": "1319"}
    entry = _deterministic_field_provenance("data.size", "hf", card, det, None)
    assert entry["evidence"] == "1319" and entry["quote"] == "1319"
