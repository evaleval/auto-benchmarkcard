"""FINISH-4 B1: OpenEval aggregate-source denylist.

An aggregate eval source (human-centered-eval/OpenEval) is bound for bold/gpqa/boolq and
injects false languages/size/licensing/org (HF) plus a false metric (EEE). These tests cover
both enforcement points: the HF-bind block in resolve_hf_repo and the aggregate-metric drop in
_process_eval_file. Network-free: the existing-repo / override paths never call HfApi; the
search-path case monkeypatches it.
"""

from auto_benchmarkcard.tools.eee import eee_tool


# helpers
def test_is_aggregate_source_repo():
    assert eee_tool._is_aggregate_source_repo("human-centered-eval/OpenEval") is True
    assert eee_tool._is_aggregate_source_repo("  human-centered-eval/openeval ") is True
    assert eee_tool._is_aggregate_source_repo("openai/gsm8k") is False
    assert eee_tool._is_aggregate_source_repo(None) is False


def test_is_aggregate_metric():
    assert eee_tool._is_aggregate_metric("openeval.bold.logprob") is True
    assert eee_tool._is_aggregate_metric("openeval.gpqa.chain-of-thought-correctness") is True
    assert eee_tool._is_aggregate_metric("accuracy") is False
    assert eee_tool._is_aggregate_metric("llm_stats.gpqa.score") is False
    assert eee_tool._is_aggregate_metric(None) is False


def test_precision_metrics_not_flagged():
    # Real metric ids from the d8 precision set + apex-v1 must never be treated as aggregate.
    for m in ["superglue.score", "apex_v1.score", "bixbench.score", "arc-agi.score",
              "accuracy", "exact_match", "AssistantBench", "llm_stats.gpqa.score", "pass@1"]:
        assert eee_tool._is_aggregate_metric(m) is False


# HF bind block
def test_existing_openeval_repo_blocked():
    # bold/gpqa/boolq bind OpenEval via the EEE-provided existing_hf_repo path.
    assert eee_tool.resolve_hf_repo("bold", "human-centered-eval/OpenEval") is None
    assert eee_tool.resolve_hf_repo("gpqa", "human-centered-eval/OpenEval") is None
    assert eee_tool.resolve_hf_repo("boolq", "human-centered-eval/OpenEval") is None


def test_non_aggregate_existing_repo_passes_through():
    # Real first-party repos (incl. apex-v1's canonical "extended" derivative) are never blocked.
    for name, repo in [
        ("apex-v1", "mercor/APEX-v1-extended"),
        ("swe-bench", "princeton-nlp/SWE-bench_Verified"),
        ("superglue", "aps/super_glue"),
        ("gsm8k", "openai/gsm8k"),
    ]:
        assert eee_tool.resolve_hf_repo(name, repo) == repo


def test_search_path_skips_aggregate_repo(monkeypatch):
    # Even if HF search surfaces OpenEval as an exact match, it is never bound.
    class _DS:
        def __init__(self, id, downloads):
            self.id, self.downloads = id, downloads

    class _Api:
        def list_datasets(self, search=None, sort=None, limit=None):
            return [_DS("human-centered-eval/OpenEval", 999999)]

    monkeypatch.setattr(eee_tool, "HfApi", lambda: _Api())
    # benchmark name "OpenEval" would be an exact normalized-name match to the repo; the skip wins.
    assert eee_tool.resolve_hf_repo("OpenEval") is None


# aggregate-metric drop at EEE collection
def _eval_file(results, eval_library="OpenEval"):
    return {
        "model_info": {"name": "m1", "developer": "d1"},
        "eval_library": {"name": eval_library},
        "evaluation_results": [
            {
                "source_data": {"dataset_name": name, "hf_repo": hf,
                                "source_type": "hf_dataset", "url": []},
                "metric_config": {"metric_id": mid},
                "evaluation_name": mid,
                "score_details": {"score": sc},
            }
            for (name, mid, sc, hf) in results
        ],
    }


def test_aggregate_metric_dropped_at_collection():
    res = eee_tool.EEEScanResult()
    eee_tool._process_eval_file(
        _eval_file([("bold", "openeval.bold.logprob", 0.5, "human-centered-eval/OpenEval")]),
        "folder", res)
    bench = res.benchmarks["bold"]
    assert bench.metrics == {}             # openeval metric not collected
    assert bench.model_scores == []        # its score not collected either
    assert eee_tool.build_evaluation_summary(bench) == {}  # nothing leaks into eval_summary


def test_real_metric_kept_alongside_dropped_aggregate():
    # gpqa-style: the openeval metric is dropped but real metrics + scores survive.
    res = eee_tool.EEEScanResult()
    eee_tool._process_eval_file(
        _eval_file([
            ("gpqa", "openeval.gpqa.chain-of-thought-correctness", 0.4, "human-centered-eval/OpenEval"),
            ("gpqa", "accuracy", 0.7, "human-centered-eval/OpenEval"),
        ]),
        "folder", res)
    bench = res.benchmarks["gpqa"]
    assert "accuracy" in bench.metrics
    assert all(not eee_tool._is_aggregate_metric(m) for m in bench.metrics)
    assert [s["metric"] for s in bench.model_scores] == ["accuracy"]


def test_non_aggregate_benchmark_unchanged():
    # Precision: a normal benchmark keeps all metrics + scores (no behavior change).
    res = eee_tool.EEEScanResult()
    eee_tool._process_eval_file(
        _eval_file([("gsm8k", "accuracy", 0.9, "openai/gsm8k")]),
        "folder", res)
    bench = res.benchmarks["gsm8k"]
    assert "accuracy" in bench.metrics
    assert len(bench.model_scores) == 1
