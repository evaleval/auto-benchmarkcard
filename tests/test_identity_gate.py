"""Identity gate for aggregator-only EEE entries (eee_workflow.gate_benchmark_identity).

Fixture shapes are trimmed copies of real EEE tool_output records: nmos (llm-stats MOS
metric listed as a benchmark), researchpulse-bench (alphaxiv chart fragments about other
datasets glued under the name), healthbench (a real llm-stats benchmark). No network:
the LLM handler is always faked.
"""

import json
import os
from types import SimpleNamespace

import pytest

import auto_benchmarkcard.eee_workflow as eee_workflow
from auto_benchmarkcard.eee_workflow import (
    _identity_prescreen,
    _is_aggregator_only,
    gate_benchmark_identity,
    run_eee_pipeline,
)


def _inputs(eee_metadata, hf_repo=None, paper_url=None):
    return {
        "extracted_ids": {"hf_repo": hf_repo, "paper_url": paper_url, "risk_tags": None},
        "hf_repo": hf_repo,
        "eee_metadata": eee_metadata,
    }


def nmos_metadata():
    return {
        "benchmark_name": "NMOS",
        "eee_source_folder": "llm-stats",
        "source_type": "url",
        "source_urls": [
            "https://llm-stats.com/models/qwen2.5-omni-7b",
            "https://llm-stats.com/benchmarks/nmos",
            "https://api.llm-stats.com/leaderboard/benchmarks/nmos",
        ],
        "eval_library": "LLM Stats",
        "metrics": {
            "llm_stats.nmos.score": {
                "evaluation_description": "NMOS evaluation benchmark for assessing model performance on specialized tasks",
                "metric_id": "llm_stats.nmos.score",
                "metric_name": "NMOS score",
                "metric_kind": "benchmark_score",
                "metric_unit": "proportion",
                "additional_details": {
                    "raw_benchmark_id": "nmos",
                    "bound_strategy": "inferred_proportion",
                    "raw_verified": "false",
                    "raw_model_count": "1",
                },
            }
        },
        "evaluation_summary": {
            "total_models_evaluated": 1,
            "primary_metric": "llm_stats.nmos.score",
            "metric_config": {"min_possible": 0.0, "max_possible": 1.0},
            "score_statistics": {"mean": 0.0451, "std_dev": 0.0, "min": 0.0451, "max": 0.0451},
            "top_performers": [{"model": "Qwen2.5-Omni-7B", "developer": "qwen", "score": 0.0451}],
        },
        "num_models_evaluated": 1,
        "benchmark_type": "single",
        "appears_in": ["llm-stats"],
        "existing_hf_repo": None,
    }


def researchpulse_metadata():
    return {
        "benchmark_name": "ResearchPulse-Bench",
        "eee_source_folder": "alphaxiv",
        "source_type": "url",
        "source_urls": ["https://huggingface.co/datasets/ResearchPulse/ResearchPulse-Bench"],
        "eval_library": "alphaxiv",
        "metrics": {
            "3d_point_cloud_registration_on_3dmatch": {
                "evaluation_description": "This benchmark compares 3D point cloud registration methods on the 3DMatch dataset. Registration Recall (RR) measures the percentage of successfully aligned point cloud fragments.",
                "metric_id": "3d_point_cloud_registration_on_3dmatch",
                "metric_name": "3D Point Cloud Registration on 3DMatch",
                "metric_kind": "score",
                "additional_details": {
                    "alphaxiv_y_axis": "Registration Recall (%)",
                    "alphaxiv_is_primary": "False",
                    "raw_evaluation_name": "3D Point Cloud Registration on 3DMatch",
                },
            },
            "image_denoising_on_sidd_2018": {
                "evaluation_description": "Performance comparison of image denoising models on the SIDD (Smartphone Image Denoising Dataset). The metric is Peak Signal-to-Noise Ratio (PSNR).",
                "metric_id": "image_denoising_on_sidd_2018",
                "metric_name": "Image Denoising on SIDD (2018)",
                "metric_kind": "score",
                "additional_details": {
                    "alphaxiv_y_axis": "PSNR (dB)",
                    "alphaxiv_is_primary": "False",
                    "raw_evaluation_name": "Image Denoising on SIDD (2018)",
                },
            },
        },
        "evaluation_summary": {
            "total_models_evaluated": 2,
            "primary_metric": "3d_point_cloud_registration_on_3dmatch",
            "metric_config": {"min_possible": 0.0, "max_possible": 100.0},
            "score_statistics": {"mean": 78.4, "std_dev": 0.0, "min": 78.4, "max": 78.4},
            "top_performers": [{"model": "3DSN", "developer": "unknown", "score": 78.4}],
        },
        "num_models_evaluated": 2,
        "benchmark_type": "single",
        "appears_in": ["alphaxiv"],
        "existing_hf_repo": None,
    }


def healthbench_metadata():
    return {
        "benchmark_name": "HealthBench",
        "eee_source_folder": "llm-stats",
        "source_type": "url",
        "source_urls": [
            "https://llm-stats.com/benchmarks/healthbench",
            "https://api.llm-stats.com/leaderboard/benchmarks/healthbench",
        ],
        "eval_library": "LLM Stats",
        "metrics": {
            "llm_stats.healthbench.score": {
                "evaluation_description": "An open-source benchmark for measuring performance and safety of large language models in healthcare, consisting of 5,000 multi-turn conversations evaluated by 262 physicians",
                "metric_id": "llm_stats.healthbench.score",
                "metric_name": "HealthBench score",
                "metric_kind": "benchmark_score",
                "additional_details": {
                    "bound_strategy": "inferred_proportion",
                    "raw_verified": "false",
                    "raw_model_count": "4",
                },
            }
        },
        "evaluation_summary": {
            "total_models_evaluated": 4,
            "primary_metric": "llm_stats.healthbench.score",
            "metric_config": {"min_possible": 0.0, "max_possible": 1.0},
            "score_statistics": {"mean": 0.52, "std_dev": 0.1, "min": 0.4, "max": 0.67},
            "top_performers": [{"model": "Kimi K2 Thinking", "developer": "moonshotai", "score": 0.67}],
        },
        "num_models_evaluated": 4,
        "benchmark_type": "single",
        "appears_in": ["llm-stats"],
        "existing_hf_repo": None,
    }


def helm_metadata():
    return {
        "benchmark_name": "CivilComments",
        "eee_source_folder": "helm_classic",
        "source_type": "url",
        "source_urls": [
            "https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/releases/v0.4.0/groups/core_scenarios.json"
        ],
        "eval_library": "helm",
        "metrics": {
            "quasi_exact_match": {
                "evaluation_description": "Toxicity classification accuracy on CivilComments",
                "metric_id": "quasi_exact_match",
                "metric_name": "Quasi exact match",
            }
        },
        "evaluation_summary": {"total_models_evaluated": 67},
        "num_models_evaluated": 67,
        "benchmark_type": "single",
        "appears_in": [],
        "existing_hf_repo": None,
    }


class FakeLLM:
    def __init__(self, response=None, exc=None):
        self.calls = []
        self.response = response
        self.exc = exc

    def generate(self, prompt, response_format=None):
        self.calls.append(prompt)
        if self.exc is not None:
            raise self.exc
        return self.response


NOT_A_BENCHMARK = json.dumps(
    {"is_benchmark": False, "confidence": 0.95,
     "reason": "NMOS is a 1-5 speech naturalness rating scale, not a benchmark"}
)
IS_A_BENCHMARK = json.dumps(
    {"is_benchmark": True, "confidence": 0.9, "reason": "Real honesty benchmark"}
)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(response=None, exc=None):
        fake = FakeLLM(response=response, exc=exc)
        monkeypatch.setattr(eee_workflow, "get_llm_handler", lambda model=None: fake)
        return fake
    return install


class TestAggregatorOnlyDetection:
    def test_llm_stats_entry_is_aggregator_only(self):
        assert _is_aggregator_only(_inputs(nmos_metadata())) is True

    def test_alphaxiv_entry_with_search_resolved_repo_is_still_aggregator_only(self):
        # Search binding is name-derived, not evidence (the HF verifier later
        # rejected exactly this binding in the real run).
        inputs = _inputs(researchpulse_metadata(), hf_repo="ResearchPulse/ResearchPulse-Bench")
        assert _is_aggregator_only(inputs) is True

    def test_helm_entry_is_not_aggregator_only(self):
        assert _is_aggregator_only(_inputs(helm_metadata())) is False

    def test_eee_bound_hf_repo_is_independent_evidence(self):
        meta = nmos_metadata()
        meta["existing_hf_repo"] = "someorg/nmos-data"
        assert _is_aggregator_only(_inputs(meta, hf_repo="someorg/nmos-data")) is False

    def test_paper_url_is_independent_evidence(self):
        assert _is_aggregator_only(
            _inputs(nmos_metadata(), paper_url="https://arxiv.org/abs/2401.00001")
        ) is False

    def test_all_aggregator_urls_without_labels(self):
        meta = nmos_metadata()
        meta["eee_source_folder"] = ""
        meta["eval_library"] = ""
        assert _is_aggregator_only(_inputs(meta)) is True


class TestPrescreen:
    def test_thin_singleton_score_fires_on_nmos_shape(self):
        assert _identity_prescreen(nmos_metadata()) == ["thin_singleton_score"]

    def test_name_detached_metrics_fires_on_researchpulse_shape(self):
        # The descriptions contain the word "benchmark"; the generic token "bench"
        # from "ResearchPulse-Bench" must not attach through it.
        assert _identity_prescreen(researchpulse_metadata()) == ["name_detached_metrics"]

    def test_healthbench_shape_is_not_suspicious(self):
        assert _identity_prescreen(healthbench_metadata()) == []

    def test_singleton_with_many_models_is_not_suspicious(self):
        meta = nmos_metadata()
        meta["num_models_evaluated"] = 19
        assert _identity_prescreen(meta) == []

    def test_name_mentioned_in_metric_id_attaches(self):
        meta = researchpulse_metadata()
        meta["metrics"]["3d_point_cloud_registration_on_3dmatch"][
            "evaluation_description"
        ] = "Citation-aware tasks from ResearchPulse-Bench"
        assert "name_detached_metrics" not in _identity_prescreen(meta)


class TestGateVerdicts:
    def test_confident_false_skips_and_writes_sidecar(self, fake_llm, tmp_path):
        fake = fake_llm(response=NOT_A_BENCHMARK)
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "skip"
        assert result["reason"].startswith("not_a_benchmark: ")
        assert "naturalness rating scale" in result["reason"]
        assert len(fake.calls) == 1
        sidecar = tmp_path / "output" / "identity_gate" / "nmos.json"
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["action"] == "skip"
        assert payload["signals"] == ["thin_singleton_score"]
        assert payload["verdict"]["is_benchmark"] is False

    def test_true_verdict_passes(self, fake_llm, tmp_path):
        fake_llm(response=IS_A_BENCHMARK)
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_called"] is True
        payload = json.loads(
            (tmp_path / "output" / "identity_gate" / "nmos.json").read_text())
        assert payload["action"] == "pass"

    def test_low_confidence_false_passes(self, fake_llm, tmp_path):
        fake_llm(response=json.dumps(
            {"is_benchmark": False, "confidence": 0.5, "reason": "unsure"}))
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_failed"] is False

    def test_fenced_json_is_parsed(self, fake_llm, tmp_path):
        fake_llm(response="```json\n" + NOT_A_BENCHMARK + "\n```")
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "skip"

    def test_llm_exception_fails_open(self, fake_llm, tmp_path):
        fake_llm(exc=RuntimeError("engine down"))
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_failed"] is True
        payload = json.loads(
            (tmp_path / "output" / "identity_gate" / "nmos.json").read_text())
        assert payload["llm_failed"] is True

    def test_unparseable_response_retries_once_then_fails_open(self, fake_llm, tmp_path):
        fake = fake_llm(response="I think this is probably not a benchmark.")
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_failed"] is True
        assert len(fake.calls) == 2

    def test_non_boolean_verdict_fails_open(self, fake_llm, tmp_path):
        fake_llm(response=json.dumps(
            {"is_benchmark": "maybe", "confidence": 0.9, "reason": "odd"}))
        result = gate_benchmark_identity("NMOS", _inputs(nmos_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_failed"] is True


class TestLLMNeverReached:
    @pytest.fixture
    def exploding_llm(self, monkeypatch):
        def boom(model=None):
            raise AssertionError("gate must not consult the LLM for this entry")
        monkeypatch.setattr(eee_workflow, "get_llm_handler", boom)

    def test_entry_with_eee_bound_hf_repo(self, exploding_llm, tmp_path):
        meta = nmos_metadata()
        meta["existing_hf_repo"] = "someorg/nmos-data"
        result = gate_benchmark_identity(
            "NMOS", _inputs(meta, hf_repo="someorg/nmos-data"), str(tmp_path))
        assert result == {"action": "pass", "reason": None, "signals": [],
                          "llm_called": False, "llm_failed": False, "verdict": None}

    def test_entry_with_paper_url(self, exploding_llm, tmp_path):
        result = gate_benchmark_identity(
            "NMOS", _inputs(nmos_metadata(), paper_url="https://arxiv.org/abs/2401.00001"),
            str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_called"] is False

    def test_non_aggregator_entry(self, exploding_llm, tmp_path):
        result = gate_benchmark_identity(
            "CivilComments", _inputs(helm_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_called"] is False

    def test_unsuspicious_aggregator_entry(self, exploding_llm, tmp_path):
        result = gate_benchmark_identity(
            "HealthBench", _inputs(healthbench_metadata()), str(tmp_path))
        assert result["action"] == "pass"
        assert result["llm_called"] is False
        assert not (tmp_path / "output" / "identity_gate").exists()


class TestBatchSkipRecording:
    def test_run_eee_pipeline_records_skip_and_telemetry(
        self, fake_llm, monkeypatch, tmp_path
    ):
        fake_llm(response=NOT_A_BENCHMARK)

        benches = {
            "nmos": SimpleNamespace(num_models_evaluated=1),
            "healthbench": SimpleNamespace(num_models_evaluated=4),
        }
        inputs_by_name = {
            "nmos": _inputs(nmos_metadata()),
            "healthbench": _inputs(healthbench_metadata()),
        }
        scan = SimpleNamespace(
            errors=[], benchmarks=benches, composites={}, total_eval_files=2)
        monkeypatch.setattr(eee_workflow, "scan_eee_folder", lambda *a, **k: scan)
        monkeypatch.setattr(
            eee_workflow, "eee_to_pipeline_inputs",
            lambda bench, benchmark_type="single", appears_in=None:
                inputs_by_name["nmos" if bench is benches["nmos"] else "healthbench"],
        )

        processed = []

        def fake_process(benchmark_name, pipeline_inputs, base_output_path=None, debug=False):
            processed.append(benchmark_name)
            return {"benchmark_card": {}}

        monkeypatch.setattr(eee_workflow, "process_single_benchmark", fake_process)

        summary = run_eee_pipeline("unused-eee-path", output_path=str(tmp_path))

        assert processed == ["healthbench"]
        assert summary["successful"] == ["healthbench"]
        assert summary["failed"] == []
        assert len(summary["skipped"]) == 1
        skip = summary["skipped"][0]
        assert skip["benchmark"] == "nmos"
        assert skip["reason"].startswith("not_a_benchmark: ")
        assert summary["identity_gate"] == {
            "checked": 2, "gated": 1, "passed": 1, "llm_failed": 0}
        assert (tmp_path / "output" / "identity_gate" / "nmos.json").exists()
