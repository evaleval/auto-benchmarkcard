"""run_composer persists composition_metadata (quote-verify counters + validator telemetry)
as a composer sidecar. Additive only: card content and the provenance sidecar are untouched.
Before this, the entire composition_metadata payload was produced and then dropped at
persistence time.
"""

import json
import os

from auto_benchmarkcard import workers as W
from auto_benchmarkcard.output import OutputManager, sanitize_benchmark_name


def _fake_result():
    return {
        "benchmark_card": {"benchmark_details": {"name": "demo", "domains": [], "languages": []}},
        "provenance": {"data": {"size": {"source": "deterministic"}}},
        "composition_metadata": {
            "model_used": "test-model",
            "quote_verify": {"emitted": 3, "verified": 2, "rejected": 1, "truncated": 0,
                             "reject_reasons": {"no_match": 1, "empty": 0, "bad_field": 0}},
            "validation": {"card": {"other_rate": 0.0, "schema_invalid": 0}},
        },
    }


def _run(tmp_path, monkeypatch, result):
    om = OutputManager("demo-bench", base_path=str(tmp_path))

    class _Tool:
        @staticmethod
        def func(**kwargs):
            return result

    monkeypatch.setattr(W, "compose_benchmark_card", _Tool)
    out = W.run_composer({"query": "demo-bench", "output_manager": om})
    return om, out


def test_composition_metadata_sidecar_written(tmp_path, monkeypatch):
    om, out = _run(tmp_path, monkeypatch, _fake_result())
    safe = sanitize_benchmark_name("demo-bench")
    composer_dir = om.get_tool_output_path("composer")
    with open(os.path.join(composer_dir, f"composition_metadata_{safe}.json")) as f:
        meta = json.load(f)
    assert meta["quote_verify"]["verified"] == 2
    assert meta["validation"]["card"]["schema_invalid"] == 0
    # provenance sidecar still written, composed card unchanged in the state result
    assert os.path.exists(os.path.join(composer_dir, f"provenance_{safe}.json"))
    assert out["composed_card"]["benchmark_card"]["benchmark_details"]["name"] == "demo"


def test_missing_metadata_writes_no_sidecar(tmp_path, monkeypatch):
    result = _fake_result()
    del result["composition_metadata"]
    om, out = _run(tmp_path, monkeypatch, result)
    safe = sanitize_benchmark_name("demo-bench")
    path = os.path.join(om.get_tool_output_path("composer"), f"composition_metadata_{safe}.json")
    assert not os.path.exists(path)
    assert out["composed_card"]["benchmark_card"]["benchmark_details"]["name"] == "demo"
