import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_judge_analysis_view.py"
SPEC = importlib.util.spec_from_file_location("build_judge_analysis_view", SCRIPT)
J = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(J)


def _card():
    card = {"possible_risks": [
        {"category": "Data bias"},
        {"category": "Reproducibility"},
    ]}
    for path in J.ALLOWLISTED_PATHS:
        section, field = path.split(".", 1)
        card.setdefault(section, {})[field] = f"value for {path}"
    return {"benchmark_card": card}


def _field(path, status="supported"):
    if status == "not_specified":
        return {
            "path": path,
            "status": status,
            "specificity": "na",
            "info_in_source": "no",
            "note": "",
        }
    return {
        "path": path,
        "status": status,
        "specificity": "specific",
        "info_in_source": "na",
        "note": "",
    }


def _risk(category):
    return {"category": category, "relevant_and_grounded": "yes", "note": ""}


def _run_meta(model="claude-sonnet-4-6"):
    return {
        "model": model,
        "prompt_version": "v2-eee",
        "transport": "api",
        "execution": "anthropic-api-sync",
        "request_params": {"model": model, "temperature": 0.0},
        "batches": [{"started": "2026-07-15T00:46:25+00:00"}],
    }


def _chartqa_fixture(tmp_path):
    card_path = tmp_path / "chartqa.json"
    card_path.write_text(json.dumps(_card()))
    fields = [_field(path) for path in J.ALLOWLISTED_PATHS
              if path != "methodology.methods"]
    fields.extend([
        _field("methodology.methods", "supported"),
        _field("benchmark_details.benchmark_type"),
        _field("benchmark_details.contains"),
        _field("benchmark_details.appears_in"),
    ])
    raw = {"chartqa": {
        "name": "chartqa",
        "field_verdicts": fields,
        "risk_verdicts": [_risk("Data bias"), _risk("Reproducibility")],
    }}
    manifest = {"cards": [{
        "name": "chartqa", "stratum": "flagged", "card_path": str(card_path),
    }]}
    return raw, manifest


def test_positive_frame_is_exact_and_excludes_structural_fields():
    assert len(J.ALLOWLISTED_PATHS) == len(set(J.ALLOWLISTED_PATHS)) == 23
    assert set(J.ALLOWLISTED_PATHS).isdisjoint(J.KNOWN_NONCONTENT_PATHS)
    for path in (
        "benchmark_details.benchmark_type",
        "benchmark_details.contains",
        "benchmark_details.appears_in",
        "benchmark_details.authors",
        "data.size",
        "methodology.metrics",
    ):
        assert path not in J.ALLOWLISTED_PATHS


def test_build_filters_noncontent_without_mutating_raw(tmp_path):
    raw, manifest = _chartqa_fixture(tmp_path)
    before = copy.deepcopy(raw)
    view = J.build_analysis_view(
        raw, manifest, _run_meta(), repo=tmp_path, label="test"
    )

    assert raw == before
    assert view["analysis_frame"] == {
        "allowlisted_paths": list(J.ALLOWLISTED_PATHS),
        "declared_exclusions": [],
        "n_rows": 23,
        "n_cards": 1,
    }
    paths = [field["path"] for field in view["per_card"]["chartqa"]["field_verdicts"]]
    assert len(paths) == len(set(paths)) == 23
    assert "methodology.methods" in paths
    assert not set(paths) & J.KNOWN_NONCONTENT_PATHS
    assert view["source_audit"]["n_excluded_raw_rows"] == 3


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "risk"])
def test_frame_validation_fails_closed(tmp_path, defect):
    raw, manifest = _chartqa_fixture(tmp_path)
    fields = raw["chartqa"]["field_verdicts"]
    if defect == "missing":
        fields[:] = [field for field in fields
                     if field["path"] != "benchmark_details.overview"]
    elif defect == "duplicate":
        fields.append(_field("benchmark_details.overview"))
    elif defect == "unknown":
        fields.append(_field("benchmark_details.future_structured_field"))
    elif defect == "risk":
        raw["chartqa"]["risk_verdicts"].pop()

    with pytest.raises(J.FrameValidationError):
        J.build_analysis_view(raw, manifest, _run_meta(), repo=tmp_path)


def test_authoritative_run_metadata_rejects_stale_model(tmp_path):
    raw, manifest = _chartqa_fixture(tmp_path)
    stale = {"label": "s150", "judge_info": {"model": "claude-fable-5"}}

    with pytest.raises(J.FrameValidationError, match="conflicts"):
        J.build_analysis_view(
            raw, manifest, _run_meta(), repo=tmp_path, metadata=stale
        )


def test_mixed_run_dates_report_latest_and_all_dates_without_changing_single_date_shape():
    single = _run_meta()
    single["batches"].append({"started": "2026-07-15T23:59:59Z"})
    assert J._judge_info_from_run_meta(single) == {
        "model": "claude-sonnet-4-6",
        "prompt_version": "v2-eee",
        "transport": "api",
        "execution": "anthropic-api-sync",
        "temperature": 0.0,
        "judged_on": "2026-07-15",
    }

    mixed = _run_meta()
    mixed["batches"] = [
        {"started": "2026-07-17T09:00:00+00:00"},
        {"started": "2026-07-15T23:59:59Z"},
        {"started": "2026-07-16T00:30:00+00:00"},
        {"started": "2026-07-17T10:00:00+00:00"},
    ]
    assert J._judge_info_from_run_meta(mixed) == {
        "model": "claude-sonnet-4-6",
        "prompt_version": "v2-eee",
        "transport": "api",
        "execution": "anthropic-api-sync",
        "temperature": 0.0,
        "judged_on": "2026-07-17",
        "judged_on_dates": ["2026-07-15", "2026-07-16", "2026-07-17"],
    }


def test_public_s150_artifacts_have_the_expected_clean_frame():
    judge_dir = REPO / "eval/s150/judge"
    raw = json.loads((judge_dir / "results.json").read_text())
    verdict_manifest = json.loads((judge_dir / "verdict_manifest.json").read_text())
    view = json.loads((judge_dir / "analysis_frame.json").read_text())

    assert len(raw) == 150
    assert len(verdict_manifest["cards"]) == 150
    assert view["analysis_frame"]["n_cards"] == 150
    assert view["analysis_frame"]["n_rows"] == 3450
    assert view["judge_info"] == {
        "model": "claude-sonnet-4-6",
        "prompt_version": "v2-eee",
        "transport": "api",
        "execution": "anthropic-api-sync",
        "temperature": 0.0,
        "judged_on": "2026-07-17",
        "judged_on_dates": ["2026-07-15", "2026-07-17"],
    }
    assert sum(len(card["field_verdicts"]) for card in view["per_card"].values()) == 3450
    assert len(view["per_card"]["chartqa"]["field_verdicts"]) == 23
    assert all(
        len(card["field_verdicts"]) == 23
        for card in view["per_card"].values()
    )
