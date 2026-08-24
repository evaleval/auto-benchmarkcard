import copy
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import calibrate_thresholds as calib  # noqa: E402

OVERVIEW = "benchmark_details.overview"
NAME = "benchmark_details.name"


def _verdict(path, status):
    return {"path": path, "status": status}


def _fixture():
    """Two flagged cards (weight 2.0) and three unflagged (weight 3.0).

    Overview: filled on f1/f2/u1/u2, not_specified on u3. Findings: one on
    (f1, overview), one on (u3, overview) which is unfilled, one on a
    non-judged field. Hand-checked overview rates: material 2/10 = 0.2,
    unsupported+partial (2+3)/10 = 0.5.
    """
    judge = {
        "per_card": {
            "f1": {
                "stratum": "flagged",
                "field_verdicts": [
                    _verdict(OVERVIEW, "supported"),
                    _verdict(NAME, "supported"),
                ],
            },
            "f2": {
                "stratum": "flagged",
                "field_verdicts": [_verdict(OVERVIEW, "unsupported")],
            },
            "u1": {
                "stratum": "unflagged",
                "field_verdicts": [_verdict(OVERVIEW, "supported")],
            },
            "u2": {
                "stratum": "unflagged",
                "field_verdicts": [_verdict(OVERVIEW, "partial")],
            },
            "u3": {
                "stratum": "unflagged",
                "field_verdicts": [_verdict(OVERVIEW, "not_specified")],
            },
        }
    }
    sample = {
        "strata": {
            "flagged": {"N": 4, "n": 2, "weight": 2.0},
            "unflagged": {"N": 9, "n": 3, "weight": 3.0},
        },
        "cards": [
            {"name": "f1", "stratum": "flagged", "weight": 2.0},
            {"name": "f2", "stratum": "flagged", "weight": 2.0},
            {"name": "u1", "stratum": "unflagged", "weight": 3.0},
            {"name": "u2", "stratum": "unflagged", "weight": 3.0},
            {"name": "u3", "stratum": "unflagged", "weight": 3.0},
        ],
    }
    findings = {
        "findings": [
            {"card": "f1", "field_path": OVERVIEW},
            {"card": "u3", "field_path": OVERVIEW},
            {"card": "f2", "field_path": "methodology.metrics"},
        ]
    }
    return judge, sample, findings


def test_weighted_rates_hand_checked():
    judge, sample, findings = _fixture()
    cards, diag = calib.build_frame(judge, sample, findings)
    points = calib.weighted_rates(cards)

    overview = points[OVERVIEW]
    assert overview["material_rate"] == pytest.approx(2.0 / 10.0)
    assert overview["unsupported_partial_rate"] == pytest.approx(5.0 / 10.0)
    assert overview["n_filled"] == 4
    assert overview["n_filled_weighted"] == pytest.approx(10.0)
    assert overview["n_material_pairs"] == 1

    name = points[NAME]
    assert name["material_rate"] == pytest.approx(0.0)
    assert name["n_filled"] == 1

    never_filled = points["methodology.judge_score_consolidation"]
    assert never_filled["material_rate"] is None
    assert never_filled["n_filled"] == 0

    assert diag["n_findings_total"] == 3
    assert diag["n_findings_joined"] == 1
    assert diag["n_findings_nonjudged_field"] == 1
    assert diag["n_findings_on_unfilled_pair"] == 1
    assert diag["findings_on_unfilled_pairs"] == [{"card": "u3", "field": OVERVIEW}]


def test_exceeds_alpha_flags():
    judge, sample, findings = _fixture()
    result = calib.calibrate(judge, sample, findings, replicates=50, now="test")
    overview = result["per_field"][OVERVIEW]
    # rate 0.2 exceeds 0.05 and 0.10 but not the 0.20 boundary itself
    assert overview["exceeds_alpha"] == {"0.05": True, "0.10": True, "0.20": False}
    empty = result["per_field"]["methodology.judge_score_consolidation"]
    assert empty["exceeds_alpha"] == {"0.05": False, "0.10": False, "0.20": False}


def test_bootstrap_deterministic_under_fixed_seed():
    judge, sample, findings = _fixture()
    cards, _ = calib.build_frame(judge, sample, findings)
    a = calib.bootstrap_cis(cards, replicates=200, seed=7)
    b = calib.bootstrap_cis(cards, replicates=200, seed=7)
    assert a == b
    lo, hi = a[OVERVIEW]["material_ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_collapses_on_homogeneous_strata():
    # Every card within a stratum identical: each replicate reproduces the
    # point estimate exactly, so the CI collapses onto it.
    judge, sample, findings = _fixture()
    judge = copy.deepcopy(judge)
    judge["per_card"]["f2"]["field_verdicts"] = [_verdict(OVERVIEW, "supported")]
    for card in ("u1", "u2", "u3"):
        judge["per_card"][card]["field_verdicts"] = [_verdict(OVERVIEW, "supported")]
    findings = {
        "findings": [
            {"card": "f1", "field_path": OVERVIEW},
            {"card": "f2", "field_path": OVERVIEW},
        ]
    }
    cards, _ = calib.build_frame(judge, sample, findings)
    point = calib.weighted_rates(cards)[OVERVIEW]["material_rate"]
    # flagged material weight 4 over total filled weight 13
    assert point == pytest.approx(4.0 / 13.0)
    cis = calib.bootstrap_cis(cards, replicates=100, seed=3)
    lo, hi = cis[OVERVIEW]["material_ci95"]
    assert lo == pytest.approx(point)
    assert hi == pytest.approx(point)


def test_join_ignores_nonjudged_field():
    judge, sample, findings = _fixture()
    without = {"findings": findings["findings"][:2]}
    r_with = calib.calibrate(judge, sample, findings, replicates=50, now="t")
    r_without = calib.calibrate(judge, sample, without, replicates=50, now="t")
    assert r_with["per_field"] == r_without["per_field"]
    assert "methodology.metrics" not in r_with["per_field"]
    assert r_with["join_diagnostics"]["nonjudged_field_paths"] == {
        "methodology.metrics": 1
    }


def test_normalize_field_path():
    assert calib.normalize_field_path("data.size_breakdown.train") == (
        "data.size_breakdown"
    )
    assert calib.normalize_field_path("benchmark_details.resources[0]") == (
        "benchmark_details.resources"
    )
    assert calib.normalize_field_path(
        'data.size_breakdown["# 3rd Party Function Call"]'
    ) == "data.size_breakdown"
    assert calib.normalize_field_path("hf_repo") == "hf_repo"


def test_duplicate_verdict_collapses_worst_status_wins():
    judge, sample, findings = _fixture()
    judge = copy.deepcopy(judge)
    judge["per_card"]["f1"]["field_verdicts"].append(_verdict(NAME, "partial"))
    cards, diag = calib.build_frame(judge, sample, {"findings": []})
    f1 = next(c for c in cards if c["name"] == "f1")
    assert f1["risky"][NAME] is True
    assert diag["duplicate_verdicts"] == [
        {"card": "f1", "field": NAME, "statuses": ["supported", "partial"]}
    ]


def test_missing_file_raises_clean_error(tmp_path):
    with pytest.raises(SystemExit, match="missing findings file"):
        calib.main(["--findings", str(tmp_path / "nope.json")])


def test_card_set_mismatch_raises():
    judge, sample, findings = _fixture()
    sample = copy.deepcopy(sample)
    sample["cards"] = sample["cards"][:-1]
    with pytest.raises(SystemExit, match="card sets differ"):
        calib.build_frame(judge, sample, findings)


def test_main_end_to_end_writes_outputs(tmp_path):
    judge, sample, findings = _fixture()
    paths = {}
    for label, payload in (
        ("judge", judge),
        ("sample", sample),
        ("findings", findings),
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = str(path)
    out_dir = tmp_path / "calibration"
    rc = calib.main(
        [
            "--judge", paths["judge"],
            "--sample", paths["sample"],
            "--findings", paths["findings"],
            "--out-dir", str(out_dir),
            "--seed", "11",
            "--replicates", "50",
            "--now", "2026-08-24T00:00:00+00:00",
        ]
    )
    assert rc == 0
    result = json.loads((out_dir / "thresholds.json").read_text(encoding="utf-8"))
    assert result["generated_at"] == "2026-08-24T00:00:00+00:00"
    assert result["design"]["seed"] == 11
    assert result["design"]["replicates"] == 50
    assert result["per_field"][OVERVIEW]["material_rate"] == pytest.approx(0.2)
    report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Caveats" in report
    assert "FROZEN pipeline" in report
    assert OVERVIEW in report
