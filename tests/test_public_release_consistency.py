import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_candidate_risk_summary_matches_frozen_judge_metric():
    release = json.loads((EVAL / "results_summary.json").read_text())
    judge = json.loads((EVAL / "s150/judge/summary.json").read_text())
    metric = judge["metrics"]["judge.risk_grounded_rate"]
    risk = release["candidate_risk_source_judge"]

    assert risk["sample_counts"] == {
        "total": 761,
        "relevant_and_grounded": 547,
        "not_relevant_or_not_grounded": 214,
    }
    assert risk["s_weighted_grounded_rate"]["value"] == metric["value"]
    assert risk["s_weighted_grounded_rate"]["ci95"] == metric["ci95"]
    assert risk["human_validated"] is False
    assert risk["headline_result"] is False


def test_source_complexity_result_names_public_inputs_and_portable_output():
    result = json.loads(
        (EVAL / "s150/source_complexity/analysis_results.json").read_text()
    )
    expected = {
        "exposures_outcome_free_csv": EVAL
        / "s150/source_complexity/exposures_outcome_free.csv",
        "sample_json": EVAL / "s150/sample.json",
        "judge_analysis_frame_json": EVAL / "s150/judge/analysis_frame.json",
        "verifier_ratings_csv": EVAL / "s150/screen/verifier_ratings.csv",
        "analysis_script": REPO / "scripts/analyze_source_complexity.py",
    }
    assert result["input_hashes"] == {
        name: _sha256(path) for name, path in expected.items()
    }
    assert result["joined_table"] == "analysis_joined_card_table.csv"


def test_eval_checksums_are_exhaustive_and_valid():
    checksum_path = EVAL / "SHA256SUMS.txt"
    recorded = {}
    for line in checksum_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest

    actual_paths = {
        path.relative_to(EVAL).as_posix(): path
        for path in EVAL.rglob("*")
        if path.is_file() and path != checksum_path
    }
    assert set(recorded) == set(actual_paths)
    assert recorded == {
        relative: _sha256(path) for relative, path in actual_paths.items()
    }
