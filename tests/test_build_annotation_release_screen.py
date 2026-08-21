"""Focused release tests for repaired screen-verifier-only packaging."""

import json
import hashlib
import os
import subprocess
import sys
import zipfile

import pytest


ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

from build_annotation_release import (  # noqa: E402
    validate_screen,
    validate_screen_instructions,
)
from build_screen_verification_worklist import derive_worklist, write_worklist  # noqa: E402


def finding(issue, category="thin"):
    return {
        "severity": "needs-fix",
        "category": category,
        "field": "data.size",
        "issue": issue,
        "ground_truth": f"truth-{issue}",
    }


def instruction_text(n_findings, n_cards, workload="10 to 13"):
    return (
        f"You will check {n_findings} possible problems across {n_cards} benchmark "
        f"cards. Expect about {workload} hours. Hard cases may take longer. Split "
        "the work across multiple sessions.\n"
        "Use confirmed-material, confirmed-trivial, not-a-defect, or unsure.\n"
        "Judge the displayed defect exactly, including its displayed category. "
        "Do not confirm it merely because you find a different problem. If you "
        "find a different problem, choose not-a-defect for this row and briefly "
        "explain it in notes.\n"
        "The card may still have a different problem.\n"
        "For confirmed-material, confirmed-trivial, or not-a-defect, provide one "
        "authoritative http(s) URL in evidence_url.\n"
        "For unsure, briefly explain why in notes. evidence_url is optional.\n"
        "If one displayed finding makes several claims about the same defect, a "
        "materially wrong claim is enough to confirm that defect. Do not use this "
        "rule to substitute a different defect or category.\n"
    )


def build_fixture(tmp_path):
    sample = {"cards": [{"name": "a"}, {"name": "b"}]}
    screen = {
        "missing": [],
        "problems": {},
        "per_card": {
            "a": {
                "card": "a",
                "verdict": "needs-fix",
                "findings": [finding("a")],
                "citations": ["https://example.test/a"],
            },
            "b": {
                "card": "b",
                "verdict": "minor",
                "findings": [finding("b", "wrong-paper")],
                "citations": ["https://example.test/b"],
            },
        },
    }
    sample_path = tmp_path / "sample.json"
    screen_path = tmp_path / "screen_results.json"
    worklist_path = tmp_path / "findings_worklist.csv"
    instructions_path = tmp_path / "INSTRUCTIONS.txt"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    rows, _ = derive_worklist(screen, sample, card_revision="fixture-revision")
    write_worklist(worklist_path, rows)
    instructions_path.write_text(instruction_text(2, 2), encoding="utf-8")
    return sample_path, screen_path, worklist_path, instructions_path


def test_screen_validation_is_source_derived_not_count_only(tmp_path):
    sample, screen, worklist, _ = build_fixture(tmp_path)
    rows, records = validate_screen(worklist, screen, sample, "fixture-revision")
    assert len(rows) == 2
    assert {record["raw_card_verdict"] for record in records} == {
        "needs-fix",
        "minor",
    }

    text = worklist.read_text(encoding="utf-8").replace("truth-a", "rewritten")
    worklist.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source-derived value"):
        validate_screen(worklist, screen, sample, "fixture-revision")


def test_screen_instructions_gate_current_counts_and_workload(tmp_path):
    path = tmp_path / "instructions.txt"
    path.write_text(instruction_text(154, 77), encoding="utf-8")
    validate_screen_instructions(path, 154, 77)
    path.write_text(path.read_text().replace("10 to 13", "9 to 11"), encoding="utf-8")
    with pytest.raises(ValueError, match="scope/workload"):
        validate_screen_instructions(path, 154, 77)


def test_screen_only_release_does_not_read_or_build_judge_packages(tmp_path):
    sample, screen, worklist, instructions = build_fixture(tmp_path)
    out = tmp_path / "release"
    command = [
        sys.executable,
        os.path.join(SCRIPTS, "build_annotation_release.py"),
        "--screen-only",
        "--screen-worklist",
        str(worklist),
        "--screen-results",
        str(screen),
        "--sample",
        str(sample),
        "--screen-instructions",
        str(instructions),
        "--card-revision",
        "fixture-revision",
        "--expected-screen-sha256",
        hashlib.sha256(screen.read_bytes()).hexdigest(),
        "--expected-sample-sha256",
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        "--out",
        str(out),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "judge packages untouched" in result.stdout
    assert {path.name for path in out.iterdir()} == {
        "staging",
        "screen_verifier_V1.zip",
        "SHA256SUMS.txt",
        "PACKAGE_AUDIT.txt",
        "HOLD_NOT_SENT.txt",
    }
    assert not list(out.glob("judge_rater_*.zip"))
    assert [path.name for path in (out / "staging").iterdir()] == [
        "screen_verifier_V1"
    ]
    with zipfile.ZipFile(out / "screen_verifier_V1.zip") as archive:
        assert archive.namelist() == ["INSTRUCTIONS.txt", "findings_worklist.csv"]
        assert not any("key.json" in name or name.startswith(".") for name in archive.namelist())
        assert archive.read("INSTRUCTIONS.txt") == instructions.read_bytes()
        assert archive.read("findings_worklist.csv") == worklist.read_bytes()
    audit = (out / "PACKAGE_AUDIT.txt").read_text(encoding="utf-8")
    assert "screen-only builder has no judge-package path" in audit
    assert "2 unique rows across 2 cards" in audit
    assert "response columns verifier_label, evidence_url, and notes" in audit
    assert "authoritative http(s) evidence URL" in audit
    assert "participant worklist SHA-256" in audit
    assert "participant instructions SHA-256" in audit
    assert "held 144-row V1" in audit
    assert "earlier repaired 154-row V1" in audit
    assert "superseded by the final evidence-capture revision" in audit
    assert "UNSENT" in (out / "HOLD_NOT_SENT.txt").read_text(encoding="utf-8")


def test_release_requires_explicit_screen_only_flag(tmp_path):
    sample, screen, worklist, instructions = build_fixture(tmp_path)
    out = tmp_path / "blocked-release"
    command = [
        sys.executable,
        os.path.join(SCRIPTS, "build_annotation_release.py"),
        "--screen-worklist",
        str(worklist),
        "--screen-results",
        str(screen),
        "--sample",
        str(sample),
        "--screen-instructions",
        str(instructions),
        "--card-revision",
        "fixture-revision",
        "--expected-screen-sha256",
        hashlib.sha256(screen.read_bytes()).hexdigest(),
        "--expected-sample-sha256",
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        "--out",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode != 0
    assert "only --screen-only mode is supported" in result.stderr
    assert not out.exists()
