"""Tests for the source-derived V1 finding worklist."""

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_screen_verification_worklist import (  # noqa: E402
    DEFAULT_CARD_REVISION,
    WORKLIST_FIELDS,
    derive_worklist,
    validate_worklist_against_source,
    write_worklist,
)


def finding(severity, category, issue):
    return {
        "severity": severity,
        "category": category,
        "field": f"field.{issue}",
        "issue": issue,
        "ground_truth": f"truth-{issue}",
    }


def fixture_data():
    sample = {
        "cards": [
            {"name": "needs-card"},
            {"name": "minor-card"},
            {"name": "plain-card"},
        ]
    }
    screen = {
        "missing": [],
        "problems": {},
        "per_card": {
            "needs-card": {
                "card": "needs-card",
                "verdict": "needs-fix",
                "findings": [
                    finding("minor", "wrong-identity", "triage-only"),
                    finding("needs-fix", "thin", "old-included"),
                ],
                "citations": ["https://example.test/needs"],
            },
            "minor-card": {
                "card": "minor-card",
                "verdict": "minor",
                "findings": [
                    finding("minor", "thin", "not-selected"),
                    finding("needs-fix", "wrong-paper", "previously-omitted"),
                ],
                "citations": ["https://example.test/minor"],
            },
            "plain-card": {
                "card": "plain-card",
                "verdict": "minor",
                "findings": [finding("minor", "other", "not-selected")],
                "citations": ["https://example.test/plain"],
            },
        },
    }
    return screen, sample


def write_sources(tmp_path, screen, sample):
    screen_path = tmp_path / "screen_results.json"
    sample_path = tmp_path / "sample.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    return screen_path, sample_path


def test_finding_level_selection_includes_severe_finding_on_minor_card():
    screen, sample = fixture_data()
    rows, records = derive_worklist(screen, sample)

    assert [(row["card"], row["finding_index"]) for row in rows] == [
        ("needs-card", "1"),
        ("minor-card", "1"),
    ]
    assert [row["row_id"] for row in rows] == ["1", "2"]
    assert WORKLIST_FIELDS[-3:] == ("verifier_label", "evidence_url", "notes")
    assert all(
        row["verifier_label"] == row["evidence_url"] == row["notes"] == ""
        for row in rows
    )
    assert rows[0]["card_reference"].endswith(
        f"/{DEFAULT_CARD_REVISION}/cards/needs-card.json"
    )
    assert [record["card"] for record in records] == ["needs-card", "minor-card"]
    assert records[1]["raw_card_verdict"] == "minor"
    assert records[1]["finding_indices"] == [1]
    assert records[1]["citations"] == ["https://example.test/minor"]


def test_source_validator_requires_exact_rows_order_and_raw_text(tmp_path):
    screen, sample = fixture_data()
    rows, _ = derive_worklist(screen, sample)
    worklist = tmp_path / "worklist.csv"
    screen_path, sample_path = write_sources(tmp_path, screen, sample)
    write_worklist(worklist, rows)

    validated, _ = validate_worklist_against_source(
        worklist, screen_path, sample_path
    )
    assert validated == rows

    reversed_rows = list(reversed(rows))
    for index, row in enumerate(reversed_rows, 1):
        row = dict(row)
        row["row_id"] = str(index)
        reversed_rows[index - 1] = row
    write_worklist(worklist, reversed_rows, overwrite=True)
    with pytest.raises(ValueError, match="differs from the frozen source-derived value"):
        validate_worklist_against_source(worklist, screen_path, sample_path)

    tampered = [dict(row) for row in rows]
    tampered[0]["issue"] = "rewritten to be easier"
    write_worklist(worklist, tampered, overwrite=True)
    with pytest.raises(ValueError, match="field 'issue' differs"):
        validate_worklist_against_source(worklist, screen_path, sample_path)


def test_source_validator_reports_missing_and_extra_finding_identities(tmp_path):
    screen, sample = fixture_data()
    rows, _ = derive_worklist(screen, sample)
    screen_path, sample_path = write_sources(tmp_path, screen, sample)
    worklist = tmp_path / "worklist.csv"
    write_worklist(worklist, rows[:-1])
    with pytest.raises(ValueError, match="finding mismatch: missing="):
        validate_worklist_against_source(worklist, screen_path, sample_path)

    duplicate = [dict(row) for row in rows]
    duplicate.append(dict(rows[0], row_id="3"))
    write_worklist(worklist, duplicate, overwrite=True)
    with pytest.raises(ValueError, match="finding identities are not unique"):
        validate_worklist_against_source(worklist, screen_path, sample_path)


def test_source_validator_rejects_prefilled_responses_and_column_changes(tmp_path):
    screen, sample = fixture_data()
    rows, _ = derive_worklist(screen, sample)
    screen_path, sample_path = write_sources(tmp_path, screen, sample)
    worklist = tmp_path / "worklist.csv"

    prefilled = [dict(row) for row in rows]
    prefilled[0]["verifier_label"] = "confirmed-material"
    write_worklist(worklist, prefilled)
    with pytest.raises(ValueError, match="prefilled response"):
        validate_worklist_against_source(worklist, screen_path, sample_path)

    evidence = [dict(row) for row in rows]
    evidence[0]["evidence_url"] = "https://example.test/evidence"
    write_worklist(worklist, evidence, overwrite=True)
    with pytest.raises(ValueError, match="prefilled response"):
        validate_worklist_against_source(worklist, screen_path, sample_path)

    whitespace = [dict(row) for row in rows]
    whitespace[0]["notes"] = " "
    write_worklist(worklist, whitespace, overwrite=True)
    with pytest.raises(ValueError, match="prefilled response"):
        validate_worklist_against_source(worklist, screen_path, sample_path)

    with worklist.open("w", newline="", encoding="utf-8") as handle:
        fields = list(WORKLIST_FIELDS) + ["hidden_key"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(row, hidden_key="secret") for row in rows)
    with pytest.raises(ValueError, match="columns differ"):
        validate_worklist_against_source(worklist, screen_path, sample_path)


def test_derive_rejects_incomplete_screen_and_blank_candidate_citations():
    screen, sample = fixture_data()
    screen["missing"] = ["plain-card"]
    with pytest.raises(ValueError, match="missing cards"):
        derive_worklist(screen, sample)

    screen, sample = fixture_data()
    screen["per_card"]["minor-card"]["citations"] = []
    with pytest.raises(ValueError, match="candidate card must have"):
        derive_worklist(screen, sample)


def test_frozen_s150_regression_is_exactly_154_findings_on_77_cards():
    repo = Path(__file__).resolve().parents[1]
    screen = json.loads(
        (repo / "eval/s150/screen/screen_results.json").read_text(encoding="utf-8")
    )
    lock = json.loads(
        (repo / "eval/s150/screen/scoring_lock.json").read_text(encoding="utf-8")
    )
    rows = lock["packet"]["rows"]

    assert len(rows) == 154
    assert len({row["card"] for row in rows}) == 77
    assert Counter(
        screen["per_card"][card]["verdict"]
        for card in {row["card"] for row in rows}
    ) == {
        "needs-fix": 69,
        "minor": 8,
    }
    expected_added = {
        ("chartqa", "0"),
        ("cyse2-vulnerability-exploit", "2"),
        ("drop", "0"),
        ("functionalmath", "0"),
        ("humanevalfim-average", "7"),
        ("lmarena-text-leaderboard", "0"),
        ("pinchbench", "0"),
        ("pinchbench", "1"),
        ("swe-bench", "0"),
        ("swe-bench", "1"),
    }
    minor_card_findings = {
        (row["card"], row["finding_index"])
        for row in rows
        if screen["per_card"][row["card"]]["verdict"] == "minor"
    }
    assert minor_card_findings == expected_added
    assert [
        (index + 1, row["card"], row["finding_index"])
        for index, row in enumerate(rows)
        if (row["card"], row["finding_index"]) in expected_added
    ] == [
        (13, "humanevalfim-average", "7"),
        (79, "lmarena-text-leaderboard", "0"),
        (84, "pinchbench", "0"),
        (85, "pinchbench", "1"),
        (103, "chartqa", "0"),
        (117, "drop", "0"),
        (139, "swe-bench", "0"),
        (140, "swe-bench", "1"),
        (142, "functionalmath", "0"),
        (154, "cyse2-vulnerability-exploit", "2"),
    ]
