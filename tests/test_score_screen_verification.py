"""Focused tests for the locked V1 return validator and scorer."""

import csv
import json
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_screen_verification_worklist import (  # noqa: E402
    PROTECTED_FIELDS,
    derive_worklist,
)
from score_screen_verification import (  # noqa: E402
    AUTHOR_RESPONSE_FIELDS,
    CONTAMINATION_CARDS,
    CONTAMINATION_WORKLIST_FIELDS,
    LABELS,
    PARTICIPANT_WORKLIST_FIELDS,
    author_agreement_block,
    card_detected_material_block,
    finding_score_block,
    prepare_scoring_lock,
    score_screen_verification,
    select_author_overlap_cards,
    validate_author_return,
    validate_contamination_return,
    validate_verifier_return,
    validation_receipt,
)


def _finding(index):
    return {
        "severity": "needs-fix",
        "category": "thin",
        "field": f"field.{index}",
        "issue": f"issue {index}",
        "ground_truth": f"truth {index}",
    }


def _full_fixture(tmp_path):
    flagged = ["sunrgbd"] + [f"flagged-{index:02d}" for index in range(74)]
    unflagged = ["mc-bench", "chartqa"] + [
        f"unflagged-{index:02d}" for index in range(73)
    ]
    sample = {
        "n": 150,
        "strata": {
            "flagged": {"N": 152, "n": 75, "weight": 2.026667},
            "unflagged": {"N": 378, "n": 75, "weight": 5.04},
        },
        "cards": [
            {"name": card, "stratum": "flagged", "weight": 2.026667}
            for card in flagged
        ] + [
            {"name": card, "stratum": "unflagged", "weight": 5.04}
            for card in unflagged
        ],
    }
    candidate_cards = set(flagged[:35] + unflagged[:42])
    per_card = {}
    for card in flagged + unflagged:
        is_contamination = card in CONTAMINATION_CARDS
        per_card[card] = {
            "card": card,
            "verdict": "needs-fix" if card in candidate_cards else "minor",
            "hf_repo_assessment": (
                "wrong-kept-CONTAMINATION" if is_contamination else "correct-kept"
            ),
            "findings": (
                [_finding(0), _finding(1)] if card in candidate_cards else []
            ),
            "citations": [f"https://example.test/{card}"],
            "summary": f"summary for {card}",
        }
    screen = {"missing": [], "problems": {}, "per_card": per_card}
    rows, _ = derive_worklist(screen, sample)
    assert len(rows) == 154
    assert len({row["card"] for row in rows}) == 77

    screen_path = tmp_path / "screen_results.json"
    sample_path = tmp_path / "sample.json"
    worklist_path = tmp_path / "findings_worklist.csv"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    with worklist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(dict(row, evidence_url="") for row in rows)
    lock, author_rows, contamination_rows = prepare_scoring_lock(
        worklist_path=worklist_path,
        screen_results_path=screen_path,
        sample_path=sample_path,
    )
    return lock, author_rows, contamination_rows


def _write_verifier_return(path, lock, labels=None):
    labels = labels or ["not-a-defect"] * len(lock["packet"]["rows"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        for row, label in zip(lock["packet"]["rows"], labels):
            writer.writerow(dict(
                row,
                verifier_label=label,
                evidence_url=(
                    "" if label == "unsure"
                    else f"https://evidence.test/v1/{row['row_id']}"
                ),
                notes="needs more evidence" if label == "unsure" else "",
            ))


def _write_author_return(path, lock, labels=None):
    row_by_id = {row["row_id"]: row for row in lock["packet"]["rows"]}
    rows = [row_by_id[item_id] for item_id in lock["author_overlap"]["row_ids"]]
    labels = labels or ["not-a-defect"] * len(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(PROTECTED_FIELDS) + AUTHOR_RESPONSE_FIELDS
        )
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(dict(
                row,
                author_label=label,
                author_evidence_url=(
                    "" if label == "unsure"
                    else f"https://evidence.test/author/{row['row_id']}"
                ),
                author_notes="needs more evidence" if label == "unsure" else "",
            ))


def _write_contamination_return(path, lock, labels=None):
    rows = lock["contamination_check"]["rows"]
    labels = labels or ["confirmed-contamination"] * len(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTAMINATION_WORKLIST_FIELDS)
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow(
                dict(
                    row,
                    author_assessment=label,
                    author_evidence_url=(
                        "" if label == "unsure"
                        else f"https://evidence.test/contamination/{row['card']}"
                    ),
                    author_notes=(
                        "needs more evidence" if label == "unsure"
                        else "documented check"
                    ),
                )
            )


def test_prepare_freezes_154_rows_and_preselects_separate_checks(tmp_path):
    lock, author_rows, contamination_rows = _full_fixture(tmp_path)

    assert lock["packet"]["n_findings"] == 154
    assert lock["packet"]["n_cards"] == 77
    assert lock["sample_design"]["n_cards"] == 150
    assert lock["sample_design"]["population_cards"] == 530
    assert len(lock["author_overlap"]["cards"]) == 10
    assert set(lock["author_overlap"]["cards"]).isdisjoint(CONTAMINATION_CARDS)
    assert lock["author_overlap"]["response_columns"] == [
        "author_label",
        "author_evidence_url",
        "author_notes",
    ]
    assert len(author_rows) == lock["author_overlap"]["n_rows"]
    assert [row["card"] for row in contamination_rows] == list(CONTAMINATION_CARDS)
    assert all(row["author_assessment"] == "" for row in contamination_rows)


def test_seeded_overlap_is_repeatable_and_balanced(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    cards = {row["card"] for row in lock["packet"]["rows"]}
    first, pools = select_author_overlap_cards(cards, lock["sample_design"])
    second, _ = select_author_overlap_cards(cards, lock["sample_design"])

    assert first == second == lock["author_overlap"]["cards"]
    strata = {
        row["card"]: row["stratum"] for row in lock["sample_design"]["cards"]
    }
    assert sum(strata[card] == "flagged" for card in first) == 5
    assert sum(strata[card] == "unflagged" for card in first) == 5
    assert pools == {"flagged": 34, "unflagged": 40}


def test_return_validator_requires_exact_identity_unique_ids_and_allowed_labels(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    returned = tmp_path / "returned.csv"
    _write_verifier_return(returned, lock)
    validated = validate_verifier_return(returned, lock)
    assert len(validated) == 154
    assert all(row["notes"] == "" for row in validated)

    lines = returned.read_text(encoding="utf-8").splitlines()
    lines[1] += ",extra"
    returned.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra cells"):
        validate_verifier_return(returned, lock)

    _write_verifier_return(returned, lock)
    lines = returned.read_text(encoding="utf-8").splitlines()
    assert lines[1].endswith(",")
    lines[1] = lines[1][:-1]
    returned.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing cells"):
        validate_verifier_return(returned, lock)

    rows = lock["packet"]["rows"]
    with returned.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            changed = dict(row)
            if index == 1:
                changed["row_id"] = rows[0]["row_id"]
            writer.writerow(dict(changed, verifier_label="not-a-defect", notes=""))
    with pytest.raises(ValueError, match="duplicate row_id"):
        validate_verifier_return(returned, lock)

    _write_verifier_return(returned, lock)
    text = returned.read_text(encoding="utf-8")
    returned.write_text(text.replace("issue 0", "rewritten issue", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="protected field 'issue'"):
        validate_verifier_return(returned, lock)

    labels = ["not-a-defect"] * 154
    labels[0] = ""
    _write_verifier_return(returned, lock, labels)
    with pytest.raises(ValueError, match="verifier_label is blank"):
        validate_verifier_return(returned, lock)

    labels[0] = "material"
    _write_verifier_return(returned, lock, labels)
    with pytest.raises(ValueError, match="invalid verifier_label"):
        validate_verifier_return(returned, lock)


def test_verifier_evidence_and_unsure_rules_are_strict(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    returned = tmp_path / "returned.csv"
    _write_verifier_return(returned, lock)
    with returned.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows[0]["evidence_url"] = ""
    with returned.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="requires evidence_url"):
        validate_verifier_return(returned, lock)

    rows[0]["evidence_url"] = "ftp://example.test/evidence"
    with returned.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="invalid evidence_url"):
        validate_verifier_return(returned, lock)

    rows[0].update(verifier_label="unsure", evidence_url="", notes="")
    with returned.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unsure requires nonempty notes"):
        validate_verifier_return(returned, lock)

    rows[0]["notes"] = "The two public sources conflict."
    with returned.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PARTICIPANT_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    validated = validate_verifier_return(returned, lock)
    assert validated[0]["evidence_url"] == ""
    assert validated[0]["notes"] == "The two public sources conflict."


def test_finding_scores_keep_unsure_separate_and_use_decided_denominator():
    rows = [
        {"verifier_label": "confirmed-material"},
        {"verifier_label": "confirmed-material"},
        {"verifier_label": "confirmed-trivial"},
        {"verifier_label": "not-a-defect"},
        {"verifier_label": "unsure"},
    ]
    score = finding_score_block(rows)
    assert score["raw_label_counts"] == {
        "confirmed-material": 2,
        "confirmed-trivial": 1,
        "not-a-defect": 1,
        "unsure": 1,
    }
    assert score["n_decided"] == 4
    assert score["n_unsure"] == 1
    assert score["candidate_defect_confirmation"] == pytest.approx(3 / 4)
    assert score["material_defect_confirmation"] == pytest.approx(2 / 4)
    assert score["non_material_severe_call_share"] == pytest.approx(2 / 4)
    assert score["non_defect_share_among_decided_screen_positives"] == pytest.approx(
        1 / 4
    )
    sensitivity = score["unsure_sensitivity_all_candidates"]
    assert sensitivity["denominator"] == 5
    assert sensitivity["candidate_defect_confirmation"]["lower_rate"] \
        == pytest.approx(3 / 5)
    assert sensitivity["candidate_defect_confirmation"]["upper_rate"] \
        == pytest.approx(4 / 5)
    assert sensitivity["material_defect_confirmation"]["lower_rate"] \
        == pytest.approx(2 / 5)
    assert sensitivity["material_defect_confirmation"]["upper_rate"] \
        == pytest.approx(3 / 5)


def test_card_result_is_weighted_with_design_interval_and_unsure_sensitivity(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    labels = ["not-a-defect"] * 154
    packet_rows = lock["packet"]["rows"]
    flagged_row = next(
        index for index, row in enumerate(packet_rows)
        if row["card"].startswith("flagged-")
    )
    unflagged_row = next(
        index for index, row in enumerate(packet_rows)
        if row["card"].startswith("unflagged-")
    )
    labels[flagged_row] = "confirmed-material"
    labels[unflagged_row] = "confirmed-material"
    unsure_row = next(
        index for index, row in enumerate(packet_rows)
        if row["card"].startswith("flagged-")
        and row["card"] != packet_rows[flagged_row]["card"]
    )
    labels[unsure_row] = "unsure"
    returned = tmp_path / "returned.csv"
    _write_verifier_return(returned, lock, labels)
    rows = validate_verifier_return(returned, lock)
    result = card_detected_material_block(rows, lock)

    assert result["n_sample_cards"] == 150
    assert result[
        "n_screen_detected_verifier_confirmed_material_cards_raw"
    ] == 2
    assert result["weighted_total_cards"] == pytest.approx(530)
    assert result["rate"] == pytest.approx(1 / 75)
    interval = result["approximate_design_interval95"]
    assert interval["lower"] <= result["rate"] <= interval["upper"]
    assert interval["standard_error"] > 0
    assert "SRSWOR" in interval["method"]
    stratum_sample_variance = (75 / 74) * (1 / 75) * (74 / 75)
    expected_variance = sum(
        (population_n / 530) ** 2
        * (1 - 75 / population_n)
        * stratum_sample_variance
        / 75
        for population_n in (152, 378)
    )
    assert interval["standard_error"] ** 2 == pytest.approx(expected_variance)
    assert result["unsure_sensitivity"]["lower_rate"] == result["rate"]
    assert result["unsure_sensitivity"]["upper_rate"] > result["rate"]
    assert "defect prevalence" in result["inference_guard"]


def test_author_return_has_exact_identity_and_reports_full_confusion(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    verifier_path = tmp_path / "verifier.csv"
    author_path = tmp_path / "author.csv"
    verifier_labels = ["not-a-defect"] * 154
    _write_verifier_return(verifier_path, lock, verifier_labels)
    author_labels = ["not-a-defect"] * lock["author_overlap"]["n_rows"]
    author_labels[0] = "confirmed-trivial"
    _write_author_return(author_path, lock, author_labels)
    verifier_rows = validate_verifier_return(verifier_path, lock)
    author_rows = validate_author_return(author_path, lock)
    block = author_agreement_block(verifier_rows, author_rows, lock)

    assert block["n_exact_agreement"] == len(author_rows) - 1
    assert block["confusion"]["confirmed-trivial"]["not-a-defect"] == 1
    assert block["confusion_orientation"] == {
        "rows": "author_label",
        "columns": "verifier_label",
    }
    assert "descriptive" in block["comparison"]
    assert "category-specific reliability" in block["inference_guard"]

    with author_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["author_evidence_url"] = ""
    with author_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(PROTECTED_FIELDS) + AUTHOR_RESPONSE_FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="requires author_evidence_url"):
        validate_author_return(author_path, lock)

    _write_author_return(author_path, lock, author_labels)
    with author_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["issue"] = "tampered"
    with author_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(PROTECTED_FIELDS) + AUTHOR_RESPONSE_FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="protected field 'issue'"):
        validate_author_return(author_path, lock)


def test_validation_receipt_exposes_no_v1_results(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    verifier_path = tmp_path / "verifier.csv"
    _write_verifier_return(verifier_path, lock)
    verifier_rows = validate_verifier_return(verifier_path, lock)
    receipt = validation_receipt(lock, verifier_rows)

    assert receipt["v1_return_validation_complete"] is True
    assert receipt["screen_embargo_clear"] is False
    serialized = json.dumps(receipt)
    assert "finding_level" not in serialized
    assert "raw_label_counts" not in serialized
    assert "confirmed-material" not in serialized
    assert "rate" not in serialized


def test_score_requires_blind_author_return_and_contamination_gates_embargo(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    verifier_path = tmp_path / "verifier.csv"
    author_path = tmp_path / "author.csv"
    contamination_path = tmp_path / "contamination.csv"
    _write_verifier_return(verifier_path, lock)
    verifier_rows = validate_verifier_return(verifier_path, lock)

    with pytest.raises(ValueError, match="metrics are withheld"):
        score_screen_verification(lock, verifier_rows, None)

    _write_author_return(author_path, lock)
    author_rows = validate_author_return(author_path, lock)
    scored = score_screen_verification(lock, verifier_rows, author_rows)
    assert scored["v1_scoring_complete"] is True
    assert scored["screen_embargo_clear"] is False
    assert scored["contamination_check"]["status"] == "pending"
    assert scored["pending_components"] == [
        "completed documented contamination return"
    ]

    _write_contamination_return(contamination_path, lock)
    contamination_rows = validate_contamination_return(contamination_path, lock)
    complete = score_screen_verification(
        lock, verifier_rows, author_rows, contamination_rows
    )
    assert complete["screen_embargo_clear"] is True
    assert complete["pending_components"] == []
    assert complete["author_v1_overlap"]["status"] == "complete"
    assert complete["author_v1_overlap"]["exact_agreement"] == 1.0
    assert complete["contamination_check"]["status"] == "complete"


def test_contamination_return_requires_exact_three_rows_and_locked_labels(tmp_path):
    lock, _, _ = _full_fixture(tmp_path)
    path = tmp_path / "contamination.csv"
    _write_contamination_return(path, lock)
    rows = validate_contamination_return(path, lock)
    assert [row["card"] for row in rows] == list(CONTAMINATION_CARDS)

    _write_contamination_return(
        path,
        lock,
        ["confirmed-contamination", "not-contamination", "maybe"],
    )
    with pytest.raises(ValueError, match="invalid author_assessment"):
        validate_contamination_return(path, lock)

    _write_contamination_return(path, lock)
    with path.open(newline="", encoding="utf-8") as handle:
        changed = list(csv.DictReader(handle))
    changed[0]["author_evidence_url"] = ""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTAMINATION_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(changed)
    with pytest.raises(ValueError, match="requires author_evidence_url"):
        validate_contamination_return(path, lock)

    _write_contamination_return(path, lock)
    with path.open(newline="", encoding="utf-8") as handle:
        changed = list(csv.DictReader(handle))
    changed[0]["screen_summary"] = "rewritten"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTAMINATION_WORKLIST_FIELDS)
        writer.writeheader()
        writer.writerows(changed)
    with pytest.raises(ValueError, match="protected field 'screen_summary'"):
        validate_contamination_return(path, lock)
