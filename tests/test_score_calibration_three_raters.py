"""Focused tests for the corrected two-arm, three-rater calibration scorer."""

import csv
import hashlib
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from score_calibration import (  # noqa: E402
    FILLED_LABELS,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    normalize_human_label,
    parse_arm_membership,
    probability_weighted_block,
    read_rating_table,
    require_distinct_rating_files,
    score_calibration,
    validate_and_join,
    validate_key,
    weighted_confusion,
    write_adjudication_worklist,
)


def key_row(item_id, card, kind, judge, arms, *, error_pi=None,
            prob_stratum=None, card_pi=None, field_pi=None, outer_weight=None):
    row = {
        "item_id": item_id,
        "card": card,
        "field_path": f"field.{item_id}",
        "kind": kind,
        "judge_label": judge,
        "arm_membership": arms,
    }
    if "error" in arms:
        row.update({
            "error_status_stratum": judge,
            "error_row_pi_within_pool": error_pi,
        })
    if "probability" in arms:
        row.update({
            "prob_flag_stratum": prob_stratum,
            "prob_card_pi_within_s": card_pi,
            "prob_field_pi_given_card": field_pi,
            "prob_row_pi_within_s": card_pi * field_pi,
            "s_to_corpus_weight": outer_weight,
        })
    return row


def packet_key():
    return {"key": [
        key_row("e1", "card-e1", "filled", "unsupported", "error",
                error_pi=0.5),
        key_row("dual", "card-dual", "filled", "partial", "error|probability",
                error_pi=0.25, prob_stratum="flagged", card_pi=0.2,
                field_pi=0.5, outer_weight=2.0),
        key_row("p1", "card-p1", "filled", "supported", "probability",
                prob_stratum="unflagged", card_pi=0.5, field_pi=1.0,
                outer_weight=1.0),
        key_row("pns", "card-pns", "not_specified", "no", "probability",
                prob_stratum="unflagged", card_pi=0.25, field_pi=0.5,
                outer_weight=1.0),
    ]}


def rating_tables(key, labels_by_rater):
    tables = []
    for labels in labels_by_rater:
        table = {}
        for row in key["key"]:
            item_id = row["item_id"]
            table[item_id] = {
                "label": labels[item_id],
                "public": {
                    "item_id": item_id,
                    "card": row["card"],
                    "field_path": row["field_path"],
                    "kind": row["kind"],
                    "field_value": f"value for {item_id}",
                },
            }
        tables.append(table)
    return tables


def majority_labels():
    return [
        {"e1": "unsupported", "dual": "partial", "p1": "unsupported", "pns": "no"},
        {"e1": "unsupported", "dual": "partial", "p1": "unsupported", "pns": "no"},
        {"e1": "partial", "dual": "unsupported", "p1": "supported", "pns": "yes_primary"},
    ]


def test_current_and_legacy_participant_labels_normalize():
    assert normalize_human_label("supported_with_registry") == "supported_by_eee_only"
    assert normalize_human_label("yes_source") == "yes_primary"
    assert normalize_human_label("yes_registry") == "yes_eee_only"
    assert normalize_human_label("supported_registry_only") == "supported_by_eee_only"
    assert normalize_human_label("yes_outside_registry") == "yes_primary"
    assert normalize_human_label("yes_registry_only") == "yes_eee_only"


def test_rating_reader_accepts_current_participant_labels(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text(
        "item_id,card,field_path,kind,field_value,human_label,human_note\n"
        "a,c,f.a,filled,value,supported_with_registry,\n"
        "b,c,f.b,not_specified,Not specified,yes_source,\n"
        "c,c,f.c,not_specified,Not specified,yes_registry,\n")
    table = read_rating_table(str(path))
    assert table["a"]["label"] == "supported_by_eee_only"
    assert table["b"]["label"] == "yes_primary"
    assert table["c"]["label"] == "yes_eee_only"


def test_arm_membership_accepts_canonical_pipe_and_defensive_list_aliases():
    assert parse_arm_membership({"item_id": "a", "arm_membership": "error|probability"}) \
        == {"error", "probability"}
    assert parse_arm_membership({"item_id": "b", "arms": ["risk", "random"]}) \
        == {"error", "probability"}


def test_nominal_three_rater_reliability_perfect_fixture():
    ratings = [
        ["supported", "supported", "supported"],
        ["partial", "partial", "partial"],
    ]
    assert fleiss_kappa(ratings, FILLED_LABELS) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal(ratings, FILLED_LABELS) == pytest.approx(1.0)


def test_nominal_three_rater_reliability_nontrivial_hand_fixture():
    ratings = [
        ["supported", "supported", "partial"],
        ["supported", "partial", "partial"],
    ]
    # Fleiss: Pbar=1/3, Pe=1/2. Krippendorff: Do=2/3, De=3/5.
    assert fleiss_kappa(ratings, FILLED_LABELS) == pytest.approx(-1 / 3)
    assert krippendorff_alpha_nominal(ratings, FILLED_LABELS) == pytest.approx(-1 / 9)


def test_key_rejects_product_inconsistency_and_uses_recorded_field_probability():
    key = packet_key()
    key["key"][1]["prob_field_pi_given_card"] = 5 / 22
    # The row probability was not updated, so the design cannot silently assume 5/23
    # or any other global field probability.
    with pytest.raises(ValueError, match="card_pi\\*field_pi"):
        validate_key(key)


def test_v7_probability_probabilities_and_low_cluster_sensitivity_metadata():
    rows = []
    for stratum, outer_weight in (("flagged", 2.026667), ("unflagged", 5.04)):
        for card_index in range(8):
            card = f"{stratum}-{card_index}"
            for field_index in range(2):
                rows.append(key_row(
                    f"{card}-{field_index}", card, "filled", "supported",
                    "probability", prob_stratum=stratum, card_pi=8 / 75,
                    field_pi=2 / 23, outer_weight=outer_weight))
    # The scorer consumes the recorded per-row product for the v7 8x2 design.
    validated = validate_key({"n_raters": 3, "key": rows})
    assert all(row["prob_row_pi"] == pytest.approx((8 / 75) * (2 / 23))
               for row in validated)
    assert all(row["prob_row_pi"] == pytest.approx(16 / 1725)
               for row in validated)
    for row in validated:
        row["human_reference"] = row["judge_label"]
    block = probability_weighted_block(validated, B=5, seed=3)
    assert block["sampled_card_clusters_by_stratum"] == {
        "flagged": 8, "unflagged": 8,
    }
    assert "realized second-stage field draw" in block["uncertainty_scope"]
    assert "not an exact two-stage design-based interval" in block["uncertainty_scope"]
    assert "low-cluster sensitivity intervals" in block["low_cluster_caution"]
    filled = block["filled"]
    assert "weighted_agreement_card_bootstrap_sensitivity_interval95" in filled
    assert "weighted_agreement_ci95_card_clustered" not in filled


def test_v7_chartqa_probability_uses_the_complete_23_field_frame():
    row = key_row(
        "chartqa-method", "chartqa", "filled", "supported", "probability",
        prob_stratum="flagged", card_pi=8 / 75, field_pi=2 / 23,
        outer_weight=2.026667)
    validated = validate_key({"key": [row]})[0]
    assert validated["prob_row_pi"] == pytest.approx((8 / 75) * (2 / 23))
    assert validated["analysis_weight"] == pytest.approx(
        2.026667 / ((8 / 75) * (2 / 23)))


def test_key_requires_dual_membership_instead_of_duplicate_packet_rows():
    key = packet_key()
    duplicate = dict(key["key"][0])
    duplicate["item_id"] = "duplicate-e1"
    key["key"].append(duplicate)
    with pytest.raises(ValueError, match="merge arm memberships"):
        validate_key(key)


def test_two_arms_are_reported_separately_and_dual_item_is_rated_once():
    key = packet_key()
    result, rows = score_calibration(
        key, rating_tables(key, majority_labels()), B=30, seed=7)

    assert result["n_unique_items"] == 4
    assert result["n_error_arm_rows"] == 2
    assert result["n_probability_arm_rows"] == 3
    assert result["n_dual_arm_rows"] == 1
    assert result["analysis_policy"]["combined_arm_estimate"] is None
    assert "design_weighted" not in result

    raw = result["judge_vs_human_reference"]
    assert raw["error_arm_raw"]["filled"]["raw_agreement"] == 1.0
    assert raw["probability_arm_raw"]["filled"]["raw_agreement"] == 0.5
    assert raw["probability_arm_raw"]["not_specified"]["raw_agreement"] == 1.0

    confirmation = result["error_arm_conditional_confirmation"]
    assert confirmation["unsupported"]["design_weighted_confirmation_rate"] == 1.0
    assert confirmation["partial"]["design_weighted_confirmation_rate"] == 1.0

    # Filled probability rows: dual has weight 2/(.2*.5)=20 and agrees; p1 has
    # weight 1/(.5*1)=2 and disagrees.  This pins the recorded row probabilities.
    weighted = result["probability_arm_corpus_weighted"]["filled"]
    assert weighted["weighted_agreement"] == pytest.approx(20 / 22)
    assert weighted["n_sampled_rows"] == 2
    assert len(rows) == 4


def test_probability_confusion_uses_outer_weight_over_recorded_row_pi():
    key = packet_key()
    rows = validate_key(key)
    tables = rating_tables(key, majority_labels())
    joined = validate_and_join(rows, tables)
    probability_rows = [row for row in joined if "probability" in row["arms"]]
    block = weighted_confusion(probability_rows, "filled")
    assert block["confusion"]["partial"]["partial"] == pytest.approx(20.0)
    assert block["confusion"]["supported"]["unsupported"] == pytest.approx(2.0)


def test_true_three_way_split_suppresses_judge_metrics_until_blind_adjudication(tmp_path):
    key = packet_key()
    labels = majority_labels()
    labels[0]["e1"] = "unsupported"
    labels[1]["e1"] = "partial"
    labels[2]["e1"] = "supported"
    tables = rating_tables(key, labels)

    result, rows = score_calibration(key, tables, B=5)
    assert result["n_true_three_way_splits"] == 1
    assert result["n_unresolved_three_way_splits"] == 1
    assert result["judge_vs_human_reference"] is None
    assert result["probability_arm_corpus_weighted"] is None

    worklist = tmp_path / "blind.csv"
    write_adjudication_worklist(worklist, rows)
    with worklist.open(newline="") as f:
        reader = csv.DictReader(f)
        written = list(reader)
    assert [row["item_id"] for row in written] == ["e1"]
    assert "judge_label" not in reader.fieldnames
    assert "ratings" not in reader.fieldnames
    assert "arm_membership" not in reader.fieldnames

    complete, _ = score_calibration(key, tables, {"e1": "unsupported"}, B=5)
    assert complete["complete_for_judge_comparison"] is True
    assert complete["judge_vs_human_reference"] is not None


def test_adjudication_is_rejected_for_an_item_that_has_a_majority():
    key = packet_key()
    with pytest.raises(ValueError, match="only for true three-way splits"):
        score_calibration(
            key, rating_tables(key, majority_labels()), {"e1": "unsupported"}, B=0)


def test_rating_files_must_have_complete_identical_item_sets_and_public_rows():
    key = packet_key()
    rows = validate_key(key)
    tables = rating_tables(key, majority_labels())
    del tables[2]["p1"]
    with pytest.raises(ValueError, match="item set mismatch"):
        validate_and_join(rows, tables)

    tables = rating_tables(key, majority_labels())
    tables[1]["dual"]["public"]["field_value"] = "different"
    with pytest.raises(ValueError, match="public row differs"):
        validate_and_join(rows, tables)


def test_returned_field_values_must_match_the_frozen_packet_hash():
    key = packet_key()
    for row in key["key"]:
        value = f"value for {row['item_id']}"
        row["field_value_chars"] = len(value)
        row["field_value_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    rows = validate_key(key)
    tables = rating_tables(key, majority_labels())
    for table in tables:
        table["p1"]["public"]["field_value"] = "same accidental edit in every file"
    with pytest.raises(ValueError, match="differs from the frozen packet"):
        validate_and_join(rows, tables)


def test_exactly_three_raters_are_required():
    key = packet_key()
    rows = validate_key(key)
    with pytest.raises(ValueError, match="exactly three"):
        validate_and_join(rows, rating_tables(key, majority_labels())[:2])


def test_cli_input_gate_requires_three_distinct_returned_files(tmp_path):
    r1 = tmp_path / "R1.csv"
    r2 = tmp_path / "R2.csv"
    r1.write_text("placeholder")
    r2.write_text("placeholder")
    with pytest.raises(ValueError, match="three distinct returned files"):
        require_distinct_rating_files([str(r1), str(r1), str(r2)])
