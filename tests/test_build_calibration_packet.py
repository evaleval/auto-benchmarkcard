import csv
import copy
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import build_calibration_packet as builder  # noqa: E402


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _synthetic_artifacts(tmp_path):
    # Put ChartQA at a deterministically selected sorted-list index so its
    # probability-field draw is exercised in the complete 23-field frame.
    flagged = (
        ["chartqa"]
        + [f"a-flagged-{i:03d}" for i in range(4)]
        + [f"z-flagged-{i:03d}" for i in range(70)]
    )
    unflagged = [f"unflagged-{i:03d}" for i in range(75)]
    outer_strata = {
        **{card: "flagged" for card in flagged},
        **{card: "unflagged" for card in unflagged},
    }

    # Determine the probability identities without consulting any judge status.
    probability_keys = []
    for stratum, cards in (("flagged", flagged), ("unflagged", unflagged)):
        card_seed = builder.derive_subseed(
            builder.MASTER_SEED, f"probability:cards:{stratum}"
        )
        selected_cards, _, _ = builder._extend_v4_card_sample(cards, card_seed)
        for card in selected_cards:
            fields = list(builder.ELIGIBLE_FIELD_PATHS)
            field_seed = builder.derive_subseed(
                builder.MASTER_SEED, f"probability:fields:{card}"
            )
            selected_fields = __import__("random").Random(field_seed).sample(
                sorted(fields), 3
            )[:2]
            probability_keys.extend((card, path) for path in selected_fields)

    # Ensure known dual memberships and both realized kinds. Selection above
    # did not consult any of these statuses.
    partial_keys = set(probability_keys[:13])
    unsupported_keys = set(probability_keys[13:23])
    not_specified_keys = set(probability_keys[23:28])
    all_available = [
        (card, path)
        for card in sorted(outer_strata)
        for path in builder.ELIGIBLE_FIELD_PATHS
    ]
    spare = [
        key for key in all_available
        if key not in partial_keys | unsupported_keys | not_specified_keys
    ]
    unsupported_keys.update(spare[:30])
    assert len(partial_keys) == 13
    assert len(unsupported_keys) == 40

    per_card = {}
    for card in sorted(outer_strata):
        verdicts = []
        for path in builder.ELIGIBLE_FIELD_PATHS:
            key = (card, path)
            status = (
                "partial" if key in partial_keys
                else "unsupported" if key in unsupported_keys
                else "not_specified" if key in not_specified_keys
                else "supported"
            )
            verdicts.append({
                "path": path,
                "status": status,
                "specificity": "na" if status == "not_specified" else "specific",
                "info_in_source": "no" if status == "not_specified" else "na",
                "note": "synthetic",
            })
        per_card[card] = {
            "stratum": outer_strata[card],
            "field_verdicts": verdicts,
            "risk_verdicts": [],
        }

    judge = {
        "analysis_frame": {
            "allowlisted_paths": list(builder.ELIGIBLE_FIELD_PATHS),
            "declared_exclusions": list(builder.EXPECTED_DECLARED_EXCLUSIONS),
            "n_rows": builder.EXPECTED_N_FRAME_ROWS,
            "n_cards": builder.EXPECTED_N_CARDS,
        },
        "per_card": per_card,
    }
    sample = {
        "n": 150,
        "strata": {
            "flagged": {"N": 152, "n": 75, "weight": 2.0},
            "unflagged": {"N": 378, "n": 75, "weight": 5.0},
        },
        "cards": [
            {
                "name": card,
                "stratum": outer_strata[card],
                "weight": 2.0 if outer_strata[card] == "flagged" else 5.0,
            }
            for card in sorted(outer_strata)
        ],
    }
    judge_path, sample_path = tmp_path / "judge.json", tmp_path / "sample.json"
    _write_json(judge_path, judge)
    _write_json(sample_path, sample)
    return judge, judge_path, sample_path, outer_strata


def _membership_keys(selected, arm, category=None):
    result = set()
    for row in selected:
        for membership in row["arm_memberships"]:
            if membership["arm"] == arm and (
                category is None or membership.get("category") == category
            ):
                result.add((row["card"], row["field_path"]))
    return result


def test_two_arm_draw_is_reproducible_independent_and_records_overlap(tmp_path):
    judge, _, sample_path, outer_strata = _synthetic_artifacts(tmp_path)
    frame = builder.validate_and_build_frame(
        judge, builder.load_outer_strata(sample_path)
    )
    selected_a, meta_a = builder.select_two_arm_sample(frame)
    selected_b, meta_b = builder.select_two_arm_sample(frame)
    assert selected_a == selected_b
    assert meta_a == meta_b
    assert meta_a["seed_derivation"]["namespaces"] == {
        "error": "human-validation-v3",
        "probability_cards": "human-validation-v4",
        "probability_fields": "human-validation-v4",
    }
    assert meta_a["seed_derivation"]["subseeds"]["error_unsupported"] == 14023610987678575025
    assert meta_a["seed_derivation"]["subseeds"]["error_partial"] == 11195151507745344204
    assert meta_a["seed_derivation"]["subseeds"]["probability_cards"] == {
        "flagged": 1322918315401641256,
        "unflagged": 12542704937546063820,
    }
    assert meta_a["membership_counts"] == {
        "error_partial": 13,
        "error_unsupported": 30,
        "probability": 32,
    }
    assert meta_a["n_dual_membership_rows"] >= 13
    assert len(_membership_keys(selected_a, "probability")) == 32
    assert len(_membership_keys(selected_a, "error", "unsupported")) == 30
    assert len(_membership_keys(selected_a, "error", "partial")) == 13

    card_draws = meta_a["probability_arm"]["cards_by_outer_stratum"]
    assert set(card_draws) == {"flagged", "unflagged"}
    for details in card_draws.values():
        assert details["pool_size"] == 75
        assert details["n_selected"] == 8
        assert details["inclusion_fraction"]["numerator"] == 8
        assert details["inclusion_fraction"]["denominator"] == 75
        assert len(details["preserved_v4_cards"]) == 5
        assert len(details["v5_extension_cards"]) == 3
    for stratum, details in card_draws.items():
        pool = sorted(
            card for card, value in outer_strata.items() if value == stratum
        )
        expected_v4_draw = __import__("random").Random(details["subseed"]).sample(
            pool, 5
        )
        assert details["preserved_v4_cards_draw_order"] == expected_v4_draw
        assert details["selected_cards_draw_order"][:5] == expected_v4_draw
    for card, details in meta_a["probability_arm"]["fields_by_card"].items():
        expected_denominator = 23
        assert details["pool_size"] == expected_denominator
        assert details["field_conditional_inclusion_fraction"] == {
            "numerator": 2,
            "denominator": expected_denominator,
            "value": 2 / expected_denominator,
        }
        assert len(details["selected_rows"]) == 2
        assert details["selected_field_paths_draw_order"] == (
            details["preserved_v4_three_field_draw_order"][:2]
        )
    assert "chartqa" in meta_a["probability_arm"]["fields_by_card"]
    assert sum(meta_a["probability_kind_counts"].values()) == 32
    assert meta_a["probability_kind_counts"] == {
        "filled": 27,
        "not_specified": 5,
    }

    # Changing only the unsupported pool cannot move the probability draw or
    # the independently seeded partial draw.
    changed = copy.deepcopy(frame)
    changed_row = next(row for row in changed if row["judge_status"] == "supported")
    changed_row["judge_status"] = "unsupported"
    selected_changed, _ = builder.select_two_arm_sample(changed)
    assert _membership_keys(selected_changed, "probability") == _membership_keys(
        selected_a, "probability"
    )
    assert _membership_keys(selected_changed, "error", "partial") == _membership_keys(
        selected_a, "error", "partial"
    )


def test_frame_validation_rejects_undeclared_gap_duplicate_and_bad_path(tmp_path):
    judge, _, sample_path, _ = _synthetic_artifacts(tmp_path)
    strata = builder.load_outer_strata(sample_path)

    duplicate = copy.deepcopy(judge)
    duplicate["per_card"]["a-flagged-000"]["field_verdicts"].append(
        copy.deepcopy(duplicate["per_card"]["a-flagged-000"]["field_verdicts"][0])
    )
    with pytest.raises(ValueError, match="expected 1 verdict"):
        builder.validate_and_build_frame(duplicate, strata)

    missing = copy.deepcopy(judge)
    missing["per_card"]["a-flagged-000"]["field_verdicts"].pop()
    with pytest.raises(ValueError, match="expected 1 verdict"):
        builder.validate_and_build_frame(missing, strata)

    typo = copy.deepcopy(judge)
    typo["analysis_frame"]["allowlisted_paths"][-1] = "purpose_and_intended_user.tasks"
    with pytest.raises(ValueError, match="23-field allowlist"):
        builder.validate_and_build_frame(typo, strata)


def test_packet_keeps_complete_nested_values_sources_and_identical_rater_csvs(tmp_path):
    judge, judge_path, sample_path, _ = _synthetic_artifacts(tmp_path)
    frame = builder.validate_and_build_frame(
        judge, builder.load_outer_strata(sample_path)
    )
    selected, _ = builder.select_two_arm_sample(frame)
    inputs_dir, out_dir = tmp_path / "inputs", tmp_path / "packet"
    inputs_dir.mkdir()
    nested_value = {
        "z": ["x" * 650, {"b": 2, "a": 1}],
        "a": "full Unicode value: café",
    }
    canonical = builder._field_value_for_csv(nested_value)
    assert len(canonical) > 500
    for card in {row["card"] for row in selected}:
        source = f"source for {card}\nUnicode: café\n"
        prepared = {
            "name": card,
            "source_text": source,
            "fields": [
                {"path": path, "value": nested_value}
                for path in builder.ELIGIBLE_FIELD_PATHS
            ],
        }
        _write_json(inputs_dir / f"{card}.json", prepared)

    meta = builder.build_packet(
        str(judge_path), str(sample_path), str(inputs_dir), str(out_dir),
        master_seed=7, raters=3,
    )
    rater_bytes = [
        (out_dir / f"ratings_rater{i}.csv").read_bytes() for i in range(1, 4)
    ]
    assert rater_bytes[0] == rater_bytes[1] == rater_bytes[2]

    with (out_dir / "ratings_rater1.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == meta["n_unique_items"]
    assert all(row["field_value"] == canonical for row in rows)
    assert all(len(row["field_value"]) > 500 for row in rows)
    assert all(row["human_label"] == row["human_note"] == "" for row in rows)

    for key_row in meta["key"]:
        assert key_row["arm_membership"] in {
            "error", "probability", "error|probability"
        }
        if key_row["arm_membership"] == "probability":
            assert key_row["error_status_stratum"] is None
            assert key_row["error_row_pi_within_pool"] is None
        if key_row["arm_membership"] == "error":
            assert key_row["prob_flag_stratum"] is None
            assert key_row["prob_card_pi_within_s"] is None
            assert key_row["prob_field_pi_given_card"] is None
            assert key_row["prob_row_pi_within_s"] is None
        expected_weight = 2.0 if key_row["outer_stratum"] == "flagged" else 5.0
        assert key_row["s_to_corpus_weight"] == expected_weight

    for card, source_meta in meta["source_files"].items():
        source_bytes = (out_dir / source_meta["file"]).read_bytes()
        assert len(source_bytes) == source_meta["bytes"]
        assert builder._sha256(source_bytes) == source_meta["sha256"]
    assert meta["eligible_frame"]["frame_hash"]["n_rows"] == 3450


def test_sample_card_weight_must_match_stratum_weight(tmp_path):
    _, _, sample_path, _ = _synthetic_artifacts(tmp_path)
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    sample["cards"][0]["weight"] = 999
    _write_json(sample_path, sample)
    with pytest.raises(ValueError, match="does not equal"):
        builder.load_sample_design(sample_path)
