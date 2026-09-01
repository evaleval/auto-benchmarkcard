import csv
import json
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from analyze_paper_extensions import (  # noqa: E402
    FIVE_STATES,
    _state,
    _write_matrix,
    matched_judged_paths,
    validate_sample_design,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ({"status": "supported", "info_in_source": "na"},
         "filled_fully_supported"),
        ({"status": "supported_by_eee_only", "info_in_source": "na"},
         "filled_fully_supported"),
        ({"status": "partial", "info_in_source": "na"},
         "filled_partially_supported"),
        ({"status": "unsupported", "info_in_source": "na"},
         "filled_unsupported"),
        ({"status": "not_specified", "info_in_source": "yes_primary"},
         "not_specified_information_available"),
        ({"status": "not_specified", "info_in_source": "yes_eee_only"},
         "not_specified_information_available"),
        ({"status": "not_specified", "info_in_source": "no"},
         "not_specified_no_information"),
    ],
)
def test_five_state_mapping_is_exhaustive_for_frozen_labels(verdict, expected):
    assert _state(verdict) == expected


def test_not_specified_with_na_information_is_rejected():
    with pytest.raises(ValueError, match="unclassified info_in_source"):
        _state({"status": "not_specified", "info_in_source": "na"})


def test_screen_locator_maps_only_exact_canonical_paths():
    locator = (
        "benchmark_details.domains / purpose_and_intended_users.tasks / "
        "methodology.metrics"
    )
    assert matched_judged_paths(locator) == [
        "benchmark_details.domains",
        "purpose_and_intended_users.tasks",
    ]


def test_broad_or_prefix_locator_does_not_create_a_field_match():
    assert matched_judged_paths("benchmark_details / data / methodology") == []
    assert matched_judged_paths("methodology.validation_notes") == []


def test_field_matrix_records_interval_method_and_raw_design_counts(tmp_path):
    state_names = (
        "filled_fully_supported",
        "filled_partially_supported",
        "filled_unsupported",
        "not_specified_information_available",
        "not_specified_no_information",
    )
    metric = {
        "value": 0.1,
        "ci95": [0.02, 0.2],
        "ci_method": "wilson-neff",
        "counts": {
            "num": 1.0,
            "den": 150.0,
            "by_stratum": {
                "flagged": {"num": 1.0, "den": 75.0},
                "unflagged": {"num": 0.0, "den": 75.0},
            },
        },
    }
    matrix = [
        {
            "path": "benchmark_details.domains",
            "states": {state: metric for state in state_names},
        }
    ]
    corpus = {
        "benchmark_details.domains": {
            "not_specified_count": 20,
            "absent_count": 0,
            "denominator": 530,
            "not_specified_rate": 20 / 530,
        }
    }
    output = tmp_path / "matrix.csv"

    _write_matrix(output, matrix, corpus)

    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    prefix = "filled_unsupported"
    assert row[f"{prefix}_ci_method"] == "wilson-neff"
    assert row[f"{prefix}_raw_denominator"] == "150"
    assert json.loads(row[f"{prefix}_by_stratum_counts"]) == {
        "flagged": {"den": 75.0, "num": 1.0},
        "unflagged": {"den": 75.0, "num": 0.0},
    }


def test_frozen_extension_artifact_has_expected_design_identities():
    with open(
        os.path.join(REPO, "eval/s150/paper_extension_analysis.json"),
        encoding="utf-8",
    ) as handle:
        artifact = json.load(handle)

    five_state = artifact["field_slot_outcomes"]["five_state"]
    expected_counts = {
        "filled_fully_supported": 1746,
        "filled_partially_supported": 259,
        "filled_unsupported": 30,
        "not_specified_information_available": 244,
        "not_specified_no_information": 1171,
    }
    assert {
        state: int(five_state[state]["counts"]["num"])
        for state in FIVE_STATES
    } == expected_counts
    assert all(five_state[state]["counts"]["den"] == 3450 for state in FIVE_STATES)
    assert sum(five_state[state]["value"] for state in FIVE_STATES) == pytest.approx(1)

    ethical = artifact["ethical_legal_coverage"]
    assert len(ethical["paths"]) == 3
    assert len(ethical["comparison_paths"]) == 20
    assert (
        ethical["held_out"]["ethical_legal_fields"]["not_specified"]["counts"]["den"]
        == 450
    )
    assert (
        ethical["held_out"]["other_20_fields"]["not_specified"]["counts"]["den"]
        == 3000
    )

    matrix = artifact["field_matrix"]
    assert len(matrix) == 23
    for row in matrix:
        assert sum(row["states"][state]["value"] for state in FIVE_STATES) == pytest.approx(
            1
        )

    human = artifact["human_confirmed_unsupported"]
    assert human["judge_unsupported_census_size"] == 30
    assert human["human_confirmed_unsupported"] == 27

    overlap = artifact["cross_instrument_overlap"]
    assert (
        overlap["confirmed_material_findings"],
        overlap["findings_naming_at_least_one_exact_judged_path"],
        overlap["matched_field_checks"],
        overlap["cards_with_matched_checks"],
    ) == (111, 43, 52, 35)
    assert overlap["source_judge_status_counts"] == {
        "supported": 20,
        "supported_by_eee_only": 9,
        "partial": 17,
        "unsupported": 6,
    }
    assert overlap["fully_supported_checks"] == 29


def test_duplicate_sample_row_is_rejected():
    with open(os.path.join(REPO, "eval/s150/sample.json"), encoding="utf-8") as handle:
        sample = json.load(handle)
    sample["cards"][-1] = dict(sample["cards"][0])

    with pytest.raises(ValueError, match="not unique"):
        validate_sample_design(sample)
