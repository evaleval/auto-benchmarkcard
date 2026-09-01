import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_numbers import build_values  # noqa: E402


def test_kappa_interval_remains_on_the_kappa_scale():
    calibration = {
        "n_unique_items": 75,
        "probability_arm_corpus_weighted": {
            "filled": {
                "weighted_agreement": 0.75,
                "weighted_agreement_card_bootstrap_sensitivity_interval95": [0.6, 0.8],
                "cohens_kappa": -0.12,
                "cohens_kappa_card_bootstrap_sensitivity_interval95": [-0.265, -0.011],
            },
            "not_specified": {},
        },
    }

    values = build_values(None, calibration, None, None)

    assert values["rq2JudgeHumanAgreementCI"][0] == "[60.0, 80.0]"
    assert values["rq2JudgeHumanKappa"][0] == "-0.12"
    assert values["rq2JudgeHumanKappaCI"][0] == "[-0.27, -0.01]"


def test_extension_rates_keep_two_decimal_precision():
    extensions = {
        "field_slot_outcomes": {
            "five_state": {
                "filled_fully_supported": {
                    "value": 0.51894,
                    "ci95": [0.47491, 0.56104],
                }
            }
        },
        "ethical_legal_coverage": {},
        "human_confirmed_unsupported": {},
        "cross_instrument_overlap": {},
    }

    values = build_values(None, None, None, None, extensions)

    assert values["rq2SlotFullSupportRate"][0] == "51.89\\%"
    assert values["rq2SlotFullSupportRateCI"][0] == "[47.49, 56.10]"
