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
