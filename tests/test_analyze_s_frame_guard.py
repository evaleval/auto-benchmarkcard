import copy
import json
import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from judge_analysis_guard import validate_judge_analysis_frame  # noqa: E402


REPO = os.path.join(os.path.dirname(__file__), "..")


def analysis_view():
    with open(os.path.join(
        REPO, "eval/s150/judge/analysis_frame.json"
    )) as f:
        return json.load(f)


def test_current_repaired_judge_view_passes_frame_guard():
    frame = validate_judge_analysis_frame(analysis_view())
    assert frame["n_cards"] == 150
    assert frame["n_rows"] == 3450


def test_legacy_unrepaired_judge_artifact_is_rejected():
    legacy = {"label": "legacy", "per_card": {}}
    with pytest.raises(ValueError, match="no analysis_frame"):
        validate_judge_analysis_frame(legacy)


def test_complete_frame_cannot_silently_drop_a_field():
    judge = analysis_view()
    bad = copy.deepcopy(judge)
    bad["per_card"]["chartqa"]["field_verdicts"] = [
        row
        for row in bad["per_card"]["chartqa"]["field_verdicts"]
        if row["path"] != "methodology.methods"
    ]
    with pytest.raises(ValueError, match="coverage does not match frame"):
        validate_judge_analysis_frame(bad)
