"""Validation guard for the repaired S150 judge analysis artifact."""


DECLARED_EXCLUSIONS = []


def validate_judge_analysis_frame(judge):
    """Reject raw/legacy judge artifacts before they can produce paper numbers."""
    frame = judge.get("analysis_frame")
    if not isinstance(frame, dict):
        raise ValueError(
            "judge artifact has no analysis_frame; build the repaired derived view "
            "with scripts/build_judge_analysis_view.py"
        )
    allowlist = frame.get("allowlisted_paths")
    exclusions = frame.get("declared_exclusions")
    if not isinstance(allowlist, list) or len(allowlist) != 23 or \
            len(set(allowlist)) != 23:
        raise ValueError("judge analysis_frame must declare 23 unique content paths")
    if exclusions != DECLARED_EXCLUSIONS:
        raise ValueError("judge analysis_frame has unexpected exclusions")
    per_card = judge.get("per_card")
    if not isinstance(per_card, dict) or len(per_card) != frame.get("n_cards"):
        raise ValueError("judge analysis_frame card count does not match per_card")
    allowed = set(allowlist)
    declared = {(row["card"], row["field_path"]) for row in exclusions}
    n_rows = 0
    for card, record in per_card.items():
        paths = [row.get("path") for row in record.get("field_verdicts", [])]
        if len(paths) != len(set(paths)) or not set(paths) <= allowed:
            raise ValueError(f"{card}: duplicate or non-allowlisted analysis verdict")
        expected = allowed - {path for name, path in declared if name == card}
        if set(paths) != expected:
            raise ValueError(f"{card}: analysis verdict coverage does not match frame")
        n_rows += len(paths)
    if n_rows != frame.get("n_rows"):
        raise ValueError("judge analysis_frame row count does not match per_card")
    return frame
