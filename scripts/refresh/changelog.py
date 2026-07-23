"""Per-card changelog and the run roll-up.

Each card gets a json record of every targeted-field decision plus its
missing_fields before/after, so any kept/replaced choice is auditable and
reconstructable. The roll-up aggregates per-field replaced / kept / floor_blocked
counts across the run. Both are written under output/refresh_runs/<run_id>/.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def card_changelog(
    slug: str,
    run_dir: str,
    b_eligible: bool,
    fields: List[Dict[str, Any]],
    missing_before: List[str],
    missing_after: List[str],
) -> Dict[str, Any]:
    return {
        "slug": slug,
        "run_dir": run_dir,
        "b_eligible": b_eligible,
        "fields": fields,
        "missing_fields_before": missing_before,
        "missing_fields_after": missing_after,
    }


def write_card_changelog(run_out_dir: str, slug: str, entry: Dict[str, Any]) -> str:
    changelog_dir = os.path.join(run_out_dir, "changelog")
    os.makedirs(changelog_dir, exist_ok=True)
    path = os.path.join(changelog_dir, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=True)
    return path


def build_rollup(card_changelogs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-field decision counts and a small run summary."""
    per_field: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"replaced": 0, "demoted": 0, "kept": 0, "floor_blocked": 0, "ineligible": 0}
    )
    totals = {"cards": 0, "cards_with_change": 0, "changes": 0}

    for cc in card_changelogs:
        totals["cards"] += 1
        changed_here = 0
        for fld in cc["fields"]:
            counts = per_field[fld["field"]]
            if fld["decision"] == "replaced":
                counts["replaced"] += 1
                changed_here += 1
            elif fld["decision"] == "demoted":
                counts["demoted"] += 1
                changed_here += 1
            elif fld["reason"] == "floor_blocked":
                counts["floor_blocked"] += 1
            elif fld["reason"] == "ineligible":
                counts["ineligible"] += 1
            else:
                counts["kept"] += 1
        if changed_here:
            totals["cards_with_change"] += 1
            totals["changes"] += changed_here

    return {"totals": totals, "per_field": dict(sorted(per_field.items()))}


def write_rollup(run_out_dir: str, rollup: Dict[str, Any]) -> str:
    os.makedirs(run_out_dir, exist_ok=True)
    path = os.path.join(run_out_dir, "rollup.json")
    with open(path, "w") as f:
        json.dump(rollup, f, indent=2, ensure_ascii=True)
    return path
