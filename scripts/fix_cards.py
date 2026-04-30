#!/usr/bin/env python3
"""Post-processing script to fix known quality issues in generated benchmark cards.

Fixes:
1. Remove -1.0 sentinel values from baseline_results text
2. Flag nonsensical score aggregations (Wordle, fibble)
3. Fix Vicuna identity issue (goal describes training, not benchmarking)

Usage:
    python scripts/fix_cards.py cards_for_avijit/
    python scripts/fix_cards.py cards_for_avijit/ --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path


def fix_negative_scores(card: dict, filename: str) -> list[str]:
    """Remove -1.0 sentinel values from baseline_results text."""
    fixes = []
    bc = card.get("benchmark_card", card)
    meth = bc.get("methodology", {})
    br = meth.get("baseline_results", "")

    if not isinstance(br, str) or "-1.0" not in br:
        return fixes

    original = br

    # Fix "ranging from -1.0 to X" → "ranging up to X"
    br = re.sub(r"ranging from -1\.0 to (\d)", r"ranging up to \1", br)
    # Fix "range from -1.0 to X" → "range up to X"
    br = re.sub(r"range from -1\.0 to (\d)", r"range up to \1", br)
    # Fix "range [-1.0, X]" → "range up to X"
    br = re.sub(r"range \[-1\.0,\s*", "range up to ", br)
    br = br.replace("]", "", 1) if "range up to " in br else br
    # Fix "scores range from -1.0 to X with" → "scores range up to X with"
    br = re.sub(r"scores range from -1\.0 to (\d)", r"scores range up to \1", br)

    if br != original:
        meth["baseline_results"] = br
        fixes.append(f"Removed -1.0 sentinel from baseline_results")

    return fixes


NONSENSICAL_SCORE_CARDS = {
    "benchmark_card_Wordle_Arena.json",
    "benchmark_card_fibble_arena_daily.json",
}

def fix_nonsensical_scores(card: dict, filename: str) -> list[str]:
    """Flag specific cards known to have nonsensical score aggregations."""
    fixes = []
    if filename not in NONSENSICAL_SCORE_CARDS:
        return fixes

    bc = card.get("benchmark_card", card)
    meth = bc.get("methodology", {})
    br = meth.get("baseline_results", "")

    if not isinstance(br, str):
        return fixes

    caveat = "\n\nNote: These scores may reflect aggregation across heterogeneous metrics (e.g., win rate, latency, attempt count) and should be interpreted with caution."
    if "should be interpreted with caution" not in br:
        meth["baseline_results"] = br + caveat
        fixes.append(f"Added caveat for nonsensical aggregated scores")

    return fixes


def fix_vicuna_identity(card: dict, filename: str) -> list[str]:
    """Fix Vicuna card goal that describes training instead of benchmarking."""
    fixes = []
    if "Vicuna" not in filename:
        return fixes

    bc = card.get("benchmark_card", card)
    purpose = bc.get("purpose_and_intended_users", {})
    goal = purpose.get("goal", "")

    if "train" in goal.lower() and "unfiltered" in goal.lower():
        purpose["goal"] = "To evaluate language model performance on conversational tasks using the Vicuna evaluation dataset, which contains multi-turn conversations collected from ShareGPT."
        fixes.append(f"Fixed goal from training description to benchmark description")

    return fixes


def process_cards(card_dir: Path, dry_run: bool = False) -> None:
    """Process all cards in directory and apply fixes."""
    fixers = [fix_negative_scores, fix_nonsensical_scores, fix_vicuna_identity]
    total_fixes = 0

    for card_file in sorted(card_dir.glob("benchmark_card_*.json")):
        with open(card_file) as f:
            card = json.load(f)

        all_fixes = []
        for fixer in fixers:
            fixes = fixer(card, card_file.name)
            all_fixes.extend(fixes)

        if all_fixes:
            total_fixes += len(all_fixes)
            print(f"\n{card_file.name}:")
            for fix in all_fixes:
                print(f"  - {fix}")

            if not dry_run:
                with open(card_file, "w") as f:
                    json.dump(card, f, indent=2, ensure_ascii=False)
                print(f"  [SAVED]")
            else:
                print(f"  [DRY RUN - not saved]")

    print(f"\nTotal fixes: {total_fixes}")


def main():
    parser = argparse.ArgumentParser(description="Fix known issues in benchmark cards")
    parser.add_argument("card_dir", help="Directory containing benchmark card JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show fixes without applying")
    args = parser.parse_args()

    card_dir = Path(args.card_dir)
    if not card_dir.exists():
        print(f"Error: {card_dir} does not exist")
        sys.exit(1)

    process_cards(card_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
