#!/usr/bin/env python3
"""One-time script to rename existing cards to canonical filenames.

Uses sanitize_benchmark_name() to ensure all card filenames follow the same
normalization rules. Also resolves via Entity Registry when available.

Usage:
    python scripts/rename_cards_to_canonical.py          # dry-run (default)
    python scripts/rename_cards_to_canonical.py --apply   # actually rename
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auto_benchmarkcard.output import sanitize_benchmark_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CARDS_REPO = "evaleval/auto-benchmarkcards"
ENTITY_REGISTRY_URL = "https://evaleval-entity-registry.hf.space/api/v1"


def resolve_canonical_id(name: str) -> str | None:
    """Resolve a benchmark name to canonical_id via Entity Registry."""
    try:
        resp = requests.post(
            f"{ENTITY_REGISTRY_URL}/resolve",
            json={"raw_value": name, "entity_type": "benchmark"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("canonical_id")
    except Exception as e:
        logger.debug("Entity Registry failed for '%s': %s", name, e)
        return None


def compute_canonical_name(current_name: str) -> str:
    """Compute the canonical filename for a card.

    Uses sanitize_benchmark_name for consistent normalization.
    Entity Registry is not used here to avoid collisions (e.g., helm_mmlu -> mmlu).
    """
    return sanitize_benchmark_name(current_name)


def main():
    parser = argparse.ArgumentParser(description="Rename cards to canonical filenames")
    parser.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    args = parser.parse_args()

    api = HfApi()

    # List all card files
    all_files = api.list_repo_files(CARDS_REPO, repo_type="dataset")
    card_files = [f for f in all_files if f.startswith("cards/") and f.endswith(".json")]
    logger.info("Found %d card files", len(card_files))

    renames = []
    already_canonical = []
    collisions = []
    seen_canonical: dict[str, str] = {}

    for card_path in sorted(card_files):
        current_name = card_path[len("cards/"):-len(".json")]
        canonical = compute_canonical_name(current_name)

        if canonical == current_name:
            already_canonical.append(current_name)
            seen_canonical[canonical] = current_name
            continue

        # Skip collisions
        if canonical in seen_canonical:
            collisions.append({
                "current_name": current_name,
                "canonical": canonical,
                "conflicts_with": seen_canonical[canonical],
            })
            continue

        seen_canonical[canonical] = current_name
        renames.append({
            "current": card_path,
            "new": f"cards/{canonical}.json",
            "current_name": current_name,
            "canonical": canonical,
        })

    # Report
    logger.info("")
    logger.info("=== Results ===")
    logger.info("Already canonical: %d", len(already_canonical))
    logger.info("Need rename: %d", len(renames))
    logger.info("Collisions (skipped): %d", len(collisions))

    if collisions:
        for c in collisions:
            logger.info("  COLLISION: %s -> %s (conflicts with %s)",
                        c["current_name"], c["canonical"], c["conflicts_with"])

    for r in renames:
        logger.info("  %s -> %s", r["current_name"], r["canonical"])

    if not renames:
        logger.info("Nothing to rename.")
        return

    if not args.apply:
        logger.info("")
        logger.info("Dry-run mode. Use --apply to actually rename.")
        return

    # Apply renames
    logger.info("")
    logger.info("Applying %d renames...", len(renames))

    for r in renames:
        try:
            card_path = api.hf_hub_download(
                CARDS_REPO, r["current"], repo_type="dataset"
            )
            card_data = json.loads(Path(card_path).read_text())

            # Inject canonical_id into card_info
            inner = card_data.get("benchmark_card", card_data)
            info = inner.get("card_info", {})
            info["canonical_id"] = r["canonical"]
            inner["card_info"] = info

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(card_data, f, indent=2)
                tmp_path = f.name

            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=r["new"],
                repo_id=CARDS_REPO,
                repo_type="dataset",
                commit_message=f"Rename card: {r['current_name']} -> {r['canonical']}",
            )

            api.delete_file(
                path_in_repo=r["current"],
                repo_id=CARDS_REPO,
                repo_type="dataset",
                commit_message=f"Remove old card: {r['current_name']} (renamed to {r['canonical']})",
            )

            logger.info("Renamed: %s -> %s", r["current_name"], r["canonical"])

        except Exception:
            logger.exception("Failed to rename %s", r["current_name"])

    logger.info("Done.")


if __name__ == "__main__":
    main()
