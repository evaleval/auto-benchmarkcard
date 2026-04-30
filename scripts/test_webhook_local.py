#!/usr/bin/env python3
"""Local dry-run of the webhook worker flow.

Simulates what the webhook does when triggered:
1. Detects all EEE folders (empty state = all are "new")
2. Downloads EEE data
3. Scans for benchmarks + composites
4. Dedup-filters against existing cards in evaleval/auto-benchmarkcards
5. Prints the list of benchmarks that would be generated

Usage:
    python scripts/test_webhook_local.py                  # dry-run only
    python scripts/test_webhook_local.py --generate       # actually generate + upload
    python scripts/test_webhook_local.py --generate -b "bench1,bench2"  # subset
"""

import argparse
import json
import logging
import os
import sys
import tempfile

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spaces", "benchmarkcard-webhook"))

from dotenv import load_dotenv
load_dotenv()

# Set persistent dir to temp so we don't pollute anything
os.environ.setdefault("PERSISTENT_DIR", "/tmp/webhook_test")

from worker import (
    detect_new_benchmarks,
    _list_existing_cards,
    _build_dedup_filter,
    _download_folders,
    process_new_benchmarks,
)
from auto_benchmarkcard.tools.eee.eee_tool import scan_eee_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_webhook")


def dry_run(benchmarks_filter: list[str] | None = None):
    """Show what would be generated without generating anything."""
    logger.info("=== Phase 1: Detecting EEE folders ===")
    all_folders = detect_new_benchmarks()
    logger.info("Total EEE folders: %d", len(all_folders))

    logger.info("=== Phase 2: Listing existing cards ===")
    existing_cards = _list_existing_cards()
    logger.info("Existing cards: %d", len(existing_cards))

    logger.info("=== Phase 3: Downloading + scanning EEE data ===")
    with tempfile.TemporaryDirectory(prefix="eee_dryrun_") as tmpdir:
        data_path = _download_folders(all_folders, tmpdir)
        scan_result = scan_eee_folder(str(data_path))

        all_names = (
            list(scan_result.benchmarks.keys())
            + list(scan_result.composites.keys())
        )
        logger.info("Scanned: %d benchmarks + %d composites = %d total",
                     len(scan_result.benchmarks), len(scan_result.composites), len(all_names))

        logger.info("=== Phase 4: Dedup filter ===")
        new_benchmarks = _build_dedup_filter(all_names, existing_cards)

        # Apply user filter if given
        if benchmarks_filter:
            filter_set = {b.lower() for b in benchmarks_filter}
            new_benchmarks = [b for b in new_benchmarks if b.lower() in filter_set]

        print("\n" + "=" * 60)
        print(f"NEW BENCHMARKS TO GENERATE: {len(new_benchmarks)}")
        print("=" * 60)

        for i, name in enumerate(sorted(new_benchmarks), 1):
            btype = "composite" if name in scan_result.composites else "single"
            bench = scan_result.benchmarks.get(name)
            hf = "unknown"
            if bench:
                hf = bench.hf_repo or bench.name
            print(f"  {i:3d}. [{btype:9s}] {name} (hf={hf})")

        print(f"\nAlready have cards: {len(all_names) - len(new_benchmarks)}")
        print(f"Existing cards in repo: {len(existing_cards)}")

    return new_benchmarks


def generate(benchmarks_filter: list[str] | None = None):
    """Actually run the worker and generate + upload cards."""
    all_folders = detect_new_benchmarks()

    if benchmarks_filter:
        filter_set = {b.lower() for b in benchmarks_filter}
        all_folders = [f for f in all_folders if f.lower() in filter_set]
        logger.info("Filtering to %d folders: %s", len(all_folders), all_folders)

    logger.info("Starting generation for %d folders...", len(all_folders))
    process_new_benchmarks(all_folders)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Local webhook worker test")
    parser.add_argument("--generate", action="store_true",
                        help="Actually generate and upload cards (default: dry-run only)")
    parser.add_argument("-b", "--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark names to filter")
    args = parser.parse_args()

    bfilter = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None

    if args.generate:
        generate(bfilter)
    else:
        dry_run(bfilter)


if __name__ == "__main__":
    main()
