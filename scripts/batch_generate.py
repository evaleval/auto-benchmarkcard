#!/usr/bin/env python3
"""Batch generate BenchmarkCards for all missing single benchmarks.

Reads the evaleval/card_backend README to find non-checkmarked benchmarks,
cross-references with the EEE datastore to filter out composites, and
generates cards for each missing single benchmark.

Usage:
    python scripts/batch_generate.py --dry-run              # list what would be generated
    python scripts/batch_generate.py --limit 2 -o test_out  # test with 2 benchmarks
    python scripts/batch_generate.py -o ./batch_output      # full overnight run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download, snapshot_download
from rapidfuzz import fuzz

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_benchmarkcard.eee_workflow import process_single_benchmark
from auto_benchmarkcard.output import sanitize_benchmark_name
from auto_benchmarkcard.tools.eee.eee_tool import (
    EEEScanResult,
    _normalize_benchmark_name,
    eee_to_pipeline_inputs,
    scan_eee_folder,
)
from auto_benchmarkcard.workflow import setup_logging_suppression

CARD_BACKEND_REPO = "evaleval/card_backend"
EEE_DATASTORE_REPO = "evaleval/EEE_datastore"
LOG_FILENAME = "run_log.txt"

# Aggregate-level entries in the README that are composite summaries, not real benchmarks
SKIP_AGGREGATE_NAMES = {
    "capabilities", "classic", "instruct", "lite", "overall", "mean",
}

logger = logging.getLogger("batch_generate")

FAILURE_REASONS = {
    "no_sources": "No paper URL or HF repo available",
    "empty_content": "Card generated but all content fields empty",
    "composer_failed": "Composer returned None or raised exception",
}



# Step 1: Parse missing benchmarks from card_backend README


def parse_missing_benchmarks() -> List[str]:
    """Download card_backend README and extract non-checkmarked benchmark names."""
    readme_path = hf_hub_download(
        CARD_BACKEND_REPO, "README.md", repo_type="dataset"
    )
    names: List[str] = []
    for line in Path(readme_path).read_text().splitlines():
        match = re.match(r"^\s*-\s*\[ \]\s*(.+)$", line)
        if match:
            name = match.group(1).strip()
            # Strip markdown bold markers (leading/trailing **)
            name = re.sub(r"^\*\*|\*\*$", "", name).strip()
            names.append(name)
    logger.info("Parsed %d non-checkmarked benchmarks from card_backend README", len(names))
    return names



# Step 2: Download EEE datastore


def download_eee_datastore() -> Path:
    """Download the EEE datastore via HF hub (cached). Returns path to data/ dir."""
    logger.info("Downloading EEE datastore (using HF cache)...")
    local = snapshot_download(
        EEE_DATASTORE_REPO,
        repo_type="dataset",
        allow_patterns=["data/**/*.json"],
    )
    data_path = Path(local) / "data"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected data/ directory at {data_path}")
    logger.info("EEE datastore available at: %s", data_path)
    return data_path



# Step 3: Match README names to EEE benchmark names


def match_benchmarks(
    readme_names: List[str],
    scan_result: EEEScanResult,
) -> List[Tuple[str, str]]:
    """Match README display names to EEE benchmark keys.

    Returns list of (readme_name, eee_key) tuples for single benchmarks only.
    Skips composites and aggregate-level entries.
    """
    # Build normalized lookup: norm_name -> eee_key
    eee_lookup: Dict[str, str] = {}
    for key in scan_result.benchmarks:
        norm = _normalize_benchmark_name(key)
        eee_lookup[norm] = key

    # Composite names to skip
    composite_norms: Set[str] = set()
    for folder in scan_result.composites:
        composite_norms.add(_normalize_benchmark_name(folder))

    matched: List[Tuple[str, str]] = []
    unmatched: List[str] = []

    for readme_name in readme_names:
        norm = _normalize_benchmark_name(readme_name)

        # Skip aggregate-level entries
        if norm in SKIP_AGGREGATE_NAMES:
            continue

        # Skip composites
        if norm in composite_norms:
            logger.debug("Skipping composite: %s", readme_name)
            continue

        # Exact match
        if norm in eee_lookup:
            matched.append((readme_name, eee_lookup[norm]))
            continue

        # Fuzzy match
        best_score = 0
        best_key = None
        for eee_norm, eee_key in eee_lookup.items():
            score = fuzz.ratio(norm, eee_norm)
            if score > best_score:
                best_score = score
                best_key = eee_key
        if best_score >= 80 and best_key:
            logger.info("Fuzzy matched: '%s' -> '%s' (score=%d)", readme_name, best_key, best_score)
            matched.append((readme_name, best_key))
            continue

        unmatched.append(readme_name)

    if unmatched:
        logger.warning(
            "%d README entries could not be matched to EEE benchmarks: %s",
            len(unmatched), unmatched,
        )

    # Deduplicate by eee_key (same benchmark may appear under different README names)
    seen: Set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for readme_name, eee_key in matched:
        if eee_key not in seen:
            seen.add(eee_key)
            deduped.append((readme_name, eee_key))

    logger.info("Matched %d single benchmarks to generate", len(deduped))
    return deduped



# Step 4: Check already generated (resumability)


def already_generated(benchmark_name: str, output_dir: Path) -> bool:
    """Check if a benchmark card has already been generated in output_dir."""
    safe = sanitize_benchmark_name(benchmark_name)
    output_base = output_dir / "output"
    if not output_base.exists():
        return False
    for d in output_base.iterdir():
        if d.is_dir() and d.name.startswith(f"{safe}_"):
            card_dir = d / "benchmarkcard"
            if card_dir.exists() and any(card_dir.glob("*.json")):
                return True
    return False



# Step 5-8: Main batch runner


def setup_file_logging(output_dir: Path, debug: bool = False) -> None:
    """Configure dual logging: console + file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / LOG_FILENAME

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)


def run_batch(
    output_dir: Path,
    limit: Optional[int] = None,
    dry_run: bool = False,
    debug: bool = False,
) -> None:
    """Run the full batch generation pipeline."""
    setup_file_logging(output_dir, debug)
    setup_logging_suppression(debug_mode=debug)

    logger.info("=" * 60)
    logger.info("BATCH GENERATE - Starting")
    logger.info("=" * 60)

    # Download data
    readme_names = parse_missing_benchmarks()
    eee_data_path = download_eee_datastore()

    # Scan EEE datastore
    logger.info("Scanning EEE datastore...")
    scan_result = scan_eee_folder(str(eee_data_path))
    logger.info(
        "Scan complete: %d benchmarks, %d composites",
        len(scan_result.benchmarks), len(scan_result.composites),
    )

    # Match
    matched = match_benchmarks(readme_names, scan_result)

    if limit:
        matched = matched[:limit]
        logger.info("Limited to %d benchmarks", limit)

    # Build appears_in map
    appears_in_map: Dict[str, List[str]] = defaultdict(list)
    for folder, comp in scan_result.composites.items():
        for sub in comp.sub_benchmarks:
            appears_in_map[sub].append(folder)

    # Dry run: just print the list
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(matched)} benchmarks would be generated:")
        print(f"{'='*60}")
        for i, (readme_name, eee_key) in enumerate(matched, 1):
            bench = scan_result.benchmarks.get(eee_key)
            hf = bench.hf_repo if bench else "?"
            appears = appears_in_map.get(eee_key, [])
            skip = " [SKIP - already exists]" if already_generated(eee_key, output_dir) else ""
            print(f"  {i:3d}. {eee_key:<40s} hf={hf or 'None':<30s} appears_in={appears}{skip}")
        return

    # Generate cards
    stats: Dict[str, Any] = {"total": len(matched), "succeeded": 0, "failed": 0, "skipped": 0}
    failed_list: List[Dict[str, str]] = []
    batch_start = time.time()

    for i, (readme_name, eee_key) in enumerate(matched, 1):
        # Resumability check
        if already_generated(eee_key, output_dir):
            logger.info("[%d/%d] SKIP %s (already exists)", i, len(matched), eee_key)
            stats["skipped"] += 1
            continue

        logger.info("[%d/%d] Generating: %s", i, len(matched), eee_key)
        start = time.time()

        try:
            bench = scan_result.benchmarks[eee_key]
            appears_in = appears_in_map.get(eee_key, [])
            inputs = eee_to_pipeline_inputs(bench, "single", appears_in)
            card = process_single_benchmark(
                benchmark_name=eee_key,
                pipeline_inputs=inputs,
                base_output_path=str(output_dir),
                debug=debug,
            )
            duration = time.time() - start

            if card:
                stats["succeeded"] += 1
                logger.info("OK %s (%.0fs)", eee_key, duration)
            else:
                stats["failed"] += 1
                hf = bench.hf_repo if bench else None
                reason = "composer_failed"
                logger.warning("FAIL %s - %s (%.0fs)", eee_key, reason, duration)
                failed_list.append({
                    "benchmark": eee_key,
                    "reason": reason,
                    "reason_description": FAILURE_REASONS.get(reason, reason),
                    "detail": "returned None",
                    "has_hf": bool(hf),
                    "appears_in": appears_in,
                })

        except Exception as e:
            duration = time.time() - start
            stats["failed"] += 1
            reason = "composer_failed"
            logger.error("ERROR %s - %s: %s (%.0fs)", eee_key, reason, e, duration)
            failed_list.append({
                "benchmark": eee_key,
                "reason": reason,
                "reason_description": FAILURE_REASONS.get(reason, reason),
                "detail": str(e),
                "has_hf": bool(bench.hf_repo if bench else None),
                "appears_in": appears_in,
            })

        # Progress update every 10 benchmarks
        if i % 10 == 0:
            elapsed = time.time() - batch_start
            attempted = stats["succeeded"] + stats["failed"]
            rate = (stats["succeeded"] / attempted * 100) if attempted > 0 else 0
            logger.info(
                "Progress: %d/%d done, %d succeeded, %d failed (%.0f%%), elapsed %.0fm",
                i, len(matched), stats["succeeded"], stats["failed"], rate, elapsed / 60,
            )

    batch_duration = time.time() - batch_start

    # Save failed benchmarks
    if failed_list:
        failed_path = output_dir / "failed_benchmarks.json"
        with open(failed_path, "w") as f:
            json.dump(failed_list, f, indent=2)
        logger.info("Failed benchmarks saved to: %s", failed_path)

    # Print summary
    hours, remainder = divmod(int(batch_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"

    logger.info("=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 60)
    logger.info("Total: %d", stats["total"])
    logger.info("Succeeded: %d", stats["succeeded"])
    logger.info("Failed: %d", stats["failed"])
    logger.info("Skipped: %d", stats["skipped"])
    logger.info("Duration: %s", duration_str)
    if failed_list:
        reason_counts = Counter(f["reason"] for f in failed_list)
        logger.info("Failure breakdown:")
        for reason, count in reason_counts.most_common():
            logger.info("  %s: %d — %s", reason, count, FAILURE_REASONS.get(reason, reason))
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate BenchmarkCards for missing benchmarks"
    )
    parser.add_argument("--output", "-o", default="batch_output", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max benchmarks to process")
    parser.add_argument("--dry-run", action="store_true", help="List benchmarks without generating")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    output_dir = Path(args.output).resolve()

    try:
        run_batch(output_dir, limit=args.limit, dry_run=args.dry_run, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Interrupted by user — partial results saved")
        sys.exit(1)


if __name__ == "__main__":
    main()
