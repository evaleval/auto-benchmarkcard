#!/usr/bin/env python3
"""Generate BenchmarkCards for benchmarks outside the EEE datastore and UnitXT.

Reads a JSON list of benchmarks with optional known sources and runs the standard
pipeline for each one. The pipeline uses Docling (for the paper), the HF worker (for
dataset metadata) and the composer to build the card. When paper_url is missing, the
paper resolver searches for one by name.

Input file format (JSON list of objects; paper_url and hf_repo are optional):
    [
      {"name": "bbh", "paper_url": "https://arxiv.org/abs/2210.09261", "hf_repo": "lukaemon/bbh"},
      {"name": "appworld", "paper_url": "https://arxiv.org/abs/2407.18901"}
    ]

Usage:
    python scripts/generate_manual.py --input benchmarks.json --dry-run
    python scripts/generate_manual.py --input benchmarks.json --limit 2 -o test_manual
    python scripts/generate_manual.py --input benchmarks.json -o manual_output
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger("generate_manual")

FAILURE_REASONS = {
    "no_sources": "No paper URL or HF repo available",
    "empty_content": "Card generated but all content fields empty",
    "composer_failed": "Composer returned None or raised exception",
}


def _classify_failure(paper_url, hf_repo, error_str):
    """Classify a failure into a reason category."""
    if not paper_url and not hf_repo:
        return "no_sources"
    if "returned None" in error_str:
        return "composer_failed"
    if "empty" in error_str.lower() or "not specified" in error_str.lower():
        return "empty_content"
    return "composer_failed"


def _make_failure_entry(name, reason, detail, paper_url, hf_repo):
    """Create a structured failure entry."""
    return {
        "benchmark": name,
        "reason": reason,
        "reason_description": FAILURE_REASONS.get(reason, reason),
        "detail": detail,
        "has_paper": bool(paper_url),
        "has_hf": bool(hf_repo),
    }


def load_benchmarks(path):
    """Read the input file into a list of (name, paper_url, hf_repo) tuples."""
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        sys.exit(f"{path}: expected a JSON list of objects")
    benchmarks = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) \
                or not entry["name"].strip():
            sys.exit(f"{path}: entry {i} needs a non-empty string 'name'")
        for key in ("paper_url", "hf_repo"):
            if entry.get(key) is not None and not isinstance(entry[key], str):
                sys.exit(f"{path}: entry {i} field '{key}' must be a string or null")
        benchmarks.append(
            (entry["name"].strip(), entry.get("paper_url") or None, entry.get("hf_repo") or None)
        )
    return benchmarks


def build_pipeline_inputs(name, paper_url, hf_repo):
    """Build pipeline_inputs dict matching the format expected by process_single_benchmark."""
    return {
        "extracted_ids": {
            "hf_repo": hf_repo,
            "paper_url": paper_url,
            "risk_tags": None,
        },
        "hf_repo": hf_repo,
        "eee_metadata": {
            "benchmark_name": name,
            "eee_source_folder": None,
            "source_type": "manual",
            "source_urls": [],
            "eval_library": None,
            "metrics": {},
            "evaluation_summary": {},
            "num_models_evaluated": 0,
            "benchmark_type": "single",
            "appears_in": [],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate BenchmarkCards from manual sources")
    parser.add_argument("--input", "-i", required=True, metavar="PATH",
                        help="JSON list of {name, paper_url?, hf_repo?} objects")
    parser.add_argument("--output", "-o", default="manual_output", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max benchmarks to process")
    parser.add_argument("--dry-run", action="store_true", help="List benchmarks without generating")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--filter", nargs="*", help="Only generate specific benchmarks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    benchmarks = load_benchmarks(args.input)
    if args.filter:
        filter_set = {f.lower() for f in args.filter}
        benchmarks = [(n, p, h) for n, p, h in benchmarks if n.lower() in filter_set]

    if args.limit:
        benchmarks = benchmarks[: args.limit]

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"DRY RUN: {len(benchmarks)} benchmarks")
        print(f"{'=' * 60}")
        for i, (name, paper, hf) in enumerate(benchmarks, 1):
            print(f"  {i:3d}. {name:<35s} paper={'yes' if paper else 'no':>3s}  hf={hf or 'None'}")
        return

    # Imported here so --help and --dry-run do not load the full pipeline.
    from auto_benchmarkcard.eee_workflow import process_single_benchmark
    from auto_benchmarkcard.workflow import setup_logging_suppression

    setup_logging_suppression(debug_mode=args.debug)
    output_dir = Path(args.output).resolve()

    stats = {"total": len(benchmarks), "succeeded": 0, "failed": 0}
    failed_list = []
    start = time.time()

    for i, (name, paper_url, hf_repo) in enumerate(benchmarks, 1):
        logger.info("[%d/%d] Generating: %s", i, len(benchmarks), name)
        t0 = time.time()

        try:
            inputs = build_pipeline_inputs(name, paper_url, hf_repo)
            card = process_single_benchmark(
                benchmark_name=name,
                pipeline_inputs=inputs,
                base_output_path=str(output_dir),
                debug=args.debug,
            )
            dt = time.time() - t0
            if card:
                stats["succeeded"] += 1
                logger.info("OK %s (%.0fs)", name, dt)
            else:
                stats["failed"] += 1
                reason = _classify_failure(paper_url, hf_repo, "returned None")
                logger.warning("FAIL %s - %s (%.0fs)", name, reason, dt)
                failed_list.append(
                    _make_failure_entry(name, reason, "returned None", paper_url, hf_repo))
        except Exception as e:
            dt = time.time() - t0
            stats["failed"] += 1
            reason = _classify_failure(paper_url, hf_repo, str(e))
            logger.error("ERROR %s - %s: %s (%.0fs)", name, reason, e, dt)
            failed_list.append(_make_failure_entry(name, reason, str(e), paper_url, hf_repo))

    duration = time.time() - start
    m, s = divmod(int(duration), 60)

    if failed_list:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "failed_benchmarks.json", "w") as f:
            json.dump(failed_list, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {stats['succeeded']}/{stats['total']} succeeded, "
          f"{stats['failed']} failed ({m}m {s}s)")
    if failed_list:
        reason_counts = Counter(f["reason"] for f in failed_list)
        print("\nFailure breakdown:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count} ({FAILURE_REASONS.get(reason, reason)})")
        print("\nFailed benchmarks:")
        for f in failed_list:
            print(f"  - {f['benchmark']}: {f['reason']} "
                  f"({f['detail'][:80] if f['detail'] else 'N/A'})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
