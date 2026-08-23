#!/usr/bin/env python3
"""Generate BenchmarkCards for benchmarks not in the EEE datastore.

Provides paper_url and hf_repo manually, then runs the standard pipeline.
The pipeline will use Docling (for paper), HF worker (for dataset metadata),
and the composer to create the card.

Usage:
    python scripts/generate_manual.py --dry-run
    python scripts/generate_manual.py --limit 2 -o test_manual
    python scripts/generate_manual.py -o manual_output
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_benchmarkcard.eee_workflow import process_single_benchmark
from auto_benchmarkcard.output import sanitize_benchmark_name
from auto_benchmarkcard.workflow import setup_logging_suppression

logger = logging.getLogger("generate_manual")

FAILURE_REASONS = {
    "no_sources": "No paper URL or HF repo available",
    "empty_content": "Card generated but all content fields empty",
    "composer_failed": "Composer returned None or raised exception",
    "sub_benchmark_no_info": "No sub-benchmark-specific info found in parent paper",
    "parent_failed": "Parent benchmark generation failed",
}


def _classify_failure(name, paper_url, hf_repo, appears_in, error_str):
    """Classify a failure into a reason category."""
    if not paper_url and not hf_repo:
        return "no_sources"
    if "returned None" in error_str:
        return "composer_failed"
    if "empty" in error_str.lower() or "not specified" in error_str.lower():
        return "empty_content"
    return "composer_failed"


def _make_failure_entry(name, reason, detail, paper_url, hf_repo, appears_in):
    """Create a structured failure entry."""
    return {
        "benchmark": name,
        "reason": reason,
        "reason_description": FAILURE_REASONS.get(reason, reason),
        "detail": detail,
        "has_paper": bool(paper_url),
        "has_hf": bool(hf_repo),
        "appears_in": appears_in,
    }


# Benchmarks with their known sources.
# Each entry: (name, paper_url, hf_repo, appears_in)
BENCHMARKS = [
    # Family-level benchmarks
    ("appworld", "https://arxiv.org/abs/2407.18901", None, []),
    ("arc_agi", "https://arxiv.org/abs/2412.04604", "fchollet/ARC", []),
    ("bfcl", None, "gorilla-llm/Berkeley-Function-Calling-Leaderboard", []),
    ("tau_bench_2", "https://arxiv.org/abs/2406.12045", None, []),
    ("rewardbench_2", "https://arxiv.org/abs/2506.01937", "allenai/reward-bench", []),
    ("sciarena", "https://arxiv.org/abs/2507.01001", None, []),
    ("la_leaderboard", "https://arxiv.org/abs/2507.00999", None, []),
    # BFCL sub-benchmarks (all share the BFCL paper)
    ("bfcl_format_sensitivity", None, None, ["bfcl"]),
    ("bfcl_live", None, None, ["bfcl"]),
    ("bfcl_memory", None, None, ["bfcl"]),
    ("bfcl_multi_turn", None, None, ["bfcl"]),
    ("bfcl_non_live", None, None, ["bfcl"]),
    ("bfcl_relevance", None, None, ["bfcl"]),
    ("bfcl_web_search", None, None, ["bfcl"]),
    # ARC AGI sub-benchmarks
    ("arc_agi_v1_public_eval", "https://arxiv.org/abs/2412.04604", "fchollet/ARC", ["arc_agi"]),
    ("arc_agi_v1_semi_private", "https://arxiv.org/abs/2412.04604", "fchollet/ARC", ["arc_agi"]),
    ("arc_agi_v2_private_eval", "https://arxiv.org/abs/2412.04604", None, ["arc_agi"]),
    ("arc_agi_v2_public_eval", "https://arxiv.org/abs/2412.04604", None, ["arc_agi"]),
    ("arc_agi_v2_semi_private", "https://arxiv.org/abs/2412.04604", None, ["arc_agi"]),
    ("arc_agi_v3_semi_private", "https://arxiv.org/abs/2412.04604", None, ["arc_agi"]),
    # Tau Bench 2 sub-benchmarks
    ("tau_bench_2_airline", "https://arxiv.org/abs/2406.12045", None, ["tau_bench_2"]),
    ("tau_bench_2_retail", "https://arxiv.org/abs/2406.12045", None, ["tau_bench_2"]),
    ("tau_bench_2_telecom", "https://arxiv.org/abs/2406.12045", None, ["tau_bench_2"]),
    # Standalone benchmarks with known papers (batch cards were sparse)
    ("bbh", "https://arxiv.org/abs/2210.09261", "lukaemon/bbh", []),
    ("helm_mmlu", "https://arxiv.org/abs/2009.03300", "cais/mmlu", []),
    ("apex_v1", "https://arxiv.org/abs/2601.14242", "Mercor/APEX-v1", []),
    ("ace", "https://arxiv.org/abs/2407.06068", None, []),
    # LiveCodeBenchPro parent + difficulty sub-benchmarks
    ("livecodebenchpro", "https://arxiv.org/abs/2403.07974", None, []),
    ("easy_problems", "https://arxiv.org/abs/2403.07974", None, ["livecodebenchpro"]),
    ("medium_problems", "https://arxiv.org/abs/2403.07974", None, ["livecodebenchpro"]),
    ("hard_problems", "https://arxiv.org/abs/2403.07974", None, ["livecodebenchpro"]),
    # Reward Bench 2 sub-benchmarks
    ("rewardbench_2_factuality", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
    ("rewardbench_2_focus", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
    ("rewardbench_2_math", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
    ("rewardbench_2_precise_if", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
    ("rewardbench_2_safety", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
    ("rewardbench_2_ties", "https://arxiv.org/abs/2506.01937", None, ["rewardbench_2"]),
]


def build_pipeline_inputs(name, paper_url, hf_repo, appears_in):
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
            "appears_in": appears_in,
        },
    }


def _get_parent_paper_url(parent_name):
    """Look up the paper URL for a parent benchmark from the BENCHMARKS list."""
    for name, paper_url, _, appears_in in BENCHMARKS:
        if name == parent_name:
            return paper_url
    return None


def _load_docling_output(benchmark_name, output_dir):
    """Load the most recent docling output for a benchmark from saved files."""
    import glob as glob_mod
    safe_name = sanitize_benchmark_name(benchmark_name)
    pattern = str(output_dir / "output" / f"{safe_name}_*" / "tool_output" / "docling" / "*.json")
    matches = sorted(glob_mod.glob(pattern))
    if not matches:
        return None
    # Use the most recent (last sorted by timestamp in dir name)
    try:
        with open(matches[-1]) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load docling output for %s: %s", benchmark_name, e)
        return None


def _load_parent_card(benchmark_name, output_dir):
    """Load the most recent benchmark card for a benchmark from saved files."""
    import glob as glob_mod
    safe_name = sanitize_benchmark_name(benchmark_name)
    pattern = str(output_dir / "output" / f"{safe_name}_*" / "benchmarkcard" / f"benchmark_card_{safe_name}.json")
    matches = sorted(glob_mod.glob(pattern))
    if not matches:
        return None
    try:
        with open(matches[-1]) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load parent card for %s: %s", benchmark_name, e)
        return None


def _load_hf_readme(benchmark_name, output_dir):
    """Load the HF worker README output for a benchmark from saved files."""
    import glob as glob_mod
    safe_name = sanitize_benchmark_name(benchmark_name)
    pattern = str(output_dir / "output" / f"{safe_name}_*" / "tool_output" / "hf_worker" / "*.json")
    matches = sorted(glob_mod.glob(pattern))
    if not matches:
        return ""
    try:
        with open(matches[-1]) as f:
            data = json.load(f)
        # HF worker output may have readme in various locations
        if isinstance(data, dict):
            readme = data.get("readme_markdown", "") or data.get("readme", "")
            if not readme and "dataset_info" in data:
                readme = data["dataset_info"].get("readme_markdown", "")
            return readme or ""
        return ""
    except Exception as e:
        logger.warning("Failed to load HF README for %s: %s", benchmark_name, e)
        return ""


def _get_parent_hf_repo(parent_name):
    """Look up the HF repo for a parent benchmark from the BENCHMARKS list."""
    for name, _, hf_repo, _ in BENCHMARKS:
        if name == parent_name:
            return hf_repo
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate BenchmarkCards from manual sources")
    parser.add_argument("--output", "-o", default="manual_output", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max benchmarks to process")
    parser.add_argument("--dry-run", action="store_true", help="List benchmarks without generating")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--filter", nargs="*", help="Only generate specific benchmarks")
    parser.add_argument("--subs-only", action="store_true", help="Skip Pass 1, only regenerate sub-benchmarks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    benchmarks = BENCHMARKS
    if args.filter:
        filter_set = {f.lower() for f in args.filter}
        benchmarks = [(n, p, h, a) for n, p, h, a in benchmarks if n.lower() in filter_set]

    if args.limit:
        benchmarks = benchmarks[: args.limit]

    if args.dry_run:
        print(f"\n{'=' * 60}")
        print(f"DRY RUN — {len(benchmarks)} benchmarks:")
        print(f"{'=' * 60}")
        parents = [(n, p, h, a) for n, p, h, a in benchmarks if not a]
        subs = [(n, p, h, a) for n, p, h, a in benchmarks if a]
        print(f"\n  Parents ({len(parents)}):")
        for i, (name, paper, hf, appears) in enumerate(parents, 1):
            print(f"    {i:3d}. {name:<35s} paper={'yes' if paper else 'no':>3s}  hf={hf or 'None':<50s}")
        print(f"\n  Sub-benchmarks ({len(subs)}):")
        for i, (name, paper, hf, appears) in enumerate(subs, 1):
            print(f"    {i:3d}. {name:<35s} paper={'yes' if paper else 'no':>3s}  parent={appears}")
        return

    setup_logging_suppression(debug_mode=args.debug)
    output_dir = Path(args.output).resolve()

    # Partition into parents and sub-benchmarks
    parents = [(n, p, h, a) for n, p, h, a in benchmarks if not a]
    subs = [(n, p, h, a) for n, p, h, a in benchmarks if a]

    stats = {"total": len(benchmarks), "succeeded": 0, "failed": 0}
    failed_list = []
    parent_results = {}  # name -> {"card": card_dict}
    start = time.time()

    # Pass 1: generate parent benchmark cards
    if getattr(args, 'subs_only', False):
        print(f"\n{'=' * 60}")
        print(f"PASS 1: SKIPPED (--subs-only). Loading parent cards from disk.")
        print(f"{'=' * 60}")
        for name, paper_url, hf_repo, appears_in in parents:
            parent_card = _load_parent_card(name, output_dir)
            if parent_card:
                parent_results[name] = {"card": parent_card}
                stats["succeeded"] += 1
                logger.info("Loaded parent card from disk: %s", name)
            else:
                logger.warning("Parent card not found on disk: %s", name)
    else:
        print(f"\n{'=' * 60}")
        print(f"PASS 1: Generating {len(parents)} parent benchmarks")
        print(f"{'=' * 60}")

        for i, (name, paper_url, hf_repo, appears_in) in enumerate(parents, 1):
            logger.info("[%d/%d] Generating parent: %s", i, len(parents), name)
            t0 = time.time()

            try:
                inputs = build_pipeline_inputs(name, paper_url, hf_repo, appears_in)
                card = process_single_benchmark(
                    benchmark_name=name,
                    pipeline_inputs=inputs,
                    base_output_path=str(output_dir),
                    debug=args.debug,
                )
                dt = time.time() - t0
                if card:
                    stats["succeeded"] += 1
                    parent_results[name] = {"card": card}
                    logger.info("OK %s (%.0fs)", name, dt)
                else:
                    stats["failed"] += 1
                    reason = _classify_failure(name, paper_url, hf_repo, appears_in, "returned None")
                    logger.warning("FAIL %s - %s (%.0fs)", name, reason, dt)
                    failed_list.append(_make_failure_entry(name, reason, "returned None", paper_url, hf_repo, appears_in))
            except Exception as e:
                dt = time.time() - t0
                stats["failed"] += 1
                reason = _classify_failure(name, paper_url, hf_repo, appears_in, str(e))
                logger.error("ERROR %s - %s: %s (%.0fs)", name, reason, e, dt)
                failed_list.append(_make_failure_entry(name, reason, str(e), paper_url, hf_repo, appears_in))

    # Sub-benchmark cards are retired: variants live on the parent card per the
    # 2026-05-06 granularity decision. The old parent-copy inheritance path bypassed
    # quote verification and produced wrong-scope cards by construction.
    if subs:
        print(f"\nSKIPPING {len(subs)} sub-benchmarks: separate sub-cards are retired "
              "(variants belong on the parent card).")
        for name, _, _, appears_in in subs:
            stats["skipped"] = stats.get("skipped", 0) + 1
            logger.info("SKIP %s - sub-benchmark of %s (sub-cards retired)", name, appears_in)

    duration = time.time() - start
    m, s = divmod(int(duration), 60)

    if failed_list:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "failed_benchmarks.json", "w") as f:
            json.dump(failed_list, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {stats['succeeded']}/{stats['total']} succeeded, {stats['failed']} failed ({m}m {s}s)")
    if failed_list:
        reason_counts = Counter(f["reason"] for f in failed_list)
        print(f"\nFailure breakdown:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count} — {FAILURE_REASONS.get(reason, reason)}")
        print(f"\nFailed benchmarks:")
        for f in failed_list:
            print(f"  - {f['benchmark']}: {f['reason']} ({f['detail'][:80] if f['detail'] else 'N/A'})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
