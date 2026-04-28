"""Compare paper resolution results across benchmarks with full LLM verification.

Bypasses cache to test fresh resolution from OpenAlex + Semantic Scholar.
Uses Entity Registry + LLM pre-query for metadata, batch verification.
Outputs a markdown table and JSON log for documentation.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_benchmarkcard.tools.eee.paper_resolver import (
    KNOWN_PAPERS,
    _build_search_queries,
    _lookup_display_name,
    _query_benchmark_metadata,
    _normalize_openalex_paper,
    _normalize_s2_paper,
    _prefilter_candidates,
    _search_openalex,
    _search_semantic_scholar,
    _batch_verify_with_llm,
    _extract_paper_url,
    _normalize_cache_key,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = [
    # Well-known benchmarks that should always resolve
    "gsm8k", "humaneval", "bfcl", "arc_agi", "winogrande",
    "drop", "triviaqa", "mbpp", "truthfulqa",
    # Additional test cases
    "bbh", "mmlu", "gpqa", "hellaswag", "musr", "ifeval",
]

SOURCES = [
    ("openalex", _search_openalex, _normalize_openalex_paper),
    ("semantic_scholar", _search_semantic_scholar, _normalize_s2_paper),
]


def resolve_fresh(suite_name: str) -> dict:
    """Run full resolution pipeline for one benchmark, bypassing cache."""
    cache_key = _normalize_cache_key(suite_name)

    result = {
        "benchmark": suite_name,
        "full_name": None,
        "domain": None,
        "queries": [],
        "resolved_url": None,
        "resolved_source": None,
        "confidence": None,
        "reasoning": None,
        "candidates_checked": 0,
    }

    # Check known-papers table first
    known = KNOWN_PAPERS.get(cache_key)
    if known:
        result["resolved_url"] = known
        result["resolved_source"] = "known_papers"
        result["confidence"] = 1.0
        result["reasoning"] = "Manually verified in KNOWN_PAPERS table"
        return result

    # Entity Registry
    full_name = _lookup_display_name(suite_name)

    # LLM pre-query for metadata
    metadata = _query_benchmark_metadata(suite_name, full_name=full_name)
    if not full_name:
        full_name = metadata.get("full_name")

    result["full_name"] = full_name
    domain = metadata.get("domain")
    result["domain"] = domain

    queries = _build_search_queries(suite_name, full_name, metadata)
    llm_full_name = metadata.get("full_name")
    if llm_full_name and llm_full_name.lower() not in {q.lower() for q in queries}:
        queries.insert(0, llm_full_name)
    result["queries"] = queries

    search_name = full_name or suite_name.replace("_", " ")
    references = [r for r in [full_name, metadata.get("paper_title")] if r]

    # Collect all candidates across queries and sources
    all_candidates = []
    seen_titles = set()

    for query in queries:
        for source_name, search_fn, normalize_fn in SOURCES:
            logger.info("  [%s] query: '%s'", source_name, query)
            raw = search_fn(query)
            if not raw:
                continue

            candidates = [normalize_fn(p) for p in raw]
            filtered = _prefilter_candidates(candidates, search_name, references, domain)
            logger.info("  [%s] %d results, %d after prefilter", source_name, len(candidates), len(filtered))

            for paper in filtered:
                title_key = (paper.get("title") or "").lower().strip()
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    paper["_query_used"] = query
                    paper["_source"] = source_name
                    all_candidates.append(paper)

    result["candidates_checked"] = len(all_candidates)

    # Batch verify
    if all_candidates:
        verify_batch = all_candidates[:10]
        batch_result = _batch_verify_with_llm(
            verify_batch, suite_name, [], [], None,
            full_name=full_name, domain=domain,
        )

        match_idx = batch_result.get("match_index")
        confidence = batch_result.get("confidence", 0.0)
        if match_idx != "none" and isinstance(match_idx, int) and 0 <= match_idx < len(verify_batch):
            if confidence >= 0.7:
                matched = verify_batch[match_idx]
                result["resolved_url"] = _extract_paper_url(matched)
                result["resolved_source"] = matched.get("_source", "unknown")
                result["confidence"] = confidence
                result["reasoning"] = batch_result.get("reasoning", "")

    return result


def main():
    print(f"\nPaper Resolver Comparison Test — {len(BENCHMARKS)} benchmarks")
    print("Known-papers + Entity Registry + LLM pre-query + batch verification")
    print("=" * 70)

    results = []
    for i, bm in enumerate(BENCHMARKS, 1):
        print(f"\n[{i}/{len(BENCHMARKS)}] {bm}")
        r = resolve_fresh(bm)
        results.append(r)
        status = r["resolved_url"] or "NOT FOUND"
        print(f"  -> {status}")
        if r["confidence"]:
            print(f"     confidence: {r['confidence']}, source: {r['resolved_source']}")

    # Print markdown table
    print("\n\n## Results\n")
    print("| # | Benchmark | Full Name | Source | URL | Confidence |")
    print("|---|-----------|-----------|--------|-----|------------|")
    for i, r in enumerate(results, 1):
        name = r["full_name"] or "-"
        source = r["resolved_source"] or "-"
        url = r["resolved_url"] or "not found"
        if len(url) > 50:
            url = url[:50] + "..."
        conf = f"{r['confidence']:.1f}" if r["confidence"] else "-"
        print(f"| {i} | {r['benchmark']} | {name} | {source} | {url} | {conf} |")

    resolved = sum(1 for r in results if r["resolved_url"])
    print(f"\n**Resolution rate: {resolved}/{len(results)} ({100*resolved/len(results):.0f}%)**\n")

    # Save JSON log
    out_path = Path(__file__).parent / "paper_resolver_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
