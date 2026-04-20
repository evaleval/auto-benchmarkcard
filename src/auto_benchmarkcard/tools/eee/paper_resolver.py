"""Resolve paper URLs for benchmark suites via OpenAlex + LLM verification.

For benchmarks without a known paper URL (especially composite suites like HELM),
searches OpenAlex for candidate papers and uses an LLM to verify the match
by comparing the paper's abstract against EEE metadata (sub-benchmarks, metrics).

Results are cached in .paper_cache.json to avoid repeated API calls.
Each verification decision is logged to output/<benchmark>/paper-verification.json
for auditability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rapidfuzz import fuzz

from auto_benchmarkcard.config import Config, get_llm_handler

logger = logging.getLogger(__name__)

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_MAILTO = "aris.hofmann@ibm.com"
OPENALEX_LIMIT = 5

CACHE_FILE = Path(__file__).resolve().parents[4] / ".paper_cache.json"

VERIFICATION_PROMPT = """You are verifying whether a paper is the ORIGINAL paper that INTRODUCES a benchmark suite.

Suite name: {suite_name}
Sub-benchmarks in this suite: {sub_benchmark_list}
Metrics: {metrics_list}
Evaluation library: {eval_library}

Paper candidate:
  Title: {paper_title}
  Abstract: {paper_abstract}
  Year: {year}

IMPORTANT: You must determine if this paper INTRODUCES or PROPOSES the benchmark itself.
A paper that merely USES or EVALUATES ON a benchmark is NOT a match.
For example, if a model paper says "we evaluate on MMLU", that does NOT match MMLU — the match is the paper that created MMLU.

Be careful with versioned suites (e.g., HELM vs HELM 2, BIG-bench vs BIG-bench Hard).

Respond ONLY with JSON (no markdown fences): {{"match": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""


def _load_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning("Failed to write paper cache: %s", e)


def _search_openalex(query: str) -> List[Dict[str, Any]]:
    """Search OpenAlex for papers matching a query."""
    try:
        resp = requests.get(
            OPENALEX_SEARCH_URL,
            params={
                "search": query,
                "per_page": OPENALEX_LIMIT,
                "mailto": OPENALEX_MAILTO,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        logger.warning("OpenAlex search failed for '%s': %s", query, e)
        return []


def _normalize_openalex_paper(work: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAlex work to a normalized paper dict."""
    # Extract abstract from inverted index if available
    abstract = ""
    inv_abstract = work.get("abstract_inverted_index")
    if inv_abstract:
        # Reconstruct abstract from inverted index
        positions = []
        for word, idxs in inv_abstract.items():
            for idx in idxs:
                positions.append((idx, word))
        positions.sort()
        abstract = " ".join(w for _, w in positions)

    # Extract IDs
    ids = work.get("ids", {})
    openalex_id = ids.get("openalex", "")
    doi = ids.get("doi", "")

    # Extract arxiv from locations or primary_location
    arxiv_id = None
    for location in work.get("locations", []):
        landing = location.get("landing_page_url") or ""
        if "arxiv.org" in landing:
            # Extract ID from URL like https://arxiv.org/abs/2211.09110
            parts = landing.rstrip("/").split("/")
            if parts:
                arxiv_id = parts[-1]
            break

    return {
        "title": work.get("title", ""),
        "abstract": abstract,
        "year": work.get("publication_year"),
        "citationCount": work.get("cited_by_count", 0),
        "openalex_id": openalex_id,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": doi or work.get("primary_location", {}).get("landing_page_url", ""),
    }


def _prefilter_candidates(
    candidates: List[Dict[str, Any]], suite_name: str,
) -> List[Dict[str, Any]]:
    """Filter candidates by title similarity and citation count."""
    filtered = []
    for paper in candidates:
        title = paper.get("title", "")
        sim = fuzz.partial_ratio(suite_name.lower(), title.lower())
        citations = paper.get("citationCount") or 0
        if sim >= 50 or citations >= 100:
            paper["_title_similarity"] = sim
            filtered.append(paper)
    return filtered


def _verify_with_llm(
    paper: Dict[str, Any],
    suite_name: str,
    sub_benchmarks: List[str],
    metrics: List[str],
    eval_library: str,
) -> Dict[str, Any]:
    """Use LLM to verify whether a paper matches a benchmark suite."""
    try:
        llm = get_llm_handler(Config.COMPOSER_MODEL)
        prompt = VERIFICATION_PROMPT.format(
            suite_name=suite_name,
            sub_benchmark_list=", ".join(sub_benchmarks[:20]),
            metrics_list=", ".join(metrics[:10]),
            eval_library=eval_library or "unknown",
            paper_title=paper.get("title", ""),
            paper_abstract=paper.get("abstract", "No abstract available"),
            year=paper.get("year", "unknown"),
        )

        text = llm.generate(prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return result

    except Exception as e:
        logger.warning("LLM verification failed: %s", e)
        return {"match": False, "confidence": 0.0, "reasoning": f"LLM error: {e}"}


def _extract_paper_url(paper: Dict[str, Any]) -> Optional[str]:
    """Extract the best URL for a paper (prefer arxiv)."""
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    doi = paper.get("doi")
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"
    return paper.get("url") or None


def _build_search_queries(suite_name: str, full_name: Optional[str] = None) -> List[str]:
    """Build a list of search queries to try, from most specific to broadest."""
    queries = []

    # If we have a human-readable full name, that's the best query
    if full_name and full_name.lower() != suite_name.lower():
        queries.append(full_name)
        queries.append(f"{full_name} benchmark")

    # Clean up suite name (underscores to spaces)
    search_name = suite_name.replace("_", " ")

    # Strip harness prefixes but keep them as context
    stripped = None
    for prefix in ("helm ", "hf ", "hfopenllm "):
        if search_name.startswith(prefix):
            stripped = search_name[len(prefix):]
            break

    if stripped and len(stripped.split()) <= 1:
        queries.append(f"{search_name} benchmark")
    else:
        queries.append(f"{search_name} benchmark evaluation")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    return unique


def resolve_paper(
    suite_name: str,
    sub_benchmarks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    eval_library: Optional[str] = None,
    full_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the paper URL for a benchmark suite.

    Tries multiple search queries against OpenAlex, pre-filters by title
    similarity, then verifies each candidate with an LLM. Results are cached.

    Args:
        suite_name: Short name / folder name (e.g., "helm_capabilities", "bbh").
        sub_benchmarks: List of sub-benchmark names for LLM context.
        metrics: List of metric names for LLM context.
        eval_library: Evaluation library name for LLM context.
        full_name: Human-readable benchmark name (e.g., "BIG-Bench Hard").
            Significantly improves search quality when available.
        output_dir: Directory to save paper-verification.json for traceability.

    Returns:
        Paper URL (arxiv, DOI, or landing page) or None if not found.
    """
    cache = _load_cache()
    if suite_name in cache:
        cached = cache[suite_name]
        logger.info("Paper URL from cache for '%s': %s", suite_name, cached.get("url"))
        return cached.get("url")

    queries = _build_search_queries(suite_name, full_name)
    search_name = full_name or suite_name.replace("_", " ")

    verification_log = {
        "suite": suite_name,
        "full_name": full_name,
        "queries_tried": queries,
        "source": "openalex",
        "candidates": [],
        "resolved_url": None,
        "resolved_from": None,
    }

    resolved_url = None
    for query in queries:
        logger.info("Searching OpenAlex for '%s'", query)
        raw_results = _search_openalex(query)
        if not raw_results:
            continue

        candidates = [_normalize_openalex_paper(w) for w in raw_results]
        filtered = _prefilter_candidates(candidates, search_name)
        if not filtered:
            continue

        for paper in filtered:
            llm_result = _verify_with_llm(
                paper,
                suite_name,
                sub_benchmarks or [],
                metrics or [],
                eval_library,
            )

            candidate_entry = {
                "title": paper.get("title", ""),
                "query_used": query,
                "openalex_id": paper.get("openalex_id", ""),
                "arxiv_id": paper.get("arxiv_id"),
                "doi": paper.get("doi", ""),
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year"),
                "title_similarity": paper.get("_title_similarity", 0),
                "llm_match": llm_result.get("match", False),
                "llm_confidence": llm_result.get("confidence", 0.0),
                "llm_reasoning": llm_result.get("reasoning", ""),
            }
            verification_log["candidates"].append(candidate_entry)

            if llm_result.get("match") and llm_result.get("confidence", 0) >= 0.7:
                resolved_url = _extract_paper_url(paper)
                verification_log["resolved_url"] = resolved_url
                verification_log["resolved_from"] = "openalex+llm_verification"
                logger.info(
                    "Paper resolved for '%s': %s (confidence: %.2f)",
                    suite_name, resolved_url, llm_result.get("confidence", 0),
                )
                break

        if resolved_url:
            break

    # Cache result (even if None, to avoid repeated lookups)
    cache[suite_name] = {"url": resolved_url}
    _save_cache(cache)

    # Save verification log for traceability
    if output_dir:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "paper-verification.json"
            log_path.write_text(json.dumps(verification_log, indent=2))
            logger.info("Paper verification log saved to %s", log_path)
        except OSError as e:
            logger.warning("Failed to save verification log: %s", e)

    return resolved_url
