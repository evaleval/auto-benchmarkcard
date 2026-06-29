"""Resolve paper URLs for benchmark suites via OpenAlex, Semantic Scholar + LLM verification.

For benchmarks without a known paper URL (especially composite suites like HELM),
searches OpenAlex and Semantic Scholar for candidate papers and uses an LLM to
verify the match by comparing the paper's abstract against EEE metadata.

Results are cached in .paper_cache.json to avoid repeated API calls.
Each verification decision is logged to output/<benchmark>/paper-verification.json
for auditability.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import requests
from rapidfuzz import fuzz

from auto_benchmarkcard.config import Config, get_llm_handler

logger = logging.getLogger(__name__)

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "aris.hofmann@gmail.com")
OPENALEX_LIMIT = 5

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,url"
S2_LIMIT = 5
_s2_last_request: float = 0.0


def _get_s2_api_key() -> Optional[str]:
    return os.environ.get("S2_API_KEY")

ENTITY_REGISTRY_URL = "https://evaleval-entity-registry.hf.space/api/v1"
_display_name_cache: Dict[str, Optional[str]] = {}
_metadata_cache: Dict[str, Dict[str, Any]] = {}

ALL_SOURCES = ["openalex", "semantic_scholar"]


METADATA_PROMPT = """Given this AI/NLP benchmark, return metadata as JSON.

Benchmark identifier: "{suite_name}"
{context_block}
Based on the information above, return:
{{
  "full_name": "the full human-readable benchmark name",
  "paper_title": "the exact title of the original paper that introduced this benchmark",
  "domain": "short domain description, e.g. math reasoning, code generation, reading comprehension",
  "year": approximate publication year or null
}}
If you are unsure about a field, set it to null. Respond ONLY with JSON (no markdown fences)."""

BATCH_VERIFICATION_PROMPT = """You are verifying which (if any) of these papers is the ORIGINAL paper that INTRODUCES a benchmark.

Benchmark info:
  Suite name: {suite_name}
  Full name: {full_name}
  Description: {overview}
  Domain: {domain}
  Sub-benchmarks: {sub_benchmark_list}
  Metrics: {metrics_list}
  Evaluation library: {eval_library}

Paper candidates:
{candidates_block}

RULES:
- A match must be the paper that INTRODUCES, PROPOSES, or CREATES the benchmark/dataset itself.
- The introducing paper often also evaluates on the benchmark — that's fine, it's still a match.
- A paper that merely USES or EVALUATES ON a benchmark without having created it is NOT a match.
  For example: a model paper saying "we evaluate on MMLU" is not the MMLU paper.
  But a paper saying "we present a new dataset..." and then evaluating on it IS a match.
- DOMAIN CHECK: If the benchmark domain is "{domain}" but the paper is about a completely different field (e.g., computer vision vs NLP, deraining vs reading comprehension), it is NOT a match regardless of name overlap.
- ACRONYM AWARENESS: Benchmark names are often acronyms. "gsm8k" = "Grade School Math 8K", "mbpp" = "Mostly Basic Python Problems", "drop" = "Discrete Reasoning Over Paragraphs". The paper title may use the full name without the acronym. This is still a match if the content aligns.
- SHORT NAME CAUTION: For benchmark names of 5 characters or fewer, require strong evidence in the abstract that this specific benchmark/dataset is being introduced. Surface-level word overlap alone is not enough.
- Be careful with versioned suites (HELM vs HELM 2, BIG-bench vs BIG-bench Hard).
- If NONE of the papers introduced this benchmark, return "none".

Respond ONLY with JSON (no markdown fences):
{{"match_index": <0-based index of the matching paper, or "none">, "confidence": 0.0-1.0, "reasoning": "..."}}"""


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


def _search_semantic_scholar(query: str) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for papers matching a query."""
    global _s2_last_request
    try:
        # Rate limit: 1 req/sec without API key
        api_key = _get_s2_api_key()
        if not api_key:
            elapsed = time.monotonic() - _s2_last_request
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": S2_LIMIT, "fields": S2_FIELDS},
            headers=headers,
            timeout=15,
        )
        _s2_last_request = time.monotonic()
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        logger.warning("Semantic Scholar search failed for '%s': %s", query, e)
        return []


def _normalize_s2_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Semantic Scholar paper to a normalized paper dict."""
    ext_ids = paper.get("externalIds") or {}
    arxiv_id = ext_ids.get("ArXiv")
    doi = ext_ids.get("DOI")

    url = paper.get("url", "")
    if doi:
        url = f"https://doi.org/{doi}"

    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract") or "",
        "year": paper.get("year"),
        "citationCount": paper.get("citationCount") or 0,
        "doi": doi or "",
        "arxiv_id": arxiv_id,
        "url": url,
    }


def _build_context_block(
    overview: Optional[str] = None,
    domains: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    full_name: Optional[str] = None,
) -> str:
    """Build a context block for the metadata prompt from card data."""
    lines = []
    if full_name:
        lines.append(f"Known full name: {full_name}")
    if overview:
        lines.append(f"Description: {overview}")
    if domains:
        clean = [d for d in domains if d.lower() != "not specified"]
        if clean:
            lines.append(f"Domains: {', '.join(clean)}")
    if metrics:
        clean = [m for m in metrics if m.lower() != "not specified"]
        if clean:
            lines.append(f"Metrics: {', '.join(clean[:10])}")
    if not lines:
        return ""
    return "Known information:\n" + "\n".join(lines)


def _query_benchmark_metadata(
    suite_name: str,
    overview: Optional[str] = None,
    domains: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    full_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask LLM for benchmark metadata to improve search queries and verification."""
    cache_key = suite_name
    if cache_key in _metadata_cache:
        return _metadata_cache[cache_key]

    context_block = _build_context_block(overview, domains, metrics, full_name)

    try:
        llm = get_llm_handler(Config.COMPOSER_MODEL)
        text = llm.generate(METADATA_PROMPT.format(
            suite_name=suite_name, context_block=context_block,
        ))
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        logger.info("LLM metadata for '%s': %s", suite_name, result)
        _metadata_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning("Benchmark metadata query failed for '%s': %s", suite_name, e)
        _metadata_cache[cache_key] = {}
        return {}


def _lookup_display_name(suite_name: str) -> Optional[str]:
    """Look up a human-readable name from the Entity Registry."""
    if suite_name in _display_name_cache:
        return _display_name_cache[suite_name]

    try:
        resp = requests.get(
            f"{ENTITY_REGISTRY_URL}/benchmarks/{suite_name}",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        # Prefer description (richer), fall back to display_name
        name = data.get("description") or data.get("display_name")
        # Strip trailing punctuation/emoji from description
        if name and " — " in name:
            name = name.split(" — ")[0].strip()
        _display_name_cache[suite_name] = name
        logger.info("Entity Registry display name for '%s': %s", suite_name, name)
        return name
    except Exception:
        logger.debug("Entity Registry lookup failed for '%s'", suite_name)
        _display_name_cache[suite_name] = None
        return None


def _prefilter_candidates(
    candidates: List[Dict[str, Any]],
    suite_name: str,
    references: Optional[List[str]] = None,
    domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter candidates by title similarity, with stricter rules for short names."""
    refs = [suite_name.lower()]
    if references:
        refs.extend(r.lower() for r in references if r)

    is_short = len(suite_name.replace("_", "").replace("-", "")) <= 5
    min_sim = 65 if is_short else 50

    # Domain terms for cross-domain filtering on short names
    domain_terms = set()
    if domain and is_short:
        domain_terms = {t for t in domain.lower().split() if len(t) > 3}

    filtered = []
    for paper in candidates:
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        sim = max(fuzz.partial_ratio(ref, title) for ref in refs)

        # Short names: require either high similarity or domain overlap
        if is_short and domain_terms and sim < 80:
            text = f"{title} {abstract}"
            if not any(term in text for term in domain_terms):
                continue

        if sim >= min_sim:
            paper["_title_similarity"] = sim
            filtered.append(paper)
    return filtered


def _batch_verify_with_llm(
    candidates: List[Dict[str, Any]],
    suite_name: str,
    sub_benchmarks: List[str],
    metrics: List[str],
    eval_library: str,
    full_name: Optional[str] = None,
    overview: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify all candidates in a single LLM call. Returns match index or 'none'."""
    # Build candidates block
    lines = []
    for i, paper in enumerate(candidates):
        abstract = paper.get("abstract", "No abstract available")
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        lines.append(
            f"[{i}] Title: {paper.get('title', '')}\n"
            f"    Year: {paper.get('year', 'unknown')}\n"
            f"    Citations: {paper.get('citationCount', 0)}\n"
            f"    Abstract: {abstract}"
        )
    candidates_block = "\n\n".join(lines)

    try:
        llm = get_llm_handler(Config.FACTREASONER_MODEL)
        prompt = BATCH_VERIFICATION_PROMPT.format(
            suite_name=suite_name,
            full_name=full_name or suite_name,
            overview=overview or "Not available",
            domain=domain or "NLP / AI evaluation",
            sub_benchmark_list=", ".join(sub_benchmarks[:20]) or "none",
            metrics_list=", ".join(metrics[:10]) or "none",
            eval_library=eval_library or "unknown",
            candidates_block=candidates_block,
        )

        text = llm.generate(prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return result

    except Exception as e:
        logger.warning("Batch LLM verification failed: %s", e)
        return {"match_index": "none", "confidence": 0.0, "reasoning": f"LLM error: {e}"}


def _extract_paper_url(paper: Dict[str, Any]) -> Optional[str]:
    """Extract the best URL for a paper (prefer arxiv, never return S2 landing pages)."""
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    doi = paper.get("doi")
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"
    url = paper.get("url") or ""
    # S2 landing pages are HTML, not useful for Docling or as reference
    if "semanticscholar.org" in url:
        return None
    return url or None


def _build_search_queries(
    suite_name: str,
    full_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build a list of search queries to try, from most specific to broadest."""
    queries = []

    # If we have a human-readable full name, that's the best query
    if full_name and full_name.lower() != suite_name.lower():
        queries.append(full_name)
        queries.append(f"{full_name} benchmark")

    # LLM-provided paper title is a powerful direct query
    if metadata and metadata.get("paper_title"):
        queries.append(metadata["paper_title"])

    # Clean up suite name (underscores to spaces)
    search_name = suite_name.replace("_", " ")

    # Strip version suffixes (v1.0, 2.0, etc.) and add base name as query
    base_name = re.sub(r'\s+v?\d+(\.\d+)*$', '', search_name).strip()
    if base_name and base_name.lower() != search_name.lower():
        queries.append(base_name)
        queries.append(f"{base_name} benchmark")

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


SearchFn = Callable[[str], List[Dict[str, Any]]]
NormalizeFn = Callable[[Dict[str, Any]], Dict[str, Any]]

_SOURCES: List[Tuple[str, SearchFn, NormalizeFn]] = [
    ("openalex", _search_openalex, _normalize_openalex_paper),
    ("semantic_scholar", _search_semantic_scholar, _normalize_s2_paper),
]


def resolve_paper(
    suite_name: str,
    sub_benchmarks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    eval_library: Optional[str] = None,
    full_name: Optional[str] = None,
    overview: Optional[str] = None,
    domains: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve the paper URL for a benchmark suite.

    Tries multiple search queries against OpenAlex and Semantic Scholar,
    pre-filters by title similarity, then verifies candidates with an LLM
    in a single batch call.

    Args:
        suite_name: Short name / folder name (e.g., "helm_capabilities", "bbh").
        sub_benchmarks: List of sub-benchmark names for LLM context.
        metrics: List of metric names for LLM context.
        eval_library: Evaluation library name for LLM context.
        full_name: Human-readable benchmark name (e.g., "BIG-Bench Hard").
            Looked up from Entity Registry if not provided.
        overview: Benchmark description from the card (prevents LLM hallucination).
        domains: Benchmark domains from the card.
        output_dir: Directory to save paper-verification.json for traceability.

    Returns:
        Dict with keys: url, abstract, title, year, citation_count.
        Returns None if no paper found.
    """
    # Resolve full_name from Entity Registry if not provided
    if not full_name:
        full_name = _lookup_display_name(suite_name)

    # LLM pre-query for paper_title and domain — pass card context to avoid
    # hallucination (e.g. BFCL = "Berkeley Function Calling Leaderboard", not guessed)
    metadata = _query_benchmark_metadata(
        suite_name, overview=overview, domains=domains,
        metrics=metrics, full_name=full_name,
    )
    llm_full_name = metadata.get("full_name")
    if not full_name:
        full_name = llm_full_name

    domain = metadata.get("domain")
    queries = _build_search_queries(suite_name, full_name, metadata)
    # Also add LLM-suggested full_name as query if it differs from what we have
    if llm_full_name and llm_full_name.lower() not in {q.lower() for q in queries}:
        queries.insert(0, llm_full_name)
    search_name = full_name or suite_name.replace("_", " ")

    # Build reference strings for prefilter (all known names/titles)
    references = [r for r in [full_name, metadata.get("paper_title")] if r]

    verification_log = {
        "suite": suite_name,
        "full_name": full_name,
        "domain": domain,
        "metadata": metadata,
        "queries_tried": queries,
        "sources_tried": [],
        "candidates": [],
        "resolved_url": None,
        "resolved_from": None,
    }

    resolved_url = None
    resolved_meta = {"abstract": "", "title": "", "year": None, "citation_count": 0}

    # Collect all candidates across queries and sources, then verify in one batch
    all_candidates: List[Dict[str, Any]] = []
    seen_titles: set = set()

    for query in queries:
        for source_name, search_fn, normalize_fn in _SOURCES:

            logger.info("Searching %s for '%s'", source_name, query)
            raw_results = search_fn(query)
            if not raw_results:
                continue

            candidates = [normalize_fn(p) for p in raw_results]
            filtered = _prefilter_candidates(candidates, search_name, references, domain)

            for paper in filtered:
                title_key = (paper.get("title") or "").lower().strip()
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    paper["_query_used"] = query
                    paper["_source"] = source_name
                    all_candidates.append(paper)

    # Batch verify all candidates in a single LLM call
    if all_candidates:
        # Cap at 10 candidates to keep the prompt reasonable
        verify_batch = all_candidates[:10]
        logger.info("Batch-verifying %d candidates for '%s'", len(verify_batch), suite_name)

        batch_result = _batch_verify_with_llm(
            verify_batch,
            suite_name,
            sub_benchmarks or [],
            metrics or [],
            eval_library,
            full_name=full_name,
            overview=overview,
            domain=domain,
        )

        # Log all candidates
        for i, paper in enumerate(verify_batch):
            is_match = batch_result.get("match_index") == i
            candidate_entry = {
                "title": paper.get("title", ""),
                "query_used": paper.get("_query_used", ""),
                "source": paper.get("_source", ""),
                "arxiv_id": paper.get("arxiv_id"),
                "doi": paper.get("doi", ""),
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year"),
                "title_similarity": paper.get("_title_similarity", 0),
                "llm_match": is_match,
                "llm_confidence": batch_result.get("confidence", 0.0) if is_match else 0.0,
                "llm_reasoning": batch_result.get("reasoning", "") if is_match else "",
            }
            verification_log["candidates"].append(candidate_entry)

        match_idx = batch_result.get("match_index")
        confidence = batch_result.get("confidence", 0.0)
        if match_idx != "none" and isinstance(match_idx, int) and 0 <= match_idx < len(verify_batch):
            if confidence >= 0.7:
                matched_paper = verify_batch[match_idx]
                resolved_url = _extract_paper_url(matched_paper)
                resolved_meta = {
                    "abstract": matched_paper.get("abstract", ""),
                    "title": matched_paper.get("title", ""),
                    "year": matched_paper.get("year"),
                    "citation_count": matched_paper.get("citationCount", 0),
                }
                verification_log["resolved_url"] = resolved_url
                verification_log["resolved_from"] = f"{matched_paper.get('_source')}+batch_verification"
                logger.info(
                    "Paper resolved for '%s': %s (confidence: %.2f, reasoning: %s)",
                    suite_name, resolved_url, confidence,
                    batch_result.get("reasoning", ""),
                )

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

    if resolved_url:
        return {"url": resolved_url, **resolved_meta}
    return None
