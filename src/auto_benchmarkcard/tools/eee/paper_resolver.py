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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import requests
from rapidfuzz import fuzz

from auto_benchmarkcard.config import Config, get_llm_handler

logger = logging.getLogger(__name__)

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO")
OPENALEX_LIMIT = 5

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,abstract,year,citationCount,externalIds,url,authors"
S2_LIMIT = 5
_s2_last_request: float = 0.0

# Shared HTTP retry policy for the author/search fetches below.
HTTP_RETRIES = 3
HTTP_BACKOFF_BASE = 0.5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# arXiv occasionally answers 200 with an empty Atom feed; retry that transient miss too.
ARXIV_PARSE_RETRIES = 2


def _http_get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with bounded retry/backoff on transient failures (network errors, 429, 5xx).

    Fail-soft: returns the final Response (caller checks status / parses) or None if every
    attempt raised. Keyed purely on HTTP semantics, no per-source logic.
    """
    kwargs.setdefault("timeout", 15)
    resp = None
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, **kwargs)
            if resp.status_code in _RETRYABLE_STATUS and attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))
                continue
            return resp
        except Exception as e:
            resp = None
            if attempt < HTTP_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))
                continue
            logger.warning("HTTP GET failed for %s after %d attempts: %s", url, HTTP_RETRIES, e)
    return resp


def _get_s2_api_key() -> Optional[str]:
    return os.environ.get("S2_API_KEY")


def _get_openalex_api_key() -> Optional[str]:
    return os.environ.get("OPENALEX_API_KEY")


def _openalex_headers() -> Dict[str, str]:
    key = _get_openalex_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _openalex_status_for(resp) -> Optional[str]:
    if resp is None:
        return "errored"
    if resp.status_code in (401, 403):
        return "auth_error"
    if resp.status_code == 429:
        return "rate_limited"
    if resp.status_code >= 400:
        return "errored"
    return None

ENTITY_REGISTRY_URL = "https://evaleval-entity-registry.hf.space/api/v1"
_display_name_cache: Dict[str, Optional[str]] = {}
_metadata_cache: Dict[str, Dict[str, Any]] = {}
# Author/title fetched per known paper URL (arXiv id / DOI), cached for the 500-card batch.
_author_cache: Dict[str, Dict[str, Any]] = {}
# OpenAlex search results / DOI lookups, cached for the batch (same lifetime as _author_cache).
_openalex_search_cache: Dict[str, List[Dict[str, Any]]] = {}
_openalex_doi_cache: Dict[str, Dict[str, Any]] = {}
# DOI-path outcome, kept separate from _SOURCE_STATUS (the per-call search map) because the DOI
# fetch runs outside the search window, so writing _SOURCE_STATUS there would be a dead write.
_OPENALEX_DOI_STATUS: Dict[str, str] = {}

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
# Modern arXiv id (NNNN.NNNNN, optional version) inside an abs/pdf URL.
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

ALL_SOURCES = ["openalex", "semantic_scholar"]

# Per-source outcome for the current resolve_paper call, surfaced in paper-verification.json so
# a None-resolution can be told apart from a rate-limited / errored source. Reset per call.
_SOURCE_STATUS: Dict[str, str] = {}

# A candidate whose title matches this closely is accepted even when LLM
# verification is unavailable or inconclusive, so a transient/misconfigured LLM
# can never silently drop a near-exact-title paper match. Prefilter admits
# candidates at sim >= 50 (65 for short names), so this floor is well above noise.
TITLE_SIMILARITY_FLOOR = 95.0

# A benchmark name shorter than this (normalized) is too ambiguous to identify a paper by title
# token containment alone, so the name-token recovery floor below skips it.
NAME_TOKEN_FLOOR_LEN = 5

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
- SAME-NAME, DIFFERENT-DOMAIN CHECK: Two unrelated benchmarks can share the SAME name (e.g. a video-language "ActionBench" vs a text-to-image "ActionBench"). A name match is NOT sufficient on its own: the benchmark the paper introduces must also cohere with this benchmark's domain ("{domain}") and metrics ({metrics_list}). If a candidate introduces a benchmark with this exact name but its subject/domain/metrics plainly differ from the target's, that is a name collision, NOT a match -- prefer "none" (or another candidate that coheres) over a same-name paper whose domain is wrong.
- ACRONYM AWARENESS: Benchmark names are often acronyms. "gsm8k" = "Grade School Math 8K", "mbpp" = "Mostly Basic Python Problems", "drop" = "Discrete Reasoning Over Paragraphs". The paper title may use the full name without the acronym. This is still a match if the content aligns.
- SHORT NAME CAUTION: For benchmark names of 5 characters or fewer, require strong evidence in the abstract that this specific benchmark/dataset is being introduced. Surface-level word overlap alone is not enough.
- Be careful with versioned suites (HELM vs HELM 2, BIG-bench vs BIG-bench Hard).
- DIFFERENT-NAME CHECK: If a candidate INTRODUCES a benchmark/dataset under a name plainly different from the target (e.g. target "Beyond AIME" but the paper introduces "MATH-Beyond"), it is NOT a match even if the titles share words. Match only the paper that introduces THIS benchmark.
- If NONE of the papers introduced this benchmark, return "none".

Respond ONLY with JSON (no markdown fences):
{{"match_index": <0-based index of the matching paper, or "none">, "confidence": 0.0-1.0, "reasoning": "..."}}"""


def _search_openalex(query: str) -> List[Dict[str, Any]]:
    """Search OpenAlex for papers matching a query."""
    if query in _openalex_search_cache:
        return _openalex_search_cache[query]
    try:
        resp = _http_get(
            OPENALEX_SEARCH_URL,
            params={
                "search": query,
                "per_page": OPENALEX_LIMIT,
                "mailto": OPENALEX_MAILTO,
            },
            headers=_openalex_headers(),
            timeout=15,
        )
        status = _openalex_status_for(resp)
        if status:
            _SOURCE_STATUS["openalex"] = status
        if status == "auth_error":
            logger.warning(
                "OpenAlex search auth failed (HTTP %s) for '%s'; falling back to other sources",
                getattr(resp, "status_code", "?"), query,
            )
        if resp is None:
            return []
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            _openalex_search_cache[query] = results
        return results
    except Exception as e:
        _SOURCE_STATUS.setdefault("openalex", "errored")
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

    # Author display names for the display-only authors field (best effort)
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])
    ]
    authors = [a for a in authors if a]

    return {
        "title": work.get("title", ""),
        "abstract": abstract,
        "year": work.get("publication_year"),
        "citationCount": work.get("cited_by_count", 0),
        "openalex_id": openalex_id,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "authors": authors,
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

        resp = _http_get(
            S2_SEARCH_URL,
            params={"query": query, "limit": S2_LIMIT, "fields": S2_FIELDS},
            headers=headers,
            timeout=15,
        )
        _s2_last_request = time.monotonic()
        if resp is None:
            _SOURCE_STATUS["semantic_scholar"] = "errored"
            return []
        if resp.status_code == 429:
            _SOURCE_STATUS["semantic_scholar"] = "rate_limited"
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        _SOURCE_STATUS.setdefault("semantic_scholar", "errored")
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

    # Author names for the display-only authors field (best effort)
    authors = [a.get("name") for a in (paper.get("authors") or []) if a.get("name")]

    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract") or "",
        "year": paper.get("year"),
        "citationCount": paper.get("citationCount") or 0,
        "doi": doi or "",
        "arxiv_id": arxiv_id,
        "authors": authors,
        "url": url,
    }


def _fetch_arxiv_meta(arxiv_id: str) -> Dict[str, Any]:
    """Fetch authors + proper-cased title + year for an arXiv id via the Atom API. Fail-soft.

    Retries a transient empty parse (200 but no <entry>) in addition to the network-level
    retries in _http_get, so one flaky arXiv response doesn't permanently drop authors.
    """
    empty = {"authors": [], "title": "", "year": None}
    for attempt in range(ARXIV_PARSE_RETRIES):
        resp = _http_get(
            ARXIV_API_URL, params={"id_list": arxiv_id, "max_results": 1}, timeout=15
        )
        if resp is None:
            return empty
        try:
            resp.raise_for_status()
            entry = ET.fromstring(resp.text).find("a:entry", _ATOM_NS)
        except Exception as e:
            logger.warning("arXiv metadata fetch failed for '%s': %s", arxiv_id, e)
            return empty
        if entry is None:
            if attempt < ARXIV_PARSE_RETRIES - 1:
                time.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))
                continue
            return empty
        authors = [(n.text or "").strip() for n in entry.findall("a:author/a:name", _ATOM_NS)]
        authors = [a for a in authors if a]
        title_el = entry.find("a:title", _ATOM_NS)
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        year = None
        pub = entry.find("a:published", _ATOM_NS)
        if pub is not None and pub.text and pub.text[:4].isdigit():
            year = int(pub.text[:4])
        return {"authors": authors, "title": title, "year": year}
    return empty


def _fetch_openalex_by_doi(doi: str) -> Dict[str, Any]:
    """Fetch authors + title + year for a DOI via OpenAlex. Fail-soft."""
    doi_clean = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I)
    if doi_clean in _openalex_doi_cache:
        return _openalex_doi_cache[doi_clean]
    try:
        resp = _http_get(
            f"{OPENALEX_SEARCH_URL}/doi:{doi_clean}",
            params={"mailto": OPENALEX_MAILTO}, headers=_openalex_headers(), timeout=15,
        )
        status = _openalex_status_for(resp)
        if status:
            _OPENALEX_DOI_STATUS[doi_clean] = status
        if status == "auth_error":
            logger.warning(
                "OpenAlex DOI auth failed (HTTP %s) for '%s'; returning empty metadata",
                getattr(resp, "status_code", "?"), doi,
            )
        if resp is None:
            return {"authors": [], "title": "", "year": None}
        resp.raise_for_status()
        norm = _normalize_openalex_paper(resp.json())
        result = {"authors": norm.get("authors") or [], "title": norm.get("title") or "",
                  "year": norm.get("year")}
        if result.get("authors"):
            _openalex_doi_cache[doi_clean] = result
        return result
    except Exception as e:
        _OPENALEX_DOI_STATUS.setdefault(doi_clean, "errored")
        logger.warning("OpenAlex DOI metadata fetch failed for '%s': %s", doi, e)
        return {"authors": [], "title": "", "year": None}


def _fetch_paper_authors(url: str) -> Dict[str, Any]:
    """Best-effort author/title/year fetch for a known paper URL (arXiv or DOI).

    General mechanism for URL-only bindings: derive the
    author list and proper-cased title from the paper's own metadata so the display-only
    authors field (and the A3 name normalizer) have a real source. Keyed purely on URL
    structure (arXiv id / DOI), never on the benchmark name. Timeout-guarded, fail-soft
    (-> empty). Only successful fetches are cached: a transient miss must not be
    negative-cached, or it would poison every same-URL card in the batch.
    """
    if not isinstance(url, str) or not url:
        return {"authors": [], "title": "", "year": None}
    if url in _author_cache:
        return _author_cache[url]
    m = _ARXIV_ID_RE.search(url)
    if m:
        result = _fetch_arxiv_meta(m.group(1))
    elif "doi.org/" in url.lower() or url.lower().startswith("10."):
        result = _fetch_openalex_by_doi(url)
    else:
        result = {"authors": [], "title": "", "year": None}
    if result.get("authors"):
        _author_cache[url] = result
    return result


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
        resp = _http_get(
            f"{ENTITY_REGISTRY_URL}/benchmarks/{suite_name}",
            timeout=5,
        )
        if resp is None:
            logger.debug("Entity Registry lookup failed for '%s'", suite_name)
            _display_name_cache[suite_name] = None
            return None
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


def _parse_verification_json(text: str) -> Dict[str, Any]:
    """Parse the verifier's JSON, tolerating markdown fences and surrounding prose.

    Raises ValueError on an empty response so the caller treats it as a failure
    (not a silent 'none') and the title-similarity fallback can take over.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


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
        # Use the engine's configured composer model, not FACTREASONER_MODEL:
        # the latter is an HF-style id (meta-llama/...) that is NOT a valid route
        # under non-HF engines (e.g. RITS 404s on it), which previously made every
        # verification silently fail. Verification is a composition-time judgement
        # and already shares the composer model with the metadata pre-query above.
        llm = get_llm_handler(Config.COMPOSER_MODEL)
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
        try:
            return _parse_verification_json(text)
        except (ValueError, json.JSONDecodeError) as parse_err:
            # A transient JSON-parse failure must not silently discard a real match.
            # Re-ask once; a second failure falls through to the error verdict below.
            logger.warning("Verification JSON parse failed for '%s', retrying once: %s",
                           suite_name, parse_err)
            text = llm.generate(prompt)
            return _parse_verification_json(text)

    except Exception as e:
        logger.warning("Batch LLM verification failed: %s", e)
        # error=True lets resolve_paper distinguish "LLM unavailable" from a
        # genuine "none" verdict and apply the title-similarity fallback.
        return {"match_index": "none", "confidence": 0.0,
                "reasoning": f"LLM error: {e}", "error": True}


def _fetch_paper_meta_for_verify(url: str) -> Dict[str, Any]:
    """Title + abstract for a bound paper URL (arXiv or DOI), so the binding can be subject-verified.

    The bypass paths (an EEE/UnitXT pre-set or an HF arxiv-tag extraction) carry only a URL, no
    abstract -- but the domain/subject check needs the paper's content. Reuse the existing arXiv Atom
    fetch (extended here to also read the <summary> abstract, which _fetch_arxiv_meta omits) and the
    OpenAlex DOI normalizer (_normalize_openalex_paper already reconstructs the abstract). Keyed purely
    on URL structure, never on the benchmark name. Timeout-guarded, fail-soft.

    Returns {"title", "abstract", "fetch_error": bool}. fetch_error is True ONLY on a network / HTTP /
    parse FAILURE (or a URL whose content cannot be fetched at all), so the caller can keep an
    unverified binding degraded instead of verifying a BLANK candidate (which would let a transient
    arXiv/OpenAlex outage drop a correct pre-set via an LLM 'none'). A SUCCESSFUL fetch returns
    fetch_error=False even when the abstract is genuinely empty (a real-empty arXiv/OpenAlex record),
    so a true thin paper is still routed to the verifier rather than masked as an outage."""
    fail = {"title": "", "abstract": "", "fetch_error": True}
    if not isinstance(url, str) or not url:
        return fail
    m = _ARXIV_ID_RE.search(url)
    if m:
        for attempt in range(ARXIV_PARSE_RETRIES):
            resp = _http_get(ARXIV_API_URL, params={"id_list": m.group(1), "max_results": 1}, timeout=15)
            if resp is None:
                return fail
            try:
                resp.raise_for_status()
                entry = ET.fromstring(resp.text).find("a:entry", _ATOM_NS)
            except Exception as e:
                logger.warning("arXiv verify-meta fetch failed for '%s': %s", m.group(1), e)
                return fail
            if entry is None:
                if attempt < ARXIV_PARSE_RETRIES - 1:
                    time.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))
                    continue
                return fail
            title_el = entry.find("a:title", _ATOM_NS)
            summ_el = entry.find("a:summary", _ATOM_NS)
            return {
                "title": " ".join((title_el.text or "").split()) if title_el is not None else "",
                "abstract": " ".join((summ_el.text or "").split()) if summ_el is not None else "",
                "fetch_error": False,
            }
        return fail
    if "doi.org/" in url.lower() or url.lower().startswith("10."):
        doi_clean = re.sub(r"^https?://(dx\.)?doi\.org/", "", url.strip(), flags=re.I)
        try:
            resp = _http_get(f"{OPENALEX_SEARCH_URL}/doi:{doi_clean}",
                             params={"mailto": OPENALEX_MAILTO}, headers=_openalex_headers(), timeout=15)
            if resp is None:
                return fail
            resp.raise_for_status()
            norm = _normalize_openalex_paper(resp.json())
            return {"title": norm.get("title") or "", "abstract": norm.get("abstract") or "",
                    "fetch_error": False}
        except Exception as e:
            logger.warning("OpenAlex verify-meta fetch failed for '%s': %s", url, e)
            return fail
    return fail


def verify_paper_binding(
    suite_name: str,
    paper_url: str,
    *,
    full_name: Optional[str] = None,
    overview: Optional[str] = None,
    domain: Optional[str] = None,
    sub_benchmarks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    eval_library: Optional[str] = None,
    paper_title: Optional[str] = None,
    paper_abstract: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a paper_url bound by a NON-resolver path (EEE/UnitXT pre-set or HF arxiv-tag extraction)
    is the paper that INTRODUCES this benchmark, parity with the HF binding verifier.

    Reuses _batch_verify_with_llm as a single-candidate call (so the strengthened DOMAIN /
    same-name-different-domain prompt applies) plus the deterministic _introduces_other_benchmark
    name guard. Fetches the paper's title/abstract from the URL when the caller did not supply them
    (the bypass paths carry only a URL). Returns a normalized verdict:

      {"is_match": bool, "confidence": float, "reasoning": str,
       "declared_other": str|None, "error": bool, "paper_title": str}

    is_match is False when the deterministic name guard fires (a plainly different declared name),
    regardless of the LLM. error=True on LLM failure lets the caller keep an authoritative binding
    degraded rather than drop it under an outage. This function stays LLM + name-guard only; the
    subject-overlap backstop for pre-set bindings lives in the worker gate, and the search path's
    accept-time subject veto lives in resolve_paper (_subject_veto_overlap)."""
    if not paper_title or paper_abstract is None:
        fetched = _fetch_paper_meta_for_verify(paper_url)
        paper_title = paper_title or fetched.get("title", "")
        if paper_abstract is None:
            paper_abstract = fetched.get("abstract", "")

    # Deterministic name guard first (0 LLM): a plainly different declared benchmark name is a reject
    # no matter what the LLM says (the beyond-aime discriminator, reused on the bound title).
    declared_other = _introduces_other_benchmark(paper_title or "", suite_name, full_name)
    if declared_other:
        return {"is_match": False, "confidence": 1.0,
                "reasoning": f"bound paper declares a different benchmark {declared_other!r}",
                "declared_other": declared_other, "error": False, "paper_title": paper_title or ""}

    candidate = {
        "title": paper_title or "",
        "abstract": paper_abstract or "",
        "year": "unknown",
        "citationCount": 0,
    }
    v = _batch_verify_with_llm(
        [candidate], suite_name, sub_benchmarks or [], metrics or [], eval_library,
        full_name=full_name, overview=overview, domain=domain,
    )
    errored = bool(v.get("error"))
    match_idx = v.get("match_index")
    is_match = (isinstance(match_idx, int) and not isinstance(match_idx, bool) and match_idx == 0)
    return {"is_match": is_match,
            "confidence": float(v.get("confidence", 0.0) or 0.0),
            "reasoning": v.get("reasoning", ""),
            "declared_other": None, "error": errored, "paper_title": paper_title or ""}


# Single-candidate HF dataset-match verifier. A separate prompt from BATCH_VERIFICATION_PROMPT
# because the question is different: not "which paper INTRODUCES this benchmark?" but "is THIS HF
# repo the dataset this benchmark evaluates on / was built from?". The signal is structural and
# semantic (domain, subject, metric coherence), never a benchmark-name list.
HF_VERIFICATION_PROMPT = """You are verifying whether a HuggingFace dataset repository IS the dataset that a given benchmark evaluates on (or was built from).

Benchmark identity:
  Name: {suite_name}
  Full name: {full_name}
  Subject / what it measures: {subject_text}
  Metrics: {metrics_list}
  Evaluation library: {eval_library}

Candidate HuggingFace dataset:
  Repo id: {repo_id}
  Card title (pretty_name): {pretty_name}
  Card description: {description}
  Task categories: {task_categories}
  Tags: {tags}
  README (leading text):
{readme_lead}

RULES:
- A MATCH means this repo IS the dataset this benchmark evaluates on or was built from -- not merely a different artifact that happens to share the name or a token.
- DOMAIN CHECK: if the benchmark's subject is one domain (e.g. text-to-image generation, reading comprehension) but the repo is plainly a different domain (e.g. 3D-mesh-from-video, image classification), it is NOT a match, regardless of name overlap. A shared name with a disjoint subject is a name collision, not a match.
- GENERIC-NAME CAUTION: a popular dataset that shares a common English name or word with the benchmark (e.g. "Titanic", "Heart Disease", "Action") is NOT a match unless the repo's content positively corroborates that it is this specific benchmark's dataset.
- METRIC / SUBJECT COHERENCE: the repo's described task and the benchmark's metrics/subject should be consistent. Incoherent task vs metrics is evidence against a match.
- ACRONYM / SHORT-NAME CAUTION: for short or acronym names, require positive content corroboration in the README/description, not just a surface name match.
- If you are uncertain, say so by returning a LOW confidence rather than guessing a match.

Respond ONLY with JSON (no markdown fences):
{{"is_match": true|false, "confidence": 0.0-1.0, "reasoning": "..."}}"""


def verify_hf_match(identity: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Verify that a single mechanically-bound HF repo is this benchmark's dataset.

    Sibling to _batch_verify_with_llm: reuses the composer LLM handler, the tolerant JSON parse,
    the retry-once-on-parse-failure loop, and the error=True fallback verdict. Returns
    {"is_match": bool, "confidence": float, "reasoning": str}; on LLM failure returns the same
    shape with "error": True (and is_match False) so the caller can apply a precision-preserving
    fallback policy instead of silently keeping or dropping the binding.

    identity keys: suite_name, subject_text, metrics (list), eval_library, optional full_name.
    candidate keys: repo_id, readme_lead, pretty_name, description, task_categories, tags.
    """
    metrics = identity.get("metrics") or []
    tags = candidate.get("tags") or []
    try:
        llm = get_llm_handler(Config.COMPOSER_MODEL)
        prompt = HF_VERIFICATION_PROMPT.format(
            suite_name=identity.get("suite_name") or "unknown",
            full_name=identity.get("full_name") or identity.get("suite_name") or "unknown",
            subject_text=identity.get("subject_text") or "Not available",
            metrics_list=", ".join(str(m) for m in metrics[:10]) or "none",
            eval_library=identity.get("eval_library") or "unknown",
            repo_id=candidate.get("repo_id") or "unknown",
            pretty_name=candidate.get("pretty_name") or "Not available",
            description=candidate.get("description") or "Not available",
            task_categories=", ".join(str(t) for t in (candidate.get("task_categories") or [])) or "none",
            tags=", ".join(str(t) for t in tags[:20]) or "none",
            readme_lead=candidate.get("readme_lead") or "Not available",
        )

        text = llm.generate(prompt)
        try:
            return _parse_verification_json(text)
        except (ValueError, json.JSONDecodeError) as parse_err:
            # A transient JSON-parse failure must not silently drop a real match. Re-ask once;
            # a second failure falls through to the error verdict below.
            logger.warning("HF match verification JSON parse failed for %r, retrying once: %s",
                           candidate.get("repo_id"), parse_err)
            text = llm.generate(prompt)
            return _parse_verification_json(text)

    except Exception as e:
        logger.warning("HF match LLM verification failed for %r: %s", candidate.get("repo_id"), e)
        # error=True lets run_hf distinguish "LLM unavailable" from a genuine "not a match"
        # verdict and apply the corroborated-coined KEEP vs generic REJECT fallback policy.
        return {"is_match": False, "confidence": 0.0,
                "reasoning": f"LLM error: {e}", "error": True}


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
    hf_repo_id: Optional[str] = None,
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

    # Also strip a short hyphenated/space letter suffix (charxiv-d, ...-lcr/-full/-lite) so the
    # base entity is searched too. General: any trailing short alpha segment, not a name list.
    suffix_base = re.sub(r'[-\s]+[a-z]{1,5}$', '', base_name, flags=re.I).strip()
    if (suffix_base and suffix_base.lower() != base_name.lower()
            and len(suffix_base) >= NAME_TOKEN_FLOOR_LEN):
        queries.append(suffix_base)
        queries.append(f"{suffix_base} benchmark")

    # Domain-disambiguated query pins a name collision to the right field (helps the
    # name-collision case where the bare name also matches an unrelated paper).
    domain = (metadata or {}).get("domain")
    if full_name and domain and domain.lower() not in full_name.lower():
        queries.append(f"{full_name} {domain}")

    # HF repo id bridges a name the index can't reach from the slug (e.g. BeyondAIME -> its host
    # paper). Split camelCase / separators into words and search them.
    if hf_repo_id and "/" in hf_repo_id:
        words = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', hf_repo_id.split("/")[-1])
        words = re.sub(r'[_\-]+', ' ', words).strip()
        if words:
            queries.append(words)

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


def _distinctive_base(name: str) -> str:
    """Version-stripped, alphanumeric-normalized benchmark name for title token-containment."""
    base = re.sub(r"\s+v?\d+(\.\d+)*$", "", (name or "").strip())
    return re.sub(r"[^a-z0-9]", "", base.lower())


def _title_bears_name(title: str, base: str) -> bool:
    """True if `base` equals a contiguous run of whole tokens in the title (whole-token aligned)."""
    if not base:
        return False
    tokens = [t for t in re.split(r"[^a-z0-9]+", (title or "").lower()) if t]
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, len(tokens)):
            joined += tokens[j]
            if joined == base:
                return True
            if len(joined) >= len(base):
                break
    return False


def _token_is_namelike(token: str) -> bool:
    """True if a title token looks like a coined benchmark name, not a common word.

    Structural markers of a coined name: an all-caps run (MATH, BOLD, SWE, GLUE), an internal
    capital / camelCase (ActivityNet, CharXiv, HellaSwag), an embedded digit (GSM8K), or a
    hyphenated caps form (MATH-Beyond, SWE-bench). Plain Title-cased words (Deep, Learning) carry
    none of these. Keyed on capitalization structure, never on a benchmark list."""
    if re.search(r"[A-Z]{2,}", token):
        return True
    if re.search(r"[a-z][A-Z]", token):
        return True
    if re.search(r"\d", token):
        return True
    return "-" in token and bool(re.search(r"[A-Z]", token))


def _declared_benchmark_name(title: str) -> Optional[str]:
    """The benchmark name a paper title plainly declares, or None.

    Many introducing papers lead with 'NAME: <description>' (BOLD:, CharXiv:, SWE-bench:). Return
    that leading segment when it is short (<=3 tokens) and bears at least one name-like token.
    Method-paper titles with no colon ('Training Verifiers to Solve Math Word Problems') and
    generic colon heads ('A Large-Scale Study of X: ...') declare no name -> None."""
    if not title or ":" not in title:
        return None
    head = title.split(":", 1)[0].strip()
    tokens = head.split()
    if not tokens or len(tokens) > 3:
        return None
    if not any(_token_is_namelike(t) for t in tokens):
        return None
    return head


def _introduces_other_benchmark(title: str, *identities: Optional[str]) -> Optional[str]:
    """The declared name of a paper that plainly introduces a DIFFERENT benchmark, or None.

    Subject-coherence guard for the beyond-aime defect: the verifier accepted 'MATH-Beyond: ...'
    for the 'Beyond AIME' benchmark because the titles overlap. Returns the paper's declared name
    only when it parses, is long enough to judge, and is contiguously disjoint from every identity
    distinctive base (neither contains the other) -- so 'MATH-Beyond' vs 'Beyond AIME' rejects
    (disjoint as strings though they share the token 'beyond'), while 'CharXiv' vs 'charxiv-d' and
    'Length-Controlled AlpacaEval' vs 'alpaca_eval' are kept (containment). None means coherent or
    undecidable: keep the match."""
    declared = _declared_benchmark_name(title)
    if not declared:
        return None
    d = _distinctive_base(declared)
    if len(d) < NAME_TOKEN_FLOOR_LEN:
        return None
    # Version-tolerant: a trailing version glued on by a hyphen (alpacaeval-2.0 -> alpacaeval20)
    # survives _distinctive_base, so also compare with trailing digits stripped. Erring toward
    # containment is precision-safe: a missed reject loses coverage, a false reject drops a fact.
    d_core = re.sub(r"\d+$", "", d)
    for ident in identities:
        b = _distinctive_base(ident or "")
        if not b:
            continue
        b_core = re.sub(r"\d+$", "", b)
        if b in d or d in b or (b_core in d_core or d_core in b_core):
            return None
    return declared


# Floor for the accept-time subject veto below. Parity with the binding gate's
# PAPER_SAMENAME_OVERLAP_FLOOR (workers.py): near-disjoint only, so a rare recoverable
# miss is traded against the wrong-paper class.
PAPER_SUBJECT_VETO_FLOOR = 0.10

# Tokens that name what a benchmark IS, not which one: never treated as a variant qualifier,
# and a parent made only of these is no parent at all. Generic vocabulary, not benchmark names.
_GENERIC_NAME_TOKENS = {
    "bench", "benches", "benchmark", "benchmarks", "eval", "evals", "evaluation",
    "evaluations", "dataset", "datasets", "data", "set", "sets", "test", "tests",
    "task", "tasks", "suite", "suites", "corpus", "corpora", "problem", "problems",
    "question", "questions",
}

_VERSION_TOKEN_RE = re.compile(r"^v?\d+(\.\d+)*$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def _identity_tokens(name: str) -> List[str]:
    """Lower-case alnum tokens of an identity name ('MS MARCO (TREC)' -> ['ms','marco','trec'])."""
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]


def _name_run_spans(text: str, base: str) -> List[List[str]]:
    """Original-case word spans in `text` whose normalized concatenation equals `base`."""
    if not text or not base:
        return []
    words = _WORD_RE.findall(text)
    norm = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    spans = []
    for i in range(len(norm)):
        if not norm[i] or not base.startswith(norm[i]):
            continue
        acc = ""
        for j in range(i, len(norm)):
            acc += norm[j]
            if acc == base:
                spans.append(words[i:j + 1])
                break
            if not base.startswith(acc):
                break
    return spans


def _adjacent_variant_mention(text: str, parent_base: str, qualifier: str) -> bool:
    """True if `text` mentions the qualified variant as a unit: a parent-run occurrence with the
    qualifier token immediately before or after it ('HELM Lite', 'MM-Mind2Web'). A bare qualifier
    word elsewhere is NOT a mention (the HELM abstract contains 'capabilities' but does not
    describe a HELM Capabilities variant)."""
    if not text:
        return False
    words = _WORD_RE.findall(text)
    norm = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    q = re.sub(r"[^a-z0-9]", "", qualifier.lower())
    for i in range(len(norm)):
        if not norm[i] or not parent_base.startswith(norm[i]):
            continue
        acc = ""
        for j in range(i, len(norm)):
            acc += norm[j]
            if acc == parent_base:
                if (i > 0 and norm[i - 1] == q) or (j + 1 < len(norm) and norm[j + 1] == q):
                    return True
                break
            if not parent_base.startswith(acc):
                break
    return False


def _is_acronym_of(qualifier: str, parent_tokens: List[str]) -> bool:
    """True if `qualifier` reads as an acronym/initialism of the parent tokens: it segments,
    in order, into non-empty prefixes of a subsequence of the parent tokens ('ifeval' =
    i(nstruction) + f(ollowing) + eval(uation)). An identity like 'Instruction-Following
    Evaluation (IFEval)' is one name written twice, expansion plus acronym, not a parent
    plus a variant qualifier."""
    q = re.sub(r"[^a-z0-9]", "", (qualifier or "").lower())
    toks = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in parent_tokens]
    toks = [t for t in toks if t]
    if not q or not toks:
        return False

    # states = reachable (chars of q consumed) after considering tokens so far
    states = {0}
    for tok in toks:
        nxt = set(states)  # skipping this token is allowed
        for consumed in states:
            for plen in range(1, len(tok) + 1):
                if q.startswith(tok[:plen], consumed):
                    nxt.add(consumed + plen)
                else:
                    break
        states = nxt
    return len(q) in states


def _parent_only_binding(suite_name: Optional[str], full_name: Optional[str],
                         title: str, abstract: str) -> Optional[Dict[str, str]]:
    """Derivative-binds-parent guard: {parent, qualifier} when a variant-qualified identity is
    bound to the PARENT's paper, else None (keep).

    A variant identity (mm-mind2web, helm_lite, helm_capabilities) splits into a parent name
    plus an edge qualifier. When the accepted paper is about the parent (its title/abstract
    bears the parent with a name-like surface such as 'HELM' or 'Mind2Web', never plain prose
    words) and neither mentions the variant itself (full identity name, or parent+qualifier
    adjacent as a unit), the binding is the parent's paper and the honest outcome is no-paper.
    Veto-only: it never promotes an accept, and it stands down when the title bears the full
    identity name (the paper IS about the variant, protects harness-prefix slugs like
    helm_air_bench whose full_name 'AIR-Bench' is title-borne by the correct paper).

    Never fires when: any identity token is version-like (v2, 2014: version refreshes
    legitimately bind the parent line), the qualifier is single-char (charxiv-d keeps the
    CharXiv paper, which documents the descriptive split), a trailing qualifier is shorter
    than 3 chars (metric suffixes _EM/_SR), the qualifier or the whole parent is generic
    benchmark vocabulary, or the parent base is shorter than 4 chars. Split-suffix slugs with
    multi-char qualifiers (the MathVerse-Mini class) abstain to no-paper by design when the
    parent paper does not name the variant, consistent with the helm_lite policy: a measured
    recall cost, not a defect."""
    identities = [n for n in (full_name, suite_name) if n]
    if not identities:
        return None
    text = f"{title or ''}\n{abstract or ''}"

    # Stand-down: the paper is about the variant itself.
    for ident in identities:
        base = _distinctive_base(ident)
        if len(base) >= 4 and _title_bears_name(title or "", base):
            return None

    for ident in identities:
        tokens = _identity_tokens(ident)
        if len(tokens) < 2 or any(_VERSION_TOKEN_RE.match(t) for t in tokens):
            continue
        for qualifier, parent_tokens, trailing in (
                (tokens[0], tokens[1:], False), (tokens[-1], tokens[:-1], True)):
            if len(qualifier) < 2 or (trailing and len(qualifier) < 3):
                continue
            if qualifier in _GENERIC_NAME_TOKENS:
                continue
            if all(t in _GENERIC_NAME_TOKENS for t in parent_tokens):
                continue
            # 'Expansion (Acronym)' identities: the qualifier is the parent's own acronym,
            # not a variant marker (the ifeval false positive).
            if _is_acronym_of(qualifier, parent_tokens):
                continue
            parent_base = _distinctive_base(" ".join(parent_tokens))
            if len(parent_base) < 4 or not re.search(r"[a-z]", parent_base):
                continue
            spans = _name_run_spans(text, parent_base)
            namelike_parent = any(any(_token_is_namelike(w) for w in span) for span in spans)
            if not namelike_parent:
                continue
            # Variant mention stands the guard down: the paper documents the variant.
            full_base = _distinctive_base(ident)
            if _name_run_spans(text, full_base):
                continue
            if _adjacent_variant_mention(text, parent_base, qualifier):
                continue
            return {"parent": parent_base, "qualifier": qualifier, "identity": ident}
    return None


def _subject_veto_overlap(abstract, identity_text, identity_names):
    """Name-blind content-word overlap between a full paper abstract and the benchmark identity.

    Comparator for the accept-time subject veto (calibrated 2026-07-04 on the ten regen
    fixtures). Differs from the composer's _subject_overlap on purpose: the abstract is NOT
    lead-truncated (the 600-char cut hides a correct paper's subject vocabulary), and the
    identity's own name tokens are excluded from both sides (in a same-name collision the
    shared name is exactly what caused the wrong accept, so it is not subject evidence).

    Returns None (abstain, never veto) when either side has fewer than 8 content words after
    name-blinding: a missing/empty abstract, or a thin identity (empty overview/domain), must
    never strip a correct binding."""
    from auto_benchmarkcard.tools.composer.composer_tool import _content_words

    name_toks = set()
    for n in identity_names:
        if not n:
            continue
        for t in re.split(r"[^a-z0-9]+", n.lower()):
            if t:
                name_toks.add(t)
        joined = re.sub(r"[^a-z0-9]+", "", n.lower())
        if joined:
            name_toks.add(joined)
    abstract_words = _content_words(abstract or "") - name_toks
    identity_words = _content_words(identity_text or "") - name_toks
    if len(abstract_words) < 8 or len(identity_words) < 8:
        return None
    inter = len(abstract_words & identity_words)
    return max(inter / len(abstract_words), inter / len(identity_words))


def _arxiv_base(arxiv_id: str) -> str:
    """Version-stripped arXiv id for equality comparison (2406.15877v2 -> 2406.15877)."""
    return re.sub(r"v\d+$", "", (arxiv_id or "").strip())


def _repo_corroboration(
    repo_id: str,
    benchmark_name: str,
    repo_arxiv_id: Optional[str] = None,
    paper_arxiv_id: Optional[str] = None,
) -> Optional[str]:
    """Check a bound HF repo against the benchmark identity.

    Returns a reason string when the repo does NOT corroborate the benchmark, so the caller
    can drop the HF-derived identity fields rather than ship a wrong repo's metadata; returns
    None when the repo corroborates (or the slug is too ambiguous to judge). Deterministic and
    general -- keyed on token structure, never on specific benchmark names.

    - arxiv agreement: if the repo's own arxiv tag and the independently resolved paper both
      carry an id and they differ (version-stripped), the repo is bound to a different entity.
    - aggregate guard: the basename bears no whole-token run of the distinctive name (a multi-
      benchmark or unrelated repo).
    - derivative guard: the basename bears the name but also carries extra distinctive tokens
      (a training-set / variant derivative). The owner org is a detail, never a sole trigger.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        return None

    if (repo_arxiv_id and paper_arxiv_id
            and _arxiv_base(repo_arxiv_id) != _arxiv_base(paper_arxiv_id)):
        return "arxiv_mismatch"

    base = _distinctive_base(benchmark_name)
    if len(base) < NAME_TOKEN_FLOOR_LEN:
        return None  # slug too short to identify the repo by name

    tokens = [t for t in re.split(r"[^a-z0-9]+", repo_id.split("/")[-1].lower()) if t]
    # Locate the contiguous whole-token run that spells the distinctive name (same alignment
    # as _title_bears_name), so the remaining tokens can be tested for derivative extras.
    slug_idx = None
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, len(tokens)):
            joined += tokens[j]
            if joined == base:
                slug_idx = set(range(i, j + 1))
                break
            if len(joined) >= len(base):
                break
        if slug_idx is not None:
            break
    if slug_idx is None:
        return "aggregate_repo"

    extra = [t for k, t in enumerate(tokens) if k not in slug_idx and len(t) >= 3]
    return "derivative_repo" if extra else None


def resolve_paper(
    suite_name: str,
    sub_benchmarks: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    eval_library: Optional[str] = None,
    full_name: Optional[str] = None,
    overview: Optional[str] = None,
    domains: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    hf_repo_id: Optional[str] = None,
    prior_binding: Optional[Dict[str, Any]] = None,
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
        Dict with keys: url, abstract, title, year, citation_count, authors.
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
    queries = _build_search_queries(suite_name, full_name, metadata, hf_repo_id=hf_repo_id)
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
    # When the search path runs as the fall-through after a pre-set paper binding was REJECTED by the
    # gate, carry that rejection forward so the sidecar still records why the bypassed paper was
    # dropped ("sidecar for every binding" -- the bypass is never silent even on fall-through).
    if prior_binding:
        verification_log["binding"] = prior_binding

    resolved_url = None
    resolved_meta = {"abstract": "", "title": "", "year": None, "citation_count": 0, "authors": []}

    # Collect all candidates across queries and sources, then verify in one batch
    all_candidates: List[Dict[str, Any]] = []
    seen_titles: set = set()
    seen_ids: set = set()
    _SOURCE_STATUS.clear()
    source_results: Dict[str, int] = {name: 0 for name, _, _ in _SOURCES}

    for query in queries:
        for source_name, search_fn, normalize_fn in _SOURCES:

            logger.info("Searching %s for '%s'", source_name, query)
            raw_results = search_fn(query)
            source_results[source_name] = source_results.get(source_name, 0) + len(raw_results)
            if not raw_results:
                continue

            candidates = [normalize_fn(p) for p in raw_results]
            filtered = _prefilter_candidates(candidates, search_name, references, domain)

            for paper in filtered:
                title_key = (paper.get("title") or "").lower().strip()
                paper_id = paper.get("arxiv_id") or paper.get("doi") or ""
                if not title_key:
                    continue
                # Dedup by title and by arxiv/DOI id, so the same paper surfaced under a slightly
                # different title across sources doesn't fill the 10-slot batch.
                if title_key in seen_titles or (paper_id and paper_id in seen_ids):
                    continue
                seen_titles.add(title_key)
                if paper_id:
                    seen_ids.add(paper_id)
                paper["_query_used"] = query
                paper["_source"] = source_name
                all_candidates.append(paper)

    # Per-source outcome: ok if it returned results, else the recorded error / rate-limit, else
    # 0-results -- so a None resolution is distinguishable from a throttled source.
    verification_log["sources_tried"] = [
        {"source": name,
         "status": "ok" if source_results.get(name) else _SOURCE_STATUS.get(name, "0-results"),
         "results": source_results.get(name, 0)}
        for name, _, _ in _SOURCES
    ]

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
        llm_errored = bool(batch_result.get("error"))
        # Record the raw verdict so an LLM failure is never invisible in the audit
        # log (the per-candidate entries blank reasoning for non-matches).
        verification_log["llm_verification"] = {
            "match_index": match_idx,
            "confidence": confidence,
            "reasoning": batch_result.get("reasoning", ""),
            "error": llm_errored,
        }

        accepted_idx: Optional[int] = None
        resolved_via: Optional[str] = None
        if (isinstance(match_idx, int) and not isinstance(match_idx, bool)
                and 0 <= match_idx < len(verify_batch) and confidence >= 0.7):
            accepted_idx = match_idx
            resolved_via = "batch_verification"
        elif match_idx == "none" and not llm_errored and confidence >= 0.7:
            # A confident, non-errored "none" is a real no-match verdict: treat it as final
            # and skip the recovery fallbacks below, which exist only to rescue inconclusive
            # verdicts. The floor mirrors the accept floor above; an errored or low-confidence
            # "none" still falls through to recovery.
            logger.info(
                "LLM confidently found no matching paper for '%s' (conf=%.2f); leaving unresolved.",
                suite_name, confidence,
            )
        else:
            # LLM gave no usable confident match (low-confidence / error / malformed index).
            # Don't silently drop a near-exact title match: accept the highest
            # title-similarity candidate if it clears the floor.
            best_i, best_sim = -1, 0.0
            for i, paper in enumerate(verify_batch):
                sim = paper.get("_title_similarity", 0) or 0
                if sim > best_sim:
                    best_i, best_sim = i, sim
            if best_i >= 0 and best_sim >= TITLE_SIMILARITY_FLOOR:
                accepted_idx = best_i
                resolved_via = "title_similarity_fallback"
                logger.warning(
                    "LLM verification inconclusive for '%s' (match_index=%s, conf=%.2f, "
                    "error=%s); accepting candidate [%d] via title-similarity floor (sim=%.1f)",
                    suite_name, match_idx, confidence, llm_errored, best_i, best_sim,
                )

            # Strong single-candidate recovery: when the LLM declines but exactly one candidate's
            # title bears the benchmark's distinctive name as a whole-token run, that candidate is
            # almost certainly the introducing paper the "introduces it?" heuristic over-rejected
            # (raw title similarity can rank an unrelated paper higher, so it is not enough alone).
            # General and structure-keyed; skips short/ambiguous names and never overrides a match.
            if accepted_idx is None:
                base = _distinctive_base(full_name or suite_name)
                if len(base) >= NAME_TOKEN_FLOOR_LEN:
                    bearers = [i for i, paper in enumerate(verify_batch)
                               if _title_bears_name(paper.get("title", ""), base)]
                    if len(bearers) == 1:
                        accepted_idx = bearers[0]
                        resolved_via = "name_token_fallback"
                        logger.warning(
                            "LLM verification declined '%s' (match_index=%s, conf=%.2f); accepting "
                            "the sole name-bearing candidate [%d] (title=%r)",
                            suite_name, match_idx, confidence, bearers[0],
                            verify_batch[bearers[0]].get("title", ""),
                        )

        # Subject-coherence guard: a verified candidate must not introduce a plainly
        # differently-named benchmark than the identity (the beyond-aime defect, where the LLM
        # accepted "MATH-Beyond" for "Beyond AIME"). Deterministic post-verify override to
        # confident-none; covers all three accept paths (batch / title-sim / name-token).
        if accepted_idx is not None:
            other = _introduces_other_benchmark(
                verify_batch[accepted_idx].get("title", ""), suite_name, full_name)
            if other:
                logger.warning(
                    "Subject-coherence: '%s' resolved a paper introducing a different benchmark "
                    "%r; leaving unresolved.", suite_name, other)
                verification_log["subject_coherence"] = {
                    "declared": other, "identity": full_name or suite_name, "action": "rejected"}
                accepted_idx = None
                resolved_via = None

        # Derivative-binds-parent guard: a variant-qualified identity (helm_lite, mm-mind2web)
        # accepted onto the PARENT's paper abstains to no-paper unless the paper itself
        # mentions the variant. Deterministic, veto-only; see _parent_only_binding.
        if accepted_idx is not None:
            parent_only = _parent_only_binding(
                suite_name, full_name,
                verify_batch[accepted_idx].get("title", ""),
                verify_batch[accepted_idx].get("abstract") or "")
            if parent_only:
                logger.warning(
                    "Derivative guard: '%s' accepted the parent %r's paper (%r) without a "
                    "variant mention; leaving unresolved.",
                    suite_name, parent_only["parent"],
                    verify_batch[accepted_idx].get("title", ""))
                verification_log["derivative_guard"] = {**parent_only, "action": "abstained"}
                accepted_idx = None
                resolved_via = None

        # Universal near-disjoint subject veto, applied to ALL THREE accept paths
        # (batch_verification, title_similarity_fallback, name_token_fallback): a paper whose
        # abstract shares no subject vocabulary with the benchmark identity is a same-name
        # collision, not a match (the echo defect: a graph-propagation "ECHO" accepted for an
        # image-generation benchmark at conf 0.95). The overlap is name-blind (the shared name
        # is what caused the collision, so it is not subject evidence) and computed over the
        # full abstract. Thin abstract or thin identity abstains, never vetoes: an outage or a
        # sparse search record must not strip a correct binding. Overlap and action are
        # recorded in the sidecar on every evaluated accept (kept, vetoed, or abstained).
        if accepted_idx is not None:
            identity_text = " ".join(x for x in (full_name, overview, domain) if x)
            veto_overlap = _subject_veto_overlap(
                verify_batch[accepted_idx].get("abstract") or "",
                identity_text, (suite_name, full_name))
            if veto_overlap is None:
                veto_action = "abstained"
            elif veto_overlap < PAPER_SUBJECT_VETO_FLOOR:
                veto_action = "vetoed"
            else:
                veto_action = "kept"
            verification_log["subject_overlap"] = {
                "overlap": veto_overlap, "floor": PAPER_SUBJECT_VETO_FLOOR,
                "resolved_via": resolved_via, "action": veto_action}
            if veto_action == "vetoed":
                logger.warning(
                    "Subject veto: '%s' accepted %r via %s but the abstract is subject-disjoint "
                    "from the identity (overlap=%.3f < %.2f); leaving unresolved.",
                    suite_name, verify_batch[accepted_idx].get("title", ""), resolved_via,
                    veto_overlap, PAPER_SUBJECT_VETO_FLOOR)
                accepted_idx = None
                resolved_via = None

        if accepted_idx is not None:
            matched_paper = verify_batch[accepted_idx]
            resolved_url = _extract_paper_url(matched_paper)
            resolved_meta = {
                "abstract": matched_paper.get("abstract", ""),
                "title": matched_paper.get("title", ""),
                "year": matched_paper.get("year"),
                "citation_count": matched_paper.get("citationCount", 0),
                "authors": matched_paper.get("authors") or [],
            }
            # Search results often omit authors; backfill from the paper's own metadata
            # (arXiv/DOI) via the URL-keyed, cached fetch so the authors field is not NS.
            if not resolved_meta["authors"] and resolved_url:
                backfill = _fetch_paper_authors(resolved_url)
                if backfill.get("authors"):
                    resolved_meta["authors"] = backfill["authors"]
                    if not resolved_meta["title"]:
                        resolved_meta["title"] = backfill.get("title", "")
                    if resolved_meta["year"] is None:
                        resolved_meta["year"] = backfill.get("year")
            verification_log["resolved_url"] = resolved_url
            verification_log["resolved_from"] = f"{matched_paper.get('_source')}+{resolved_via}"
            # Reflect the accepted candidate in the per-candidate log.
            if accepted_idx < len(verification_log["candidates"]):
                entry = verification_log["candidates"][accepted_idx]
                entry["llm_match"] = True
                entry["resolved_via"] = resolved_via
                entry["llm_confidence"] = confidence if resolved_via == "batch_verification" else 0.0
                entry["llm_reasoning"] = batch_result.get("reasoning", "")
            logger.info(
                "Paper resolved for '%s': %s (via %s, confidence: %.2f)",
                suite_name, resolved_url, resolved_via, confidence,
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
