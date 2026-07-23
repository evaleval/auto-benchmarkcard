"""Fetch a GitHub repository README as an extraction source.

Some benchmarks (e.g. biolp) are source-starved: no HF card and a paywalled paper,
but a public repo whose README describes the dataset. Routing that README into the
evidence/RAG sources gives data.* fields real text to ground on.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

MAX_README_CHARS = 20_000
FETCH_TIMEOUT = 15  # seconds; never an unbounded GET (batch-hang risk)

# github.com paths whose first segment is not a repo owner
_RESERVED_OWNERS = {
    "orgs", "sponsors", "features", "about", "topics", "collections",
    "marketplace", "settings", "notifications", "explore", "search", "apps",
}

_GITHUB_URL_RE = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)


def derive_github_repo(*sources: Any) -> Optional[str]:
    """Find a github.com/{owner}/{repo} URL in structured URLs or free text.

    Scans every source (URL strings and prose blobs alike) because some benchmarks
    name their repo only in the paper abstract, not in a structured link field.
    """
    for src in sources:
        if not isinstance(src, str) or "github.com" not in src.lower():
            continue
        m = _GITHUB_URL_RE.search(src)
        if not m:
            continue
        owner, repo = m.group(1), m.group(2)
        # Strip trailing prose punctuation and a .git suffix
        repo = repo.rstrip(".,;:)]\"'").removesuffix(".git")
        if owner.lower() in _RESERVED_OWNERS or not repo:
            continue
        return f"https://github.com/{owner}/{repo}"
    return None


def fetch_github_readme(repo_url: str, timeout: int = FETCH_TIMEOUT) -> Dict[str, Any]:
    """Fetch a repo's README via the GitHub API, bounded by a hard timeout.

    Returns {success, text, url, http_status, error?}. Never raises: a fetch failure
    is reported, not propagated, so the batch never hangs or aborts on one repo.
    """
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if parsed.netloc.lower() != "github.com" or len(parts) < 2:
        return {"success": False, "text": "", "url": repo_url,
                "http_status": None, "error": "not_a_github_repo"}

    owner, repo = parts[0], parts[1]
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "auto-benchmarkcard"}
    # Optional token lifts the unauthenticated 60 req/hr limit, which a 500-card batch
    # would otherwise exhaust into fetch_blocked. Absent -> plain anonymous request.
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(api_url, headers=headers, timeout=timeout)
    except requests.Timeout:
        return {"success": False, "text": "", "url": repo_url,
                "http_status": None, "error": "timeout"}
    except Exception as e:
        logger.warning("GitHub README fetch failed for %s: %s", repo_url, e)
        return {"success": False, "text": "", "url": repo_url,
                "http_status": None, "error": str(e)}

    status = resp.status_code
    if status != 200:
        return {"success": False, "text": "", "url": repo_url,
                "http_status": status, "error": f"http_{status}"}

    text = (resp.text or "")[:MAX_README_CHARS]
    if not text.strip():
        return {"success": False, "text": "", "url": repo_url,
                "http_status": status, "error": "empty_readme"}

    return {"success": True, "text": text, "url": repo_url, "http_status": status}
