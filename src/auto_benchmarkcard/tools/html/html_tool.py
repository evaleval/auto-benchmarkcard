"""Extract clean content from web pages using trafilatura."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import trafilatura
from langchain.tools import tool

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 20_000

# Pages on these hosts carry a model leaderboard table and a "related benchmarks"
# carousel that trafilatura keeps as main content. The carousel describes OTHER
# benchmarks and leaks into the card, so we strip it at ingestion.
_FURNITURE_NETLOCS = ("llm-stats.com",)
_PROGRESS_MARKER = "progress over time"
_LEADERBOARD_HEADING_RE = re.compile(r"\bleaderboard\b\s*$", re.IGNORECASE)
_MD_TABLE_RE = re.compile(r"^\s*\|")


def _extract_title(html: str, url: str) -> str:
    """Extract page title from raw HTML, falling back to domain."""
    import re

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return urlparse(url).netloc


def _is_pdf_or_arxiv(url: str) -> bool:
    """Check if a URL points to a PDF or arxiv paper."""
    lower = url.lower()
    parsed = urlparse(lower)
    return (
        lower.endswith(".pdf")
        or "arxiv.org" in parsed.netloc
        or parsed.path.endswith(".pdf")
    )


def _is_furniture_netloc(url: str) -> bool:
    """True for hosts whose pages carry leaderboard/related-benchmarks furniture."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return any(netloc == d or netloc.endswith("." + d) for d in _FURNITURE_NETLOCS)


def strip_page_furniture(text: str, url: str) -> tuple[str, int]:
    """Drop llm-stats page furniture from extracted text. Returns (clean_text, removed_chars).

    The genuine benchmark description is the hero summary at the top of the page; everything
    from the first furniture marker onward is the progress chart, the model leaderboard table,
    and the related-benchmarks carousel (unlabeled descriptions of OTHER benchmarks). We cut at
    the earliest of: a "Progress Over Time" line, a short "<Name> Leaderboard" heading, or the
    first markdown-table row. The cut is positional, never keyword-based, so a benchmark whose
    own description mentions another field (e.g. biology) keeps its genuine content.

    Scoped to llm-stats netlocs. If the page is not an llm-stats page, no marker is found, or
    the marker is the very first line (no hero above it), the text is returned unchanged.
    """
    if not text or not _is_furniture_netloc(url):
        return text, 0

    lines = text.split("\n")
    cut = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower() == _PROGRESS_MARKER:
            cut = i
            break
        if _MD_TABLE_RE.match(line):
            cut = i
            break
        if len(stripped) <= 40 and _LEADERBOARD_HEADING_RE.search(stripped):
            cut = i
            break

    if not cut:  # None (no marker) or 0 (no hero content above marker) -> leave unchanged
        return text, 0

    clean = "\n".join(lines[:cut]).rstrip("\n")
    return clean, len(text) - len(clean)


@tool("extract_html_content")
def extract_html_content(url: str) -> Dict[str, Any]:
    """Extract clean main content from a web page.

    Uses trafilatura to strip navigation, ads, and boilerplate,
    returning only the main textual content.

    Args:
        url: The web page URL to extract content from.

    Returns:
        Dict with keys: text, url, title, success, error (optional).
    """
    if _is_pdf_or_arxiv(url):
        return {
            "text": "",
            "url": url,
            "title": "",
            "success": False,
            "error": f"URL appears to be PDF/arxiv — use docling instead: {url}",
        }

    try:
        logger.info("Fetching HTML content from: %s", url)

        config = trafilatura.settings.use_config()
        config.set("DEFAULT", "DOWNLOAD_TIMEOUT", "30")

        downloaded = trafilatura.fetch_url(url, config=config)
        if not downloaded:
            return {
                "text": "",
                "url": url,
                "title": "",
                "success": False,
                "error": f"Failed to download page: {url}",
            }

        title = _extract_title(downloaded, url)

        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_links=False,
            include_comments=False,
            include_images=False,
            favor_precision=True,
        )

        if not text or len(text.strip()) < 50:
            return {
                "text": "",
                "url": url,
                "title": title,
                "success": False,
                "error": "Extracted content too short or empty",
            }

        text, removed = strip_page_furniture(text, url)
        if removed:
            logger.info("Stripped %d chars of leaderboard/related-benchmarks furniture from %s", removed, url)

        if len(text) > MAX_CONTENT_CHARS:
            text = text[:MAX_CONTENT_CHARS]
            logger.info("Truncated HTML content to %d chars", MAX_CONTENT_CHARS)

        logger.info(
            "Extracted %d chars from %s (title: %s)", len(text), url, title
        )

        return {
            "text": text,
            "url": url,
            "title": title,
            "success": True,
        }

    except Exception as e:
        logger.warning("HTML extraction failed for %s: %s", url, e)
        return {
            "text": "",
            "url": url,
            "title": "",
            "success": False,
            "error": str(e),
        }
