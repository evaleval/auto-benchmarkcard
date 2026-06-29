"""Extract clean content from web pages using trafilatura."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import trafilatura
from langchain.tools import tool

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 20_000


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
