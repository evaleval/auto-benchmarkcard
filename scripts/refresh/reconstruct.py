"""Reconstruct composer kwargs from a run dir's cached tool_output.

All published cards came through the EEE path, so unitxt_metadata is always None.
Every input file is optional: a shell card has no docling/html, many have no hf.
Missing or unreadable files degrade to None/empty without crashing.

Documented cache gap (D3, low blast radius): the paper resolver sidecar persists
only the resolved URL, not paper_title / paper_abstract / authors. Those three
extracted_ids fields are therefore NOT cache-recoverable here; none of them is an
A/B/C target, so the refresh cannot regress them. extracted_ids is filled
best-effort with paper_url only.
"""

import json
import os
from typing import Any, Dict, Optional

from .resolve import tool_output_path


def _load_json(path: str) -> Optional[Any]:
    """Load a json file, or None if absent/empty/unreadable."""
    try:
        if os.path.getsize(path) == 0:
            return None
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_eee_metadata(run_dir: str, slug: str) -> Dict[str, Any]:
    """The EEE sidecar drives the whole reconstruction; {} if somehow missing."""
    data = _load_json(tool_output_path(run_dir, "eee", f"{slug}.json"))
    return data if isinstance(data, dict) else {}


def _load_extracted_ids(run_dir: str) -> Dict[str, Any]:
    """Best-effort extracted_ids from the paper-verification sidecar (paper_url only)."""
    pv = _load_json(tool_output_path(run_dir, "paper_resolver", "paper-verification.json"))
    if not isinstance(pv, dict):
        return {}
    url = pv.get("resolved_url")
    return {"paper_url": url} if url else {}


def reconstruct_kwargs(slug: str, run_dir: str) -> Dict[str, Any]:
    """Build the compose_benchmark_card.func kwargs from cache.

    Mirrors workers.run_composer's call: unitxt is None on the EEE path, query is
    the EEE benchmark_name, and every source is read from tool_output/.
    """
    eee_metadata = load_eee_metadata(run_dir, slug)
    query = eee_metadata.get("benchmark_name") or slug

    docling = _load_json(tool_output_path(run_dir, "docling", f"{slug}.json"))
    hf = _load_json(tool_output_path(run_dir, "hf", f"{slug}.json"))
    html = _load_json(tool_output_path(run_dir, "html", f"{slug}.json"))
    github = _load_json(tool_output_path(run_dir, "github", f"{slug}.json"))

    return {
        "unitxt_metadata": None,
        "hf_metadata": hf if hf else None,
        "extracted_ids": _load_extracted_ids(run_dir),
        "docling_output": docling,
        "query": query,
        "eee_metadata": eee_metadata if eee_metadata else None,
        "html_content": html,
        "github_readme": github,
    }
