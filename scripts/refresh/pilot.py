"""Deterministic stratified pilot selector (no RNG).

Strata are four cache facts per card: docling present, hf non-empty, html
non-empty, and EEE benchmark_type composite-vs-single. Known anchors are always
included; the rest is filled by taking the alphabetically-first unused slug from
each stratum in a fixed round-robin until the target size is reached. Same index
in, same pilot set out.
"""

import os
from typing import Dict, List, Tuple

from .reconstruct import load_eee_metadata
from .resolve import tool_output_path

ANCHORS_GOOD = ["agieval", "bbq", "attaq", "androidworld"]
ANCHORS_THIN = ["ai2d", "activitynet", "imdb", "ruler", "vcr", "multi-swe-bench"]


def _nonempty(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def cache_facts(slug: str, run_dir: str) -> Tuple[bool, bool, bool, bool]:
    """(docling_present, hf_nonempty, html_nonempty, is_composite) for one card."""
    docling = _nonempty(tool_output_path(run_dir, "docling", f"{slug}.json"))
    hf = _nonempty(tool_output_path(run_dir, "hf", f"{slug}.json"))
    html = _nonempty(tool_output_path(run_dir, "html", f"{slug}.json"))
    btype = str(load_eee_metadata(run_dir, slug).get("benchmark_type", "single")).strip().lower()
    return docling, hf, html, btype == "composite"


def select_pilot(index: Dict[str, str], target: int = 25) -> List[str]:
    """Pick ~target slugs spanning the cache strata, anchors first. Deterministic."""
    facts = {slug: cache_facts(slug, run_dir) for slug, run_dir in index.items()}

    strata: Dict[Tuple[bool, bool, bool, bool], List[str]] = {}
    for slug, key in sorted(facts.items()):
        strata.setdefault(key, []).append(slug)

    selected: List[str] = []
    seen = set()
    for anchor in ANCHORS_GOOD + ANCHORS_THIN:
        if anchor in index and anchor not in seen:
            selected.append(anchor)
            seen.add(anchor)

    ordered_keys = sorted(strata)
    depth = 0
    max_depth = max((len(v) for v in strata.values()), default=0)
    while len(selected) < target and depth < max_depth:
        for key in ordered_keys:
            if len(selected) >= target:
                break
            bucket = strata[key]
            if depth < len(bucket) and bucket[depth] not in seen:
                selected.append(bucket[depth])
                seen.add(bucket[depth])
        depth += 1

    return sorted(selected)
