"""Locate the cache run dir for each published card and load the live card.

The mapping slug -> run_dir is established by CONTENT-MATCHING: a published card
may correspond to several timestamped run dirs, so we pick the one whose produced
benchmark_card is byte-identical to the published card. The result is cached to
json (rebuild with build_index(rebuild=True)).
"""

import glob
import json
import os
from typing import Dict, Optional


def repo_root() -> str:
    """The integration worktree root (holds output/broad_run)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO = repo_root()
PUBLISHED_ROOT = os.environ.get("AUTO_BENCHMARKCARD_MAIN_ROOT", REPO)
PUB_DIR = os.environ.get(
    "REFRESH_PUB_DIR",
    os.path.join(PUBLISHED_ROOT, "output", "auto-benchmarkcards-v2", "cards"),
)
RUN_ROOT = os.environ.get(
    "REFRESH_RUN_ROOT", os.path.join(REPO, "output", "broad_run", "output")
)
# REFRESH_OUT is where run outputs land. It is overridable (REFRESH_OUT_DIR) so parallel
# recall shards can each write to their own dir without colliding on the timestamped run name.
# INDEX_PATH stays in the default location so all shards share one cached resolve index.
_DEFAULT_REFRESH_OUT = os.path.join(REPO, "output", "refresh_runs")
REFRESH_OUT = os.environ.get("REFRESH_OUT_DIR", _DEFAULT_REFRESH_OUT)
INDEX_PATH = os.path.join(_DEFAULT_REFRESH_OUT, "resolve_index.json")


def _serialize(card: dict) -> str:
    """Canonical serialization used for both content-matching and final writes."""
    return json.dumps(card, indent=2, ensure_ascii=True)


def published_slugs() -> list:
    """All published card slugs, sorted."""
    return sorted(
        os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PUB_DIR, "*.json"))
    )


def load_live_card(slug: str) -> dict:
    """Load the published card (the {"benchmark_card": ...} overlay base)."""
    with open(os.path.join(PUB_DIR, f"{slug}.json")) as f:
        return json.load(f)


def _inner(card: dict) -> dict:
    return card.get("benchmark_card", card)


def _match_run_dir(slug: str) -> Optional[str]:
    """Return the run dir whose produced card matches the published card, else None."""
    pub_inner = _serialize(_inner(load_live_card(slug)))
    cands = sorted(
        glob.glob(
            os.path.join(RUN_ROOT, f"{slug}_*", "benchmarkcard", f"benchmark_card_{slug}.json")
        )
    )
    for c in cands:
        try:
            with open(c) as f:
                produced = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if _serialize(_inner(produced)) == pub_inner:
            return os.path.dirname(os.path.dirname(c))
    return None


def build_index(rebuild: bool = False) -> Dict[str, str]:
    """Build {slug: run_dir} by content-matching, caching to INDEX_PATH.

    Unmatched slugs are omitted and reported by the caller. A run_dir is the
    timestamped directory under RUN_ROOT (its tool_output/ holds the cache).
    """
    if not rebuild and os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            return json.load(f)

    index: Dict[str, str] = {}
    for slug in published_slugs():
        run_dir = _match_run_dir(slug)
        if run_dir is not None:
            index[slug] = run_dir

    os.makedirs(REFRESH_OUT, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    return index


def tool_output_path(run_dir: str, tool: str, filename: str) -> str:
    return os.path.join(run_dir, "tool_output", tool, filename)


def is_b_eligible(run_dir: str, slug: str) -> bool:
    """True when docling/<slug>.json is present and non-empty (paper-backed).

    B abstains on shells (no paper text to recall from); A runs on all cards.
    """
    p = tool_output_path(run_dir, "docling", f"{slug}.json")
    try:
        return os.path.getsize(p) > 0
    except OSError:
        return False
