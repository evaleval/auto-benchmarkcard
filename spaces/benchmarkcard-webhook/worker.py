"""Background worker for benchmark card generation.

Detects new benchmark folders in EEE_datastore, generates cards via
run_eee_pipeline(), and uploads them to evaleval/auto-benchmarkcards.

Uses Jenny's Entity Registry for canonical ID resolution and dedup.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import requests
from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger("worker")

EEE_REPO = "evaleval/EEE_datastore"
CARDS_REPO = "evaleval/auto-benchmarkcards"
ENTITY_REGISTRY_URL = "https://evaleval-entity-registry.hf.space/api/v1"

# Persistent storage on HF Spaces (mounted volume).
# Falls back to local /tmp for development.
PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_DIR", "/data"))
STATE_FILE = PERSISTENT_DIR / "state.json"

FORCE_REGENERATE = os.environ.get("FORCE_REGENERATE", "").lower() in ("1", "true", "yes")


# -- Retry decorator for transient failures --

def retry(max_attempts=3, delay=5, backoff=2):
    """Retry decorator with exponential backoff for transient failures."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay * (backoff ** attempt)
                    logger.warning(
                        "%s failed (attempt %d/%d), retrying in %ds: %s",
                        func.__name__, attempt + 1, max_attempts, wait, e,
                    )
                    time.sleep(wait)
        return wrapper
    return decorator


# -- State management (atomic writes) --

def load_state() -> dict:
    """Load persistent state (known folders, job history, pending queue)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            logger.exception("Failed to read state file, starting fresh")
    return {"known_folders": [], "jobs": [], "pending_folders": []}


def save_state(state: dict) -> None:
    """Save persistent state atomically (write-then-rename)."""
    PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


def save_pending(folders: list[str]) -> None:
    """Add folders to the pending queue."""
    state = load_state()
    pending = state.get("pending_folders", [])
    for f in folders:
        if f not in pending:
            pending.append(f)
    state["pending_folders"] = pending
    save_state(state)


def pop_pending() -> list[str]:
    """Pop all pending folders from the queue."""
    state = load_state()
    pending = state.pop("pending_folders", [])
    state["pending_folders"] = []
    save_state(state)
    return pending


# -- Entity Registry --

_canonical_cache: dict[str, Optional[str]] = {}


def resolve_canonical_id(benchmark_name: str) -> Optional[str]:
    """Resolve benchmark name to canonical_id via Entity Registry.

    Returns canonical_id string (e.g. "math") or None if not found.
    Uses an in-memory cache to avoid repeated API calls within a job.
    """
    if benchmark_name in _canonical_cache:
        return _canonical_cache[benchmark_name]

    try:
        resp = requests.post(
            f"{ENTITY_REGISTRY_URL}/resolve",
            json={"raw_value": benchmark_name, "entity_type": "benchmark"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        canonical_id = data.get("canonical_id")
        _canonical_cache[benchmark_name] = canonical_id
        if canonical_id:
            logger.info("Entity Registry: '%s' -> '%s'", benchmark_name, canonical_id)
        return canonical_id
    except Exception:
        logger.debug("Entity Registry lookup failed for '%s'", benchmark_name)
        _canonical_cache[benchmark_name] = None
        return None


def resolve_canonical_ids_batch(names: list[str]) -> dict[str, Optional[str]]:
    """Batch-resolve benchmark names to canonical_ids."""
    # Check cache first, only query uncached names
    uncached = [n for n in names if n not in _canonical_cache]
    if not uncached:
        return {n: _canonical_cache[n] for n in names}

    try:
        payload = [{"raw_value": n, "entity_type": "benchmark"} for n in uncached]
        resp = requests.post(
            f"{ENTITY_REGISTRY_URL}/resolve/batch",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()
        for name, result in zip(uncached, results):
            canonical_id = result.get("canonical_id")
            _canonical_cache[name] = canonical_id
            if canonical_id:
                logger.info("Entity Registry: '%s' -> '%s'", name, canonical_id)
    except Exception:
        logger.warning("Entity Registry batch resolve failed, using fallback")
        for name in uncached:
            _canonical_cache[name] = None

    return {n: _canonical_cache.get(n) for n in names}


def _get_card_filename(benchmark_name: str) -> str:
    """Get the canonical filename for a benchmark card.

    Uses Entity Registry canonical_id when available, falls back to
    sanitize_benchmark_name from the main package.
    """
    canonical = resolve_canonical_id(benchmark_name)
    if canonical:
        return canonical

    from auto_benchmarkcard.output import sanitize_benchmark_name
    return sanitize_benchmark_name(benchmark_name).lower()


# -- EEE folder detection --

def _extract_folders(file_list: list[str]) -> set[str]:
    """Extract unique top-level folder names under data/."""
    folders = set()
    for path in file_list:
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] == "data":
            folders.add(parts[1])
    return folders


@retry(max_attempts=3, delay=5)
def detect_new_benchmarks() -> list[str]:
    """Compare current EEE_datastore file listing against known state."""
    api = HfApi()
    all_files = api.list_repo_files(EEE_REPO, repo_type="dataset")

    current_folders = _extract_folders(all_files)
    state = load_state()
    known = set(state.get("known_folders", []))

    new_folders = sorted(current_folders - known)
    if new_folders:
        logger.info("Detected %d new folders: %s", len(new_folders), new_folders)
    else:
        logger.info("No new folders (known: %d, current: %d)", len(known), len(current_folders))

    return new_folders


# -- Download & upload --

@retry(max_attempts=3, delay=10)
def _download_folders(folder_names: list[str], target_dir: str) -> Path:
    """Download EEE folders into a shared temp directory."""
    patterns = [f"data/{f}/**/*.json" for f in folder_names]
    logger.info("Downloading %d EEE folders to %s", len(folder_names), target_dir)

    snapshot_download(
        repo_id=EEE_REPO,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=patterns,
    )

    return Path(target_dir) / "data"


@retry(max_attempts=3, delay=5)
def _upload_card(card: dict, benchmark_name: str, canonical_id: Optional[str] = None) -> bool:
    """Upload a generated card to evaleval/auto-benchmarkcards."""
    api = HfApi()
    filename = canonical_id or _get_card_filename(benchmark_name)
    remote_path = f"cards/{filename}.json"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(card, f, indent=2)
        tmp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=remote_path,
            repo_id=CARDS_REPO,
            repo_type="dataset",
            commit_message=f"Auto-generated card: {benchmark_name}",
        )
        logger.info("Uploaded card to %s/%s", CARDS_REPO, remote_path)
        return True
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@retry(max_attempts=2, delay=5)
def _list_existing_cards() -> set[str]:
    """List all card filenames (without extension) in the cards repo."""
    api = HfApi()
    all_files = api.list_repo_files(CARDS_REPO, repo_type="dataset")
    cards = set()
    for path in all_files:
        if path.startswith("cards/") and path.endswith(".json"):
            name = path[len("cards/"):-len(".json")]
            cards.add(name)
    return cards


# -- Main processing --

def _build_dedup_filter(
    benchmark_names: list[str],
    existing_cards: set[str],
) -> list[str]:
    """Return list of benchmark names that don't already have cards.

    Checks in order: Entity Registry canonical_id, exact fallback name,
    and parent prefix match (for 'Parent - Child' pattern).
    """
    if FORCE_REGENERATE:
        logger.info("FORCE_REGENERATE=true, skipping dedup")
        return benchmark_names

    # Batch-resolve all names
    canonical_map = resolve_canonical_ids_batch(benchmark_names)

    from auto_benchmarkcard.output import sanitize_benchmark_name

    new_benchmarks = []
    for name in benchmark_names:
        canonical = canonical_map.get(name)
        fallback = sanitize_benchmark_name(name).lower()

        # 1. Entity Registry canonical_id match
        if canonical and canonical in existing_cards:
            logger.info("Skipping '%s' (card exists as '%s')", name, canonical)
            continue

        # 2. Exact fallback name match
        if fallback in existing_cards:
            logger.info("Skipping '%s' (card exists as '%s')", name, fallback)
            continue

        # 3. Prefix match: 'MGSM - Bengali' -> check if 'mgsm' card exists
        if " - " in name:
            parent = name.split(" - ", 1)[0].strip()
            parent_lower = sanitize_benchmark_name(parent).lower()
            if parent_lower in existing_cards:
                logger.info("Skipping '%s' (parent card exists as '%s')", name, parent_lower)
                continue

        new_benchmarks.append(name)

    logger.info("Dedup: %d total, %d new, %d existing",
                len(benchmark_names), len(new_benchmarks),
                len(benchmark_names) - len(new_benchmarks))
    return new_benchmarks


def process_new_benchmarks(new_folders: list[str]) -> None:
    """Generate and upload cards for benchmarks in new folders.

    Delegates to run_eee_pipeline() for the actual generation, using a
    callback to upload each card as it's generated.
    """
    from auto_benchmarkcard.eee_workflow import run_eee_pipeline
    from auto_benchmarkcard.tools.eee.eee_tool import scan_eee_folder
    from auto_benchmarkcard.workflow import setup_logging_suppression

    setup_logging_suppression(debug_mode=False)

    state = load_state()
    job_record: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "folders": new_folders,
        "results": [],
    }

    # Pre-fetch existing cards for dedup
    try:
        existing_cards = _list_existing_cards()
        logger.info("Found %d existing cards in %s", len(existing_cards), CARDS_REPO)
    except Exception:
        logger.warning("Failed to list existing cards, dedup disabled for this job")
        existing_cards = set()

    # Download all folders into one shared temp dir
    with tempfile.TemporaryDirectory(prefix="eee_batch_") as tmpdir:
        try:
            data_path = _download_folders(new_folders, tmpdir)
        except Exception:
            logger.exception("Failed to download EEE folders")
            job_record["results"].append({
                "folders": new_folders, "status": "download_failed",
            })
            job_record["completed_at"] = datetime.now(timezone.utc).isoformat()
            state["jobs"].append(job_record)
            state["jobs"] = state["jobs"][-50:]
            save_state(state)
            return

        # Scan to discover benchmark names for dedup
        try:
            scan_result = scan_eee_folder(str(data_path))
        except Exception:
            logger.exception("Failed to scan EEE data")
            job_record["results"].append({
                "folders": new_folders, "status": "scan_failed",
            })
            job_record["completed_at"] = datetime.now(timezone.utc).isoformat()
            state["jobs"].append(job_record)
            state["jobs"] = state["jobs"][-50:]
            save_state(state)
            return

        all_names = (
            list(scan_result.benchmarks.keys())
            + list(scan_result.composites.keys())
        )
        benchmarks_to_generate = _build_dedup_filter(all_names, existing_cards)

        if not benchmarks_to_generate:
            logger.info("All benchmarks already have cards, nothing to generate")
            job_record["results"].append({"status": "all_existing"})
        else:
            # Upload callback: called by run_eee_pipeline for each generated card
            def _on_card_generated(name: str, card: dict) -> None:
                canonical = resolve_canonical_id(name)

                # Enrich card metadata
                inner = card.get("benchmark_card", card)
                info = inner.get("card_info", {})
                info["source"] = "webhook"
                if canonical:
                    info["canonical_id"] = canonical
                inner["card_info"] = info

                try:
                    _upload_card(card, name, canonical_id=canonical)
                    job_record["results"].append({
                        "benchmark": name,
                        "canonical_id": canonical,
                        "status": "uploaded",
                    })
                except Exception:
                    logger.exception("Failed to upload card for %s", name)
                    job_record["results"].append({
                        "benchmark": name, "status": "upload_failed",
                    })

            # Run the unified pipeline
            summary = run_eee_pipeline(
                eee_path=str(data_path),
                output_path=str(PERSISTENT_DIR / "output"),
                benchmarks_filter=benchmarks_to_generate,
                on_card_generated=_on_card_generated,
            )

            # Record skipped/failed from pipeline summary
            for item in summary.get("skipped", []):
                job_record["results"].append({
                    "benchmark": item.get("benchmark", "unknown"),
                    "status": f"skipped:{item.get('reason', 'unknown')}",
                })
            for name in summary.get("failed", []):
                # Only add if not already recorded by callback
                existing = {r.get("benchmark") for r in job_record["results"]}
                if name not in existing:
                    job_record["results"].append({
                        "benchmark": name, "status": "generation_failed",
                    })

    # Mark folders as known
    for folder_name in new_folders:
        if folder_name not in state["known_folders"]:
            state["known_folders"].append(folder_name)

    job_record["completed_at"] = datetime.now(timezone.utc).isoformat()

    results = job_record["results"]
    uploaded = sum(1 for r in results if r.get("status") == "uploaded")
    failed = sum(1 for r in results if "failed" in r.get("status", ""))
    skipped = sum(1 for r in results if r.get("status", "").startswith("skipped"))
    logger.info("Job complete: %d uploaded, %d failed, %d skipped", uploaded, failed, skipped)

    state["jobs"].append(job_record)
    state["jobs"] = state["jobs"][-50:]
    save_state(state)
