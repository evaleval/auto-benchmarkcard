"""Background worker for benchmark card generation.

Detects new benchmark folders in EEE_datastore, generates cards via
process_single_benchmark(), and uploads them to evaleval/auto-benchmarkcards.
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger("worker")

EEE_REPO = "evaleval/EEE_datastore"
CARDS_REPO = "evaleval/auto-benchmarkcards"
MAX_BENCHMARKS_PER_JOB = 5

# Persistent storage on HF Spaces (mounted volume).
# Falls back to local /tmp for development.
PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_DIR", "/data"))
STATE_FILE = PERSISTENT_DIR / "state.json"


def load_state() -> dict:
    """Load persistent state (known folders, job history)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            logger.exception("Failed to read state file, starting fresh")
    return {"known_folders": [], "jobs": []}


def save_state(state: dict) -> None:
    """Save persistent state to disk (atomic write via temp + rename)."""
    PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


def _normalize_name(name: str) -> str:
    """Normalize benchmark name for comparison (lowercase, hyphens to underscores)."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _extract_folders(file_list: list[str]) -> set[str]:
    """Extract unique top-level folder names under data/."""
    folders = set()
    for path in file_list:
        parts = path.split("/")
        # Only include entries that have files beneath them (depth > 2)
        if len(parts) >= 3 and parts[0] == "data":
            folders.add(parts[1])
    return folders


def _get_existing_cards() -> set[str]:
    """List benchmark names that already have a card in the target dataset."""
    api = HfApi()
    try:
        files = api.list_repo_files(CARDS_REPO, repo_type="dataset")
    except Exception:
        logger.exception("Failed to list existing cards")
        return set()

    names = set()
    for f in files:
        if f.startswith("cards/") and f.endswith(".json"):
            # cards/mmlu.json -> mmlu
            names.add(f[len("cards/"):-len(".json")])
    return names


def detect_new_benchmarks() -> list[str]:
    """Find EEE folders that don't have a card yet.

    Compares EEE_datastore folders against existing cards in the
    target dataset, so we never regenerate what's already there.
    """
    api = HfApi()
    try:
        all_files = api.list_repo_files(EEE_REPO, repo_type="dataset")
    except Exception:
        logger.exception("Failed to list EEE_datastore files")
        return []

    current_folders = _extract_folders(all_files)
    existing_cards = _get_existing_cards()

    # Normalize both sides for comparison (arc-agi == arc_agi)
    normalized_cards = {_normalize_name(c) for c in existing_cards}
    new_folders = sorted(
        f for f in current_folders
        if _normalize_name(f) not in normalized_cards
    )
    if not new_folders:
        logger.info("All %d folders already have cards", len(current_folders))
        return []

    if len(new_folders) > MAX_BENCHMARKS_PER_JOB:
        logger.info(
            "Found %d folders without cards, limiting to %d per job",
            len(new_folders), MAX_BENCHMARKS_PER_JOB,
        )
        new_folders = new_folders[:MAX_BENCHMARKS_PER_JOB]

    logger.info("Processing %d folders: %s", len(new_folders), new_folders)
    return new_folders


def _download_folder(folder_name: str) -> Path:
    """Download a single EEE folder to a temp directory."""
    target_dir = tempfile.mkdtemp(prefix=f"eee_{folder_name}_")
    logger.info("Downloading EEE folder '%s' to %s", folder_name, target_dir)

    snapshot_download(
        repo_id=EEE_REPO,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=[f"data/{folder_name}/**/*.json"],
    )

    data_path = Path(target_dir) / "data"
    return data_path


def _upload_card(card: dict, benchmark_name: str) -> bool:
    """Upload a generated card to the auto-benchmarkcards dataset."""
    api = HfApi()
    safe_name = benchmark_name.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    remote_path = f"cards/{safe_name}.json"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(card, f, indent=2)
            tmp_path = f.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=remote_path,
            repo_id=CARDS_REPO,
            repo_type="dataset",
            commit_message=f"Auto-generated card: {benchmark_name}",
        )
        logger.info("Uploaded card to %s/%s", CARDS_REPO, remote_path)
        return True

    except Exception:
        logger.exception("Failed to upload card for %s", benchmark_name)
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def process_new_benchmarks(new_folders: list[str]) -> None:
    """Generate and upload cards for all benchmarks in the new folders.

    This runs in a background thread, called from app.py.
    """
    from auto_benchmarkcard.tools.eee.eee_tool import (
        scan_eee_folder,
        eee_to_pipeline_inputs,
    )
    from auto_benchmarkcard.eee_workflow import process_single_benchmark
    from auto_benchmarkcard.workflow import setup_logging_suppression

    setup_logging_suppression(debug_mode=False)

    state = load_state()
    job_record: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "folders": new_folders,
        "results": [],
    }

    for folder_name in new_folders:
        logger.info("Processing folder: %s", folder_name)

        tmp_root = None
        try:
            data_path = _download_folder(folder_name)
            tmp_root = data_path.parent
        except Exception:
            logger.exception("Failed to download folder %s", folder_name)
            job_record["results"].append({
                "folder": folder_name, "status": "download_failed",
            })
            continue

        try:
            scan_result = scan_eee_folder(str(data_path))

            for name, bench in sorted(scan_result.benchmarks.items()):
                inputs = eee_to_pipeline_inputs(bench)

                if not inputs.get("hf_repo"):
                    logger.warning("Skipping %s: no HF repo", name)
                    job_record["results"].append({
                        "folder": folder_name, "benchmark": name, "status": "no_hf_repo",
                    })
                    continue

                card = process_single_benchmark(
                    benchmark_name=name,
                    pipeline_inputs=inputs,
                    base_output_path=str(PERSISTENT_DIR / "output"),
                    debug=False,
                )

                if card:
                    uploaded = _upload_card(card, name)
                    job_record["results"].append({
                        "folder": folder_name, "benchmark": name,
                        "status": "uploaded" if uploaded else "upload_failed",
                    })
                else:
                    job_record["results"].append({
                        "folder": folder_name, "benchmark": name, "status": "generation_failed",
                    })

        except Exception:
            logger.exception("Failed to process folder %s", folder_name)
            job_record["results"].append({
                "folder": folder_name, "status": "scan_failed",
            })

        finally:
            if tmp_root and tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)

        if folder_name not in state["known_folders"]:
            state["known_folders"].append(folder_name)

    job_record["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Summarize
    results = job_record["results"]
    uploaded = sum(1 for r in results if r["status"] == "uploaded")
    failed = sum(1 for r in results if r["status"] not in ("uploaded", "no_hf_repo"))
    skipped = sum(1 for r in results if r["status"] == "no_hf_repo")
    logger.info("Job complete: %d uploaded, %d failed, %d skipped", uploaded, failed, skipped)

    state["jobs"].append(job_record)
    state["jobs"] = state["jobs"][-50:]
    save_state(state)
