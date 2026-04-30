"""HF Webhook receiver for auto-benchmarkcard generation.

Listens for PR merge events on evaleval/EEE_datastore and triggers
card generation for new benchmarks in a background thread.
Queued folders are persisted so nothing is lost between webhook events.
"""

import hmac
import logging
import os
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from worker import (
    detect_new_benchmarks,
    process_new_benchmarks,
    load_state,
    save_pending,
    pop_pending,
    PERSISTENT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webhook")

app = FastAPI(title="BenchmarkCard Webhook")

# Max time a generation job can run before we allow new jobs (1 hour)
MAX_JOB_DURATION_SECONDS = 3600

# Track active generation thread (max 1 concurrent)
_active_job: dict = {"thread": None, "started_at": None, "folders": []}
_job_lock = threading.Lock()


def _verify_secret(request_secret: str) -> bool:
    """Verify webhook secret from X-Webhook-Secret header."""
    expected = os.environ.get("WEBHOOK_SECRET", "")
    if not expected:
        logger.warning("WEBHOOK_SECRET not set, skipping verification")
        return True
    return hmac.compare_digest(expected, request_secret)


def _is_merged_pr(payload: dict) -> bool:
    """Check if the webhook payload represents a merged PR."""
    discussion = payload.get("discussion", {})
    return (
        discussion.get("isPullRequest", False)
        and discussion.get("status") == "merged"
    )


def _is_job_timed_out() -> bool:
    """Check if the active job has exceeded the timeout."""
    started = _active_job.get("started_at")
    if not started:
        return False
    try:
        started_dt = datetime.fromisoformat(started)
        elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
        return elapsed > MAX_JOB_DURATION_SECONDS
    except (ValueError, TypeError):
        return False


def _run_generation(new_folders: list[str]):
    """Background worker: generate cards, then drain pending queue."""
    try:
        logger.info("Background generation started for %d folders: %s", len(new_folders), new_folders)
        process_new_benchmarks(new_folders)
        logger.info("Background generation completed")

        # Drain pending queue: process any folders that arrived while we were busy
        while True:
            pending = pop_pending()
            if not pending:
                break
            logger.info("Draining pending queue: %d folders: %s", len(pending), pending)
            # Re-detect to catch any new folders since the pending was saved
            process_new_benchmarks(pending)

    except Exception:
        logger.exception("Background generation failed")
    finally:
        with _job_lock:
            _active_job["thread"] = None
            _active_job["started_at"] = None
            _active_job["folders"] = []


@app.post("/webhook")
async def webhook(request: Request):
    """Receive HF webhook events and trigger card generation."""
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _verify_secret(secret):
        return JSONResponse(status_code=403, content={"error": "invalid secret"})

    payload = await request.json()

    if not _is_merged_pr(payload):
        event_scope = payload.get("event", {}).get("scope", "unknown")
        discussion = payload.get("discussion", {})
        status = discussion.get("status", "unknown")
        return JSONResponse(
            status_code=200,
            content={"action": "ignored", "reason": f"not a merged PR (scope={event_scope}, status={status})"},
        )

    discussion = payload.get("discussion", {})
    pr_title = discussion.get("title", "unknown")
    pr_num = discussion.get("num", "?")
    logger.info("Merged PR detected: #%s '%s'", pr_num, pr_title)

    new_folders = detect_new_benchmarks()
    if not new_folders:
        return JSONResponse(
            status_code=200,
            content={"action": "no_new_benchmarks", "pr": f"#{pr_num} {pr_title}"},
        )

    with _job_lock:
        thread_alive = _active_job["thread"] is not None and _active_job["thread"].is_alive()
        timed_out = thread_alive and _is_job_timed_out()

        if timed_out:
            logger.warning(
                "Active job timed out (started %s), allowing new job",
                _active_job["started_at"],
            )
            # Don't kill old thread (daemon, will die on exit), just reset tracking
            _active_job["thread"] = None
            _active_job["started_at"] = None
            _active_job["folders"] = []
            thread_alive = False

        if thread_alive:
            # Persist to pending queue so they get processed after current job
            save_pending(new_folders)
            logger.info("Job in progress, queued %d folders to pending", len(new_folders))
            return JSONResponse(
                status_code=200,
                content={
                    "action": "queued",
                    "reason": "generation in progress, folders saved to pending queue",
                    "active_folders": _active_job["folders"],
                    "queued_folders": new_folders,
                },
            )

        thread = threading.Thread(target=_run_generation, args=(new_folders,), daemon=True)
        _active_job["thread"] = thread
        _active_job["started_at"] = datetime.now(timezone.utc).isoformat()
        _active_job["folders"] = new_folders
        thread.start()

    return JSONResponse(
        status_code=200,
        content={
            "action": "generation_started",
            "pr": f"#{pr_num} {pr_title}",
            "new_folders": new_folders,
        },
    )


@app.get("/status")
async def status():
    """Return recent job history and current state."""
    state = load_state()

    with _job_lock:
        active = None
        if _active_job["thread"] is not None and _active_job["thread"].is_alive():
            active = {
                "started_at": _active_job["started_at"],
                "folders": _active_job["folders"],
            }

    return {
        "active_job": active,
        "known_folders": len(state.get("known_folders", [])),
        "pending_folders": state.get("pending_folders", []),
        "jobs": state.get("jobs", [])[-20:],
    }


@app.post("/init-known-folders")
async def init_known_folders(request: Request):
    """One-time: mark all current EEE folders as known."""
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _verify_secret(secret):
        return JSONResponse(status_code=403, content={"error": "invalid secret"})

    from huggingface_hub import HfApi
    from worker import save_state

    api = HfApi()
    all_files = api.list_repo_files("evaleval/EEE_datastore", repo_type="dataset")
    current_folders = sorted({
        p.split("/")[1] for p in all_files
        if p.startswith("data/") and len(p.split("/")) >= 2
    })

    state = load_state()
    state["known_folders"] = current_folders
    save_state(state)
    return {"action": "initialized", "known_folders_count": len(current_folders)}


@app.get("/")
async def root():
    """Root endpoint for HF Space UI."""
    return {"service": "BenchmarkCard Webhook", "endpoints": ["/webhook", "/status", "/health", "/init-known-folders"]}


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}
