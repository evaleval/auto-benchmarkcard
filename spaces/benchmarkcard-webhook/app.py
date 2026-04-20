"""HF Webhook receiver for auto-benchmarkcard generation.

Listens for PR merge events on evaleval/EEE_datastore and triggers
card generation for new benchmarks in a background thread.
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
    save_state,
    PERSISTENT_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webhook")

app = FastAPI(title="BenchmarkCard Webhook")

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


def _run_generation(new_folders: list[str]):
    """Background worker: generate cards for new benchmark folders."""
    try:
        logger.info("Background generation started for %d folders: %s", len(new_folders), new_folders)
        process_new_benchmarks(new_folders)
        logger.info("Background generation completed")
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
    # Verify secret
    secret = request.headers.get("X-Webhook-Secret", "")
    if not _verify_secret(secret):
        return JSONResponse(status_code=403, content={"error": "invalid secret"})

    payload = await request.json()

    # Only act on merged PRs
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

    # Detect new benchmark folders
    new_folders = detect_new_benchmarks()
    if not new_folders:
        return JSONResponse(
            status_code=200,
            content={"action": "no_new_benchmarks", "pr": f"#{pr_num} {pr_title}"},
        )

    # Spawn background thread if not already running
    with _job_lock:
        if _active_job["thread"] is not None and _active_job["thread"].is_alive():
            return JSONResponse(
                status_code=200,
                content={
                    "action": "queued",
                    "reason": "generation already in progress",
                    "active_folders": _active_job["folders"],
                    "new_folders": new_folders,
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
        "jobs": state.get("jobs", [])[-20:],
    }


@app.get("/")
async def root():
    """Root endpoint for HF Space UI."""
    return {"service": "BenchmarkCard Webhook", "endpoints": ["/webhook", "/status", "/health"]}


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}
