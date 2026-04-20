---
title: BenchmarkCard Webhook
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BenchmarkCard Webhook

Receives HF webhook events when PRs are merged on `evaleval/EEE_datastore`,
detects new benchmark folders, and automatically generates benchmark cards.

## Endpoints

- `POST /webhook` — HF webhook receiver
- `GET /status` — Job history and active state
- `GET /health` — Health check

## Environment Variables

- `HF_TOKEN` — HuggingFace token with write access to `evaleval/auto-benchmarkcards`
- `WEBHOOK_SECRET` — Secret for webhook verification
- `PERSISTENT_DIR` — Path to persistent storage (default: `/data`)
