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

Webhook receiver that auto-generates benchmark cards when new evaluation data
is merged into [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore).

Generated cards are uploaded to [evaleval/auto-benchmarkcards](https://huggingface.co/datasets/evaleval/auto-benchmarkcards).

## Endpoints

- `POST /webhook` — receives HF webhook events
- `GET /status` — shows recent job history
- `GET /health` — health check

## Setup

Required Space secrets:
- `HF_TOKEN` — HuggingFace token with write access to evaleval/auto-benchmarkcards
- `WEBHOOK_SECRET` — shared secret for webhook verification
