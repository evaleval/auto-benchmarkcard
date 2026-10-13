#!/usr/bin/env bash
# Run batch generation over a backlog with independent timeout-protected shards.
#
# Usage:
#   scripts/run_sharded.sh <backlog> <outdir> [nshards] [timeout_seconds]
#
# Engine and model selection come from the environment (.env in the repo root is
# sourced if present). Optional: PYTHON_BIN selects the interpreter.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BACKLOG="${1:?usage: run_sharded.sh <backlog> <outdir> [nshards] [timeout]}"
OUT="${2:?usage: run_sharded.sh <backlog> <outdir> [nshards] [timeout]}"
NSHARDS="${3:-4}"
PER_CARD_TIMEOUT="${4:-1200}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v gtimeout)"
elif command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v timeout)"
else
  printf 'error: install GNU timeout (gtimeout on macOS) or provide it on PATH\n' >&2
  exit 2
fi

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

mkdir -p "$OUT"

run_shard() {
  local shard_id="$1"
  local one rc
  one="$(mktemp)"
  while IFS= read -r card; do
    [ -z "$card" ] && continue
    printf '%s\n' "$card" > "$one"
    printf '[shard %s] >>> %s\n' "$shard_id" "$card"
    "$TIMEOUT_BIN" "$PER_CARD_TIMEOUT" env PYTHONPATH=src \
      "$PYTHON_BIN" scripts/batch_generate.py \
      --backlog "$one" \
      -o "$OUT" \
      --no-download \
      >> "$OUT/shard_${shard_id}.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      printf '[shard %s] !!! FAIL rc=%s: %s\n' "$shard_id" "$rc" "$card"
    fi
  done
  rm -f "$one"
}

SHARD_DIR="$(mktemp -d)"
trap 'rm -rf "$SHARD_DIR"' EXIT
grep -vE '^#|^$' "$BACKLOG" |
  awk -v n="$NSHARDS" -v d="$SHARD_DIR" \
    '{ print >> (d "/shard_" (NR % n)) }'

pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
  shard_file="$SHARD_DIR/shard_$i"
  [ -f "$shard_file" ] || continue
  run_shard "$i" < "$shard_file" &
  pids+=("$!")
done
wait "${pids[@]}"

printf 'Done. Cards in %s/output: ' "$OUT"
find "$OUT/output" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
printf 'Field status: PYTHONPATH=src %s scripts/field_status.py --run-dir %s/output\n' \
  "$PYTHON_BIN" "$OUT"
