#!/usr/bin/env bash
# Sharded batch card generator. Paths are relative to this checkout.
# Usage: run_sharded.sh <backlog> <outdir> [nshards] [per_card_timeout_sec]
#
# Splits the backlog round-robin into N shards run concurrently. Each card runs in its own
# batch_generate process under `gtimeout` (RITS hang watchdog), and all shards write to one
# output dir. already_generated() makes this idempotent/resumable: re-run the same command to
# retry only the timed-out/missing cards. Per-shard stdout goes to <outdir>/shard_<i>.log.
set -u
cd "$(dirname "$0")" || exit 1

BACKLOG="${1:?usage: run_sharded.sh <backlog> <outdir> [nshards] [timeout]}"
OUT="${2:?usage: run_sharded.sh <backlog> <outdir> [nshards] [timeout]}"
NSHARDS="${3:-4}"
PER_CARD_TIMEOUT="${4:-1200}"

# Override these for an environment or credentials file outside this checkout.
VENV="${BENCHMARKCARD_PYTHON:-$PWD/.venv/bin/python}"
ENV_FILE="${BENCHMARKCARD_ENV_FILE:-$PWD/.env}"
# Model IDENTITY (card_info.llm) stays the corpus checkpoint. RITS retired this deployment
# on 2026-07-04, so the composer is SERVED via OpenRouter (same checkpoint). Provider pinned
# to Baidu fp8 (supports server-enforced json_schema, matching broad_run's vLLM guided
# decoding; Novita/SiliconFlow do not), no fallbacks. Pre-authorized fallback if Baidu
# misbehaves: AkashML. FactReasoner and everything else stay on RITS.
MODEL=deepseek-ai/DeepSeek-V4-Flash
OR_MODEL_ID=deepseek/deepseek-v4-flash
# Baidu-only by default: silent failover to a misbehaving provider returns empty
# 200s that gut extraction without any error (2026-08-24 incident). A loud 429 is
# recoverable; a quiet empty parse is not. Override OR_PROVIDERS deliberately.
OR_PROVIDERS="${OR_PROVIDERS:-Baidu}"
OR_QUANT=fp8

# Load API keys into child processes; an exported environment is sufficient too.
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
elif [ -n "${BENCHMARKCARD_ENV_FILE:-}" ]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

if [ ! -x "$VENV" ]; then
  echo "Python not executable: $VENV (set BENCHMARKCARD_PYTHON)" >&2
  exit 1
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
    gtimeout "$PER_CARD_TIMEOUT" env RITS_COMPOSER_MODEL="$MODEL" LLM_ENGINE_TYPE=rits \
      COMPOSER_ENGINE_TYPE=openrouter OPENROUTER_MODEL_ID="$OR_MODEL_ID" \
      OPENROUTER_PROVIDER_ORDER="$OR_PROVIDERS" OPENROUTER_QUANTIZATION="$OR_QUANT" \
      PYTHONPATH=src "$VENV" scripts/batch_generate.py --backlog "$one" -o "$OUT" --no-download \
      >> "$OUT/shard_${shard_id}.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && printf '[shard %s] !!! FAIL rc=%s: %s\n' "$shard_id" "$rc" "$card"
  done
  rm -f "$one"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
grep -vE '^#|^$' "$BACKLOG" | awk -v n="$NSHARDS" -v d="$tmp" '{ print >> (d "/shard_" (NR % n)) }'

pids=()
for i in $(seq 0 $((NSHARDS - 1))); do
  f="$tmp/shard_$i"
  [ -f "$f" ] || continue
  run_shard "$i" < "$f" &
  pids+=("$!")
done
wait "${pids[@]}"

echo
echo "Done. Cards in $OUT/output:"
ls -d "$OUT"/output/*/ 2>/dev/null | wc -l
echo "Gate it:  PYTHONPATH=src $VENV scripts/compute_gate_metrics.py --run-dir $OUT/output"
