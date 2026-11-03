#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <smoke|development|held-out|no-regression|logit-lens> <run-dir>" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

PHASE="$1"
RUN_DIR="$2"
PYTHON_BIN="${OPEN_VLM_PYTHON:-python3}"
DESTINATION="${OPEN_VLM_ARTIFACT_DESTINATION:-}"
UPLOADER="${OPEN_VLM_UPLOADER:-}"
DELETER="${OPEN_VLM_DELETER:-}"
STOPPER="${OPEN_VLM_STOPPER:-}"
POD_ID="${OPEN_VLM_POD_ID:-}"
TIMEOUT_SECONDS="${OPEN_VLM_TIMEOUT_SECONDS:-3600}"
WATCHDOG_CLI="${OPEN_VLM_WATCHDOG_CLI:-runpodctl}"
BALANCE_BEFORE_USD="${OPEN_VLM_BALANCE_BEFORE_USD:-}"
SPEND_CAP_USD="${OPEN_VLM_SPEND_CAP_USD:-}"
WATCHDOG_TIMEOUT_SECONDS="${OPEN_VLM_WATCHDOG_TIMEOUT_SECONDS:-3600}"
WATCHDOG_POLL_SECONDS="${OPEN_VLM_WATCHDOG_POLL_SECONDS:-30}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_PID=""

case "$PHASE" in
  smoke|development|held-out|no-regression|logit-lens) ;;
  *) echo "unsupported phase: $PHASE" >&2; exit 2 ;;
esac

if [[ ! -d "$RUN_DIR" ]]; then
  echo "run directory does not exist: $RUN_DIR" >&2
  exit 2
fi

positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v number="$1" 'BEGIN { exit !(number > 0) }'
}

if [[ -n "$POD_ID" ]]; then
  if ! command -v "$WATCHDOG_CLI" >/dev/null 2>&1; then
    echo "watchdog CLI not found: $WATCHDOG_CLI" >&2
    exit 2
  fi
  if ! positive_number "$BALANCE_BEFORE_USD" || ! positive_number "$SPEND_CAP_USD" || \
     ! awk -v balance="$BALANCE_BEFORE_USD" -v cap="$SPEND_CAP_USD" 'BEGIN { exit !(cap <= balance) }'; then
    echo "watchdog requires a positive registered balance and spend cap no greater than balance" >&2
    exit 2
  fi
  if [[ ! "$WATCHDOG_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ||
        ! "$WATCHDOG_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "watchdog timeout and poll interval must be positive integers" >&2
    exit 2
  fi
  if ! "$WATCHDOG_CLI" user >/dev/null || ! "$WATCHDOG_CLI" pod list >/dev/null; then
    echo "watchdog RunPod preflight failed" >&2
    exit 2
  fi
fi

artifact_checksum() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode('utf-8'))
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
}

cleanup() {
  local exit_code=$?
  local local_checksum remote_checksum
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
  if [[ -z "$DESTINATION" || -z "$UPLOADER" || -z "$DELETER" || -z "$POD_ID" ]]; then
    echo "CLEANUP_BLOCKED: upload destination, uploader, deleter, and pod id are required" >&2
    if [[ -n "$STOPPER" && -n "$POD_ID" ]]; then
      "$STOPPER" "$POD_ID" --preserve-volume || true
      echo "STOPPED_CLEANUP_BLOCKED: pod stopped; volume preserved" >&2
    fi
    exit 3
  fi
  local_checksum="$(artifact_checksum "$RUN_DIR")"
  if ! remote_checksum="$("$UPLOADER" "$RUN_DIR" "$DESTINATION")"; then
    remote_checksum=""
  fi
  if [[ "$remote_checksum" != "$local_checksum" ]]; then
    echo "CLEANUP_BLOCKED: remote checksum mismatch" >&2
    if [[ -n "$STOPPER" ]]; then
      "$STOPPER" "$POD_ID" --preserve-volume || true
      echo "STOPPED_CLEANUP_BLOCKED: pod stopped; volume preserved" >&2
    fi
    exit 4
  fi
  "$DELETER" "$POD_ID"
  exit "$exit_code"
}
trap cleanup EXIT

if [[ -n "$POD_ID" ]]; then
  WATCHDOG_LOG="${OPEN_VLM_WATCHDOG_LOG:-$RUN_DIR/watchdog.jsonl}"
  "$PYTHON_BIN" "$SCRIPT_DIR/scripts/open_vlm_runpod_guard.py" \
    --cli-path "$WATCHDOG_CLI" \
    --pod-id "$POD_ID" \
    --balance-before-usd "$BALANCE_BEFORE_USD" \
    --spend-cap-usd "$SPEND_CAP_USD" \
    --timeout-seconds "$WATCHDOG_TIMEOUT_SECONDS" \
    --poll-seconds "$WATCHDOG_POLL_SECONDS" \
    --log-path "$WATCHDOG_LOG" > "$RUN_DIR/watchdog-process.log" 2>&1 &
  WATCHDOG_PID=$!
  sleep 1
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    wait "$WATCHDOG_PID" 2>/dev/null || true
    WATCHDOG_PID=""
    echo "watchdog exited during startup; refusing to run the phase" >&2
    exit 5
  fi
fi

if [[ -n "${OPEN_VLM_COMMAND:-}" ]]; then
  COMMAND=("$OPEN_VLM_COMMAND" "$PHASE" "$RUN_DIR")
else
  case "$PHASE" in
    smoke) PHASE_FLAG="--smoke" ;;
    development) PHASE_FLAG="--development" ;;
    held-out) PHASE_FLAG="--held-out" ;;
    no-regression) PHASE_FLAG="--no-regression" ;;
    logit-lens) PHASE_FLAG="--logit-lens" ;;
  esac
  export PYTHONPATH="$SCRIPT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
  COMMAND=("$PYTHON_BIN" "$SCRIPT_DIR/scripts/open_vlm_pilot.py" "$PHASE_FLAG" --run-dir "$RUN_DIR")
fi

if command -v timeout >/dev/null 2>&1; then
  timeout --signal=TERM "$TIMEOUT_SECONDS" "${COMMAND[@]}"
else
  "${COMMAND[@]}"
fi
