#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <smoke|development|held-out|no-regression> <run-dir>" >&2
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

case "$PHASE" in
  smoke|development|held-out|no-regression) ;;
  *) echo "unsupported phase: $PHASE" >&2; exit 2 ;;
esac

if [[ ! -d "$RUN_DIR" ]]; then
  echo "run directory does not exist: $RUN_DIR" >&2
  exit 2
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

if [[ -n "${OPEN_VLM_COMMAND:-}" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM "$TIMEOUT_SECONDS" "$OPEN_VLM_COMMAND" "$PHASE" "$RUN_DIR"
  else
    "$OPEN_VLM_COMMAND" "$PHASE" "$RUN_DIR"
  fi
fi
