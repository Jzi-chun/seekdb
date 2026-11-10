#!/bin/bash
# Run the public mysqltest package against a locally built binary (no docker).
# Results land in round2/testout/<name> so they are readable from Windows too.
set -u
cd "$(dirname "$0")"
BIN=${1:-build/mvdb}
NAME=${2:-public}
shift 2 2>/dev/null || true
mkdir -p testout
OUT="testout/$NAME"
rm -rf "$OUT"
python3 protocol_check/run.py \
  --command "[\"$BIN\",\"--data-dir\",\"{data_dir}\"]" \
  --output "$OUT" "$@"
