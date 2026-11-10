#!/bin/bash
# Repeat the public suite N times; results under round2/testout/rep-<i>.
set -u
cd "$(dirname "$0")"
BIN=${BIN:-build/mvdb}
N=${1:-1}
shift || true
mkdir -p testout
for i in $(seq 1 "$N"); do
  bash run_public.sh "$BIN" "rep-$i" "$@" > "testout/rep-$i.log" 2>&1
  echo "run $i: $(grep -c ': PASS' "testout/rep-$i.log") passed, $(grep -c ': FAIL' "testout/rep-$i.log") failed, $(grep -c ': TIMEOUT' "testout/rep-$i.log") timeout"
  grep -E ': (FAIL|TIMEOUT)' "testout/rep-$i.log" | head -5
done
