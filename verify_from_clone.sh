#!/bin/bash
# End-to-end dry run of what the judge does:
#   clone the branch (from the local bundle), docker build, run protocol_check.
set -u
W=/mnt/c/Users/m1956/Documents/deepseek-harness/default-workspace
LOG="$W/round2/testout/clone_build.log"
mkdir -p "$W/round2/testout"
: > "$LOG"

echo "[1/5] docker daemon" | tee -a "$LOG"
if ! docker info >/dev/null 2>&1; then
  rm -f /var/run/docker.pid
  nohup dockerd --iptables=false --bridge=none >/tmp/dockerd_run.log 2>&1 &
  for i in $(seq 1 40); do sleep 2; docker info >/dev/null 2>&1 && break; done
fi
docker info --format 'server {{.ServerVersion}}' 2>&1 | tee -a "$LOG"

echo "[2/5] materialise the committed tree (git archive of HEAD)" | tee -a "$LOG"
rm -rf /tmp/judge
mkdir -p /tmp/judge
git -C "$W/round2" archive HEAD | tar -x -C /tmp/judge || exit 1
cd /tmp/judge || exit 1
SHA=$(git -C "$W/round2" rev-parse HEAD)
echo "tree of $SHA materialised" | tee -a "$LOG"
du -sh /tmp/judge | tee -a "$LOG"

echo "[3/5] docker build from the cloned tree" | tee -a "$LOG"
if ! docker image inspect ubuntu:24.04 >/dev/null 2>&1; then
  docker pull docker.1panel.live/library/ubuntu:24.04 >>"$LOG" 2>&1 \
    && docker tag docker.1panel.live/library/ubuntu:24.04 ubuntu:24.04 >>"$LOG" 2>&1
fi
docker build --network=host -t mvdb-judge:dry . >>"$LOG" 2>&1
rc=$?
echo "build rc=$rc" | tee -a "$LOG"
[ $rc -ne 0 ] && { tail -20 "$LOG"; exit 1; }

echo "[4/5] handshake" | tee -a "$LOG"
cp -r "$W/round2/protocol_check" /tmp/judge/pc 2>/dev/null
cd /tmp/judge/pc || exit 1
rm -rf out-handshake
python3 run.py --image mvdb-judge:dry --handshake-only --output out-handshake >>"$LOG" 2>&1
echo "handshake rc=$?" | tee -a "$LOG"

echo "[5/5] public cases through the image" | tee -a "$LOG"
rm -rf out-public
python3 run.py --image mvdb-judge:dry --package /tmp/judge/mysqltest --output out-public >>"$LOG" 2>&1
echo "public rc=$?" | tee -a "$LOG"
grep -E 'PASS|FAIL|TIMEOUT' "$LOG" | tail -22
