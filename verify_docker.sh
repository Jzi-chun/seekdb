#!/bin/bash
# Builds the competition image in WSL docker and runs the official protocol
# checker against the image (this is exactly what the judge does).
set -u
cd "$(dirname "$0")"
LOG=testout/docker_verify.log
mkdir -p testout
: > "$LOG"

echo "[1/4] starting dockerd" | tee -a "$LOG"
if ! docker info >/dev/null 2>&1; then
  rm -f /var/run/docker.pid
  nohup dockerd --iptables=false --bridge=none > /tmp/dockerd_run.log 2>&1 &
  for i in $(seq 1 40); do
    sleep 2
    if docker info >/dev/null 2>&1; then break; fi
  done
fi
docker info --format 'server {{.ServerVersion}} driver {{.Driver}}' 2>&1 | tee -a "$LOG"

echo "[2/4] docker build" | tee -a "$LOG"
if ! docker image inspect ubuntu:24.04 >/dev/null 2>&1; then
  docker pull docker.1panel.live/library/ubuntu:24.04 >>"$LOG" 2>&1 \
    && docker tag docker.1panel.live/library/ubuntu:24.04 ubuntu:24.04 >>"$LOG" 2>&1
fi
docker build --network=host -t minivectordb:ci . >>"$LOG" 2>&1
rc=$?
echo "build rc=$rc" | tee -a "$LOG"
if [ $rc -ne 0 ]; then
  tail -30 "$LOG"
  exit 1
fi

echo "[3/4] handshake through the image" | tee -a "$LOG"
rm -rf testout/docker-handshake
python3 protocol_check/run.py --image minivectordb:ci --handshake-only \
  --output testout/docker-handshake >>"$LOG" 2>&1
echo "handshake rc=$?" | tee -a "$LOG"

echo "[4/4] all public cases through the image" | tee -a "$LOG"
rm -rf testout/docker-public
python3 protocol_check/run.py --image minivectordb:ci \
  --output testout/docker-public >>"$LOG" 2>&1
echo "public rc=$?" | tee -a "$LOG"
tail -25 testout/docker-public/../docker_verify.log | grep -E 'PASS|FAIL|TIMEOUT|rc=' | tail -25
