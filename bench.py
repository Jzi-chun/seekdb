#!/usr/bin/env python3
"""MiniVectorDB benchmark modelled on the official round-2 workload:
10,000 rows x 128 dims, batches of 100 rows per commit, then 1000 top-K queries
(K=10), half of them with an equality filter on a normal column.

Reports wall time, on-disk bytes and peak RSS, plus A/B runs where SIMD or the
second worker thread are switched off (MVDB_SIMD / MVDB_THREADS).

  python3 tests/bench.py --binary build/mvdb
"""
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


class Client:
    def __init__(self, binary, data_dir, env=None):
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self.proc = subprocess.Popen([binary, "--data-dir", data_dir],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=self.env,
                                     start_new_session=True)
        self.seq = 0
        if self._read() != {"type": "ready"}:
            raise RuntimeError("bad handshake")

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("database exited")
        return json.loads(line.decode())

    def req(self, sql, mode="exact"):
        self.seq += 1
        self.proc.stdin.write((json.dumps({"id": self.seq, "op": "execute", "sql": sql,
                                           "search_mode": mode}) + "\n").encode())
        self.proc.stdin.flush()
        resp = self._read()
        if not resp.get("ok"):
            raise RuntimeError(f"{sql[:80]}: {resp}")
        return resp

    def shutdown(self):
        self.seq += 1
        self.proc.stdin.write((json.dumps({"id": self.seq, "op": "shutdown"}) + "\n").encode())
        self.proc.stdin.flush()
        self._read()
        self.proc.wait(timeout=60)


def rss_peak_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
               if os.path.isfile(os.path.join(path, f)))


def make_dataset(n, dim, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        v = [round(rng.gauss(0, 1), 4) for _ in range(dim)]
        rows.append((i, f"c{i % 8}", ",".join(f"{x:g}" for x in v)))
    return rows


def one_run(binary, n, dim, nq, batch, seed, env, workroot, keep=False):
    d = tempfile.mkdtemp(prefix="mvdb-bench-", dir=workroot)
    rows = make_dataset(n, dim, seed)
    c = Client(binary, d, env)
    c.req("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
          "content VARCHAR(16), embedding VECTOR(%d));" % dim)
    t0 = time.perf_counter()
    for start in range(0, n, batch):
        chunk = rows[start:start + batch]
        vals = ",".join(f"({i},'{cat}','d{i}','[{v}]')" for i, cat, v in chunk)
        c.req(f"INSERT INTO docs VALUES {vals};")
    t_write = time.perf_counter() - t0

    rng = random.Random(seed + 1)
    queries = []
    for j in range(nq):
        qv = [round(rng.gauss(0, 1), 4) for _ in range(dim)]
        lit = "'[" + ",".join(f"{x:g}" for x in qv) + "]'"
        cat = f"c{j % 8}"
        if j % 2 == 0:
            sql = (f"SELECT id, L2_DISTANCE(embedding, {lit}) AS distance FROM docs "
                   f"WHERE category = '{cat}' ORDER BY distance LIMIT 10;")
        else:
            sql = (f"SELECT id, L2_DISTANCE(embedding, {lit}) AS distance FROM docs "
                   f"ORDER BY distance LIMIT 10;")
        queries.append(sql)

    # warm up (platform does a warmup that is not counted in T)
    for sql in queries[:20]:
        c.req(sql)
    t0 = time.perf_counter()
    for sql in queries:
        resp = c.req(sql)
        if len(resp["rows"]) != 10:
            raise RuntimeError("wrong result size")
    t_query = time.perf_counter() - t0
    bytes_on_disk = dir_bytes(d)
    peak = rss_peak_kb(c.proc.pid)
    c.shutdown()
    if not keep:
        shutil.rmtree(d, ignore_errors=True)
    return dict(t_write=t_write, t_query=t_query, bytes=bytes_on_disk, peak_kb=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/mvdb")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    workroot = tempfile.mkdtemp(prefix="mvdb-benchroot-")
    variants = [
        ("tuned (avx2+2 threads)", {}),
        ("no SIMD", {"MVDB_SIMD": "0"}),
        ("1 thread", {"MVDB_THREADS": "0"}),
        ("1 thread, no SIMD", {"MVDB_THREADS": "0", "MVDB_SIMD": "0"}),
    ]
    results = {}
    try:
        for label, env in variants:
            runs = [one_run(args.binary, args.rows, args.dim, args.queries, args.batch,
                            args.seed, env, workroot) for _ in range(args.rounds)]
            med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            results[label] = med
            print(f"{label:26s} write {med['t_write']*1000:8.0f} ms  "
                  f"query {med['t_query']*1000:8.0f} ms  "
                  f"disk {med['bytes']/1e6:6.1f} MB  peakRSS {med['peak_kb']/1024:6.1f} MB")
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    base = results["tuned (avx2+2 threads)"]
    print("\nrelative to the unoptimised variant (1 thread, no SIMD):")
    slow = results["1 thread, no SIMD"]
    print(f"  write speedup x{slow['t_write']/base['t_write']:.2f}, "
          f"query speedup x{slow['t_query']/base['t_query']:.2f}")
    with open("bench_result.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
