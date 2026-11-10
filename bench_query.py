#!/usr/bin/env python3
"""Focused query benchmark: separates filtered and unfiltered top-K so the
effect of the lazy equality index is visible.

  python3 tests/bench_query.py --binary build/mvdb --rows 10000 --dim 128
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
        e = dict(os.environ)
        if env:
            e.update(env)
        self.proc = subprocess.Popen([binary, "--data-dir", data_dir],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=e, start_new_session=True)
        self.seq = 0
        assert self._read() == {"type": "ready"}

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
        r = self._read()
        if not r.get("ok"):
            raise RuntimeError(f"{sql[:60]}: {r}")
        return r

    def shutdown(self):
        self.seq += 1
        self.proc.stdin.write((json.dumps({"id": self.seq, "op": "shutdown"}) + "\n").encode())
        self.proc.stdin.flush()
        self._read()
        self.proc.wait(timeout=60)


def run(binary, n, dim, nq, seed, env, workroot, filtered, cardinality):
    d = tempfile.mkdtemp(prefix="mvdb-q-", dir=workroot)
    c = Client(binary, d, env)
    c.req(f"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
          f"content VARCHAR(16), embedding VECTOR({dim}));")
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        v = ",".join(f"{rng.gauss(0, 1):.4f}" for _ in range(dim))
        rows.append(f"({i},'c{i % cardinality}','d{i}','[{v}]')")
    for start in range(0, n, 100):
        c.req("INSERT INTO docs VALUES " + ",".join(rows[start:start + 100]) + ";")
    qrng = random.Random(seed + 5)
    sqls = []
    for j in range(nq):
        v = ",".join(f"{qrng.gauss(0, 1):.4f}" for _ in range(dim))
        lit = f"'[{v}]'"
        if filtered:
            sqls.append(f"SELECT id, L2_DISTANCE(embedding, {lit}) AS distance FROM docs "
                        f"WHERE category = 'c{j % cardinality}' ORDER BY distance LIMIT 10;")
        else:
            sqls.append(f"SELECT id, L2_DISTANCE(embedding, {lit}) AS distance FROM docs "
                        f"ORDER BY distance LIMIT 10;")
    for s in sqls[:10]:
        c.req(s)
    t0 = time.perf_counter()
    for s in sqls:
        c.req(s)
    dt = time.perf_counter() - t0
    c.shutdown()
    shutil.rmtree(d, ignore_errors=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/mvdb")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="mvdb-qroot-")
    try:
        for label, filtered, card in [("unfiltered", False, 8), ("filtered (8 values)", True, 8),
                                      ("filtered (1000 values)", True, 1000)]:
            for envlabel, env in [("index+simd", {}), ("no index", {"MVDB_NO_FILTER_INDEX": "1"})]:
                ts = [run(args.binary, args.rows, args.dim, args.queries, args.seed, env, work,
                          filtered, card) for _ in range(args.rounds)]
                med = statistics.median(ts)
                print(f"{label:24s} {envlabel:11s} {med*1000:8.1f} ms  "
                      f"{med/args.queries*1e6:7.0f} us/query")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
