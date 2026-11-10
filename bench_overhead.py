#!/usr/bin/env python3
"""Measures the protocol round-trip overhead that is independent of the data
(small table, vector literal of the same size) so the query numbers can be read
correctly."""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time


def client(binary, d):
    p = subprocess.Popen([binary, "--data-dir", d], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert json.loads(p.stdout.readline()) == {"type": "ready"}
    return p


def req(p, seq, sql):
    p.stdin.write((json.dumps({"id": seq, "op": "execute", "sql": sql}) + "\n").encode())
    p.stdin.flush()
    return json.loads(p.stdout.readline())


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "build/mvdb"
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    nq = 2000
    d = tempfile.mkdtemp()
    p = client(binary, d)
    req(p, 0, f"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
              f"content VARCHAR(16), embedding VECTOR({dim}));")
    vec = ",".join("0.1234" for _ in range(dim))
    req(p, 1, f"INSERT INTO docs VALUES (1,'c0','d0','[{vec}]');")
    sql = (f"SELECT id, L2_DISTANCE(embedding, '[{vec}]') AS distance FROM docs "
           f"ORDER BY distance LIMIT 10;")
    for i in range(20):
        req(p, 100 + i, sql)
    t0 = time.perf_counter()
    for i in range(nq):
        req(p, 1000 + i, sql)
    dt = time.perf_counter() - t0
    print(f"round-trip overhead with {dim}-dim literal: {dt/nq*1e6:.1f} us/query")
    # parse-only cost: same SQL but no distance function
    sql2 = "SELECT id FROM docs ORDER BY id LIMIT 10;"
    t0 = time.perf_counter()
    for i in range(nq):
        req(p, 500000 + i, sql2)
    dt2 = time.perf_counter() - t0
    print(f"minimal query overhead:                 {dt2/nq*1e6:.1f} us/query")


main()
