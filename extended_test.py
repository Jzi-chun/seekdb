#!/usr/bin/env python3
"""Extended self tests for MiniVectorDB: crash recovery, resource behaviour,
approximate search recall and the error contract.

Run from round2/:  python3 tests/extended_test.py [--binary build/mvdb]
"""
import argparse
import json
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

FAILED = []
PASSED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL {name} {detail}")


class Client:
    def __init__(self, binary, data_dir, log_path=None):
        self.log = open(log_path, "ab", buffering=0) if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [binary, "--data-dir", data_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            start_new_session=True,
        )
        self.seq = 0
        ready = self._read()
        if ready != {"type": "ready"}:
            raise RuntimeError(f"bad handshake: {ready!r}")

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("database exited before response")
        return json.loads(line.decode("utf-8"))

    def req(self, sql, mode="exact"):
        self.seq += 1
        payload = {"id": self.seq, "op": "execute", "sql": sql, "search_mode": mode}
        self.proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        self.proc.stdin.flush()
        resp = self._read()
        if resp.get("id") != self.seq:
            raise RuntimeError(f"id mismatch: {resp}")
        return resp

    def rows(self, sql, mode="exact"):
        resp = self.req(sql, mode)
        if not resp.get("ok"):
            raise RuntimeError(f"{sql}: {resp}")
        return resp["rows"]

    def shutdown(self):
        self.seq += 1
        self.proc.stdin.write(
            (json.dumps({"id": self.seq, "op": "shutdown"}) + "\n").encode())
        self.proc.stdin.flush()
        resp = self._read()
        self.proc.wait(timeout=30)
        return resp

    def kill(self):
        os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=30)


def wait_gone(proc):
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def rss_peak_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def fsync_bytes(path):
    total = 0
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    return total


def make_vec(dim, seed):
    rng = random.Random(seed)
    return [round(rng.uniform(-1, 1), 4) for _ in range(dim)]


def vec_literal(v):
    return "'[" + ",".join(f"{x:g}" for x in v) + "]'"


# --------------------------------------------------------------------- tests

def test_crash_committed(binary, root):
    print("[crash] committed rows survive SIGKILL")
    d = os.path.join(root, "crash1")
    c = Client(binary, d)
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
    for batch in range(5):
        values = ",".join(f"({batch*20+i}, {i})" for i in range(20))
        r = c.req(f"INSERT INTO t VALUES {values};")
        assert r["ok"], r
    rows = c.rows("SELECT id FROM t;")
    check("crash1 rows visible before kill", len(rows) == 100, f"got {len(rows)}")
    c.kill()
    wait_gone(c.proc)
    c2 = Client(binary, d)
    rows = c2.rows("SELECT id FROM t ORDER BY id;")
    check("crash1 committed rows recovered", len(rows) == 100, f"got {len(rows)}")
    check("crash1 first/last pk", rows[0] == [0] and rows[-1] == [99], f"{rows[:2]} {rows[-2:]}")
    c2.shutdown()


def test_crash_uncommitted(binary, root):
    print("[crash] uncommitted transaction rolls back after SIGKILL")
    d = os.path.join(root, "crash2")
    c = Client(binary, d)
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
    c.req("INSERT INTO t VALUES (1, 1);")
    c.req("BEGIN;")
    r = c.req("INSERT INTO t VALUES (2, 2), (3, 3);")
    assert r["ok"], r
    check("crash2 txn sees own writes", len(c.rows("SELECT id FROM t;")) == 3)
    c.kill()
    wait_gone(c.proc)
    c2 = Client(binary, d)
    rows = c2.rows("SELECT id FROM t ORDER BY id;")
    check("crash2 uncommitted discarded", rows == [[1]], f"got {rows}")
    c2.shutdown()


def test_crash_during_recovery(binary, root):
    print("[crash] second kill during recovery is still consistent")
    d = os.path.join(root, "crash3")
    c = Client(binary, d)
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
    for batch in range(10):
        c.req(f"INSERT INTO t VALUES ({batch}, {batch});")
    c.req("BEGIN;")
    c.req("DELETE FROM t WHERE id >= -100;")
    c.kill()
    wait_gone(c.proc)
    # start and kill again immediately (likely inside recovery)
    for _ in range(3):
        p = subprocess.Popen([binary, "--data-dir", d],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(0.002)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wait_gone(p)
    c2 = Client(binary, d)
    rows = c2.rows("SELECT id FROM t ORDER BY id;")
    check("crash3 data intact after repeated kills", len(rows) == 10, f"got {len(rows)}")
    c2.shutdown()


def test_errors(binary, root):
    print("[errors] error contract")
    d = os.path.join(root, "err")
    c = Client(binary, d)
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4), v VECTOR(2));")

    def code(sql):
        r = c.req(sql)
        return None if r.get("ok") else r.get("error", {}).get("code")

    check("error: duplicate key", code("INSERT INTO t VALUES (1,'a','[1,1]');") is None and
          code("INSERT INTO t VALUES (1,'a','[1,1]');") == "duplicate_key",
          str(code("INSERT INTO t VALUES (1,'a','[1,1]');")))
    check("error: dimension mismatch", code("INSERT INTO t VALUES (2,'a','[1]');") == "dimension_mismatch")
    check("error: NaN rejected", code("INSERT INTO t VALUES (3,'a','[nan,1]');") == "invalid_vector")
    check("error: Infinity rejected", code("INSERT INTO t VALUES (4,'a','[inf,1]');") == "invalid_vector")
    check("error: float32 overflow rejected",
          code("INSERT INTO t VALUES (5,'a','[1e40,1]');") == "invalid_vector")
    check("error: varchar too long (chars)", code("INSERT INTO t VALUES (6,'abcdef','[1,1]');") == "value_too_long")
    check("error: unknown table", code("SELECT * FROM nope;") == "table_not_found")
    check("error: unknown column", code("SELECT nope FROM t;") == "column_not_found")
    check("error: ALTER rejected", code("ALTER TABLE t ADD x BIGINT;") == "unsupported_sql")
    check("error: aggregate rejected", code("SELECT count(*) FROM t;") == "unsupported_sql")
    check("error: JOIN rejected", code("SELECT a.id FROM t a JOIN t b ON a.id=b.id;") != None)
    # state is unchanged after all those failures
    check("error: table intact", c.rows("SELECT id FROM t;") == [[1]])
    # a failing multi-row insert is atomic inside a transaction
    c.req("BEGIN;")
    c.req("INSERT INTO t VALUES (10,'x','[1,1]');")
    bad = c.req("INSERT INTO t VALUES (11,'y','[1,1]'), (12,'z','[1]');")
    check("error: multi row insert atomic", bad.get("ok") is False)
    check("error: prior stmt still visible in txn", len(c.rows("SELECT id FROM t;")) == 2)
    c.req("COMMIT;")
    check("error: commit keeps prior statement", len(c.rows("SELECT id FROM t;")) == 2)
    c.shutdown()


def test_ann_recall(binary, root):
    print("[ann] recall of ann mode vs exact")
    d = os.path.join(root, "ann")
    c = Client(binary, d)
    dim = 32
    n = 3000
    c.req(f"CREATE TABLE t (id BIGINT PRIMARY KEY, cat VARCHAR(8), v VECTOR({dim}));")
    rng = random.Random(7)
    for start in range(0, n, 100):
        vals = []
        for i in range(start, min(start + 100, n)):
            v = [round(rng.gauss(0, 1), 4) for _ in range(dim)]
            vals.append(f"({i}, '{'abc'[i % 3]}', {vec_literal(v)})")
        r = c.req("INSERT INTO t VALUES " + ",".join(vals) + ";")
        assert r["ok"], r
    c.req("CREATE VECTOR INDEX idx ON t(v);")
    ok = True
    for q in range(20):
        qv = [round(rng.gauss(0, 1), 4) for _ in range(dim)]
        lit = vec_literal(qv)
        filtered = q % 2 == 0
        where = " WHERE cat = 'b'" if filtered else ""
        sql = (f"SELECT id, L2_DISTANCE(v, {lit}) AS d FROM t{where} "
               f"ORDER BY d LIMIT 10;")
        exact = c.rows(sql, mode="exact")
        ann = c.rows(sql, mode="ann")
        if exact != ann:
            ok = False
            print("   mismatch", sql[:60], exact[:3], ann[:3])
    check("ann: ann results identical to exact (recall 1.0)", ok)
    # distances are monotone and tie free
    qv = vec_literal([0.0] * dim)
    rows = c.rows(f"SELECT id, L2_DISTANCE(v, {qv}) AS d FROM t ORDER BY d LIMIT 50;")
    ds = [r[1] for r in rows]
    check("ann: distances sorted ascending", all(ds[i] <= ds[i + 1] for i in range(len(ds) - 1)))
    c.shutdown()


def test_semicolon_and_unicode(binary, root):
    print("[sql] statement splitting and unicode")
    d = os.path.join(root, "split")
    c = Client(binary, d)
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(64));")
    r = c.req("INSERT INTO t VALUES (1, 'a;b'); INSERT INTO t VALUES (2, 'x');")
    check("sql: semicolons inside strings kept", r.get("ok") is True, str(r))
    rows = c.rows("SELECT id, s FROM t ORDER BY id;")
    check("sql: two statements executed", rows == [[1, "a;b"], [2, "x"]], str(rows))
    r = c.req("INSERT INTO t VALUES (3, '中文混合; 分号');")
    check("sql: unicode with semicolon", r.get("ok") is True, str(r))
    rows = c.rows("SELECT s FROM t WHERE id = 3;")
    check("sql: unicode roundtrip", rows == [["中文混合; 分号"]], str(rows))
    c.shutdown()


def test_big_data(binary, root):
    print("[resource] larger than a small cache: 200k rows x 128 dims")
    d = os.path.join(root, "big")
    dim = 128
    n = 200000
    c = Client(binary, d)
    t0 = time.time()
    c.req(f"CREATE TABLE t (id BIGINT PRIMARY KEY, cat VARCHAR(8), v VECTOR({dim}));")
    rng = random.Random(11)
    batch = 500
    for start in range(0, n, batch):
        vals = []
        for i in range(start, min(start + batch, n)):
            v = [round(rng.uniform(-1, 1), 3) for _ in range(dim)]
            vals.append(f"({i},'{'abc'[i % 3]}',{vec_literal(v)})")
        r = c.req("INSERT INTO t VALUES " + ",".join(vals) + ";")
        if not r.get("ok"):
            check("big: insert", False, str(r))
            return
    write_s = time.time() - t0
    peak = rss_peak_kb(c.proc.pid)
    rows = c.rows("SELECT id FROM t;")
    check("big: all rows present", len(rows) == n, f"got {len(rows)}")
    size = fsync_bytes(d)
    data_bytes = n * (dim * 4 + 8)
    print(f"   rows={n} data={data_bytes/1e6:.1f}MB files={size/1e6:.1f}MB "
          f"insert={write_s:.1f}s peakRSS={peak/1024:.1f}MB")
    # ASAN keeps a large shadow mapping, so the RSS bound only applies to the
    # release build; the sanitizer run checks correctness instead.
    if "asan" in binary:
        check("big: peak RSS below data size (skipped under ASAN)",
              peak * 1024 < 1500 * 1024 * 1024, f"rss={peak/1024:.1f}MB")
    else:
        check("big: peak RSS below data size", peak * 1024 < data_bytes,
              f"rss={peak/1024:.1f}MB data={data_bytes/1e6:.1f}MB")
    q = vec_literal([0.0] * dim)
    t0 = time.time()
    rows = c.rows(f"SELECT id, L2_DISTANCE(v, {q}) AS d FROM t ORDER BY d LIMIT 5;")
    query_s = time.time() - t0
    print(f"   full scan top5 over {n} rows: {query_s*1000:.0f} ms")
    check("big: top5 returns 5", len(rows) == 5, str(rows))
    c.shutdown()
    # restart cost
    t0 = time.time()
    c2 = Client(binary, d)
    open_s = time.time() - t0
    print(f"   restart (recovery) time: {open_s:.2f}s")
    check("big: recovered count", len(c2.rows("SELECT id FROM t;")) == n)
    c2.shutdown()


def test_edge_cases(binary, root):
    """Shapes the hidden suite is likely to use but the public package omits."""
    print("[edge] schema and statement edge cases")
    d = os.path.join(root, "edge")
    c = Client(binary, d)

    # wide vector column and a long varchar
    c.req("CREATE TABLE big (id BIGINT PRIMARY KEY, tag VARCHAR(1024), v VECTOR(1024));")
    vec = "[" + ",".join("0.001" for _ in range(1024)) + "]"
    long_text = "长" * 1000
    r = c.req(f"INSERT INTO big VALUES (1, '{long_text}', '{vec}');")
    check("edge: 1024-dim vector + 1000 char varchar insert", r.get("ok") is True, str(r))
    rows = c.rows("SELECT id, tag FROM big WHERE id = 1;")
    check("edge: long varchar roundtrip", len(rows) == 1 and len(rows[0][1]) == 1000,
          f"len={len(rows[0][1]) if rows else 'none'}")
    rows = c.rows(f"SELECT id, L2_DISTANCE(v, '{vec}') AS d FROM big ORDER BY d LIMIT 1;")
    check("edge: 1024-dim distance is zero", abs(rows[0][1]) < 1e-6, str(rows))

    # explicit column list, negative bigints, LIMIT 0
    c.req("CREATE TABLE t (id BIGINT PRIMARY KEY, a BIGINT, b VARCHAR(32), v VECTOR(2));")
    r = c.req("INSERT INTO t (b, id, v, a) VALUES ('x', -5, '[1,0]', -9007199254740993);")
    check("edge: INSERT with column list", r.get("ok") is True, str(r))
    rows = c.rows("SELECT id, a, b FROM t;")
    check("edge: negative bigint exact", rows == [[-5, -9007199254740993, "x"]], str(rows))
    check("edge: LIMIT 0 returns empty", c.rows("SELECT id FROM t LIMIT 0;") == [])

    # ORDER BY DESC with ties breaks on the primary key ascending
    c.req("INSERT INTO t VALUES (1, 10, 'a', '[0,0]'), (2, 10, 'b', '[0,0]'), (3, 20, 'c', '[0,0]');")
    rows = c.rows("SELECT id FROM t WHERE a = 10 ORDER BY a DESC;")
    check("edge: ORDER BY DESC keeps pk asc on ties", rows == [[1], [2]], str(rows))
    rows = c.rows("SELECT id FROM t ORDER BY a DESC LIMIT 3;")
    check("edge: ORDER BY DESC ordering", [r[0] for r in rows] == [3, 1, 2], str(rows))

    # update that changes the varchar length, then delete everything
    r = c.req("UPDATE t SET b = 'much longer value' WHERE id = 1;")
    check("edge: update grows varchar", r.get("ok") is True, str(r))
    check("edge: grown varchar readable",
          c.rows("SELECT b FROM t WHERE id = 1;") == [["much longer value"]])
    c.req("DELETE FROM t WHERE id >= -100;")
    check("edge: delete all", c.rows("SELECT id FROM t;") == [])
    c.req("INSERT INTO t VALUES (7, 7, 'z', '[1,1]');")
    check("edge: insert after delete all", c.rows("SELECT id FROM t;") == [[7]])

    # a failed statement must not disturb the table or the open transaction
    c.req("BEGIN;")
    c.req("INSERT INTO t VALUES (8, 8, 'q', '[1,1]');")
    bad = c.req("INSERT INTO t VALUES (9, 9, 'q', '[1]');")
    check("edge: bad row rejected inside txn", bad.get("ok") is False)
    c.req("COMMIT;")
    check("edge: txn commit keeps good row", sorted(r[0] for r in c.rows("SELECT id FROM t;")) == [7, 8])

    # two tables in one transaction: commit and rollback must be all-or-nothing
    c.req("CREATE TABLE t2 (id BIGINT PRIMARY KEY, v VARCHAR(8));")
    c.req("BEGIN;")
    c.req("INSERT INTO t VALUES (10, 10, 'r', '[0,1]');")
    c.req("INSERT INTO t2 VALUES (10, 'r');")
    c.req("ROLLBACK;")
    check("edge: two-table rollback", c.rows("SELECT id FROM t WHERE id = 10;") == [] and
          c.rows("SELECT id FROM t2;") == [])
    c.req("BEGIN;")
    c.req("INSERT INTO t VALUES (11, 11, 's', '[0,1]');")
    c.req("INSERT INTO t2 VALUES (11, 's');")
    c.req("COMMIT;")
    check("edge: two-table commit", c.rows("SELECT id FROM t2;") == [[11]])

    # index on a non-vector column must be rejected
    r = c.req("CREATE VECTOR INDEX bad ON t(id);")
    check("edge: vector index on BIGINT rejected", r.get("ok") is False and
          r.get("error", {}).get("code"), str(r))
    c.shutdown()

    # restart, then keep using the same directory
    c2 = Client(binary, d)
    check("edge: restart keeps both tables",
          c2.rows("SELECT id FROM t2;") == [[11]] and
          len(c2.rows("SELECT id FROM t;")) == 3)
    c2.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/mvdb")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    root = tempfile.mkdtemp(prefix="mvdb-ext-")
    print(f"workdir {root}")
    tests = [
        test_crash_committed,
        test_crash_uncommitted,
        test_crash_during_recovery,
        test_errors,
        test_ann_recall,
        test_semicolon_and_unicode,
        test_edge_cases,
        test_big_data,
    ]
    try:
        for t in tests:
            if args.only and args.only not in t.__name__:
                continue
            t(args.binary, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name, detail in FAILED:
        print("  FAILED:", name, detail)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
