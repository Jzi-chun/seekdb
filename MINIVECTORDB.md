# MiniVectorDB

A single-node vector database written for the `2026-competition-round2` extra
round. It speaks the JSON-lines protocol from `STDIO_PROTOCOL.md` on
stdin/stdout, keeps everything in `/data`, supports the required SQL subset,
transactions and crash recovery, and serves exact L2 top-K queries with an
AVX2 kernel.

```
main.cpp      protocol: JSONL in, JSONL out, statement splitting
engine.h      public engine interface
internal.h    SQL AST
sql.cpp       lexer + recursive descent parser
engine.cpp    storage, recovery, transactions, executor, vector scan
numparse.h    fast float scanner for vector literals
conversation.json         Agent prompt material (same file as agent/conversation.json)
extended_test.py          crash recovery, error contract, ANN recall, edge cases, big data
bench.py                  the round-2 workload (10k x 128, 1000 top-K) with A/B knobs
bench_query.py            filtered vs unfiltered top-K
bench_overhead.py         protocol round-trip cost
```

## Build and run

Local build (no docker):

```bash
g++ -std=c++17 -O2 -pthread -o build/mvdb main.cpp engine.cpp sql.cpp
printf '%s\n' '{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, s VARCHAR(8));"}' \
              '{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1, '"'"'hi'"'"');"}' \
              '{"id":3,"op":"execute","sql":"SELECT * FROM docs;"}' \
              '{"id":4,"op":"shutdown"}' | ./build/mvdb --data-dir /tmp/mvdb-data
```

Container (exactly the judge's shape: stdin attached, no TTY, data dir mounted):

```bash
docker build -t minivectordb:latest .
mkdir -p /tmp/mvdb-data
docker run --rm -i --network=none --cpus=2 --memory=2g \
  --mount type=bind,src=/tmp/mvdb-data,dst=/data \
  minivectordb:latest
```

The program never reads a port, never daemonises, and writes only under the
data directory (`VECTORDB_DATA_DIR` or `--data-dir`, default `/data`).

## Supported SQL

| Statement | Notes |
| --- | --- |
| `CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(n), v VECTOR(d), ...)` | one `BIGINT` primary key (inline or `PRIMARY KEY (id)`), at most one vector column |
| `INSERT INTO t [(cols)] VALUES (...), (...)` | multi-row, all-or-nothing |
| `SELECT cols \| * \| L2_DISTANCE(v,'[...]') AS d FROM t [WHERE ...] [ORDER BY k [DESC]] [LIMIT n]` | `=`, `<`, `<=`, `>`, `>=`, `AND`; ties break on the primary key ascending |
| `UPDATE t SET c = v, ... [WHERE ...]` | primary key updates are rejected |
| `DELETE FROM t [WHERE ...]` | unconditional `DELETE` is rejected |
| `CREATE [VECTOR] INDEX name ON t(v)` | recorded, maintained, persisted |
| `BEGIN` / `START TRANSACTION`, `COMMIT`, `ROLLBACK` | one explicit transaction at a time |
| `DROP TABLE [IF EXISTS] t` | convenience for local tests |

Anything else (`ALTER TABLE`, aggregates, `JOIN`, sub-queries, `NULL`, ...)
answers `unsupported_sql` instead of being silently treated as something else.

Error codes are stable snake_case strings: `duplicate_key`,
`dimension_mismatch`, `invalid_vector`, `value_too_long`, `type_mismatch`,
`column_count_mismatch`, `table_not_found`, `column_not_found`,
`table_exists`, `unsupported_sql`, `syntax_error`, `transaction_in_progress`.

## Storage layout

`/data/data.log` is a single append-only file:

```
file header   [u32 magic][u16 version][u16 reserved]                 8 B
record        [u32 padded_body_len][u8 kind][u32 table_id]
              [u64 rowid][u64 pk]                                   32 B
body          [vector float32[d]][columns in declaration order]        n B
```

`kind` is `1` row image, `2` tombstone, `3` commit marker. The record header is
32 bytes so the body starts on an 8 byte boundary and the `float32` vectors are
naturally aligned; bodies are padded to a multiple of 8 bytes, which keeps the
next record aligned as well.

* Rows are copy-on-write images. An update appends a new image and re-points
  `rowid -> offset`, a delete appends a tombstone.
* `rowid` is the insertion sequence per table; the in-memory directory is a
  `vector<uint64_t>` of live offsets plus a `pk -> rowid` hash map. Memory is
  therefore a few tens of bytes per row, not the row data: 200k rows of 128
  dims (104 MB of vectors) run with a 16.5 MB peak RSS.
* `/data/catalog` holds the schemas. It is written to `catalog.tmp`, fsynced and
  renamed, so a table definition is durable before the statement is acknowledged.

## Transactions and recovery

`BEGIN` snapshots (log length, undo depth). Every statement applies its changes
to memory and the log immediately, remembering an undo record per touched row,
so a transaction reads its own writes. `COMMIT` appends a commit marker and
`fdatasync`s once; only then is the response written. `ROLLBACK`, a failed
statement, or a failed statement inside a transaction truncates the log back to
the mark and replays the undo records in reverse.

Recovery scans the log from the beginning, buffers records until a commit
marker, and applies them; everything after the last commit marker is truncated,
which implements "uncommitted transactions cannot be partially visible". A crash
in the middle of recovery simply repeats the same scan. If more than two thirds
of the log is garbage, a compaction rewrites the live images into a new file and
`rename()`s it into place.

## Vector search

`SELECT ... ORDER BY distance LIMIT k` runs an exact scan: the query vector is
compared against every live row with `l2Avx2` (AVX2 + FMA, two accumulators,
runtime detection via `__builtin_cpu_supports`, scalar fallback), distances are
accumulated in float32 and the final square root is taken in double. A bounded
max-heap of size `k` keeps the best rows ordered by `(distance, primary key)`.

`ann` mode is served by the same exact scan: the round-2 rule only requires
valid, duplicate-free, filter-respecting ids with `Recall@10 >= 0.95`, and an
exact scan cannot miss. `CREATE VECTOR INDEX` is accepted, persisted and
maintained (it is recorded in the catalog) but the planner deliberately keeps
the exact path, because for 10k rows the exact scan with SIMD is faster than
maintaining a graph index and still gives recall 1.0.

Two measured optimisations (see `bench.py`, `bench_query.py`):

* AVX2 kernel: ~1.6x-3x less query time than the scalar loop.
* Lazy equality index for `WHERE col = value`: a filtered top-K touches only the
  matching rowids instead of the whole table (~1.4x-2x on the filtered half of
  the workload). It is rebuilt whenever `Table::epoch` changes, i.e. after any
  insert/update/delete/rollback, and the predicate is re-checked on every
  candidate so a stale index can never produce a wrong row.
* A second worker thread is implemented (`MVDB_THREADS=1`) but off by default:
  at 10k x 128 the scan is memory-bandwidth bound and the hand-off cost per
  query was larger than the gain.

## Verification

```bash
# 20 public cases from the official package
bash run_public.sh build/mvdb public

# crash recovery, error contract, ANN recall, edge shapes, 200k x 128 resource test
python3 extended_test.py --binary build/mvdb

# the round-2 workload, with SIMD/thread/index A/B variants
python3 bench.py --binary build/mvdb --rounds 3

# build the image and run the official checker against the container
bash verify_docker.sh
```

Measured on this machine (WSL Ubuntu 22.04, g++ 11.4, 16 cores):

| check | result |
| --- | --- |
| `tests/extended_test.py` | 49/49 |
| `protocol_check/run.py` (local binary) | 20/20, repeated runs |
| `protocol_check/run.py --image` (real container, read-only rootfs, no network) | handshake + 20/20 |
| 200k rows x 128 dims (104 MB of vectors) | peak RSS 16.7 MB, full scan 13 ms, restart 0.03 s |
| 10k x 128, 1000 top-K | 218 ms query, 519 ms write, 5.7 MB on disk, 9.9 MB peak RSS |

Known limits: no `NULL` values, no joins/aggregates/sub-queries, no multi-column
primary keys, no concurrent clients, one explicit transaction at a time (as
allowed by the task), and `VARCHAR` values are limited to 65535 bytes.
