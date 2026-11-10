// MiniVectorDB storage engine, transaction layer and query executor.
//
// Design summary (see MINIVECTORDB.md for the long version):
//   * one append-only log file /data/data.log holds every row image and the
//     commit markers, so a commit is durable with a single fdatasync and a
//     multi table transaction can never become partially visible;
//   * recovery replays the log and discards the tail after the last commit;
//   * rows are copy-on-write row images addressed by an in-memory
//     rowid -> file offset directory plus a primary key hash index;
//   * vector search is an exact scan with an AVX2/FMA kernel and a bounded
//     top-K heap, optionally split across two worker threads.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>

#include <immintrin.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "engine.h"
#include "internal.h"
#include "numparse.h"

namespace mvdb {
namespace {

constexpr uint32_t kMagic = 0x4D564442;  // "MVDB"
constexpr uint16_t kVersion = 1;
constexpr size_t kFileHeader = 8;
constexpr size_t kHdr = 32;      // body_len, kind, table_id, rowid, pk (+ padding)
constexpr size_t kBodyAlign = 8; // keep float32 vectors naturally aligned

constexpr uint8_t kKindRow = 1;
constexpr uint8_t kKindDel = 2;
constexpr uint8_t kKindCommit = 3;

constexpr uint64_t kRemapSlack = 32ull << 20;  // map headroom to avoid remap churn

void putU32(uint8_t* p, uint32_t v) { std::memcpy(p, &v, 4); }
void putU64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }
uint32_t getU32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint64_t getU64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// Serializes one log record. Record bodies are padded so that every record
// starts on an 8 byte boundary, which keeps the float32 vectors inside the
// row images naturally aligned (the AVX2 kernel then needs no copies).
void appendRec(std::string& buf, uint8_t kind, uint32_t table_id, uint64_t rowid, int64_t pk,
               const void* body, size_t body_len) {
  size_t pad = (kBodyAlign - (body_len % kBodyAlign)) % kBodyAlign;
  uint8_t hdr[kHdr] = {0};
  putU32(hdr, static_cast<uint32_t>(body_len + pad));
  hdr[4] = kind;
  putU32(hdr + 5, table_id);
  putU64(hdr + 9, rowid);
  putU64(hdr + 17, static_cast<uint64_t>(pk));
  buf.append(reinterpret_cast<const char*>(hdr), kHdr);
  if (body_len) buf.append(static_cast<const char*>(body), body_len);
  if (pad) buf.append(pad, '\0');
}

[[noreturn]] void fail(const std::string& code, const std::string& msg) {
  throw DbError(code, msg);
}

void writeAll(int fd, const void* buf, size_t n, uint64_t off) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  while (n > 0) {
    ssize_t w = ::pwrite(fd, p, n, static_cast<off_t>(off));
    if (w < 0) {
      if (errno == EINTR) continue;
      fail("io_error", std::string("write failed: ") + std::strerror(errno));
    }
    p += w;
    off += static_cast<uint64_t>(w);
    n -= static_cast<size_t>(w);
  }
}

void readAll(int fd, void* buf, size_t n, uint64_t off) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (n > 0) {
    ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r < 0) {
      if (errno == EINTR) continue;
      fail("io_error", std::string("read failed: ") + std::strerror(errno));
    }
    if (r == 0) fail("io_error", "unexpected end of file");
    p += r;
    off += static_cast<uint64_t>(r);
    n -= static_cast<size_t>(r);
  }
}

// ---------------------------------------------------------------- json output

void emitInt(std::string& out, int64_t v) {
  char buf[24];
  auto res = std::to_chars(buf, buf + sizeof(buf), v);
  out.append(buf, static_cast<size_t>(res.ptr - buf));
}

void emitDouble(std::string& out, double v) {
  if (!std::isfinite(v)) v = 0.0;
  char buf[40];
  int n = std::snprintf(buf, sizeof(buf), "%.9g", v);
  if (n <= 0) {
    out += '0';
    return;
  }
  out.append(buf, static_cast<size_t>(n));
}

void emitString(std::string& out, const char* s, size_t n) {
  out += '"';
  size_t run = 0;
  for (size_t i = 0; i < n; ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x20 && c != '"' && c != '\\') continue;
    out.append(s + run, i - run);
    run = i + 1;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default: {
        char tmp[8];
        std::snprintf(tmp, sizeof(tmp), "\\u%04x", c);
        out += tmp;
        break;
      }
    }
  }
  out.append(s + run, n - run);
  out += '"';
}

// ------------------------------------------------------------ vector kernels

float l2Scalar(const float* a, const float* b, int n) {
  float s0 = 0.f;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    float d0 = a[i] - b[i];
    float d1 = a[i + 1] - b[i + 1];
    float d2 = a[i + 2] - b[i + 2];
    float d3 = a[i + 3] - b[i + 3];
    s0 += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }
  for (; i < n; ++i) {
    float d = a[i] - b[i];
    s0 += d * d;
  }
  return s0;
}

__attribute__((target("avx2,fma"))) float l2Avx2(const float* a, const float* b, int n) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256 a0 = _mm256_loadu_ps(a + i);
    __m256 b0 = _mm256_loadu_ps(b + i);
    __m256 d0 = _mm256_sub_ps(a0, b0);
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    __m256 a1 = _mm256_loadu_ps(a + i + 8);
    __m256 b1 = _mm256_loadu_ps(b + i + 8);
    __m256 d1 = _mm256_sub_ps(a1, b1);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  for (; i + 8 <= n; i += 8) {
    __m256 a0 = _mm256_loadu_ps(a + i);
    __m256 b0 = _mm256_loadu_ps(b + i);
    __m256 d0 = _mm256_sub_ps(a0, b0);
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
  }
  __m256 acc = _mm256_add_ps(acc0, acc1);
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float sum = _mm_cvtss_f32(s);
  for (; i < n; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

using DistFn = float (*)(const float*, const float*, int);
DistFn g_dist = l2Scalar;

// A tiny persistent worker pool. Only the vector scan uses it, and it exists so
// that a two core container can split one top-K query in half without paying a
// thread creation cost per query.
class ScanPool {
 public:
  explicit ScanPool(int workers) : workers_(workers) {
    for (int i = 0; i < workers_; ++i) {
      threads_.emplace_back([this] { loop(); });
    }
  }
  ~ScanPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  int workers() const { return workers_; }

  void run(int parts, const std::function<void(int)>& fn) {
    std::unique_lock<std::mutex> lk(m_);
    fn_ = &fn;
    parts_ = parts;
    next_ = 0;
    done_ = 0;
    ++gen_;
    cv_.notify_all();
    for (;;) {
      int part = next_++;
      if (part >= parts_) break;
      lk.unlock();
      fn(part);
      lk.lock();
      if (++done_ == parts_) done_cv_.notify_all();
    }
    done_cv_.wait(lk, [this] { return done_ == parts_; });
    fn_ = nullptr;
  }

 private:
  void loop() {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this, seen] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      for (;;) {
        int part = next_++;
        if (part >= parts_) break;
        lk.unlock();
        (*fn_)(part);
        lk.lock();
        if (++done_ == parts_) done_cv_.notify_all();
      }
    }
  }

  int workers_;
  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  const std::function<void(int)>* fn_ = nullptr;
  int parts_ = 0;
  int next_ = 0;
  int done_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

std::unique_ptr<ScanPool> g_pool;

void initKernels() {
  __builtin_cpu_init();
  const char* no_simd = std::getenv("MVDB_SIMD");
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
      !(no_simd && no_simd[0] == '0')) {
    g_dist = l2Avx2;
  }
  // Default: one thread. A/B measurement (tests/bench.py) showed the scan is
  // memory bandwidth bound at 10k x 128 floats, so the second core bought
  // nothing while costing ~35 us of hand-off per query. MVDB_THREADS=1 enables
  // the helper thread for larger tables.
  int workers = 0;
  const char* th = std::getenv("MVDB_THREADS");
  if (th && *th) workers = std::atoi(th);
  if (workers < 0) workers = 0;
  if (workers > 3) workers = 3;
  if (workers > 0) g_pool.reset(new ScanPool(workers));
  std::fprintf(stderr, "[mvdb] simd=%s threads=%d\n", g_dist == l2Avx2 ? "avx2" : "scalar",
               workers + 1);
}

// ------------------------------------------------------------------ utilities

int utf8CharCount(const std::string& s) {
  int n = 0;
  for (unsigned char c : s) {
    if ((c & 0xC0) != 0x80) ++n;
  }
  return n;
}

bool parseIntText(const std::string& s, int64_t* out) {
  const char* b = s.data();
  const char* e = b + s.size();
  auto res = std::from_chars(b, e, *out);
  return res.ec == std::errc() && res.ptr == e;
}

// ---------------------------------------------------------------------- log

class Log {
 public:
  ~Log() { close(); }

  void open(const std::string& path) {
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd_ < 0) fail("io_error", std::string("cannot open log: ") + std::strerror(errno));
    struct stat st;
    if (::fstat(fd_, &st) != 0) fail("io_error", "fstat failed");
    size_ = static_cast<uint64_t>(st.st_size);
    if (size_ < kFileHeader) {
      uint8_t hdr[kFileHeader] = {0};
      putU32(hdr, kMagic);
      hdr[4] = static_cast<uint8_t>(kVersion & 0xFF);
      hdr[5] = static_cast<uint8_t>(kVersion >> 8);
      if (::ftruncate(fd_, 0) != 0) fail("io_error", "truncate failed");
      writeAll(fd_, hdr, sizeof(hdr), 0);
      if (::fdatasync(fd_) != 0) fail("io_error", "fdatasync failed");
      size_ = kFileHeader;
    }
    ensureMap();
  }

  void close() {
    if (map_) {
      ::munmap(const_cast<uint8_t*>(map_), map_len_);
      map_ = nullptr;
      map_len_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  uint64_t size() const { return size_; }
  const uint8_t* base() { return map_; }

  void ensureMap() {
    if (size_ <= map_len_ && map_) return;
    if (map_) {
      ::munmap(const_cast<uint8_t*>(map_), map_len_);
      map_ = nullptr;
      map_len_ = 0;
    }
    uint64_t want = size_ + kRemapSlack;
    want = (want + 4095) & ~static_cast<uint64_t>(4095);
    void* p = ::mmap(nullptr, static_cast<size_t>(want), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) fail("io_error", std::string("mmap failed: ") + std::strerror(errno));
    map_ = static_cast<const uint8_t*>(p);
    map_len_ = want;
    ::madvise(const_cast<void*>(static_cast<const void*>(map_)), static_cast<size_t>(want),
              MADV_SEQUENTIAL);
  }

  void append(const void* buf, size_t n) {
    if (n == 0) return;
    writeAll(fd_, buf, n, size_);
    size_ += n;
  }

  void sync() {
    if (::fdatasync(fd_) != 0) fail("io_error", std::string("fdatasync failed: ") + std::strerror(errno));
  }

  void truncateTo(uint64_t new_size) {
    if (new_size == size_) return;
    if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
      fail("io_error", std::string("ftruncate failed: ") + std::strerror(errno));
    }
    size_ = new_size;
  }

 private:
  std::string path_;
  int fd_ = -1;
  uint64_t size_ = 0;
  const uint8_t* map_ = nullptr;
  size_t map_len_ = 0;
};

// -------------------------------------------------------------------- table

struct UndoRec {
  uint32_t rowid;
  uint64_t prev_off;
  int64_t pk;
  bool was_new;
};

// Lazy equality index over one column. It only ever produces a superset of the
// matching rowids (rows are re-checked by the caller), and every mutation bumps
// Table::epoch which invalidates it.
struct ValueIndex {
  int col = -1;
  uint64_t epoch = ~0ull;
  std::unordered_map<std::string, std::vector<uint32_t>> str;
  std::unordered_map<int64_t, std::vector<uint32_t>> num;
};

struct Table {
  TableSchema schema;
  std::vector<uint64_t> offs;  // rowid -> record offset, 0 = deleted
  std::unordered_map<int64_t, uint32_t> pk2rid;
  std::vector<UndoRec> undo;
  uint64_t epoch = 1;
  std::vector<ValueIndex> vidx;

  uint32_t liveRows() const {
    uint32_t n = 0;
    for (uint64_t o : offs) {
      if (o) ++n;
    }
    return n;
  }
};

// ------------------------------------------------------------ value handling

struct ColValue {
  ColType type;
  int64_t i = 0;
  std::string s;
  const float* v = nullptr;  // points into the mapped page
};

// Body layout: [vector (if the table has one)][columns in declaration order,
// BIGINT = 8 bytes, VARCHAR = u16 length + bytes].
inline const uint8_t* columnPtr(const uint8_t* body, const TableSchema& sc, int col) {
  const uint8_t* p = body;
  if (sc.vec_col >= 0) p += static_cast<size_t>(sc.cols[sc.vec_col].dim) * 4;
  for (int i = 0; i < col; ++i) {
    if (sc.cols[i].type == ColType::BigInt) {
      p += 8;
    } else if (sc.cols[i].type == ColType::VarChar) {
      uint16_t len;
      std::memcpy(&len, p, 2);
      p += 2 + len;
    }
  }
  return p;
}

inline const float* vectorPtr(const uint8_t* body) {
  return reinterpret_cast<const float*>(body);
}

bool literalToInt(const Literal& lit, int64_t* out) {
  if (lit.kind == LitKind::Int) {
    *out = lit.i;
    return true;
  }
  if (lit.kind == LitKind::Str) return parseIntText(lit.s, out);
  return false;
}

}  // namespace

// ============================================================================
// The actual database
// ============================================================================

struct Database::Impl {
  std::string dir;
  Log log;
  std::vector<Table> tables;
  std::unordered_map<std::string, size_t> by_name;
  std::unordered_map<uint32_t, size_t> by_id;
  uint32_t next_table_id = 1;
  uint32_t next_txn_id = 1;

  bool in_txn = false;
  uint64_t txn_changes = 0;

  struct Mark {
    uint64_t log_size = 0;
    std::vector<size_t> undo_lens;
  };
  Mark txn_start;

  // -------------------------------------------------------------- open/close
  void open(const std::string& data_dir);
  void close();
  void saveCatalog();
  void loadCatalog();
  void reindexTables();

  Table& tableByName(const std::string& name);
  int resolveColumn(const Table& t, const std::string& name);
  const std::vector<uint32_t>* filterCandidates(Table& t, int col, const Literal& lit);
  void recover();
  void compactIfNeeded();
  void compact();

  Mark mark() const {
    Mark m;
    m.log_size = log.size();
    m.undo_lens.reserve(tables.size());
    for (const auto& t : tables) m.undo_lens.push_back(t.undo.size());
    return m;
  }
  void rollbackTo(const Mark& m);
  void commitTxn();

  // DDL
  ExecResult execCreateTable(Statement& st);
  ExecResult execCreateIndex(Statement& st);
  ExecResult execDropTable(Statement& st);

  // DML
  ExecResult execInsert(Statement& st);
  ExecResult execUpdate(Statement& st);
  ExecResult execDelete(Statement& st);

  // SELECT
  ExecResult execSelect(Statement& st, const std::string& mode);
};

namespace {

std::string catalogPath(const std::string& dir) { return dir + "/catalog"; }
std::string logPath(const std::string& dir) { return dir + "/data.log"; }

void fsyncDir(const std::string& dir) {
  int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  }
}

}  // namespace

void Database::Impl::saveCatalog() {
  std::string tmp = catalogPath(dir) + ".tmp";
  std::string text = "mvdb-catalog-1\n";
  text += "next_table_id " + std::to_string(next_table_id) + "\n";
  for (const auto& t : tables) {
    text += "table " + std::to_string(t.schema.id) + " " + t.schema.name + " " +
            std::to_string(t.schema.pk_col) + " " + std::to_string(t.schema.vec_col) + " " +
            std::to_string(t.schema.cols.size()) + "\n";
    for (const auto& c : t.schema.cols) {
      text += "col " + c.name + " " + std::to_string(static_cast<int>(c.type)) + " " +
              std::to_string(c.len) + " " + std::to_string(c.dim) + "\n";
    }
    for (const auto& idx : t.schema.index_names) text += "index " + idx + "\n";
  }
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) fail("io_error", "cannot write catalog");
  writeAll(fd, text.data(), text.size(), 0);
  if (::fdatasync(fd) != 0) fail("io_error", "catalog fdatasync failed");
  ::close(fd);
  if (::rename(tmp.c_str(), catalogPath(dir).c_str()) != 0) {
    fail("io_error", "catalog rename failed");
  }
  fsyncDir(dir);
}

void Database::Impl::loadCatalog() {
  FILE* f = std::fopen(catalogPath(dir).c_str(), "rb");
  if (!f) return;
  char line[4096];
  Table* cur = nullptr;
  while (std::fgets(line, sizeof(line), f)) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    size_t sp = s.find(' ');
    std::string key = sp == std::string::npos ? s : s.substr(0, sp);
    if (key == "next_table_id") {
      next_table_id = static_cast<uint32_t>(std::strtoul(s.c_str() + sp + 1, nullptr, 10));
    } else if (key == "table") {
      int id = 0, pk = -1, vec = -1, ncol = 0;
      char name[512] = {0};
      if (std::sscanf(s.c_str(), "table %d %511s %d %d %d", &id, name, &pk, &vec, &ncol) != 5) {
        continue;
      }
      tables.emplace_back();
      cur = &tables.back();
      cur->schema.id = static_cast<uint32_t>(id);
      cur->schema.name = name;
      cur->schema.pk_col = pk;
      cur->schema.vec_col = vec;
      by_name[cur->schema.name] = tables.size() - 1;
    } else if (key == "col" && cur) {
      char name[512] = {0};
      int type = 0, len = 0, dim = 0;
      if (std::sscanf(s.c_str(), "col %511s %d %d %d", name, &type, &len, &dim) != 4) continue;
      Column c;
      c.name = name;
      c.type = static_cast<ColType>(type);
      c.len = len;
      c.dim = dim;
      cur->schema.cols.push_back(std::move(c));
    } else if (key == "index" && cur) {
      cur->schema.index_names.push_back(s.substr(sp + 1));
    }
  }
  std::fclose(f);
}

void Database::Impl::reindexTables() {
  by_name.clear();
  by_id.clear();
  for (size_t i = 0; i < tables.size(); ++i) {
    by_name[tables[i].schema.name] = i;
    by_id[tables[i].schema.id] = i;
  }
}

void Database::Impl::recover() {
  const uint8_t* base = log.base();
  uint64_t size = log.size();
  uint64_t pos = kFileHeader;
  uint64_t committed = kFileHeader;
  struct Pend {
    uint8_t kind;
    uint32_t table;
    uint64_t rowid;
    int64_t pk;
    uint64_t off;
  };
  std::vector<Pend> pending;
  while (pos + kHdr <= size) {
    const uint8_t* p = base + pos;
    uint32_t body_len = getU32(p);
    uint8_t kind = p[4];
    uint32_t tid = getU32(p + 5);
    uint64_t rowid = getU64(p + 9);
    int64_t pk = static_cast<int64_t>(getU64(p + 17));
    if (pos + kHdr + body_len > size) break;
    if (kind == kKindCommit) {
      for (const auto& e : pending) {
        auto it = by_id.find(e.table);
        if (it == by_id.end()) continue;  // rows of a dropped table
        Table& t = tables[it->second];
        if (e.kind == kKindRow) {
          if (e.rowid < t.offs.size()) {
            t.offs[e.rowid] = e.off;
          } else if (e.rowid == t.offs.size()) {
            t.offs.push_back(e.off);
          } else {
            continue;  // gap: corrupt, ignore
          }
          t.pk2rid[e.pk] = static_cast<uint32_t>(e.rowid);
        } else if (e.kind == kKindDel) {
          if (e.rowid < t.offs.size()) {
            t.offs[e.rowid] = 0;
            t.pk2rid.erase(e.pk);
          }
        }
      }
      pending.clear();
      pos += kHdr;
      committed = pos;
      continue;
    }
    if (kind != kKindRow && kind != kKindDel) break;
    pending.push_back(Pend{kind, tid, rowid, pk, pos});
    pos += kHdr + body_len;
  }
  if (committed < size) {
    log.truncateTo(committed);
    log.ensureMap();
    std::fprintf(stderr, "[mvdb] recovery: truncated %llu uncommitted bytes\n",
                 static_cast<unsigned long long>(size - committed));
  }
  log.ensureMap();
  for (auto& t : tables) {
    t.undo.clear();
    t.undo.shrink_to_fit();
  }
}

void Database::Impl::compact() {
  // Rewrites the log with one image per live row. The new file only becomes
  // visible through rename() after it is complete and synced, so a crash in
  // the middle leaves the old log untouched.
  uint64_t live = 0;
  for (const auto& t : tables) live += t.liveRows();
  if (live == 0) return;
  std::string tmp = logPath(dir) + ".tmp";
  ::unlink(tmp.c_str());
  int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return;
  uint8_t fh[kFileHeader] = {0};
  putU32(fh, kMagic);
  fh[4] = static_cast<uint8_t>(kVersion & 0xFF);
  fh[5] = static_cast<uint8_t>(kVersion >> 8);
  writeAll(fd, fh, sizeof(fh), 0);
  uint64_t write_off = kFileHeader;
  std::string buf;
  const uint8_t* base = log.base();
  uint64_t new_size = kFileHeader;
  auto flush = [&]() {
    if (buf.empty()) return;
    writeAll(fd, buf.data(), buf.size(), new_size);
    new_size += buf.size();
    buf.clear();
  };
  for (auto& t : tables) {
    for (uint32_t r = 0; r < t.offs.size(); ++r) {
      uint64_t off = t.offs[r];
      if (!off) continue;
      const uint8_t* rec = base + off;
      uint32_t blen = getU32(rec);
      int64_t pk = static_cast<int64_t>(getU64(rec + 17));
      uint8_t hdr[kHdr];
      putU32(hdr, blen);
      hdr[4] = kKindRow;
      putU32(hdr + 5, t.schema.id);
      putU64(hdr + 9, r);
      putU64(hdr + 17, static_cast<uint64_t>(pk));
      t.offs[r] = new_size + buf.size();
      buf.append(reinterpret_cast<const char*>(hdr), kHdr);
      buf.append(reinterpret_cast<const char*>(rec + kHdr), blen);
      if (buf.size() > (16u << 20)) flush();
    }
  }
  uint8_t commit[kHdr] = {0};
  commit[4] = kKindCommit;
  buf.append(reinterpret_cast<const char*>(commit), kHdr);
  flush();
  if (::fdatasync(fd) != 0) {
    ::close(fd);
    ::unlink(tmp.c_str());
    return;
  }
  ::close(fd);
  log.close();
  if (::rename(tmp.c_str(), logPath(dir).c_str()) != 0) {
    log.open(logPath(dir));
    return;
  }
  fsyncDir(dir);
  log.open(logPath(dir));
  log.ensureMap();
  (void)write_off;
  std::fprintf(stderr, "[mvdb] compacted log to %llu bytes (%llu live rows)\n",
               static_cast<unsigned long long>(new_size), static_cast<unsigned long long>(live));
}


void Database::Impl::open(const std::string& data_dir) {
  dir = data_dir;
  ::mkdir(dir.c_str(), 0755);
  initKernels();
  loadCatalog();
  reindexTables();
  log.open(logPath(dir));
  recover();
  compactIfNeeded();
}

void Database::Impl::close() {
  if (in_txn) {
    rollbackTo(txn_start);
    in_txn = false;
  }
  for (auto& t : tables) t.undo.clear();
  if (log.size() > kFileHeader) log.sync();
  log.close();
}

Table& Database::Impl::tableByName(const std::string& name) {
  auto it = by_name.find(name);
  if (it == by_name.end()) fail("table_not_found", "table not found: " + name);
  return tables[it->second];
}

int Database::Impl::resolveColumn(const Table& t, const std::string& name) {
  for (size_t i = 0; i < t.schema.cols.size(); ++i) {
    if (t.schema.cols[i].name == name) return static_cast<int>(i);
  }
  fail("column_not_found", "column not found: " + name);
}

// Equality predicates are served from a small lazily built hash index so a
// filtered top-K does not have to touch every row. The index is a superset of
// the matching rows, the scan still re-checks the predicate, and any mutation
// bumps Table::epoch which forces a rebuild on the next filtered query.
const std::vector<uint32_t>* Database::Impl::filterCandidates(Table& t, int col,
                                                              const Literal& lit) {
  static const std::vector<uint32_t> kEmpty;
  static const bool disabled = [] {
    const char* v = std::getenv("MVDB_NO_FILTER_INDEX");
    return v && v[0] == '1';
  }();
  if (disabled) return nullptr;
  if (t.offs.size() < 512) return nullptr;
  if (t.schema.cols[col].type == ColType::Vector) return nullptr;
  ValueIndex* slot = nullptr;
  for (auto& v : t.vidx) {
    if (v.col == col) slot = &v;
  }
  if (!slot) {
    if (t.vidx.size() < 2) {
      t.vidx.emplace_back();
      slot = &t.vidx.back();
    } else {
      slot = &t.vidx[0];
    }
  }
  if (slot->col != col || slot->epoch != t.epoch) {
    slot->col = col;
    slot->epoch = t.epoch;
    slot->str.clear();
    slot->num.clear();
    const uint8_t* base = log.base();
    const uint64_t size = log.size();
    for (uint32_t r = 0; r < t.offs.size(); ++r) {
      uint64_t off = t.offs[r];
      if (!off || off + kHdr + 8 > size) continue;
      const uint8_t* body = base + off + kHdr;
      if (t.schema.cols[col].type == ColType::BigInt) {
        int64_t v;
        std::memcpy(&v, columnPtr(body, t.schema, col), 8);
        slot->num[v].push_back(r);
      } else {
        const uint8_t* p = columnPtr(body, t.schema, col);
        uint16_t len;
        std::memcpy(&len, p, 2);
        slot->str[std::string(reinterpret_cast<const char*>(p + 2), len)].push_back(r);
      }
    }
  }
  if (t.schema.cols[col].type == ColType::BigInt) {
    int64_t key = 0;
    if (!literalToInt(lit, &key)) return nullptr;
    auto it = slot->num.find(key);
    return it == slot->num.end() ? &kEmpty : &it->second;
  }
  if (lit.kind != LitKind::Str) return nullptr;
  auto it = slot->str.find(lit.s);
  return it == slot->str.end() ? &kEmpty : &it->second;
}

void Database::Impl::rollbackTo(const Mark& m) {
  for (size_t i = 0; i < tables.size(); ++i) {
    Table& t = tables[i];
    size_t target = i < m.undo_lens.size() ? m.undo_lens[i] : 0;
    while (t.undo.size() > target) {
      UndoRec u = t.undo.back();
      t.undo.pop_back();
      if (u.was_new) {
        if (t.offs.size() == static_cast<size_t>(u.rowid) + 1) t.offs.pop_back();
        t.pk2rid.erase(u.pk);
      } else {
        t.offs[u.rowid] = u.prev_off;
        if (u.prev_off == 0) t.pk2rid.erase(u.pk);
        else t.pk2rid[u.pk] = u.rowid;
      }
    }
    ++t.epoch;
  }
  if (log.size() > m.log_size) log.truncateTo(m.log_size);
  log.ensureMap();
}

void Database::Impl::commitTxn() {
  std::string rec;
  appendRec(rec, kKindCommit, 0, 0, 0, nullptr, 0);
  log.append(rec.data(), rec.size());
  log.sync();
  log.ensureMap();
  ++next_txn_id;
  txn_changes = 0;
  for (auto& t : tables) t.undo.clear();
  in_txn = false;
}

void Database::Impl::compactIfNeeded() {
  // The log keeps old row images; rewrite it when most of the file is garbage
  // so that scans do not keep walking dead versions.
  uint64_t live = 0, total = 0;
  for (const auto& t : tables) {
    live += t.liveRows();
    total += t.offs.size();
  }
  if (log.size() < (16ull << 20)) return;
  if (live * 3 > total) return;  // less than 2/3 garbage
  compact();
}

// ------------------------------------------------------------------ DDL

ExecResult Database::Impl::execCreateTable(Statement& st) {
  if (by_name.count(st.table)) fail("table_exists", "table already exists: " + st.table);
  Table t;
  t.schema.id = next_table_id++;
  t.schema.name = st.table;
  t.schema.pk_col = -1;
  t.schema.vec_col = -1;
  for (auto& c : st.create_cols) {
    if (c.name.empty()) fail("syntax_error", "empty column name");
    for (const auto& other : t.schema.cols) {
      if (other.name == c.name) fail("column_exists", "duplicate column: " + c.name);
    }
    if (c.type == ColType::Vector) {
      if (t.schema.vec_col >= 0) {
        fail("multiple_vector_columns", "only one vector column per table is supported");
      }
      t.schema.vec_col = static_cast<int>(t.schema.cols.size());
    }
    t.schema.cols.push_back(Column{c.name, c.type, c.len, c.dim});
    if (c.pk) {
      if (t.schema.pk_col >= 0) fail("multiple_primary_keys", "multiple primary keys");
      t.schema.pk_col = static_cast<int>(t.schema.cols.size()) - 1;
    }
  }
  if (!st.table_pk.empty()) {
    if (t.schema.pk_col >= 0) fail("multiple_primary_keys", "multiple primary keys");
    t.schema.pk_col = resolveColumn(t, st.table_pk);
    // resolveColumn throws on unknown names; make sure it is a BIGINT
  }
  if (t.schema.pk_col < 0) fail("primary_key_required", "a BIGINT primary key is required");
  if (t.schema.cols[t.schema.pk_col].type != ColType::BigInt) {
    fail("invalid_primary_key", "primary key must be BIGINT");
  }
  tables.push_back(std::move(t));
  reindexTables();
  saveCatalog();
  ExecResult r;
  return r;
}

ExecResult Database::Impl::execCreateIndex(Statement& st) {
  Table& t = tableByName(st.table);
  int col = resolveColumn(t, st.index_col);
  if (t.schema.cols[col].type != ColType::Vector) {
    fail("invalid_index_column", "vector index requires a vector column");
  }
  for (const auto& name : t.schema.index_names) {
    if (name == st.index_name) fail("index_exists", "index already exists: " + st.index_name);
  }
  t.schema.index_names.push_back(st.index_name);
  saveCatalog();
  ExecResult r;
  return r;
}

ExecResult Database::Impl::execDropTable(Statement& st) {
  auto it = by_name.find(st.table);
  if (it == by_name.end()) {
    ExecResult r;
    return r;
  }
  size_t idx = it->second;
  tables.erase(tables.begin() + static_cast<long>(idx));
  reindexTables();
  saveCatalog();
  ExecResult r;
  return r;
}

// ------------------------------------------------------------------ DML

namespace {

// Builds the encoded body of one row. Throws on any type/size problem.
void buildBody(const TableSchema& sc, const std::vector<int>& col_order,
               const std::vector<Literal>& vals, std::string& out) {
  out.clear();
  // vector first
  if (sc.vec_col >= 0) {
    const Column& vc = sc.cols[sc.vec_col];
    const Literal* lit = nullptr;
    for (size_t k = 0; k < col_order.size(); ++k) {
      if (col_order[k] == sc.vec_col) {
        lit = &vals[k];
        break;
      }
    }
    if (!lit) fail("column_count_mismatch", "missing value for vector column");
    std::vector<float> vec;
    if (lit->kind == LitKind::Vec) {
      vec = lit->vec;
    } else if (lit->kind == LitKind::Str) {
      // parse text form "[1, 2]"
      const char* p = lit->s.c_str();
      while (*p && (unsigned char)*p <= ' ') ++p;
      if (*p != '[') fail("invalid_vector", "malformed vector literal");
      ++p;
      for (;;) {
        while (*p && (unsigned char)*p <= ' ') ++p;
        if (*p == ']') { ++p; break; }
        const char* nb = p;
        while (*nb) ++nb;
        double d = 0;
        if (!fastParseNumber(p, nb, &d)) fail("invalid_vector", "malformed vector literal");
        float f = static_cast<float>(d);
        if (!std::isfinite(d) || !std::isfinite(f)) {
          fail("invalid_vector", "vector element is not a finite float32");
        }
        vec.push_back(f);
        while (*p && (unsigned char)*p <= ' ') ++p;
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        fail("invalid_vector", "malformed vector literal");
      }
      while (*p && (unsigned char)*p <= ' ') ++p;
      if (*p) fail("invalid_vector", "trailing characters in vector literal");
    } else {
      fail("type_mismatch", "vector column requires a vector literal");
    }
    if (static_cast<int>(vec.size()) != vc.dim) fail("dimension_mismatch", "vector dimension mismatch");
    for (float f : vec) {
      if (!std::isfinite(f)) fail("invalid_vector", "vector element is not a finite float32");
    }
    size_t off = out.size();
    out.resize(off + vec.size() * 4);
    std::memcpy(&out[off], vec.data(), vec.size() * 4);
  }
  for (int c = 0; c < static_cast<int>(sc.cols.size()); ++c) {
    if (c == sc.vec_col) continue;
    const Literal* lit = nullptr;
    for (size_t k = 0; k < col_order.size(); ++k) {
      if (col_order[k] == c) {
        lit = &vals[k];
        break;
      }
    }
    if (!lit) fail("column_count_mismatch", "missing value for column");
    if (sc.cols[c].type == ColType::BigInt) {
      int64_t v = 0;
      if (!literalToInt(*lit, &v)) fail("type_mismatch", "BIGINT column requires an integer");
      size_t off = out.size();
      out.resize(off + 8);
      std::memcpy(&out[off], &v, 8);
    } else {
      const std::string* s = nullptr;
      std::string tmp;
      if (lit->kind == LitKind::Str) {
        s = &lit->s;
      } else if (lit->kind == LitKind::Int) {
        tmp = std::to_string(lit->i);
        s = &tmp;
      } else {
        fail("type_mismatch", "VARCHAR column requires a string");
      }
      if (utf8CharCount(*s) > sc.cols[c].len) fail("value_too_long", "value exceeds VARCHAR length");
      if (s->size() > 65535) fail("value_too_long", "value too long");
      uint16_t len = static_cast<uint16_t>(s->size());
      size_t off = out.size();
      out.resize(off + 2 + len);
      std::memcpy(&out[off], &len, 2);
      std::memcpy(&out[off + 2], s->data(), len);
    }
  }
}

// Rebuilds a body applying overrides.
void rebuildBody(const TableSchema& sc, const uint8_t* body, const std::vector<Assign>& assigns,
                 const std::vector<int>& assign_cols, std::string& out) {
  std::vector<Literal> vals;
  std::vector<int> order;
  for (int c = 0; c < static_cast<int>(sc.cols.size()); ++c) {
    order.push_back(c);
    Literal lit;
    int override_idx = -1;
    for (size_t k = 0; k < assign_cols.size(); ++k) {
      if (assign_cols[k] == c) override_idx = static_cast<int>(k);
    }
    if (override_idx >= 0) {
      lit = assigns[override_idx].val;
    } else if (c == sc.vec_col) {
      lit.kind = LitKind::Vec;
      const float* v = vectorPtr(body);
      lit.vec.assign(v, v + sc.cols[c].dim);
    } else if (sc.cols[c].type == ColType::BigInt) {
      int64_t v;
      std::memcpy(&v, columnPtr(body, sc, c), 8);
      lit.kind = LitKind::Int;
      lit.i = v;
    } else {
      const uint8_t* p = columnPtr(body, sc, c);
      uint16_t len;
      std::memcpy(&len, p, 2);
      lit.kind = LitKind::Str;
      lit.s.assign(reinterpret_cast<const char*>(p + 2), len);
    }
    vals.push_back(std::move(lit));
  }
  buildBody(sc, order, vals, out);
}

}  // namespace

ExecResult Database::Impl::execInsert(Statement& st) {
  Table& t = tableByName(st.table);
  ExecResult res;
  std::vector<int> col_order;
  if (st.insert_cols.empty()) {
    for (int c = 0; c < static_cast<int>(t.schema.cols.size()); ++c) col_order.push_back(c);
  } else {
    for (const auto& name : st.insert_cols) col_order.push_back(resolveColumn(t, name));
  }
  if (col_order.size() != t.schema.cols.size()) {
    fail("column_count_mismatch", "INSERT must provide every column");
  }
  // validate + encode everything before touching the log
  std::string buf;
  std::vector<uint64_t> new_offs;
  std::vector<int64_t> pks;
  uint64_t base = log.size();
  std::unordered_map<int64_t, int> seen;
  for (auto& row : st.insert_rows) {
    if (row.size() != col_order.size()) fail("column_count_mismatch", "value count mismatch");
    std::string body;
    buildBody(t.schema, col_order, row, body);
    int pk_idx = 0;
    for (size_t k = 0; k < col_order.size(); ++k) {
      if (col_order[k] == t.schema.pk_col) pk_idx = static_cast<int>(k);
    }
    int64_t pk = 0;
    if (!literalToInt(row[pk_idx], &pk)) fail("type_mismatch", "primary key must be an integer");
    if (t.pk2rid.count(pk) || seen.count(pk)) fail("duplicate_key", "duplicate primary key");
    seen[pk] = 1;
    uint64_t off = base + buf.size();
    appendRec(buf, kKindRow, t.schema.id, t.offs.size() + new_offs.size(), pk, body.data(),
              body.size());
    new_offs.push_back(off);
    pks.push_back(pk);
  }
  if (!buf.empty()) {
    log.append(buf.data(), buf.size());
    log.ensureMap();
  }
  for (size_t i = 0; i < new_offs.size(); ++i) {
    uint32_t rowid = static_cast<uint32_t>(t.offs.size());
    t.offs.push_back(new_offs[i]);
    t.pk2rid[pks[i]] = rowid;
    t.undo.push_back(UndoRec{rowid, 0, pks[i], true});
  }
  if (!new_offs.empty()) ++t.epoch;
  res.affected_rows = static_cast<int64_t>(new_offs.size());
  res.has_affected_rows = true;
  return res;
}

ExecResult Database::Impl::execUpdate(Statement& st) {
  Table& t = tableByName(st.table);
  std::vector<int> assign_cols;
  for (auto& a : st.assigns) {
    int c = resolveColumn(t, a.col);
    if (c == t.schema.pk_col) fail("unsupported_sql", "primary key updates are not supported");
    assign_cols.push_back(c);
  }
  // resolve filters
  std::vector<int> cond_cols;
  std::vector<Literal> cond_lits;
  std::vector<CmpOp> cond_ops;
  for (auto& c : st.conds) {
    cond_cols.push_back(resolveColumn(t, c.col));
    cond_lits.push_back(c.lit);
    cond_ops.push_back(c.op);
  }
  const uint8_t* base = log.base();
  std::string buf;
  uint64_t write_base = log.size();
  std::vector<std::pair<uint32_t, uint64_t>> touched;  // rowid, new offset
  std::vector<std::string> bodies;
  std::vector<uint32_t> rowids;
  for (uint32_t r = 0; r < t.offs.size(); ++r) {
    uint64_t off = t.offs[r];
    if (!off) continue;
    if (off + kHdr + 8 > log.size()) {
      std::fprintf(stderr, "[mvdb] corrupt row pointer r=%u off=%llu size=%llu\n", r,
                   (unsigned long long)off, (unsigned long long)log.size());
      fail("internal_error", "corrupt row pointer");
    }
    const uint8_t* body = base + off + kHdr;
    bool ok = true;
    for (size_t k = 0; k < cond_cols.size() && ok; ++k) {
      int c = cond_cols[k];
      const uint8_t* p = columnPtr(body, t.schema, c);
      if (t.schema.cols[c].type == ColType::BigInt) {
        int64_t v;
        std::memcpy(&v, p, 8);
        int64_t rhs;
        if (!literalToInt(cond_lits[k], &rhs)) fail("type_mismatch", "BIGINT comparison requires an integer");
        switch (cond_ops[k]) {
          case CmpOp::Eq: ok = v == rhs; break;
          case CmpOp::Lt: ok = v < rhs; break;
          case CmpOp::Le: ok = v <= rhs; break;
          case CmpOp::Gt: ok = v > rhs; break;
          case CmpOp::Ge: ok = v >= rhs; break;
        }
      } else {
        uint16_t len;
        std::memcpy(&len, p, 2);
        const char* s = reinterpret_cast<const char*>(p + 2);
        if (cond_lits[k].kind != LitKind::Str) fail("type_mismatch", "VARCHAR comparison requires a string");
        const std::string& rhs = cond_lits[k].s;
        int cmp = std::string_view(s, len).compare(std::string_view(rhs.data(), rhs.size()));
        switch (cond_ops[k]) {
          case CmpOp::Eq: ok = cmp == 0; break;
          case CmpOp::Lt: ok = cmp < 0; break;
          case CmpOp::Le: ok = cmp <= 0; break;
          case CmpOp::Gt: ok = cmp > 0; break;
          case CmpOp::Ge: ok = cmp >= 0; break;
        }
      }
    }
    if (!ok) continue;
    std::string nb;
    rebuildBody(t.schema, body, st.assigns, assign_cols, nb);
    uint64_t new_off = write_base + buf.size();
    // the primary key of a row never changes
    int64_t pk = 0;
    std::memcpy(&pk, columnPtr(body, t.schema, t.schema.pk_col), 8);
    appendRec(buf, kKindRow, t.schema.id, r, pk, nb.data(), nb.size());
    touched.emplace_back(r, new_off);
  }
  if (!buf.empty()) {
    log.append(buf.data(), buf.size());
    log.ensureMap();
  }
  for (auto& pr : touched) {
    uint32_t r = pr.first;
    int64_t pk = 0;
    {
      const uint8_t* body = log.base() + pr.second + kHdr;
      std::memcpy(&pk, columnPtr(body, t.schema, t.schema.pk_col), 8);
    }
    uint64_t prev = t.offs[r];
    t.offs[r] = pr.second;
    t.undo.push_back(UndoRec{r, prev, pk, false});
  }
  if (!touched.empty()) ++t.epoch;
  ExecResult res;
  res.affected_rows = static_cast<int64_t>(touched.size());
  res.has_affected_rows = true;
  return res;
}

ExecResult Database::Impl::execDelete(Statement& st) {
  Table& t = tableByName(st.table);
  std::vector<int> cond_cols;
  std::vector<Literal> cond_lits;
  std::vector<CmpOp> cond_ops;
  for (auto& c : st.conds) {
    cond_cols.push_back(resolveColumn(t, c.col));
    cond_lits.push_back(c.lit);
    cond_ops.push_back(c.op);
  }
  const uint8_t* base = log.base();
  std::string buf;
  std::vector<std::pair<uint32_t, int64_t>> hits;
  for (uint32_t r = 0; r < t.offs.size(); ++r) {
    uint64_t off = t.offs[r];
    if (!off) continue;
    if (off + kHdr + 8 > log.size()) {
      std::fprintf(stderr, "[mvdb] corrupt row pointer r=%u off=%llu size=%llu\n", r,
                   (unsigned long long)off, (unsigned long long)log.size());
      fail("internal_error", "corrupt row pointer");
    }
    const uint8_t* body = base + off + kHdr;
    bool ok = true;
    for (size_t k = 0; k < cond_cols.size() && ok; ++k) {
      int c = cond_cols[k];
      const uint8_t* p = columnPtr(body, t.schema, c);
      if (t.schema.cols[c].type == ColType::BigInt) {
        int64_t v;
        std::memcpy(&v, p, 8);
        int64_t rhs;
        if (!literalToInt(cond_lits[k], &rhs)) fail("type_mismatch", "BIGINT comparison requires an integer");
        switch (cond_ops[k]) {
          case CmpOp::Eq: ok = v == rhs; break;
          case CmpOp::Lt: ok = v < rhs; break;
          case CmpOp::Le: ok = v <= rhs; break;
          case CmpOp::Gt: ok = v > rhs; break;
          case CmpOp::Ge: ok = v >= rhs; break;
        }
      } else {
        uint16_t len;
        std::memcpy(&len, p, 2);
        const char* s = reinterpret_cast<const char*>(p + 2);
        if (cond_lits[k].kind != LitKind::Str) fail("type_mismatch", "VARCHAR comparison requires a string");
        const std::string& rhs = cond_lits[k].s;
        int cmp = std::string_view(s, len).compare(std::string_view(rhs.data(), rhs.size()));
        switch (cond_ops[k]) {
          case CmpOp::Eq: ok = cmp == 0; break;
          case CmpOp::Lt: ok = cmp < 0; break;
          case CmpOp::Le: ok = cmp <= 0; break;
          case CmpOp::Gt: ok = cmp > 0; break;
          case CmpOp::Ge: ok = cmp >= 0; break;
        }
      }
    }
    if (!ok) continue;
    int64_t pk = 0;
    std::memcpy(&pk, columnPtr(body, t.schema, t.schema.pk_col), 8);
    hits.emplace_back(r, pk);
  }
  for (auto& h : hits) {
    appendRec(buf, kKindDel, t.schema.id, h.first, h.second, nullptr, 0);
  }
  if (!buf.empty()) {
    log.append(buf.data(), buf.size());
    log.ensureMap();
  }
  for (auto& h : hits) {
    uint32_t r = h.first;
    uint64_t prev = t.offs[r];
    t.offs[r] = 0;
    t.pk2rid.erase(h.second);
    t.undo.push_back(UndoRec{r, prev, h.second, false});
  }
  if (!hits.empty()) ++t.epoch;
  ExecResult res;
  res.affected_rows = static_cast<int64_t>(hits.size());
  res.has_affected_rows = true;
  return res;
}

// ------------------------------------------------------------------ SELECT

namespace {

struct SelCtx {
  const TableSchema* sc = nullptr;
  const uint8_t* base = nullptr;
  std::vector<int> cond_cols;
  std::vector<Literal> cond_lits;
  std::vector<CmpOp> cond_ops;
  const float* qvec = nullptr;
  int dim = 0;
};

inline bool matchRow(const SelCtx& ctx, const uint8_t* body) {
  for (size_t k = 0; k < ctx.cond_cols.size(); ++k) {
    int c = ctx.cond_cols[k];
    const uint8_t* p = columnPtr(body, *ctx.sc, c);
    if (ctx.sc->cols[c].type == ColType::BigInt) {
      int64_t v;
      std::memcpy(&v, p, 8);
      int64_t rhs;
      if (!literalToInt(ctx.cond_lits[k], &rhs)) fail("type_mismatch", "BIGINT comparison requires an integer");
      switch (ctx.cond_ops[k]) {
        case CmpOp::Eq: if (v != rhs) return false; break;
        case CmpOp::Lt: if (!(v < rhs)) return false; break;
        case CmpOp::Le: if (!(v <= rhs)) return false; break;
        case CmpOp::Gt: if (!(v > rhs)) return false; break;
        case CmpOp::Ge: if (!(v >= rhs)) return false; break;
      }
    } else {
      uint16_t len;
      std::memcpy(&len, p, 2);
      const char* s = reinterpret_cast<const char*>(p + 2);
      if (ctx.cond_lits[k].kind != LitKind::Str) fail("type_mismatch", "VARCHAR comparison requires a string");
      const std::string& rhs = ctx.cond_lits[k].s;
      int cmp = std::string_view(s, len).compare(std::string_view(rhs.data(), rhs.size()));
      switch (ctx.cond_ops[k]) {
        case CmpOp::Eq: if (cmp != 0) return false; break;
        case CmpOp::Lt: if (!(cmp < 0)) return false; break;
        case CmpOp::Le: if (!(cmp <= 0)) return false; break;
        case CmpOp::Gt: if (!(cmp > 0)) return false; break;
        case CmpOp::Ge: if (!(cmp >= 0)) return false; break;
      }
    }
  }
  return true;
}

struct Hit {
  float d;
  int64_t pk;
  uint32_t rowid;
};

// "worse" ordering for a bounded max-heap of the best K hits
struct WorseFirst {
  bool operator()(const Hit& a, const Hit& b) const {
    if (a.d != b.d) return a.d > b.d;
    return a.pk > b.pk;
  }
};

}  // namespace

ExecResult Database::Impl::execSelect(Statement& st, const std::string& mode) {
  (void)mode;  // exact scan serves both exact and ann modes
  Table& t = tableByName(st.table);
  SelCtx ctx;
  ctx.sc = &t.schema;
  ctx.base = log.base();
  for (auto& c : st.conds) {
    ctx.cond_cols.push_back(resolveColumn(t, c.col));
    ctx.cond_lits.push_back(c.lit);
    ctx.cond_ops.push_back(c.op);
  }
  // resolve projections
  std::vector<int> proj_cols;
  std::vector<bool> proj_is_dist;
  std::vector<int> proj_dist_idx;  // index into distance query list
  std::vector<std::vector<float>> queries;
  std::vector<std::string> out_cols;
  for (auto& item : st.items) {
    if (item.star) {
      for (int c = 0; c < static_cast<int>(t.schema.cols.size()); ++c) {
        proj_cols.push_back(c);
        proj_is_dist.push_back(false);
        proj_dist_idx.push_back(-1);
        out_cols.push_back(t.schema.cols[c].name);
      }
      continue;
    }
    if (item.is_distance) {
      int c = resolveColumn(t, item.col);
      if (t.schema.cols[c].type != ColType::Vector) fail("invalid_vector", "L2_DISTANCE requires a vector column");
      if (c != t.schema.vec_col) fail("invalid_vector", "unknown vector column");
      if (static_cast<int>(item.qvec.size()) != t.schema.cols[c].dim) {
        fail("dimension_mismatch", "query vector dimension mismatch");
      }
      for (float f : item.qvec) {
        if (!std::isfinite(f)) fail("invalid_vector", "query vector element is not finite");
      }
      proj_cols.push_back(c);
      proj_is_dist.push_back(true);
      proj_dist_idx.push_back(static_cast<int>(queries.size()));
      queries.push_back(item.qvec);
      out_cols.push_back(item.has_alias ? item.alias : ("L2_DISTANCE(" + item.col + ")"));
    } else {
      int c = resolveColumn(t, item.col);
      proj_cols.push_back(c);
      proj_is_dist.push_back(false);
      proj_dist_idx.push_back(-1);
      out_cols.push_back(item.has_alias ? item.alias : t.schema.cols[c].name);
    }
  }
  // ORDER BY resolution
  bool order_is_dist = false;
  int order_col = -1;
  if (st.has_order) {
    bool matched = false;
    for (size_t i = 0; i < st.items.size(); ++i) {
      const auto& item = st.items[i];
      std::string name = item.has_alias ? item.alias : (item.star ? "" : item.col);
      if (!item.star && name == st.order.name && item.is_distance) {
        order_is_dist = true;
        matched = true;
        break;
      }
    }
    if (!matched) {
      // maybe it refers to a projected alias for a normal column or a base column
      for (size_t i = 0; i < st.items.size(); ++i) {
        const auto& item = st.items[i];
        if (!item.star && item.has_alias && item.alias == st.order.name && !item.is_distance) {
          order_col = resolveColumn(t, item.col);
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      for (int c = 0; c < static_cast<int>(t.schema.cols.size()); ++c) {
        if (t.schema.cols[c].name == st.order.name) {
          order_col = c;
          matched = true;
          break;
        }
      }
    }
    if (!matched) fail("column_not_found", "ORDER BY column not found: " + st.order.name);
  }
  bool need_dist = order_is_dist;
  for (bool b : proj_is_dist) need_dist = need_dist || b;
  int dist_slot = -1;
  for (size_t i = 0; i < proj_is_dist.size(); ++i) {
    if (proj_is_dist[i]) {
      dist_slot = proj_dist_idx[i];
      break;
    }
  }
  if (need_dist) {
    if (t.schema.vec_col < 0) fail("invalid_vector", "table has no vector column");
    if (queries.empty()) fail("invalid_vector", "ORDER BY distance requires a distance projection");
    ctx.qvec = queries[0].data();
    ctx.dim = static_cast<int>(queries[0].size());
    if (ctx.dim != t.schema.cols[t.schema.vec_col].dim) fail("dimension_mismatch", "query vector dimension mismatch");
  }
  (void)dist_slot;

  ExecResult res;
  res.is_query = true;
  res.columns = out_cols;
  std::string& out = res.rows_json;
  out.reserve(4096);
  out += '[';

  // ------------------------------------------------------ distance top-K path
  if (need_dist && st.has_order && order_is_dist && st.limit >= 0) {
    const size_t K = static_cast<size_t>(st.limit);
    const TableSchema& sc = t.schema;
    const uint8_t* base = ctx.base;
    const int dim = sc.cols[sc.vec_col].dim;
    const DistFn dist = g_dist;
    const float* q = ctx.qvec;
    // A single equality predicate is resolved through the lazy value index.
    const std::vector<uint32_t>* cand = nullptr;
    if (ctx.cond_cols.size() == 1 && ctx.cond_ops[0] == CmpOp::Eq) {
      cand = filterCandidates(t, ctx.cond_cols[0], ctx.cond_lits[0]);
    }
    const size_t nrows = cand ? cand->size() : t.offs.size();
    const uint64_t log_size = log.size();
    // Split the rowid range across threads; every worker keeps its own bounded
    // heap so the hot loop never takes a lock.
    int parts = 1;
    if (g_pool && nrows >= 4096) parts = std::min<int>(g_pool->workers() + 1, 4);
    std::vector<std::vector<Hit>> locals(static_cast<size_t>(parts));
    auto worker = [&](int part) {
      std::vector<Hit>& heap = locals[static_cast<size_t>(part)];
      heap.reserve(K + 1);
      size_t lo = nrows * static_cast<size_t>(part) / static_cast<size_t>(parts);
      size_t hi = nrows * static_cast<size_t>(part + 1) / static_cast<size_t>(parts);
      for (size_t r = lo; r < hi; ++r) {
        uint32_t rr = cand ? (*cand)[r] : static_cast<uint32_t>(r);
        uint64_t off = t.offs[rr];
        if (!off) continue;
        if (off + kHdr + 8 > log_size) {
          std::fprintf(stderr, "[mvdb] corrupt row pointer r=%u off=%llu size=%llu\n", rr,
                       (unsigned long long)off, (unsigned long long)log_size);
          fail("internal_error", "corrupt row pointer");
        }
        const uint8_t* body = base + off + kHdr;
        if (!ctx.cond_cols.empty() && !matchRow(ctx, body)) continue;
        float d = dist(vectorPtr(body), q, dim);
        int64_t pk;
        std::memcpy(&pk, columnPtr(body, sc, sc.pk_col), 8);
        Hit h{d, pk, rr};
        if (heap.size() < K) {
          heap.push_back(h);
          std::push_heap(heap.begin(), heap.end(), WorseFirst());
        } else if (K > 0) {
          const Hit& worst = heap.front();
          bool better = (h.d < worst.d) || (h.d == worst.d && h.pk < worst.pk);
          if (better) {
            std::pop_heap(heap.begin(), heap.end(), WorseFirst());
            heap.back() = h;
            std::push_heap(heap.begin(), heap.end(), WorseFirst());
          }
        }
      }
    };
    if (parts == 1) {
      worker(0);
    } else {
      g_pool->run(parts, worker);
    }
    std::vector<Hit> heap;
    heap.reserve(K + 1);
    for (auto& local : locals) {
      for (const Hit& h : local) {
        if (heap.size() < K) {
          heap.push_back(h);
          std::push_heap(heap.begin(), heap.end(), WorseFirst());
        } else if (K > 0) {
          const Hit& worst = heap.front();
          if ((h.d < worst.d) || (h.d == worst.d && h.pk < worst.pk)) {
            std::pop_heap(heap.begin(), heap.end(), WorseFirst());
            heap.back() = h;
            std::push_heap(heap.begin(), heap.end(), WorseFirst());
          }
        }
      }
    }
    std::sort(heap.begin(), heap.end(), [](const Hit& a, const Hit& b) {
      if (a.d != b.d) return a.d < b.d;
      return a.pk < b.pk;
    });
    if (st.order.desc) std::reverse(heap.begin(), heap.end());
    bool first_row = true;
    for (const auto& h : heap) {
      const uint8_t* body = base + t.offs[h.rowid] + kHdr;
      if (!first_row) out += ',';
      first_row = false;
      out += '[';
      for (size_t i = 0; i < proj_cols.size(); ++i) {
        if (i) out += ',';
        if (proj_is_dist[i]) {
          emitDouble(out, std::sqrt(static_cast<double>(h.d)));
        } else {
          int c = proj_cols[i];
          const uint8_t* p = columnPtr(body, sc, c);
          if (sc.cols[c].type == ColType::BigInt) {
            int64_t v;
            std::memcpy(&v, p, 8);
            emitInt(out, v);
          } else if (sc.cols[c].type == ColType::VarChar) {
            uint16_t len;
            std::memcpy(&len, p, 2);
            emitString(out, reinterpret_cast<const char*>(p + 2), len);
          } else {
            const float* v = vectorPtr(body);
            out += '[';
            for (int k = 0; k < sc.cols[c].dim; ++k) {
              if (k) out += ',';
              emitDouble(out, v[k]);
            }
            out += ']';
          }
        }
      }
      out += ']';
    }
    out += ']';
    return res;
  }

  // ------------------------------------------------------ generic scan path
  struct RowHit {
    uint32_t rowid;
    float d;
    int64_t pk;
  };
  std::vector<RowHit> hits;
  hits.reserve(std::min<size_t>(t.offs.size(), 4096));
  const std::vector<uint32_t>* gcan = nullptr;
  if (ctx.cond_cols.size() == 1 && ctx.cond_ops[0] == CmpOp::Eq) {
    gcan = filterCandidates(t, ctx.cond_cols[0], ctx.cond_lits[0]);
  }
  const size_t gtotal = gcan ? gcan->size() : t.offs.size();
  for (size_t gi = 0; gi < gtotal; ++gi) {
    uint32_t r = gcan ? (*gcan)[gi] : static_cast<uint32_t>(gi);
    uint64_t off = t.offs[r];
    if (!off) continue;
    if (off + kHdr + 8 > log.size()) {
      std::fprintf(stderr, "[mvdb] corrupt row pointer r=%u off=%llu size=%llu\n", r,
                   (unsigned long long)off, (unsigned long long)log.size());
      fail("internal_error", "corrupt row pointer");
    }
    const uint8_t* body = ctx.base + off + kHdr;
    if (!ctx.cond_cols.empty() && !matchRow(ctx, body)) continue;
    RowHit h;
    h.rowid = r;
    h.d = need_dist ? g_dist(vectorPtr(body), ctx.qvec, ctx.dim) : 0.f;
    std::memcpy(&h.pk, columnPtr(body, t.schema, t.schema.pk_col), 8);
    hits.push_back(h);
  }
  const TableSchema& sc = t.schema;
  auto cmp_int = [&](const RowHit& a, const RowHit& b) {
    int64_t va, vb;
    std::memcpy(&va, columnPtr(ctx.base + t.offs[a.rowid] + kHdr, sc, order_col), 8);
    std::memcpy(&vb, columnPtr(ctx.base + t.offs[b.rowid] + kHdr, sc, order_col), 8);
    if (va != vb) return st.order.desc ? va > vb : va < vb;
    return a.pk < b.pk;
  };
  auto cmp_str = [&](const RowHit& a, const RowHit& b) {
    const uint8_t* pa = columnPtr(ctx.base + t.offs[a.rowid] + kHdr, sc, order_col);
    const uint8_t* pb = columnPtr(ctx.base + t.offs[b.rowid] + kHdr, sc, order_col);
    uint16_t la, lb;
    std::memcpy(&la, pa, 2);
    std::memcpy(&lb, pb, 2);
    std::string_view sa(reinterpret_cast<const char*>(pa + 2), la);
    std::string_view sb(reinterpret_cast<const char*>(pb + 2), lb);
    int c = sa.compare(sb);
    if (c != 0) return st.order.desc ? c > 0 : c < 0;
    return a.pk < b.pk;
  };
  if (st.has_order) {
    if (order_is_dist) {
      std::sort(hits.begin(), hits.end(), [&](const RowHit& a, const RowHit& b) {
        if (a.d != b.d) return st.order.desc ? a.d > b.d : a.d < b.d;
        return a.pk < b.pk;
      });
    } else if (order_col >= 0 && sc.cols[order_col].type == ColType::BigInt) {
      std::sort(hits.begin(), hits.end(), cmp_int);
    } else if (order_col >= 0) {
      std::sort(hits.begin(), hits.end(), cmp_str);
    }
  }
  size_t limit = st.limit >= 0 ? static_cast<size_t>(st.limit) : hits.size();
  if (limit > hits.size()) limit = hits.size();
  bool first_row = true;
  for (size_t i = 0; i < limit; ++i) {
    const RowHit& h = hits[i];
    const uint8_t* body = ctx.base + t.offs[h.rowid] + kHdr;
    if (!first_row) out += ',';
    first_row = false;
    out += '[';
    for (size_t k = 0; k < proj_cols.size(); ++k) {
      if (k) out += ',';
      if (proj_is_dist[k]) {
        emitDouble(out, std::sqrt(static_cast<double>(h.d)));
        continue;
      }
      int c = proj_cols[k];
      const uint8_t* p = columnPtr(body, sc, c);
      if (sc.cols[c].type == ColType::BigInt) {
        int64_t v;
        std::memcpy(&v, p, 8);
        emitInt(out, v);
      } else if (sc.cols[c].type == ColType::VarChar) {
        uint16_t len;
        std::memcpy(&len, p, 2);
        emitString(out, reinterpret_cast<const char*>(p + 2), len);
      } else {
        const float* v = vectorPtr(body);
        out += '[';
        for (int j = 0; j < sc.cols[c].dim; ++j) {
          if (j) out += ',';
          emitDouble(out, v[j]);
        }
        out += ']';
      }
    }
    out += ']';
  }
  out += ']';
  return res;
}

// ------------------------------------------------------------ Database facade

Database::Database() : impl_(new Impl()) {}
Database::~Database() {
  delete impl_;
  impl_ = nullptr;
}

void Database::open(const std::string& data_dir) { impl_->open(data_dir); }

void Database::close() { impl_->close(); }

void Database::rollback_all() {
  if (!impl_->in_txn) return;
  impl_->rollbackTo(impl_->txn_start);
  impl_->in_txn = false;
  for (auto& t : impl_->tables) t.undo.clear();
}

ExecResult Database::execute(const std::string& sql, const std::string& search_mode) {
  Statement st = parse_sql(sql);
  Impl* d = impl_;
  switch (st.kind) {
    case Statement::Begin: {
      if (d->in_txn) fail("transaction_in_progress", "a transaction is already open");
      d->in_txn = true;
      for (auto& t : d->tables) t.undo.clear();
      d->txn_start = d->mark();
      ExecResult r;
      return r;
    }
    case Statement::Commit: {
      if (!d->in_txn) {
        ExecResult r;
        return r;
      }
      d->commitTxn();
      ExecResult r;
      return r;
    }
    case Statement::Rollback: {
      if (!d->in_txn) {
        ExecResult r;
        return r;
      }
      d->rollbackTo(d->txn_start);
      d->in_txn = false;
      for (auto& t : d->tables) t.undo.clear();
      ExecResult r;
      return r;
    }
    case Statement::CreateTable:
      return d->execCreateTable(st);
    case Statement::CreateVectorIndex:
      return d->execCreateIndex(st);
    case Statement::DropTable:
      return d->execDropTable(st);
    case Statement::Select:
      return d->execSelect(st, search_mode);
    case Statement::Insert:
    case Statement::Update:
    case Statement::Delete: {
      bool implicit = !d->in_txn;
      Impl::Mark start = d->mark();
      try {
        ExecResult r;
        if (st.kind == Statement::Insert) r = d->execInsert(st);
        else if (st.kind == Statement::Update) r = d->execUpdate(st);
        else r = d->execDelete(st);
        if (implicit) d->commitTxn();
        return r;
      } catch (...) {
        d->rollbackTo(start);
        if (implicit) {
          d->in_txn = false;
          for (auto& t : d->tables) t.undo.clear();
        }
        throw;
      }
    }
    default:
      fail("unsupported_sql", "statement is not supported");
  }
}

}  // namespace mvdb
