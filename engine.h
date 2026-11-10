// MiniVectorDB - lightweight single-node vector database.
// Public interface of the storage/query engine.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mvdb {

// Every failure path in the engine is reported with a stable machine readable
// code; the wire protocol only forwards the code and a human message.
struct DbError {
  std::string code;
  std::string msg;
  DbError(std::string c, std::string m) : code(std::move(c)), msg(std::move(m)) {}
};

enum class ColType : uint8_t { BigInt = 0, VarChar = 1, Vector = 2 };

struct Column {
  std::string name;
  ColType type = ColType::BigInt;
  int len = 0;  // maximum number of unicode characters (VARCHAR)
  int dim = 0;  // number of float32 elements (VECTOR)
};

struct TableSchema {
  uint32_t id = 0;
  std::string name;
  std::vector<Column> cols;
  int pk_col = -1;
  int vec_col = -1;
  std::vector<std::string> index_names;
};

// Result of one statement. Query results keep the already serialized JSON row
// array so that the protocol layer does not need to walk the rows again.
struct ExecResult {
  bool is_query = false;
  std::vector<std::string> columns;
  std::string rows_json;  // e.g. [[1,"a"],[2,"b"]]
  int64_t affected_rows = 0;
  bool has_affected_rows = false;
};

class Database {
 public:
  Database();
  ~Database();

  void open(const std::string& data_dir);
  void close();

  // Executes exactly one SQL statement. Throws DbError on failure.
  ExecResult execute(const std::string& sql, const std::string& search_mode);

  // Rolls back an open transaction (used on shutdown).
  void rollback_all();

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace mvdb
