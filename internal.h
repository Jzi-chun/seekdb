// Shared internal definitions: SQL AST and parser entry points.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine.h"

namespace mvdb {

enum class LitKind : uint8_t { Int = 0, Str = 1, Vec = 2 };

struct Literal {
  LitKind kind = LitKind::Int;
  int64_t i = 0;
  std::string s;
  std::vector<float> vec;
};

enum class CmpOp : uint8_t { Eq = 0, Lt = 1, Le = 2, Gt = 3, Ge = 4 };

struct Cond {
  std::string col;
  CmpOp op = CmpOp::Eq;
  Literal lit;
};

struct SelItem {
  bool star = false;
  bool is_distance = false;
  std::string col;   // projected column (or the vector column of the distance)
  std::vector<float> qvec;
  bool has_alias = false;
  std::string alias;
};

struct OrderKey {
  std::string name;
  bool desc = false;
};

struct CreateCol {
  std::string name;
  ColType type = ColType::BigInt;
  int len = 0;
  int dim = 0;
  bool pk = false;
};

struct Assign {
  std::string col;
  Literal val;
};

struct Statement {
  enum Kind {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
    CreateVectorIndex,
    DropTable,
    Begin,
    Commit,
    Rollback,
    Unsupported
  } kind = Unsupported;

  std::string table;
  std::string index_name;
  std::string index_col;

  // CREATE TABLE
  std::vector<CreateCol> create_cols;
  std::string table_pk;  // table level PRIMARY KEY (col)

  // INSERT
  std::vector<std::string> insert_cols;  // empty -> positional over all columns
  std::vector<std::vector<Literal>> insert_rows;

  // UPDATE
  std::vector<Assign> assigns;

  // SELECT / UPDATE / DELETE
  std::vector<Cond> conds;

  // SELECT
  std::vector<SelItem> items;
  bool has_order = false;
  OrderKey order;
  int64_t limit = -1;
};

// Parses one statement (a trailing semicolon is allowed). Throws DbError.
Statement parse_sql(const std::string& sql);

}  // namespace mvdb
