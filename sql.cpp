// Lexer and recursive descent parser for the SQL subset of MiniVectorDB.
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "internal.h"
#include "numparse.h"

namespace mvdb {
namespace {

struct Token {
  enum Kind { End, Ident, Num, Str, Punct, Op } kind = End;
  std::string text;  // identifier text / string content / operator
  int64_t num = 0;
  bool overflow = false;
  size_t pos = 0;
};

[[noreturn]] void syntax(const std::string& what) {
  throw DbError("syntax_error", what);
}

bool is_ident_start(unsigned char c) {
  return std::isalpha(c) || c == '_' || c == '$' || c >= 0x80;
}
bool is_ident_char(unsigned char c) {
  return std::isalnum(c) || c == '_' || c == '$' || c >= 0x80;
}

std::vector<Token> lex(const std::string& s) {
  std::vector<Token> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (std::isspace(c)) {
      ++i;
      continue;
    }
    // -- line comment and /* */ block comment
    if (c == '-' && i + 1 < n && s[i + 1] == '-') {
      while (i < n && s[i] != '\n') ++i;
      continue;
    }
    if (c == '/' && i + 1 < n && s[i + 1] == '*') {
      i += 2;
      while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) ++i;
      i = (i + 1 < n) ? i + 2 : n;
      continue;
    }
    Token t;
    t.pos = i;
    if (c == '\'') {
      ++i;
      std::string v;
      while (i < n) {
        char ch = s[i];
        if (ch == '\'') {
          if (i + 1 < n && s[i + 1] == '\'') {  // doubled quote
            v.push_back('\'');
            i += 2;
            continue;
          }
          ++i;
          break;
        }
        if (ch == '\\' && i + 1 < n) {
          char e = s[i + 1];
          switch (e) {
            case 'n': v.push_back('\n'); break;
            case 't': v.push_back('\t'); break;
            case 'r': v.push_back('\r'); break;
            case '0': v.push_back('\0'); break;
            case 'b': v.push_back('\b'); break;
            default: v.push_back(e); break;
          }
          i += 2;
          continue;
        }
        v.push_back(ch);
        ++i;
      }
      t.kind = Token::Str;
      t.text = std::move(v);
      out.push_back(std::move(t));
      continue;
    }
    if (c == '`') {
      ++i;
      std::string v;
      while (i < n && s[i] != '`') v.push_back(s[i++]);
      if (i < n) ++i;
      t.kind = Token::Ident;
      t.text = std::move(v);
      out.push_back(std::move(t));
      continue;
    }
    if (std::isdigit(c)) {
      uint64_t v = 0;
      bool ovf = false;
      while (i < n && std::isdigit(static_cast<unsigned char>(s[i]))) {
        unsigned d = static_cast<unsigned>(s[i] - '0');
        if (v > (static_cast<uint64_t>(INT64_MAX) - d) / 10) ovf = true;
        else v = v * 10 + d;
        ++i;
      }
      t.kind = Token::Num;
      t.num = static_cast<int64_t>(v);
      t.overflow = ovf;
      out.push_back(std::move(t));
      continue;
    }
    if (is_ident_start(c)) {
      size_t start = i;
      while (i < n && is_ident_char(static_cast<unsigned char>(s[i]))) ++i;
      t.kind = Token::Ident;
      t.text = s.substr(start, i - start);
      out.push_back(std::move(t));
      continue;
    }
    if ((c == '<' || c == '>' || c == '!') && i + 1 < n) {
      char d = s[i + 1];
      if (d == '=' || (c == '<' && d == '>') || (c == '!' && d == '=')) {
        t.kind = Token::Op;
        t.text = std::string(1, static_cast<char>(c)) + d;
        i += 2;
        out.push_back(std::move(t));
        continue;
      }
    }
    if (c == '=' || c == '<' || c == '>') {
      t.kind = Token::Op;
      t.text = std::string(1, static_cast<char>(c));
      ++i;
      out.push_back(std::move(t));
      continue;
    }
    t.kind = Token::Punct;
    t.text = std::string(1, static_cast<char>(c));
    ++i;
    out.push_back(std::move(t));
  }
  Token end;
  end.kind = Token::End;
  end.pos = n;
  out.push_back(std::move(end));
  return out;
}

std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool parse_vector_text(const std::string& text, std::vector<float>* out) {
  const char* p = text.c_str();
  while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
  if (*p != '[') return false;
  ++p;
  out->clear();
  for (;;) {
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (*p == ']') { ++p; break; }
    const char* endp = p;
    while (*endp) ++endp;
    double v = 0;
    if (!fastParseNumber(p, endp, &v)) return false;
    out->push_back(static_cast<float>(v));
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (*p == ',') { ++p; continue; }
    if (*p == ']') { ++p; break; }
    if (*p == '\0') return false;
    return false;
  }
  while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
  return *p == '\0';
}

class Parser {
 public:
  explicit Parser(std::vector<Token> toks) : toks_(std::move(toks)) {}

  Statement parse_statement() {
    Statement st;
    if (peek().kind == Token::End) syntax("empty statement");
    if (peek().kind != Token::Ident) syntax("expected statement keyword");
    std::string kw = lower(peek().text);
    if (kw == "select") {
      ++pos_;
      st.kind = Statement::Select;
      parse_select(st);
    } else if (kw == "insert") {
      ++pos_;
      st.kind = Statement::Insert;
      parse_insert(st);
    } else if (kw == "update") {
      ++pos_;
      st.kind = Statement::Update;
      parse_update(st);
    } else if (kw == "delete") {
      ++pos_;
      st.kind = Statement::Delete;
      parse_delete(st);
    } else if (kw == "create") {
      ++pos_;
      parse_create(st);
    } else if (kw == "begin") {
      ++pos_;
      st.kind = Statement::Begin;
      if (peek_ident("work") || peek_ident("transaction")) ++pos_;
    } else if (kw == "start") {
      ++pos_;
      if (!peek_ident("transaction")) syntax("expected TRANSACTION");
      ++pos_;
      st.kind = Statement::Begin;
    } else if (kw == "commit") {
      ++pos_;
      st.kind = Statement::Commit;
      if (peek_ident("work")) ++pos_;
    } else if (kw == "rollback") {
      ++pos_;
      st.kind = Statement::Rollback;
      if (peek_ident("work")) ++pos_;
    } else if (kw == "drop") {
      ++pos_;
      if (!peek_ident("table")) throw DbError("unsupported_sql", "only DROP TABLE is supported");
      ++pos_;
      if (peek_ident("if")) {
        ++pos_;
        if (!peek_ident("exists")) syntax("expected EXISTS");
        ++pos_;
      }
      st.kind = Statement::DropTable;
      st.table = expect_ident();
    } else if (kw == "alter" || kw == "truncate" || kw == "rename" || kw == "with" ||
               kw == "show" || kw == "use" || kw == "set" || kw == "replace" ||
               kw == "analyze" || kw == "explain" || kw == "vacuum" || kw == "grant") {
      throw DbError("unsupported_sql", "statement is not supported: " + kw);
    } else {
      throw DbError("unsupported_sql", "unknown statement: " + kw);
    }
    if (peek().kind == Token::Punct && peek().text == ";") ++pos_;
    if (peek().kind != Token::End) syntax("unexpected trailing input: " + peek().text);
    return st;
  }

 private:
  const Token& peek(size_t k = 0) const {
    size_t i = pos_ + k;
    return i < toks_.size() ? toks_[i] : toks_.back();
  }
  bool peek_ident(const char* kw) const {
    return peek().kind == Token::Ident && lower(peek().text) == kw;
  }
  bool peek_punct(char c) const { return peek().kind == Token::Punct && peek().text[0] == c; }
  bool accept_ident(const char* kw) {
    if (peek_ident(kw)) { ++pos_; return true; }
    return false;
  }
  void expect_punct(char c) {
    if (!peek_punct(c)) syntax(std::string("expected '") + c + "'");
    ++pos_;
  }
  std::string expect_ident() {
    if (peek().kind != Token::Ident) syntax("expected identifier");
    std::string v = lower(peek().text);
    ++pos_;
    return v;
  }
  int64_t expect_int() {
    bool neg = false;
    if (peek().kind == Token::Punct && peek().text == "-") { neg = true; ++pos_; }
    if (peek().kind != Token::Num) syntax("expected integer");
    if (peek().overflow) syntax("integer literal out of range");
    int64_t v = peek().num;
    ++pos_;
    return neg ? -v : v;
  }

  void parse_select(Statement& st) {
    // projection list
    for (;;) {
      SelItem item;
      if (peek_punct('*')) {
        ++pos_;
        item.star = true;
        st.items.push_back(std::move(item));
      } else {
        if (peek().kind != Token::Ident) syntax("expected projection expression");
        std::string name = lower(peek().text);
        if (name == "count" || name == "sum" || name == "avg" || name == "min" ||
            name == "max") {
          throw DbError("unsupported_sql", "aggregate functions are not supported");
        }
        if ((name == "l2_distance" || name == "l2_distance_sq" || name == "distance" ||
             name == "cosine_distance") &&
            peek(1).kind == Token::Punct && peek(1).text == "(") {
          if (name != "l2_distance") {
            throw DbError("unsupported_sql", "only L2_DISTANCE is supported");
          }
          ++pos_;
          expect_punct('(');
          item.is_distance = true;
          item.col = expect_ident();
          if (!peek_punct(',')) syntax("expected ',' in L2_DISTANCE");
          ++pos_;
          Literal lit = parse_literal();
          if (lit.kind == LitKind::Int) {
            throw DbError("type_mismatch", "L2_DISTANCE requires a vector literal");
          }
          if (lit.kind == LitKind::Vec) {
            item.qvec = lit.vec;
          } else if (lit.kind == LitKind::Str) {
            std::vector<float> v;
            if (!parse_vector_text(lit.s, &v)) {
              throw DbError("invalid_vector", "malformed vector literal");
            }
            item.qvec = std::move(v);
          }
          expect_punct(')');
        } else {
          item.col = expect_ident();
          // allow table-qualified column: t.col
          if (peek_punct('.') && peek(1).kind == Token::Ident) {
            ++pos_;
            item.col = expect_ident();
          }
        }
        if (accept_ident("as")) {
          item.has_alias = true;
          item.alias = expect_alias();
        }
        st.items.push_back(std::move(item));
      }
      if (peek_punct(',')) { ++pos_; continue; }
      break;
    }
    if (!accept_ident("from")) syntax("expected FROM");
    st.table = expect_ident();
    parse_tail(st);
  }

  std::string expect_alias() {
    if (peek().kind == Token::Ident) {
      std::string v = lower(peek().text);
      ++pos_;
      return v;
    }
    if (peek().kind == Token::Str) {
      std::string v = peek().text;
      ++pos_;
      return v;
    }
    syntax("expected alias name");
  }

  void parse_tail(Statement& st) {
    if (accept_ident("where")) parse_conds(st);
    if (accept_ident("order")) {
      if (!accept_ident("by")) syntax("expected BY");
      st.has_order = true;
      st.order.name = expect_order_name();
      if (peek_punct('.')) {
        ++pos_;
        st.order.name = expect_ident();
      }
      if (accept_ident("desc")) st.order.desc = true;
      else accept_ident("asc");
    }
    if (accept_ident("limit")) {
      st.limit = expect_int();
      if (st.limit < 0) syntax("negative LIMIT");
      if (peek_punct(',')) {  // LIMIT offset, count
        ++pos_;
        st.limit = expect_int();
      }
    }
  }

  std::string expect_order_name() {
    if (peek().kind == Token::Ident) {
      std::string v = lower(peek().text);
      ++pos_;
      return v;
    }
    syntax("expected ORDER BY key");
  }

  void parse_conds(Statement& st) {
    for (;;) {
      Cond c;
      // column on the left, or literal on the left (20 <= value)
      if (peek().kind == Token::Ident) {
        c.col = expect_ident();
        if (peek_punct('.')) { ++pos_; c.col = expect_ident(); }
        c.op = parse_cmp_op();
        c.lit = parse_literal();
      } else {
        Literal left = parse_literal();
        CmpOp op = parse_cmp_op();
        c.col = expect_ident();
        // mirror the operator so that the literal stays on the right side
        switch (op) {
          case CmpOp::Lt: c.op = CmpOp::Gt; break;
          case CmpOp::Le: c.op = CmpOp::Ge; break;
          case CmpOp::Gt: c.op = CmpOp::Lt; break;
          case CmpOp::Ge: c.op = CmpOp::Le; break;
          default: c.op = CmpOp::Eq; break;
        }
        c.lit = std::move(left);
      }
      st.conds.push_back(std::move(c));
      if (accept_ident("and")) continue;
      break;
    }
  }

  CmpOp parse_cmp_op() {
    if (peek().kind != Token::Op) syntax("expected comparison operator");
    std::string op = peek().text;
    ++pos_;
    if (op == "=") return CmpOp::Eq;
    if (op == "<") return CmpOp::Lt;
    if (op == "<=") return CmpOp::Le;
    if (op == ">") return CmpOp::Gt;
    if (op == ">=") return CmpOp::Ge;
    if (op == "!=" || op == "<>") throw DbError("unsupported_sql", "operator " + op);
    syntax("unknown operator " + op);
  }

  Literal parse_literal() {
    Literal lit;
    bool neg = false;
    if (peek().kind == Token::Punct && peek().text == "-") {
      neg = true;
      ++pos_;
    }
    if (peek().kind == Token::Num) {
      if (peek().overflow) syntax("integer literal out of range");
      lit.kind = LitKind::Int;
      lit.i = neg ? -peek().num : peek().num;
      ++pos_;
      return lit;
    }
    if (neg) syntax("expected number after '-'");
    if (peek().kind == Token::Str) {
      lit.kind = LitKind::Str;
      lit.s = peek().text;
      ++pos_;
      return lit;
    }
    if (peek_punct('[')) {
      ++pos_;
      lit.kind = LitKind::Vec;
      for (;;) {
        bool lneg = false;
        if (peek().kind == Token::Punct && peek().text == "-") { lneg = true; ++pos_; }
        if (peek().kind != Token::Num) syntax("expected number in vector literal");
        int64_t v = peek().num;
        ++pos_;
        float f = static_cast<float>(lneg ? -v : v);
        while (peek_punct('.')) {  // allow [1.5, ...]
          ++pos_;
          if (peek().kind != Token::Num) syntax("expected fraction digits");
          ++pos_;
        }
        lit.vec.push_back(f);
        if (peek_punct(',')) { ++pos_; continue; }
        break;
      }
      expect_punct(']');
      return lit;
    }
    if (peek_ident("null")) {
      throw DbError("unsupported_sql", "NULL literal is not supported");
    }
    syntax("expected literal value");
  }

  void parse_insert(Statement& st) {
    if (!accept_ident("into")) syntax("expected INTO");
    st.table = expect_ident();
    if (peek_punct('(')) {
      ++pos_;
      for (;;) {
        st.insert_cols.push_back(expect_ident());
        if (peek_punct(',')) { ++pos_; continue; }
        break;
      }
      expect_punct(')');
    }
    if (!accept_ident("values") && !accept_ident("value")) syntax("expected VALUES");
    for (;;) {
      expect_punct('(');
      std::vector<Literal> row;
      for (;;) {
        row.push_back(parse_literal());
        if (peek_punct(',')) { ++pos_; continue; }
        break;
      }
      expect_punct(')');
      st.insert_rows.push_back(std::move(row));
      if (peek_punct(',')) { ++pos_; continue; }
      break;
    }
  }

  void parse_update(Statement& st) {
    st.table = expect_ident();
    if (!accept_ident("set")) syntax("expected SET");
    for (;;) {
      Assign a;
      a.col = expect_ident();
      if (peek().kind != Token::Op || peek().text != "=") syntax("expected '=' in SET");
      ++pos_;
      a.val = parse_literal();
      st.assigns.push_back(std::move(a));
      if (peek_punct(',')) { ++pos_; continue; }
      break;
    }
    if (accept_ident("where")) parse_conds(st);
  }

  void parse_delete(Statement& st) {
    if (!accept_ident("from")) syntax("expected FROM");
    st.table = expect_ident();
    if (accept_ident("where")) parse_conds(st);
    else if (peek().kind != Token::End && !peek_punct(';')) {
      throw DbError("unsupported_sql", "DELETE without WHERE clause is not supported");
    }
  }

  void parse_create(Statement& st) {
    if (accept_ident("table")) {
      if (peek_ident("if")) {
        ++pos_;
        if (!accept_ident("not")) syntax("expected NOT");
        if (!accept_ident("exists")) syntax("expected EXISTS");
      }
      st.kind = Statement::CreateTable;
      st.table = expect_ident();
      expect_punct('(');
      for (;;) {
        if (peek_ident("primary")) {
          ++pos_;
          if (!accept_ident("key")) syntax("expected KEY");
          expect_punct('(');
          st.table_pk = expect_ident();
          expect_punct(')');
        } else {
          CreateCol col;
          col.name = expect_ident();
          std::string type = expect_ident();
          if (type == "bigint" || type == "int" || type == "integer") {
            col.type = ColType::BigInt;
            if (type != "bigint") {
              throw DbError("unsupported_sql", "only BIGINT is supported");
            }
          } else if (type == "varchar" || type == "char") {
            col.type = ColType::VarChar;
            expect_punct('(');
            int64_t len = expect_int();
            if (len <= 0) syntax("invalid VARCHAR length");
            col.len = static_cast<int>(len);
            expect_punct(')');
          } else if (type == "vector") {
            col.type = ColType::Vector;
            expect_punct('(');
            int64_t dim = expect_int();
            if (dim <= 0) syntax("invalid VECTOR dimension");
            col.dim = static_cast<int>(dim);
            expect_punct(')');
          } else {
            throw DbError("unsupported_sql", "unsupported column type: " + type);
          }
          // column level constraints
          for (;;) {
            if (accept_ident("primary")) {
              if (!accept_ident("key")) syntax("expected KEY");
              col.pk = true;
            } else if (accept_ident("not")) {
              if (!accept_ident("null")) syntax("expected NULL");
            } else if (accept_ident("null")) {
              // nullable by default
            } else if (accept_ident("unique")) {
              throw DbError("unsupported_sql", "UNIQUE constraint is not supported");
            } else if (accept_ident("default")) {
              throw DbError("unsupported_sql", "DEFAULT is not supported");
            } else if (accept_ident("comment")) {
              if (peek().kind == Token::Str) ++pos_;
            } else {
              break;
            }
          }
          st.create_cols.push_back(std::move(col));
        }
        if (peek_punct(',')) { ++pos_; continue; }
        break;
      }
      expect_punct(')');
      return;
    }
    if (accept_ident("vector")) {
      if (!accept_ident("index")) syntax("expected INDEX");
      st.kind = Statement::CreateVectorIndex;
      st.index_name = expect_ident();
      if (!accept_ident("on")) syntax("expected ON");
      st.table = expect_ident();
      expect_punct('(');
      st.index_col = expect_ident();
      expect_punct(')');
      return;
    }
    if (accept_ident("index")) {
      st.kind = Statement::CreateVectorIndex;
      st.index_name = expect_ident();
      if (!accept_ident("on")) syntax("expected ON");
      st.table = expect_ident();
      expect_punct('(');
      st.index_col = expect_ident();
      expect_punct(')');
      return;
    }
    throw DbError("unsupported_sql", "unsupported CREATE statement");
  }

  std::vector<Token> toks_;
  size_t pos_ = 0;
};

}  // namespace

Statement parse_sql(const std::string& sql) {
  Parser p(lex(sql));
  return p.parse_statement();
}

}  // namespace mvdb
