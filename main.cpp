// MiniVectorDB protocol front end: JSONL over stdin/stdout (see STDIO_PROTOCOL.md).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <string>

#include "engine.h"

namespace {

void writeOut(const std::string& s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fflush(stdout);
}

// ---------------------------------------------------------------- json input

void appendUtf8(std::string& out, unsigned cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool parseJsonString(const char*& p, const char* end, std::string& out) {
  if (p >= end || *p != '"') return false;
  ++p;
  out.clear();
  while (p < end) {
    unsigned char c = static_cast<unsigned char>(*p);
    if (c == '"') {
      ++p;
      return true;
    }
    if (c == '\\') {
      ++p;
      if (p >= end) return false;
      char e = *p++;
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (p + 4 > end) return false;
          unsigned cp = 0;
          for (int i = 0; i < 4; ++i) {
            char h = *p++;
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
            else return false;
          }
          // surrogate pair
          if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= end && p[0] == '\\' && p[1] == 'u') {
            const char* save = p;
            p += 2;
            unsigned lo = 0;
            for (int i = 0; i < 4; ++i) {
              char h = *p++;
              lo <<= 4;
              if (h >= '0' && h <= '9') lo |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') lo |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') lo |= static_cast<unsigned>(h - 'A' + 10);
              else { p = save; lo = 0; break; }
            }
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              p = save;
            }
          }
          appendUtf8(out, cp);
          break;
        }
        default: return false;
      }
      continue;
    }
    out.push_back(static_cast<char>(c));
    ++p;
  }
  return false;
}

bool skipJsonValue(const char*& p, const char* end) {
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  if (p >= end) return false;
  if (*p == '"') {
    std::string tmp;
    return parseJsonString(p, end, tmp);
  }
  if (*p == '{' || *p == '[') {
    int depth = 0;
    while (p < end) {
      if (*p == '"') {
        std::string tmp;
        if (!parseJsonString(p, end, tmp)) return false;
        continue;
      }
      if (*p == '{' || *p == '[') ++depth;
      else if (*p == '}' || *p == ']') {
        --depth;
        ++p;
        if (depth == 0) return true;
        continue;
      }
      ++p;
    }
    return false;
  }
  const char* start = p;
  while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n' && *p != '\r') ++p;
  return p > start;
}

struct Request {
  std::string id_raw = "null";
  std::string op;
  std::string sql;
  std::string search_mode = "exact";
  bool ok = false;
};

Request parseRequest(const char* data, size_t len) {
  Request req;
  const char* p = data;
  const char* end = data + len;
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  if (p >= end || *p != '{') return req;
  ++p;
  bool first = true;
  while (p < end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    if (p < end && *p == ',') {
      ++p;
      continue;
    }
    if (p < end && *p == '}') {
      ++p;
      req.ok = true;
      return req;
    }
    std::string key;
    if (!parseJsonString(p, end, key)) return req;
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    if (p >= end || *p != ':') return req;
    ++p;
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    const char* vstart = p;
    if (key == "sql" || key == "op" || key == "search_mode") {
      std::string val;
      if (!parseJsonString(p, end, val)) return req;
      if (key == "sql") req.sql = std::move(val);
      else if (key == "op") req.op = std::move(val);
      else req.search_mode = std::move(val);
    } else {
      if (!skipJsonValue(p, end)) return req;
      if (key == "id") req.id_raw = std::string(vstart, static_cast<size_t>(p - vstart));
    }
    first = false;
    (void)first;
  }
  return req;
}

std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char tmp[8];
          std::snprintf(tmp, sizeof(tmp), "\\u%04x", c);
          out += tmp;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

// Splits a SQL text on top level semicolons; keeps statements inside quotes.
std::vector<std::string> splitStatements(const std::string& sql) {
  std::vector<std::string> out;
  std::string cur;
  bool in_str = false;
  bool in_ident = false;
  for (size_t i = 0; i < sql.size(); ++i) {
    char c = sql[i];
    if (in_str) {
      cur.push_back(c);
      if (c == '\'') {
        if (i + 1 < sql.size() && sql[i + 1] == '\'') {
          cur.push_back(sql[++i]);
        } else {
          in_str = false;
        }
      } else if (c == '\\' && i + 1 < sql.size()) {
        cur.push_back(sql[++i]);
      }
      continue;
    }
    if (in_ident) {
      cur.push_back(c);
      if (c == '`') in_ident = false;
      continue;
    }
    if (c == '\'') {
      in_str = true;
      cur.push_back(c);
      continue;
    }
    if (c == '`') {
      in_ident = true;
      cur.push_back(c);
      continue;
    }
    if (c == ';') {
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

bool blank(const std::string& s) {
  for (char c : s) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return false;
  }
  return true;
}

std::string dataDirFromArgs(int argc, char** argv) {
  std::string dir;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) dir = argv[++i];
  }
  if (dir.empty()) {
    const char* env = std::getenv("VECTORDB_DATA_DIR");
    if (env && *env) dir = env;
  }
  if (dir.empty()) dir = "/data";
  return dir;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = dataDirFromArgs(argc, argv);
  mvdb::Database db;
  try {
    db.open(dir);
  } catch (const mvdb::DbError& e) {
    std::fprintf(stderr, "[mvdb] startup failed: %s (%s)\n", e.msg.c_str(), e.code.c_str());
    return 1;
  }
  writeOut("{\"type\":\"ready\"}\n");

  char* line = nullptr;
  size_t cap = 0;
  ssize_t n;
  for (;;) {
    n = ::getline(&line, &cap, stdin);
    if (n < 0) break;
    if (n == 0 || blank(std::string(line, static_cast<size_t>(n)))) continue;
    Request req = parseRequest(line, static_cast<size_t>(n));
    if (!req.ok) {
      writeOut(std::string("{\"id\":") + req.id_raw +
               ",\"ok\":false,\"error\":{\"code\":\"bad_request\",\"message\":\"invalid JSON request\"}}\n");
      continue;
    }
    if (req.op == "shutdown") {
      try {
        db.rollback_all();
        db.close();
      } catch (const mvdb::DbError& e) {
        std::fprintf(stderr, "[mvdb] shutdown error: %s\n", e.msg.c_str());
      }
      writeOut(std::string("{\"id\":") + req.id_raw + ",\"ok\":true}\n");
      std::free(line);
      return 0;
    }
    if (req.op != "execute") {
      writeOut(std::string("{\"id\":") + req.id_raw +
               ",\"ok\":false,\"error\":{\"code\":\"unsupported_op\",\"message\":\"unknown op\"}}\n");
      continue;
    }
    std::string out;
    try {
      mvdb::ExecResult res;
      auto parts = splitStatements(req.sql);
      bool any = false;
      for (const auto& part : parts) {
        if (blank(part)) continue;
        res = db.execute(part, req.search_mode);
        any = true;
      }
      if (!any) throw mvdb::DbError("empty_statement", "empty SQL");
      out = "{\"id\":" + req.id_raw + ",\"ok\":true";
      if (res.is_query) {
        out += ",\"columns\":[";
        for (size_t i = 0; i < res.columns.size(); ++i) {
          if (i) out += ',';
          out += '"';
          out += jsonEscape(res.columns[i]);
          out += '"';
        }
        out += "],\"rows\":";
        out += res.rows_json;
      } else if (res.has_affected_rows) {
        out += ",\"affected_rows\":";
        char buf[24];
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(res.affected_rows));
        out += buf;
      }
      out += "}\n";
    } catch (const mvdb::DbError& e) {
      out = "{\"id\":" + req.id_raw + ",\"ok\":false,\"error\":{\"code\":\"" + jsonEscape(e.code) +
            "\",\"message\":\"" + jsonEscape(e.msg) + "\"}}\n";
    } catch (const std::exception& e) {
      out = "{\"id\":" + req.id_raw +
            ",\"ok\":false,\"error\":{\"code\":\"internal_error\",\"message\":\"" + jsonEscape(e.what()) +
            "\"}}\n";
    }
    writeOut(out);
  }
  std::free(line);
  try {
    db.rollback_all();
    db.close();
  } catch (...) {
  }
  return 0;
}
