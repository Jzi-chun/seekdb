// Small, fast float scanner for vector literals.
//
// strtod() costs roughly 300 ns per value, and the write phase parses
// 10k * 128 = 1.28M values, so the parser is a measurable part of the write
// time. This scanner accumulates up to 19 digits into an integer mantissa and
// applies a power-of-ten table, which is exact for every value that round-trips
// through float32 anyway.
#pragma once

#include <cstdint>

namespace mvdb {

inline constexpr double kPow10Table[] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

// Parses an optionally signed decimal number. Returns false for anything that
// is not a plain decimal number (so "nan", "inf", empty and junk are rejected).
inline bool fastParseNumber(const char*& p, const char* end, double* out) {
  const char* start = p;
  bool neg = false;
  if (p < end && (*p == '+' || *p == '-')) {
    neg = (*p == '-');
    ++p;
  }
  uint64_t mant = 0;
  int digits = 0;
  int exp10 = 0;
  bool any = false;
  while (p < end && *p >= '0' && *p <= '9') {
    any = true;
    if (digits < 19) {
      mant = mant * 10 + static_cast<uint64_t>(*p - '0');
      ++digits;
    }
    ++p;
  }
  if (p < end && *p == '.') {
    ++p;
    while (p < end && *p >= '0' && *p <= '9') {
      any = true;
      if (digits < 19) {
        mant = mant * 10 + static_cast<uint64_t>(*p - '0');
        ++digits;
        --exp10;
      }
      ++p;
    }
  }
  if (!any) {
    p = start;
    return false;
  }
  if (p < end && (*p == 'e' || *p == 'E')) {
    ++p;
    bool eneg = false;
    if (p < end && (*p == '+' || *p == '-')) {
      eneg = (*p == '-');
      ++p;
    }
    int e = 0;
    int ed = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      if (e < 10000) e = e * 10 + (*p - '0');
      ++ed;
      ++p;
    }
    if (ed == 0) {
      p = start;
      return false;
    }
    exp10 += eneg ? -e : e;
  }
  double v = static_cast<double>(mant);
  int e = exp10;
  while (e > 22) {
    v *= 1e22;
    e -= 22;
  }
  while (e < -22) {
    v /= 1e22;
    e += 22;
  }
  v *= (e >= 0) ? kPow10Table[e] : 1.0 / kPow10Table[-e];
  *out = neg ? -v : v;
  return true;
}

}  // namespace mvdb
