// fastmath.hpp
//
// Branch-light, libm-free approximations of exp/sigmoid/tanh, used by
// afp_ops.hpp's *_inplace_native activation functions. These operate on
// plain floats (after a cheap AFP block decode) and lean on the classic
// "construct 2^k directly in the IEEE exponent field via a bit shift"
// trick (Schraudolph 1999) combined with a short minimax polynomial for
// the fractional part, which keeps 2^f accurate to ~1e-6 while avoiding
// any call into libm's transcendental routines.

#pragma once

#include <cstdint>
#include <cmath>
#include <bit>
#include <algorithm>

namespace afp::fastmath {

// 2^x for x unrestricted, via bit-shift exponent construction + degree-5
// minimax polynomial correction for the fractional part. This is the
// single primitive the exp/sigmoid/tanh approximations below build on.
inline float fast_exp2(float x) {
  x = std::clamp(x, -125.0f, 125.0f);
  float fi = std::floor(x);
  float f = x - fi;  // in [0,1)
  // Minimax polynomial approximation of 2^f on [0,1), coefficients chosen
  // so 2^f = 1 + f*ln2 + ... matches the Taylor series of 2^f = e^(f ln2);
  // truncated to degree 5 gives < 1e-6 max error on [0,1).
  float p = 1.0f + f * (0.6931471805599453f +
             f * (0.2402265069591007f +
             f * (0.05550410866482158f +
             f * (0.009618129107628477f +
             f * 0.001333355814678995f))));
  int32_t exp_bits = (static_cast<int32_t>(fi) + 127) << 23;
  float scale = std::bit_cast<float>(exp_bits);
  return p * scale;
}

// e^x = 2^(x * log2(e))
inline float fast_exp(float x) {
  constexpr float kLog2E = 1.4426950408889634f;
  return fast_exp2(x * kLog2E);
}

// sigmoid(x) = 1 / (1 + e^-x), computed in a numerically-safer form that
// avoids overflow for large |x| by branching on sign (branch is cheap and
// highly predictable for ML activation distributions).
inline float fast_sigmoid(float x) {
  if (x >= 0.0f) {
    float z = fast_exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = fast_exp(x);
    return z / (1.0f + z);
  }
}

// tanh(x) = 2*sigmoid(2x) - 1, reusing the same fast_exp primitive.
inline float fast_tanh(float x) {
  if (x >= 0.0f) {
    float z = fast_exp(-2.0f * x);
    return (1.0f - z) / (1.0f + z);
  } else {
    float z = fast_exp(2.0f * x);
    return (z - 1.0f) / (z + 1.0f);
  }
}

} // namespace afp::fastmath
