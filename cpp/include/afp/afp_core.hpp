// afp_core.hpp
//
// Paper-faithful implementation of the Adaptive Floating Point (AFP) format
// from:
//   Thomas Y. Yeh, Maxwell R. Sterner, Zerlina Lai, Brandon Y. Chuang,
//   Alexander Ihler. "Be Like Water: Adaptive Floating Point for Machine
//   Learning." ICML 2022 (PMLR 162:25490-25500).
//   https://proceedings.mlr.press/v162/yeh22a/yeh22a.pdf
//
// ---------------------------------------------------------------------------
// FORMAT SUMMARY (Section 3 of the paper)
// ---------------------------------------------------------------------------
// AFP groups values into blocks of BLOCK_SIZE = 16 elements (Section 3.1).
//
// SHARED FIELDS (2 bytes / block, Section 3.3):
//   byte 0 : e*      - 8-bit shared exponent = max IEEE-754 exponent (biased
//                       by 127, exactly like FP32) among the block's values.
//   byte 1 : characterization byte, packed as (Section 3.3.1, 3.3.2):
//              bit 0 : positive flag, lower half   (elements 0..7)
//              bit 1 : positive flag, upper half   (elements 8..15)
//              bit 2 : zero flag, offset-0 group, lower half
//              bit 3 : zero flag, offset-0 group, upper half
//              bit 4 : zero flag, offset-1 group, lower half
//              bit 5 : zero flag, offset-1 group, upper half
//              bits 6,7 : reserved (0) -- paper allocates a full byte "to
//                         support byte alignment and enable future
//                         optimizations".
//
// PRIVATE FIELDS (9 bits / value, Section 3.2, "AFP8" fixed-width variant
// used for the paper's headline density/accuracy numbers in Table 1):
//   1 sign bit + 3 offset bits (t) + 5 mantissa bits, nominally.
//
//   Total block size = 16 bits (shared) + 16 * 9 bits (private)
//                     = 160 bits = 20 bytes for 16 values
//                     = 10 bits/value average.
//   32 bits/value (FP32) / 10 bits/value (AFP) = 3.2x memory density,
//   exactly matching the paper's reported 3.2x density vs. FP32.
//
// AUTO FOCUS / OFFSET (Section 3.2.1, 3.2.2):
//   t in [0,7] measures how many exponent steps a value's own exponent is
//   below the block's shared maximum exponent e*: value's exponent = e* - t.
//   For t < 7 the mantissa carries an implicit leading one (normalized,
//   just like IEEE-754). For t == 7 the value is treated as "denormal":
//   no implicit leading one is used, giving graceful underflow instead of
//   an abrupt jump to zero (mirrors IEEE-754 subnormals). Exact zero is
//   represented as t == 0b111 and mantissa == 0 (Section 3.2.3).
//
// POSITIVE FIELD (Section 3.3.1):
//   If every value in a half-block (8 elements) is >= 0, the sign bit
//   position of each of those 8 values is *repurposed* as an extra
//   mantissa bit (5 -> 6 bits), for free extra precision at no extra
//   storage cost. This is why the paper reports "7 effective mantissa
//   bits" on average despite only storing 5.
//
// ZERO FIELD (Section 3.3.2):
//   For elements with offset t == 0 or t == 1 within a half-block: if the
//   top fractional mantissa bit (the bit immediately following the
//   implicit leading one) is guaranteed zero for *every* member of that
//   offset-group, that bit does not need to be stored. The freed bit is
//   used to extend the mantissa by one more (less significant) bit,
//   again for free extra precision. Per the paper, this optimization is
//   only applied when the positive-field optimization is *not* already
//   active for that half ("the zero bits are ignored when the all
//   positive bit is on").
//
// Everywhere the paper leaves an implementation detail unspecified (e.g.
// the exact reconstruction formula for denormals), this file states the
// convention chosen and why, in a comment at the point of use.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <bit>

namespace afp {

// ============================================================================
// Tunable constants
// ============================================================================

// Block size the paper settles on as the sweet spot between accuracy and
// hardware complexity (Section 3.2, end of Section 3.2.2).
static constexpr int kBlockSize = 16;

static constexpr int kOffsetBits   = 3;              // 0..7
static constexpr int kBaseMantissaBits = 5;           // "AFP8" baseline
static constexpr int kMaxOffset    = (1 << kOffsetBits) - 1; // 7
static constexpr int kPrivateBits  = 1 + kOffsetBits + kBaseMantissaBits; // 9
static constexpr int kSharedHeaderBits = 16;           // 8 exponent + 8 char.
static constexpr int kExponentBias = 127;              // matches IEEE-754 FP32

// Total packed size (bytes) of one 16-element AFP block.
static constexpr int kBlockPrivateBits = kPrivateBits * kBlockSize;   // 144
static constexpr int kBlockTotalBits   = kSharedHeaderBits + kBlockPrivateBits; // 160
static constexpr int kBlockBytes       = kBlockTotalBits / 8;                  // 20

static_assert(kBlockTotalBits % 8 == 0, "Block must be byte-aligned");

// ============================================================================
// Low level bit stream helpers (MSB-first packing), written with shifts only
// ============================================================================

// A tiny fixed-capacity bit writer/reader operating directly on a byte
// buffer with shift/mask operations -- no branches in the hot path other
// than the byte-boundary crossing check.
class BitWriter {
 public:
  explicit BitWriter(uint8_t* buf) : buf_(buf) {}

  // Write the low `nbits` bits of `value` (nbits <= 32), MSB first.
  inline void write(uint32_t value, int nbits) {
    value &= (nbits == 32) ? 0xFFFFFFFFu : ((1u << nbits) - 1u);
    int remaining = nbits;
    while (remaining > 0) {
      int byte_idx = bit_pos_ >> 3;
      int bit_in_byte = bit_pos_ & 7;              // 0 = MSB of byte
      int free_bits = 8 - bit_in_byte;
      int take = remaining < free_bits ? remaining : free_bits;
      // Extract the top `take` bits (of the remaining `remaining`) from value.
      uint32_t chunk = (value >> (remaining - take)) & ((1u << take) - 1u);
      int shift = free_bits - take;
      buf_[byte_idx] = static_cast<uint8_t>(buf_[byte_idx] | (chunk << shift));
      bit_pos_ += take;
      remaining -= take;
    }
  }

  size_t bit_position() const { return bit_pos_; }

 private:
  uint8_t* buf_;
  size_t bit_pos_ = 0;
};

class BitReader {
 public:
  explicit BitReader(const uint8_t* buf) : buf_(buf) {}

  inline uint32_t read(int nbits) {
    uint32_t result = 0;
    int remaining = nbits;
    while (remaining > 0) {
      int byte_idx = bit_pos_ >> 3;
      int bit_in_byte = bit_pos_ & 7;
      int free_bits = 8 - bit_in_byte;
      int take = remaining < free_bits ? remaining : free_bits;
      int shift = free_bits - take;
      uint32_t chunk = (buf_[byte_idx] >> shift) & ((1u << take) - 1u);
      result = (result << take) | chunk;
      bit_pos_ += take;
      remaining -= take;
    }
    return result;
  }

  size_t bit_position() const { return bit_pos_; }

 private:
  const uint8_t* buf_;
  size_t bit_pos_ = 0;
};

// ============================================================================
// Fast float <-> (sign, exponent, mantissa) bit tricks
// ============================================================================
//
// FP32 bit layout: [sign:1][exponent:8 (bias 127)][mantissa:23]
// We use std::bit_cast (C++20) for strict-aliasing-safe reinterpretation,
// then pure shift/mask to pull out fields. This avoids any libm calls
// (frexp/ldexp) on the hot path.

struct FloatBits {
  uint32_t sign;      // 0 or 1
  int32_t  exponent;  // unbiased, i.e. real exponent such that value =
                       // (-1)^sign * 1.mantissa * 2^exponent  (normal case)
  uint32_t mantissa;  // 23-bit raw mantissa (no implicit leading one)
  bool is_zero;
};

inline uint32_t float_to_u32(float f) { return std::bit_cast<uint32_t>(f); }
inline float u32_to_float(uint32_t u) { return std::bit_cast<float>(u); }

inline FloatBits decompose_float(float f) {
  uint32_t bits = float_to_u32(f);
  FloatBits fb;
  fb.sign = bits >> 31;
  uint32_t raw_exp = (bits >> 23) & 0xFFu;
  fb.mantissa = bits & 0x7FFFFFu;
  fb.is_zero = (raw_exp == 0 && fb.mantissa == 0);
  if (raw_exp == 0) {
    // FP32 subnormal input: extremely small, effectively zero for AFP's
    // dynamic range (AFP's own minimum is far above FP32 subnormal range
    // for the exponents ML tensors actually use). Treat as zero.
    fb.exponent = -127;
    fb.is_zero = true;
  } else {
    fb.exponent = static_cast<int32_t>(raw_exp) - kExponentBias;
  }
  return fb;
}

// ============================================================================
// A single decoded AFP scalar, kept in a hardware-plausible fixed-point
// form: sign + integer significand (with implicit bit already folded in,
// when applicable) + a binary exponent for the significand's scale.
// This is the type the AFP-native math kernels (afp_ops.hpp) operate on
// directly via integer/shift arithmetic, without ever materializing an
// IEEE float, mirroring Figure 10 of the paper.
// ============================================================================
struct AFPScalar {
  int32_t sign;          // +1 or -1
  int32_t significand;   // unsigned integer, `sig_bits` wide (5,6,or 7 incl.
                          // implicit leading one for normalized values)
  int32_t sig_bits;      // number of fractional bits below the significand's
                          // radix point, i.e. value = sign * significand *
                          // 2^(exponent - sig_bits)
  int32_t exponent;       // = e* - t  (unbiased)
  bool is_zero;
};

inline float afp_scalar_to_float(const AFPScalar& v) {
  if (v.is_zero) return 0.0f;
  // Build the float directly via bit shifts instead of calling ldexp/pow:
  // value = sign * significand * 2^(exponent - sig_bits)
  //       = sign * (significand as fixed point Q(sig_bits)) * 2^exponent
  // We normalize `significand` so its MSB sits at the implicit-one position
  // of an IEEE mantissa (bit 23), then let the FP32 exponent field absorb
  // the rest. This keeps everything in integer shifts.
  int32_t sig = v.significand;
  int32_t exp = v.exponent;
  if (sig == 0) return 0.0f;
  // Normalize so the leading set bit of `sig` is exactly at position
  // `sig_bits` (i.e. sig in [2^sig_bits, 2^(sig_bits+1)) ), adjusting exp.
  while (sig >= (1 << (v.sig_bits + 1))) { sig >>= 1; exp += 1; }
  while (sig < (1 << v.sig_bits)) { sig <<= 1; exp -= 1; }
  // sig now has its implicit leading '1' at bit position sig_bits.
  uint32_t frac23;
  if (v.sig_bits <= 23) {
    frac23 = static_cast<uint32_t>(sig & ((1 << v.sig_bits) - 1)) << (23 - v.sig_bits);
  } else {
    frac23 = static_cast<uint32_t>(sig >> (v.sig_bits - 23)) & 0x7FFFFFu;
  }
  int32_t biased_exp = exp + kExponentBias;
  if (biased_exp <= 0) return 0.0f;         // underflow to zero
  if (biased_exp >= 255) biased_exp = 254;  // clamp instead of inf
  uint32_t bits = (static_cast<uint32_t>(v.sign < 0) << 31) |
                   (static_cast<uint32_t>(biased_exp) << 23) | frac23;
  return u32_to_float(bits);
}

// ============================================================================
// AFPBlock: packed 20-byte representation of 16 values.
// ============================================================================
struct AFPBlock {
  std::array<uint8_t, kBlockBytes> bytes{};

  // ---- shared-field accessors -------------------------------------------
  inline uint8_t exponent_byte() const { return bytes[0]; }
  inline void set_exponent_byte(uint8_t e) { bytes[0] = e; }
  inline int32_t shared_exponent() const {
    return static_cast<int32_t>(bytes[0]) - kExponentBias;
  }

  inline uint8_t characterization() const { return bytes[1]; }
  inline void set_characterization(uint8_t c) { bytes[1] = c; }

  inline bool positive_flag(int half) const {              // half: 0 or 1
    return (bytes[1] >> half) & 1u;
  }
  inline void set_positive_flag(int half, bool v) {
    bytes[1] = static_cast<uint8_t>(v ? (bytes[1] | (1u << half))
                                       : (bytes[1] & ~(1u << half)));
  }
  // group: 0 -> offset-0 zero-field, 1 -> offset-1 zero-field
  inline bool zero_flag(int group, int half) const {
    int bit = 2 + group * 2 + half;
    return (bytes[1] >> bit) & 1u;
  }
  inline void set_zero_flag(int group, int half, bool v) {
    int bit = 2 + group * 2 + half;
    bytes[1] = static_cast<uint8_t>(v ? (bytes[1] | (1u << bit))
                                       : (bytes[1] & ~(1u << bit)));
  }

  // Raw 9-bit private field access (offset from bit 16 = start of private
  // section), used by the encoder/decoder below.
  inline uint32_t read_private(int idx) const {
    BitReader br(bytes.data());
    // Fast-forward: we don't want a linear re-scan per element, so callers
    // that need many fields should use BitReader directly; this helper is
    // for convenience/testing.
    br.read(kSharedHeaderBits); // skip header (no-op on state other than pos)
    for (int i = 0; i < idx; ++i) br.read(kPrivateBits);
    return br.read(kPrivateBits);
  }
};

} // namespace afp
