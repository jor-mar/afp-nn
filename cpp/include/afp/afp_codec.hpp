// afp_codec.hpp
//
// Encoder/decoder for a single 16-element AFP block. See afp_core.hpp for
// the bit-layout reference and a summary of the paper's design.
//
// This file implements, faithfully to Section 3.2/3.3 of the paper:
//   - Auto Focus: per-value 3-bit offset t from the shared max exponent e*.
//   - Round-to-nearest mantissa quantization (paper section 3.2.3: "AFP
//     uses nearest rounding on the least significant bit for inference").
//   - Denormal encoding for t == 7 (no implicit leading one -> graceful
//     underflow instead of abrupt truncation).
//   - Positive-field optimization (Section 3.3.1): reuse the sign bit as
//     an extra mantissa bit when a half-block (8 values) is all >= 0.
//   - Zero-field optimization (Section 3.3.2): reuse a mantissa bit that
//     is provably zero across an entire offset-0 or offset-1 group within
//     a half-block, for a free extra bit of precision.
//
// Everything here is branch-light, integer/shift based -- no libm calls on
// the hot path.

#pragma once

#include "afp_core.hpp"

namespace afp {

struct EncodeStats {
  // Diagnostics accumulated during encode() calls, useful for validating
  // against the paper's reported statistics (Sec 3.2.2, 3.3.1, 3.3.2):
  // e.g. "1% of weight blocks are all-positive", "50% of blocks get a
  // zero-field bonus bit", "<1% of values truncate to zero".
  uint64_t blocks = 0;
  uint64_t values = 0;
  uint64_t values_truncated_to_zero = 0;
  uint64_t values_offset_exceeds_7 = 0; // would-be offset >=7 (denormal path taken)
  uint64_t half_blocks_all_positive = 0;
  uint64_t half_blocks_total = 0;
  uint64_t zero_bonus_bits_used = 0; // elements that got the +1 zero-field bit
  uint64_t positive_bonus_bits_used = 0; // elements that got the +1 positive-field bit
};

struct EncodeOptions {
  bool enable_positive_field = true;
  bool enable_zero_field = true;
};

namespace detail {

// Round the 23-bit fractional mantissa (representing 1.mantissa) to
// `width` bits, round-to-nearest (ties away from zero). Returns the
// rounded value in [0, 2^width]; caller must handle the carry-out case
// where the result equals 2^width (rounds up to the next power of two).
inline uint32_t round_frac(uint32_t mantissa23, int width) {
  int shift = 23 - width;
  uint32_t half_ulp = 1u << (shift - 1);
  return (mantissa23 + half_ulp) >> shift;
}

// Round an arbitrary Q23 fixed-point value (not assumed to have an
// implicit leading one -- used for denormal quantization) to `width`
// unsigned bits, clamped to the representable range.
inline uint32_t round_q23(uint32_t q23, int width) {
  int shift = 23 - width;
  uint32_t half_ulp = shift > 0 ? (1u << (shift - 1)) : 0;
  uint32_t r = (q23 + half_ulp) >> shift;
  uint32_t max_val = (1u << width) - 1u;
  return r > max_val ? max_val : r;
}

struct PerElementPlan {
  uint32_t sign;       // 0/1 (0 = positive)
  uint32_t offset;      // 0..7 (t)
  uint32_t mantissa;    // width-bit quantized fraction (post rounding/carry)
  bool is_zero;
};

// First pass: compute exact (sign, offset, mantissa-at-`width`-bits) for one
// element, given the block's shared exponent e*. Handles the carry (round
// up to next power of two) and denormal (offset==7) cases.
inline PerElementPlan plan_element(const FloatBits& fb, int32_t e_star, int width) {
  PerElementPlan p{};
  if (fb.is_zero) {
    p.sign = 0; p.offset = kMaxOffset; p.mantissa = 0; p.is_zero = true;
    return p;
  }
  p.sign = fb.sign;
  p.is_zero = false;
  int32_t raw_offset = e_star - fb.exponent;
  if (raw_offset < 0) raw_offset = 0; // shouldn't happen: e* is the block max
  if (raw_offset <= kMaxOffset - 1) {
    // Normalized path: implicit leading one, quantize the fraction.
    uint32_t rounded = round_frac(fb.mantissa, width);
    if (rounded == (1u << width)) {
      // Rounds up to exactly the next power of two, i.e. the true value
      // rounds to 2^(e_i+1). Normally that is exactly representable one
      // offset step closer to e* (bump exponent by one == decrement the
      // offset by one, mantissa resets to 0). The single exception is an
      // element that was *already* at offset 0 (e_i == e*): decrementing
      // would require offset -1, which the format cannot express (e* is
      // by definition the block's largest exponent). In that boundary
      // case we clamp to the largest representable mantissa at offset 0
      // instead of wrapping to a mantissa of 0, which would silently
      // halve the value -- e.g. 0.9997 must round to ~1.999*2^e*, not to
      // 1.0*2^e*.
      if (raw_offset > 0) {
        rounded = 0;
        raw_offset -= 1;
      } else {
        rounded = (1u << width) - 1u;
      }
    }
    p.offset = static_cast<uint32_t>(raw_offset);
    p.mantissa = rounded;
  } else {
    // Denormal path (t == 7): mirrors IEEE-754 subnormals. The smallest
    // *normalized* bucket is t == kMaxOffset-1 == 6, whose minimum
    // representable magnitude is 1.0 * 2^(e*-6). Denormals share that
    // same reference exponent (e*-6) but store an unnormalized fraction
    // directly (no implicit leading one) instead of 1.fraction, giving a
    // seamless, gradually-underflowing continuation down toward zero --
    // exactly as IEEE-754 subnormals share the minimum normal exponent.
    constexpr int32_t kDenormalRefShift = kMaxOffset - 1; // 6
    int shift_extra = raw_offset - kDenormalRefShift; // >= 1 here
    // Full-precision significand (with implicit 1) in Q23, then shifted
    // right by shift_extra to rebase it from exponent e_i onto the
    // denormal reference exponent (e* - 6).
    uint64_t full_q23 = (uint64_t(1) << 23) | fb.mantissa;
    uint64_t shifted = shift_extra >= 63 ? 0 : (full_q23 >> shift_extra);
    if (shifted > 0xFFFFFFFFu) shifted = 0xFFFFFFFFu;
    uint32_t mant = round_q23(static_cast<uint32_t>(shifted), width);
    p.offset = kMaxOffset;
    p.mantissa = mant;
  }
  return p;
}

} // namespace detail

// Encode 16 floats (in[0..15]) into a packed AFPBlock.
inline AFPBlock encode_block(const float* in, EncodeOptions opts, EncodeStats* stats = nullptr) {
  AFPBlock block{};

  FloatBits fb[kBlockSize];
  int32_t e_star = -1000000000;
  bool any_nonzero = false;
  for (int i = 0; i < kBlockSize; ++i) {
    fb[i] = decompose_float(in[i]);
    if (!fb[i].is_zero) {
      any_nonzero = true;
      if (fb[i].exponent > e_star) e_star = fb[i].exponent;
    }
  }
  if (!any_nonzero) e_star = 0;

  // -- Determine positive flags per half (Section 3.3.1) -------------------
  bool positive_half[2] = {true, true};
  for (int h = 0; h < 2; ++h) {
    for (int k = 0; k < 8; ++k) {
      int i = h * 8 + k;
      if (!fb[i].is_zero && fb[i].sign != 0) { positive_half[h] = false; break; }
    }
    if (!opts.enable_positive_field) positive_half[h] = false;
  }

  // -- Baseline (5-bit) plan for every element, used to test zero-field --
  detail::PerElementPlan base_plan[kBlockSize];
  for (int i = 0; i < kBlockSize; ++i)
    base_plan[i] = detail::plan_element(fb[i], e_star, kBaseMantissaBits);

  // -- Determine zero flags per (group in {0,1}) x (half) (Section 3.3.2) --
  // A group is eligible if, for every element in that half with offset==
  // group, the top fractional bit of the *6*-bit quantization (the bit
  // that would be dropped) is zero. We test using a 6-bit quantization
  // (the width we'd store if the flag is granted) computed straight from
  // the original float, not from the baseline 5-bit plan, so the decision
  // is made on the true data rather than an already-rounded value.
  bool zero_flag[2][2] = {{true, true}, {true, true}}; // [group][half]
  int members[2][2] = {{0, 0}, {0, 0}};
  for (int h = 0; h < 2; ++h) {
    if (positive_half[h]) { zero_flag[0][h] = zero_flag[1][h] = false; continue; }
    for (int k = 0; k < 8; ++k) {
      int i = h * 8 + k;
      if (fb[i].is_zero) continue;
      auto p6 = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
      for (int g = 0; g < 2; ++g) {
        if (static_cast<int>(p6.offset) == g) {
          members[g][h]++;
          uint32_t top_bit = (p6.mantissa >> kBaseMantissaBits) & 1u;
          if (top_bit != 0) zero_flag[g][h] = false;
        }
      }
    }
    for (int g = 0; g < 2; ++g)
      if (members[g][h] == 0) zero_flag[g][h] = false; // no members -> no benefit
    if (!opts.enable_zero_field) { zero_flag[0][h] = zero_flag[1][h] = false; }
  }

  // -- Write shared header --------------------------------------------------
  int32_t biased = e_star + kExponentBias;
  if (biased < 1) biased = 1;
  if (biased > 254) biased = 254;
  block.set_exponent_byte(static_cast<uint8_t>(biased));
  uint8_t charac = 0;
  for (int h = 0; h < 2; ++h) if (positive_half[h]) charac |= (1u << h);
  for (int g = 0; g < 2; ++g)
    for (int h = 0; h < 2; ++h)
      if (zero_flag[g][h]) charac |= (1u << (2 + g * 2 + h));
  block.set_characterization(charac);

  // -- Write private fields --------------------------------------------------
  BitWriter bw(block.bytes.data());
  bw.write(block.exponent_byte(), 8);
  bw.write(charac, 8);

  for (int i = 0; i < kBlockSize; ++i) {
    int h = i / 8;
    bool use_positive = positive_half[h];
    detail::PerElementPlan pe;
    if (use_positive) {
      pe = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
      bw.write(pe.offset, kOffsetBits);
      bw.write(pe.mantissa, kBaseMantissaBits + 1); // 6 bits, no sign stored
      if (!pe.is_zero && stats) stats->positive_bonus_bits_used++;
    } else {
      int g = (base_plan[i].offset == 0) ? 0 : (base_plan[i].offset == 1 ? 1 : -1);
      bool use_zero = (g >= 0) && zero_flag[g][h] && !base_plan[i].is_zero;
      if (use_zero) {
        pe = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
        // Top fractional bit is guaranteed 0 by the zero-field precondition;
        // store only the lower kBaseMantissaBits bits.
        uint32_t low_bits = pe.mantissa & ((1u << kBaseMantissaBits) - 1u);
        bw.write(pe.sign, 1);
        bw.write(pe.offset, kOffsetBits);
        bw.write(low_bits, kBaseMantissaBits);
        if (stats) stats->zero_bonus_bits_used++;
      } else {
        pe = base_plan[i];
        bw.write(pe.sign, 1);
        bw.write(pe.offset, kOffsetBits);
        bw.write(pe.mantissa, kBaseMantissaBits);
      }
    }
    if (stats) {
      stats->values++;
      if (pe.is_zero) stats->values_truncated_to_zero++;
      if (pe.offset == kMaxOffset && !pe.is_zero) stats->values_offset_exceeds_7++;
    }
  }

  if (stats) {
    stats->blocks++;
    stats->half_blocks_total += 2;
    stats->half_blocks_all_positive += (positive_half[0] ? 1 : 0) + (positive_half[1] ? 1 : 0);
  }

  return block;
}

// Decode a packed AFPBlock back into 16 floats.
inline void decode_block(const AFPBlock& block, float* out) {
  BitReader br(block.bytes.data());
  br.read(8); // exponent (already have via block.shared_exponent())
  uint8_t charac = static_cast<uint8_t>(br.read(8));
  int32_t e_star = block.shared_exponent();
  bool positive_half[2] = { (charac & 1u) != 0, (charac & 2u) != 0 };
  bool zero_flag[2][2] = {
      { (charac & (1u << 2)) != 0, (charac & (1u << 3)) != 0 },
      { (charac & (1u << 4)) != 0, (charac & (1u << 5)) != 0 },
  };

  for (int i = 0; i < kBlockSize; ++i) {
    int h = i / 8;
    AFPScalar sc{};
    if (positive_half[h]) {
      uint32_t offset = br.read(kOffsetBits);
      uint32_t mant6 = br.read(kBaseMantissaBits + 1);
      bool denormal = (offset == static_cast<uint32_t>(kMaxOffset));
      sc.sign = 1;
      sc.sig_bits = kBaseMantissaBits + 1;
      sc.exponent = e_star - (denormal ? (kMaxOffset - 1) : static_cast<int32_t>(offset));
      sc.significand = denormal ? static_cast<int32_t>(mant6)
                                 : static_cast<int32_t>((1u << sc.sig_bits) | mant6);
      sc.is_zero = (mant6 == 0) && denormal;
    } else {
      uint32_t sign = br.read(1);
      uint32_t offset = br.read(kOffsetBits);
      uint32_t stored = br.read(kBaseMantissaBits);
      bool denormal = (offset == static_cast<uint32_t>(kMaxOffset));
      int g = (offset == 0) ? 0 : (offset == 1 ? 1 : -1);
      bool zf = (g >= 0) && zero_flag[g][h];
      sc.sign = sign ? -1 : 1;
      sc.exponent = e_star - (denormal ? (kMaxOffset - 1) : static_cast<int32_t>(offset));
      if (zf) {
        // Stored bits are the *lower* kBaseMantissaBits of a 6-bit fraction
        // whose top bit is known to be zero.
        sc.sig_bits = kBaseMantissaBits + 1;
        uint32_t mant6 = stored; // top bit (value 0) implicitly restored
        sc.significand = denormal ? static_cast<int32_t>(mant6)
                                   : static_cast<int32_t>((1u << sc.sig_bits) | mant6);
        sc.is_zero = (mant6 == 0) && denormal;
      } else {
        sc.sig_bits = kBaseMantissaBits;
        sc.significand = denormal ? static_cast<int32_t>(stored)
                                   : static_cast<int32_t>((1u << sc.sig_bits) | stored);
        sc.is_zero = (stored == 0) && denormal;
      }
    }
    out[i] = afp_scalar_to_float(sc);
  }
}

// Convenience: decode straight into AFPScalar (fixed-point) form for
// AFP-native math kernels that want to avoid floats entirely (afp_ops.hpp).
inline void decode_block_scalars(const AFPBlock& block, AFPScalar* out) {
  BitReader br(block.bytes.data());
  br.read(8);
  uint8_t charac = static_cast<uint8_t>(br.read(8));
  int32_t e_star = block.shared_exponent();
  bool positive_half[2] = { (charac & 1u) != 0, (charac & 2u) != 0 };
  bool zero_flag[2][2] = {
      { (charac & (1u << 2)) != 0, (charac & (1u << 3)) != 0 },
      { (charac & (1u << 4)) != 0, (charac & (1u << 5)) != 0 },
  };
  for (int i = 0; i < kBlockSize; ++i) {
    int h = i / 8;
    AFPScalar& sc = out[i];
    if (positive_half[h]) {
      uint32_t offset = br.read(kOffsetBits);
      uint32_t mant6 = br.read(kBaseMantissaBits + 1);
      bool denormal = (offset == static_cast<uint32_t>(kMaxOffset));
      sc.sign = 1;
      sc.sig_bits = kBaseMantissaBits + 1;
      sc.exponent = e_star - (denormal ? (kMaxOffset - 1) : static_cast<int32_t>(offset));
      sc.significand = denormal ? static_cast<int32_t>(mant6)
                                 : static_cast<int32_t>((1u << sc.sig_bits) | mant6);
      sc.is_zero = (mant6 == 0) && denormal;
    } else {
      uint32_t sign = br.read(1);
      uint32_t offset = br.read(kOffsetBits);
      uint32_t stored = br.read(kBaseMantissaBits);
      bool denormal = (offset == static_cast<uint32_t>(kMaxOffset));
      int g = (offset == 0) ? 0 : (offset == 1 ? 1 : -1);
      bool zf = (g >= 0) && zero_flag[g][h];
      sc.sign = sign ? -1 : 1;
      sc.exponent = e_star - (denormal ? (kMaxOffset - 1) : static_cast<int32_t>(offset));
      if (zf) {
        sc.sig_bits = kBaseMantissaBits + 1;
        sc.significand = denormal ? static_cast<int32_t>(stored)
                                   : static_cast<int32_t>((1u << sc.sig_bits) | stored);
        sc.is_zero = (stored == 0) && denormal;
      } else {
        sc.sig_bits = kBaseMantissaBits;
        sc.significand = denormal ? static_cast<int32_t>(stored)
                                   : static_cast<int32_t>((1u << sc.sig_bits) | stored);
        sc.is_zero = (stored == 0) && denormal;
      }
    }
  }
}

} // namespace afp
