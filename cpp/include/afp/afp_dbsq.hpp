// afp_dbsq.hpp
//
// Dynamic Block-Size Quantization (DBSQ) for AFP, inspired by:
//   Xiao Ju, Jianfei Yang, Mingyu Wen, Junyi He, Jiayi Feng, Ming Tang,
//   Zhaolin Chen, Yiran Shi. "SmartBlock: Adaptive Block Floating Point
//   Quantization for Efficient DNN Acceleration." ICPP 2025.
//   https://dl.acm.org/doi/full/10.1145/3754598.3754660
//
// The ACM paper page is largely paywalled beyond the abstract/intro, so
// unlike afp_core.hpp/afp_codec.hpp (which follow "Be Like Water"'s fully
// public bit layout exactly), this file implements a documented,
// good-faith *interpretation* of DBSQ's stated design goal, built on top
// of the paper-faithful AFP element codec:
//
//   "We assign larger block sizes (e.g. 256 or 512) to regions without
//    outliers to minimize hardware overhead, and smaller block sizes
//    (e.g. 8) around outliers to reduce quantization error... the first
//    work to incorporate dynamic block sizing into BFP-based
//    quantization," with "a hardware-friendly block size parameter
//    encoding scheme."
//
// Design (documented, not paper-verbatim):
//
//   1. Candidate block sizes are powers of two times the AFP base group
//      of 8 elements: {8, 16, 32, 64, 128, 256, 512, 1024} -- exactly 8
//      options, so the per-block "size code" fits in 3 bits (the
//      "hardware-friendly ... encoding scheme" analog: a fixed 3-bit
//      field selects the block's element count, just like this file's
//      sibling afp_codec.hpp uses a fixed 3-bit offset field per value).
//   2. An outlier-aware greedy partitioner (plan_block_sizes) scans a
//      tensor's per-8-element exponent range and *doubles* the candidate
//      block size as long as the merged exponent range stays within a
//      configurable threshold (default: kMaxOffset, the same 7-step
//      window AFP's own offset field can represent without truncation).
//      Regions with a wide exponent spread (outliers) stop doubling
//      early and fall back toward the 8-element minimum; smooth regions
//      grow all the way to the largest candidate, amortizing the
//      per-block header over many more values.
//   3. Each variable-length block reuses the exact same per-element
//      codec (afp::detail::plan_element, sign/offset/mantissa, the
//      positive-field and zero-field bonus-bit optimizations) as the
//      fixed-16 AFP format, just generalized from "2 halves of 8" to
//      "N/8 groups of 8".
//
// This keeps DBSQ strictly additive: afp_core.hpp/afp_codec.hpp/
// afp_tensor.hpp (fixed 16-element blocks) are unchanged and still the
// paper-faithful "Be Like Water" implementation; this file is a separate,
// opt-in encoding built from the same primitives.

#pragma once

#include "afp_core.hpp"
#include "afp_codec.hpp" // for afp::detail::plan_element/round_frac/round_q23

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>

namespace afp::dbsq {

// 8 candidate sizes -> exactly fits a 3-bit "size code".
inline constexpr int kNumSizeCodes = 8;
inline constexpr int kGroupSize = 8; // AFP's base positive/zero-field granularity
inline constexpr int kSizeCodeTable[kNumSizeCodes] = {8, 16, 32, 64, 128, 256, 512, 1024};

inline int size_for_code(int code) { return kSizeCodeTable[code]; }
inline int groups_for_code(int code) { return kSizeCodeTable[code] / kGroupSize; }

struct DBSQConfig {
  // Max allowed (max_exponent - min_exponent) among a candidate block's
  // *nonzero* elements before we refuse to grow the block further.
  // kMaxOffset (7) is the natural choice: AFP's own 3-bit offset field
  // cannot represent a larger spread without truncating to zero anyway,
  // so this threshold asks "would growing the block start truncating
  // values that a smaller block could have preserved?"
  int32_t outlier_exponent_threshold = kMaxOffset;
  int min_size_code = 0;                 // 8 elements
  int max_size_code = kNumSizeCodes - 1;  // 1024 elements
};

// ============================================================================
// Variable-length block storage
// ============================================================================
struct DBSQBlock {
  std::vector<uint8_t> bytes;
  int32_t size = 0; // number of logical elements this block encodes
};

namespace detail_dbsq {

inline size_t header_bits(int num_groups) { return 8 + 3 + 3 * size_t(num_groups); }
inline size_t header_bytes(int num_groups) { return (header_bits(num_groups) + 7) / 8; }
inline size_t total_bits(int size) {
  int groups = size / kGroupSize;
  return header_bits(groups) + size_t(size) * kPrivateBits;
}
inline size_t total_bytes(int size) { return (total_bits(size) + 7) / 8; }

} // namespace detail_dbsq

// ============================================================================
// Per-block encode/decode (generalizes afp_codec.hpp's fixed 16-block logic
// to an arbitrary size that is a multiple of kGroupSize).
// ============================================================================

inline DBSQBlock encode_dbsq_block(const float* in, int size, EncodeOptions opts,
                                    EncodeStats* stats = nullptr) {
  int num_groups = size / kGroupSize;
  DBSQBlock block;
  block.size = size;
  block.bytes.assign(detail_dbsq::total_bytes(size), 0);

  std::vector<FloatBits> fb(size);
  int32_t e_star = -1000000000;
  bool any_nonzero = false;
  for (int i = 0; i < size; ++i) {
    fb[i] = decompose_float(in[i]);
    if (!fb[i].is_zero) { any_nonzero = true; e_star = std::max(e_star, fb[i].exponent); }
  }
  if (!any_nonzero) e_star = 0;

  std::vector<uint8_t> positive_group(num_groups, 1);
  for (int g = 0; g < num_groups; ++g) {
    bool pos = true;
    for (int k = 0; k < kGroupSize; ++k) {
      int i = g * kGroupSize + k;
      if (!fb[i].is_zero && fb[i].sign != 0) { pos = false; break; }
    }
    if (!opts.enable_positive_field) pos = false;
    positive_group[g] = pos ? 1 : 0;
  }

  std::vector<detail::PerElementPlan> base_plan(size);
  for (int i = 0; i < size; ++i) base_plan[i] = detail::plan_element(fb[i], e_star, kBaseMantissaBits);

  // zero_flag[group_type][g] : group_type 0 -> offset-0 group, 1 -> offset-1 group
  std::vector<std::array<uint8_t, 2>> zero_flag(num_groups, {0, 0});
  for (int g = 0; g < num_groups; ++g) {
    if (positive_group[g]) continue;
    bool ok0 = true, ok1 = true;
    int n0 = 0, n1 = 0;
    for (int k = 0; k < kGroupSize; ++k) {
      int i = g * kGroupSize + k;
      if (fb[i].is_zero) continue;
      auto p6 = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
      uint32_t top_bit = (p6.mantissa >> kBaseMantissaBits) & 1u;
      if (p6.offset == 0) { n0++; if (top_bit) ok0 = false; }
      else if (p6.offset == 1) { n1++; if (top_bit) ok1 = false; }
    }
    zero_flag[g][0] = (opts.enable_zero_field && n0 > 0 && ok0) ? 1 : 0;
    zero_flag[g][1] = (opts.enable_zero_field && n1 > 0 && ok1) ? 1 : 0;
  }

  int32_t biased = std::clamp(e_star + kExponentBias, 1, 254);
  int size_code = 0;
  for (int c = 0; c < kNumSizeCodes; ++c) if (kSizeCodeTable[c] == size) { size_code = c; break; }

  BitWriter bw(block.bytes.data());
  bw.write(static_cast<uint32_t>(biased), 8);
  bw.write(static_cast<uint32_t>(size_code), 3);
  for (int g = 0; g < num_groups; ++g) {
    uint32_t flags = (positive_group[g] << 2) | (zero_flag[g][0] << 1) | zero_flag[g][1];
    bw.write(flags, 3);
  }

  for (int i = 0; i < size; ++i) {
    int g = i / kGroupSize;
    detail::PerElementPlan pe;
    if (positive_group[g]) {
      pe = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
      bw.write(pe.offset, kOffsetBits);
      bw.write(pe.mantissa, kBaseMantissaBits + 1);
      if (!pe.is_zero && stats) stats->positive_bonus_bits_used++;
    } else {
      int gt = (base_plan[i].offset == 0) ? 0 : (base_plan[i].offset == 1 ? 1 : -1);
      bool use_zero = (gt >= 0) && zero_flag[g][gt] && !base_plan[i].is_zero;
      if (use_zero) {
        pe = detail::plan_element(fb[i], e_star, kBaseMantissaBits + 1);
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
    }
  }
  if (stats) stats->blocks++;
  return block;
}

inline void decode_dbsq_block_scalars(const DBSQBlock& block, AFPScalar* out) {
  BitReader br(block.bytes.data());
  int32_t e_star = static_cast<int32_t>(br.read(8)) - kExponentBias;
  int size_code = static_cast<int>(br.read(3));
  int size = size_for_code(size_code);
  int num_groups = size / kGroupSize;
  std::vector<uint8_t> positive_group(num_groups), zero0(num_groups), zero1(num_groups);
  for (int g = 0; g < num_groups; ++g) {
    uint32_t flags = br.read(3);
    positive_group[g] = (flags >> 2) & 1u;
    zero0[g] = (flags >> 1) & 1u;
    zero1[g] = flags & 1u;
  }
  for (int i = 0; i < size; ++i) {
    int g = i / kGroupSize;
    AFPScalar& sc = out[i];
    if (positive_group[g]) {
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
      bool zf = (offset == 0 && zero0[g]) || (offset == 1 && zero1[g]);
      sc.sign = sign ? -1 : 1;
      sc.exponent = e_star - (denormal ? (kMaxOffset - 1) : static_cast<int32_t>(offset));
      sc.sig_bits = zf ? kBaseMantissaBits + 1 : kBaseMantissaBits;
      sc.significand = denormal ? static_cast<int32_t>(stored)
                                 : static_cast<int32_t>((1u << sc.sig_bits) | stored);
      sc.is_zero = (stored == 0) && denormal;
    }
  }
}

inline void decode_dbsq_block(const DBSQBlock& block, float* out) {
  std::vector<AFPScalar> sc(block.size);
  decode_dbsq_block_scalars(block, sc.data());
  for (int i = 0; i < block.size; ++i) out[i] = afp_scalar_to_float(sc[i]);
}

// ============================================================================
// Outlier-aware block-size planning
// ============================================================================

struct GroupStat {
  int32_t min_exp = std::numeric_limits<int32_t>::max();
  int32_t max_exp = std::numeric_limits<int32_t>::min();
  bool has_data = false;
};

inline std::vector<int> plan_block_sizes(const float* data, size_t n, const DBSQConfig& cfg) {
  size_t num_groups = (n + kGroupSize - 1) / kGroupSize;
  std::vector<GroupStat> groups(num_groups);
  for (size_t g = 0; g < num_groups; ++g) {
    GroupStat s;
    for (int k = 0; k < kGroupSize; ++k) {
      size_t i = g * kGroupSize + k;
      if (i >= n) break;
      float v = data[i];
      if (v == 0.0f) continue;
      auto fb = decompose_float(v);
      if (fb.is_zero) continue;
      s.has_data = true;
      s.min_exp = std::min(s.min_exp, fb.exponent);
      s.max_exp = std::max(s.max_exp, fb.exponent);
    }
    groups[g] = s;
  }

  int max_groups = groups_for_code(cfg.max_size_code);
  int min_groups = groups_for_code(cfg.min_size_code);
  std::vector<int> block_sizes; // in elements
  size_t g = 0;
  while (g < num_groups) {
    int k = std::max(1, min_groups); // groups included so far (>= min candidate)
    if (g + size_t(k) > num_groups) k = static_cast<int>(num_groups - g); // last partial run
    GroupStat merged = groups[g];
    for (int t = 1; t < k && g + t < num_groups; ++t) {
      auto& s2 = groups[g + t];
      if (s2.has_data) {
        merged.has_data = true;
        merged.min_exp = std::min(merged.min_exp, s2.min_exp);
        merged.max_exp = std::max(merged.max_exp, s2.max_exp);
      }
    }
    // Greedily double the run length while the merged exponent range stays
    // within the outlier threshold and we haven't hit the max block size.
    while (k * 2 <= max_groups && g + size_t(k) * 2 <= num_groups) {
      GroupStat added; added.has_data = false;
      for (size_t t = g + k; t < g + size_t(k) * 2; ++t) {
        auto& s2 = groups[t];
        if (s2.has_data) {
          added.has_data = true;
          added.min_exp = std::min(added.min_exp, s2.min_exp);
          added.max_exp = std::max(added.max_exp, s2.max_exp);
        }
      }
      GroupStat cand = merged;
      if (added.has_data) {
        if (!cand.has_data) { cand = added; }
        else {
          cand.min_exp = std::min(cand.min_exp, added.min_exp);
          cand.max_exp = std::max(cand.max_exp, added.max_exp);
        }
      }
      bool within = !cand.has_data || (cand.max_exp - cand.min_exp) <= cfg.outlier_exponent_threshold;
      if (!within) break;
      merged = cand;
      k *= 2;
    }
    block_sizes.push_back(k * kGroupSize);
    g += k;
  }
  return block_sizes;
}

// ============================================================================
// DBSQTensor: a flat array encoded as a sequence of variable-length blocks.
// ============================================================================
class DBSQTensor {
 public:
  static DBSQTensor encode(const float* data, size_t n, const DBSQConfig& cfg = {},
                            EncodeOptions opts = {}, EncodeStats* stats = nullptr) {
    DBSQTensor t;
    t.n_ = n;
    t.block_sizes_ = plan_block_sizes(data, n, cfg);
    std::vector<float> padded;
    size_t total = 0;
    for (int s : t.block_sizes_) total += s;
    padded.assign(total, 0.0f);
    std::copy(data, data + n, padded.begin());
    size_t off = 0;
    t.blocks_.reserve(t.block_sizes_.size());
    for (int s : t.block_sizes_) {
      t.blocks_.push_back(encode_dbsq_block(&padded[off], s, opts, stats));
      off += s;
    }
    return t;
  }

  void decode(float* out) const {
    size_t off = 0;
    std::vector<float> tmp;
    for (auto& b : blocks_) {
      tmp.resize(b.size);
      decode_dbsq_block(b, tmp.data());
      size_t count = std::min<size_t>(b.size, n_ - off);
      std::copy(tmp.begin(), tmp.begin() + count, out + off);
      off += count;
      if (off >= n_) break;
    }
  }

  // Flat per-element decode used by the dot product below: unlike the
  // fixed-block AFPTensor, two DBSQTensors are not guaranteed to share
  // block boundaries (a static weight row and a dynamic activation vector
  // are planned independently), so the native dot product operates
  // per-element rather than per-block-pair.
  std::vector<AFPScalar> decode_scalars_flat() const {
    std::vector<AFPScalar> out(n_);
    size_t off = 0;
    std::vector<AFPScalar> tmp;
    for (auto& b : blocks_) {
      tmp.resize(b.size);
      decode_dbsq_block_scalars(b, tmp.data());
      size_t count = std::min<size_t>(b.size, n_ - off);
      std::copy(tmp.begin(), tmp.begin() + count, out.begin() + off);
      off += count;
      if (off >= n_) break;
    }
    return out;
  }

  size_t size() const { return n_; }
  const std::vector<DBSQBlock>& blocks() const { return blocks_; }
  const std::vector<int>& block_sizes() const { return block_sizes_; }

  size_t packed_bytes() const {
    size_t total = 0;
    for (auto& b : blocks_) total += b.bytes.size();
    return total;
  }
  size_t fp32_bytes() const { return n_ * sizeof(float); }
  double compression_ratio() const {
    return packed_bytes() == 0 ? 0.0 : double(fp32_bytes()) / double(packed_bytes());
  }
  // Fraction of blocks that landed at each candidate size -- useful to
  // sanity-check that DBSQ is actually adapting (e.g. mostly-uniform
  // tensors should skew toward large blocks; outlier-heavy ones toward 8).
  std::array<size_t, kNumSizeCodes> size_histogram() const {
    std::array<size_t, kNumSizeCodes> hist{};
    for (int s : block_sizes_)
      for (int c = 0; c < kNumSizeCodes; ++c) if (kSizeCodeTable[c] == s) { hist[c]++; break; }
    return hist;
  }

 private:
  size_t n_ = 0;
  std::vector<int> block_sizes_;
  std::vector<DBSQBlock> blocks_;
};

// Native dot product across two (independently block-planned) DBSQTensors:
// same integer significand * significand + shift-accumulate arithmetic as
// afp::dot_product_native, just indexed per element instead of per aligned
// block pair (see decode_scalars_flat() note above).
inline double dot_product_native(const DBSQTensor& a, const DBSQTensor& b) {
  auto sa = a.decode_scalars_flat();
  auto sb = b.decode_scalars_flat();
  size_t n = std::min(sa.size(), sb.size());
  double total = 0.0;
  // Process in chunks so we still get the fixed-point-accumulate-then-
  // ldexp-once benefit afp::dot_product_native uses, rather than an ldexp
  // per element.
  const size_t kChunk = 64;
  for (size_t base = 0; base < n; base += kChunk) {
    size_t end = std::min(n, base + kChunk);
    int32_t min_v_exp = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> v_exp(end - base);
    bool any = false;
    for (size_t i = base; i < end; ++i) {
      if (sa[i].is_zero || sb[i].is_zero) continue;
      int32_t ve = sa[i].exponent + sb[i].exponent - sa[i].sig_bits - sb[i].sig_bits;
      v_exp[i - base] = ve;
      min_v_exp = std::min(min_v_exp, ve);
      any = true;
    }
    if (!any) continue;
    int64_t fixed_sum = 0;
    for (size_t i = base; i < end; ++i) {
      if (sa[i].is_zero || sb[i].is_zero) continue;
      int32_t shift = v_exp[i - base] - min_v_exp;
      if (shift > 40) continue;
      int64_t prod = int64_t(sa[i].significand) * int64_t(sb[i].significand) * (sa[i].sign * sb[i].sign);
      fixed_sum += prod << shift;
    }
    total += std::ldexp(double(fixed_sum), min_v_exp);
  }
  return total;
}

} // namespace afp::dbsq
