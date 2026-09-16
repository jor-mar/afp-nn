// afp_tensor.hpp
//
// AFPTensor: stores a flat array of values as a sequence of 16-element AFP
// blocks (Section 4 of the paper: "we flatten and reshape the tensors into
// blocks of 16 elements ... For tensors that do not divide evenly into
// 16-element blocks, we pad the end of the tensor with zeros and remove
// these padded zeros at the end of rounding").

#pragma once

#include "afp_core.hpp"
#include "afp_codec.hpp"

#include <vector>
#include <cstddef>

namespace afp {

class AFPTensor {
 public:
  AFPTensor() = default;

  static AFPTensor encode(const float* data, size_t n, EncodeOptions opts = {},
                           EncodeStats* stats = nullptr) {
    AFPTensor t;
    t.n_ = n;
    size_t n_blocks = (n + kBlockSize - 1) / kBlockSize;
    t.blocks_.resize(n_blocks);
    std::vector<float> padded(n_blocks * kBlockSize, 0.0f);
    std::copy(data, data + n, padded.begin());
    for (size_t b = 0; b < n_blocks; ++b) {
      t.blocks_[b] = encode_block(&padded[b * kBlockSize], opts, stats);
    }
    return t;
  }

  void decode(float* out) const {
    std::vector<float> tmp(kBlockSize);
    for (size_t b = 0; b < blocks_.size(); ++b) {
      decode_block(blocks_[b], tmp.data());
      size_t base = b * kBlockSize;
      size_t count = std::min<size_t>(kBlockSize, n_ - base);
      for (size_t i = 0; i < count; ++i) out[base + i] = tmp[i];
    }
  }

  std::vector<float> decode() const {
    std::vector<float> out(n_);
    decode(out.data());
    return out;
  }

  size_t size() const { return n_; }
  size_t num_blocks() const { return blocks_.size(); }
  const std::vector<AFPBlock>& blocks() const { return blocks_; }
  std::vector<AFPBlock>& blocks() { return blocks_; }

  // Total packed storage in bytes (paper: 20 bytes / 16-element block =
  // 10 bits/value average).
  size_t packed_bytes() const { return blocks_.size() * sizeof(AFPBlock); }

  // What the same logical tensor would cost in plain FP32.
  size_t fp32_bytes() const { return n_ * sizeof(float); }

  double compression_ratio() const {
    return packed_bytes() == 0 ? 0.0 : double(fp32_bytes()) / double(packed_bytes());
  }

 private:
  size_t n_ = 0;
  std::vector<AFPBlock> blocks_;
};

} // namespace afp
