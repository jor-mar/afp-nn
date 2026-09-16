// gen_testvectors.cpp
//
// Produces the hex test-vector files tb_afp_vs_fp32.v loads via
// $readmemh, plus the expected (software, float64) dot-product result,
// for a random NUM_BLOCKS*16-element vector pair.
//
// Bit-ordering convention (must match afp_element_decode.v /
// afp_block_dot.v exactly): within a block's 144-bit fields_i bus,
// element i's 9-bit field occupies bits [9*i+8 : 9*i] -- i.e. element 0
// in the *least* significant 9 bits, element 15 in the most significant.
// This generator builds that same layout explicitly (independent of
// afp_codec.hpp's own MSB-first serial byte-packing, which is an
// unrelated storage-format detail) so there is exactly one place
// (this comment, mirrored in afp_block_dot.v) where the convention is
// defined.
//
// The RTL models the *baseline* AFP8 layout (no positive-field/zero-field
// bonus bits -- see cpp/rtl/README.md), so vectors are encoded with both
// optimizations disabled to keep the hardware and the test vectors
// consistent.
//
// Usage:
//   g++ -std=c++20 -O2 -I../include gen_testvectors.cpp -o gen_testvectors
//   ./gen_testvectors [num_blocks] [seed]
//   (writes ./testvectors/*.hex and ./testvectors/expected_dot.txt)

#include "afp/afp_core.hpp"
#include "afp/afp_codec.hpp"
#include "afp/afp_tensor.hpp"

#include <bitset>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string nibble_hex(const std::bitset<144>& bits) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (int k = 35; k >= 0; --k) {
    int base = k * 4;
    int val = (bits[base + 3] << 3) | (bits[base + 2] << 2) | (bits[base + 1] << 1) | bits[base];
    out += digits[val];
  }
  return out;
}

// Extract element i's raw 9-bit private field (sign,offset,mantissa) from
// an AFPBlock encoded with both bonus-bit optimizations disabled (so the
// layout is exactly 1+3+5 bits per element, no reinterpretation needed).
static uint32_t extract_field(const afp::AFPBlock& block, int i) {
  afp::BitReader br(block.bytes.data());
  br.read(afp::kSharedHeaderBits);
  for (int k = 0; k < i; ++k) br.read(afp::kPrivateBits);
  return br.read(afp::kPrivateBits);
}

int main(int argc, char** argv) {
  int num_blocks = argc > 1 ? std::atoi(argv[1]) : 8;
  unsigned seed = argc > 2 ? static_cast<unsigned>(std::atol(argv[2])) : 42;
  int n = num_blocks * afp::kBlockSize;

  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> a(n), b(n);
  for (auto& v : a) v = dist(rng);
  for (auto& v : b) v = dist(rng);

  afp::EncodeOptions baseline; // both bonus-bit optimizations off, matching the RTL
  baseline.enable_positive_field = false;
  baseline.enable_zero_field = false;

  fs::create_directories("testvectors");
  std::ofstream fa_fields("testvectors/afp_a_fields.hex");
  std::ofstream fb_fields("testvectors/afp_b_fields.hex");
  std::ofstream fa_exp("testvectors/afp_a_exp.hex");
  std::ofstream fb_exp("testvectors/afp_b_exp.hex");
  std::ofstream fa32("testvectors/fp32_a.hex");
  std::ofstream fb32("testvectors/fp32_b.hex");

  for (int blk = 0; blk < num_blocks; ++blk) {
    afp::AFPBlock ba = afp::encode_block(&a[blk * afp::kBlockSize], baseline);
    afp::AFPBlock bb = afp::encode_block(&b[blk * afp::kBlockSize], baseline);

    std::bitset<144> bits_a, bits_b;
    for (int i = 0; i < afp::kBlockSize; ++i) {
      uint32_t fa = extract_field(ba, i);
      uint32_t fb = extract_field(bb, i);
      for (int bit = 0; bit < 9; ++bit) {
        bits_a[i * 9 + bit] = (fa >> bit) & 1u;
        bits_b[i * 9 + bit] = (fb >> bit) & 1u;
      }
    }
    fa_fields << nibble_hex(bits_a) << "\n";
    fb_fields << nibble_hex(bits_b) << "\n";
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02x", ba.exponent_byte());
    fa_exp << buf << "\n";
    std::snprintf(buf, sizeof(buf), "%02x", bb.exponent_byte());
    fb_exp << buf << "\n";
  }

  for (int i = 0; i < n; ++i) {
    uint32_t bits_a = afp::float_to_u32(a[i]);
    uint32_t bits_b = afp::float_to_u32(b[i]);
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", bits_a); fa32 << buf << "\n";
    std::snprintf(buf, sizeof(buf), "%08x", bits_b); fb32 << buf << "\n";
  }

  // Expected result: decode the *baseline*-encoded (RTL-equivalent)
  // vectors back to float and compute the reference dot product in
  // double precision, so "expected" reflects what the RTL *should*
  // produce given AFP's quantization (not the unquantized FP32 dot
  // product, which would differ by the normal AFP quantization error).
  std::vector<float> a_dec(n), b_dec(n);
  {
    afp::AFPTensor ta = afp::AFPTensor::encode(a.data(), n, baseline);
    afp::AFPTensor tb = afp::AFPTensor::encode(b.data(), n, baseline);
    ta.decode(a_dec.data());
    tb.decode(b_dec.data());
  }
  double expected = 0.0;
  for (int i = 0; i < n; ++i) expected += double(a_dec[i]) * double(b_dec[i]);

  std::ofstream fexp("testvectors/expected_dot.txt");
  fexp.precision(10);
  fexp << expected << "\n";

  std::printf("Wrote test vectors for %d blocks (%d elements) to testvectors/\n", num_blocks, n);
  std::printf("Expected AFP-quantized dot product: %.6f\n", expected);
  double true_dot = 0;
  for (int i = 0; i < n; ++i) true_dot += double(a[i]) * double(b[i]);
  std::printf("(unquantized FP32 dot product, for reference: %.6f)\n", true_dot);
  return 0;
}
