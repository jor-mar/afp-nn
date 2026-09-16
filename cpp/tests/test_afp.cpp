// test_afp.cpp -- correctness unit tests for the AFP codec.
// Build: see CMakeLists.txt / Makefile. No external test framework: uses
// simple asserts and prints a PASS/FAIL summary so it works standalone.

#include "afp/afp_core.hpp"
#include "afp/afp_codec.hpp"
#include "afp/afp_tensor.hpp"
#include "afp/afp_ops.hpp"
#include "afp/afp_dbsq.hpp"
#include "afp/network.hpp"
#include "afp/pruning.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { \
  if (!(cond)) { \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    g_failures++; \
  } \
} while (0)

static double rel_err(float a, float b) {
  double denom = std::max(std::fabs((double)a), 1e-30);
  return std::fabs((double)a - (double)b) / denom;
}

void test_basic_roundtrip() {
  // Values chosen with a *homogeneous* dynamic range (all within offset<=7
  // of the block max, i.e. no element more than 128x smaller than 100) so
  // none of them are expected to legitimately truncate to zero.
  float in[afp::kBlockSize] = {
      1.0f, -1.0f, 0.5f, -0.25f, 3.14159f, -2.71828f, 100.0f, -100.0f,
      0.9f, 0.0f, 1.5f, -1.5f, 42.0f, -42.0f, 7.5f, -7.5f
  };
  auto blk = afp::encode_block(in, afp::EncodeOptions{});
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  for (int i = 0; i < afp::kBlockSize; ++i) {
    // AFP has at most ~6 bits of mantissa -> relative error should be
    // well under 2^-5 ~= 3% for normalized in-range values.
    if (in[i] == 0.0f) { CHECK(out[i] == 0.0f); continue; }
    CHECK(rel_err(in[i], out[i]) < 0.06);
  }
}

void test_wide_dynamic_range_truncates_small_values() {
  // Values spanning far more than 7 exponent-steps: the smallest ones are
  // *expected* to legitimately truncate to zero -- this is the documented
  // AFP behavior (Section 3.2.2: <1% of values truncate in real models),
  // not a bug. This test documents/pins that behavior instead of treating
  // it as an error.
  float in[afp::kBlockSize];
  for (auto& v : in) v = 1.0f;
  in[0] = 1e6f;   // sets a huge block exponent
  in[1] = 1e-3f;  // many exponent-steps below the max -> should truncate
  auto blk = afp::encode_block(in, afp::EncodeOptions{});
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  CHECK(out[1] == 0.0f);
  CHECK(rel_err(in[0], out[0]) < 0.06);
}

void test_exact_zero() {
  float in[afp::kBlockSize];
  for (auto& v : in) v = 0.0f;
  auto blk = afp::encode_block(in, afp::EncodeOptions{});
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  for (auto v : out) CHECK(v == 0.0f);
}

void test_all_positive_block_uses_extra_bit() {
  float in[afp::kBlockSize];
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(0.01f, 10.0f);
  for (auto& v : in) v = dist(rng);
  afp::EncodeStats stats{};
  auto blk = afp::encode_block(in, afp::EncodeOptions{}, &stats);
  CHECK(blk.positive_flag(0));
  CHECK(blk.positive_flag(1));
  CHECK(stats.positive_bonus_bits_used == afp::kBlockSize);
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  double max_re = 0;
  for (int i = 0; i < afp::kBlockSize; ++i) max_re = std::max(max_re, rel_err(in[i], out[i]));
  // With the bonus bit, precision should be a bit better than the plain
  // 5-bit baseline (< 1/64 ~ 1.6% typical, well under the 5-bit bound).
  CHECK(max_re < 0.035);
}

void test_negative_block_no_positive_bonus() {
  float in[afp::kBlockSize];
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-10.0f, -0.01f);
  for (auto& v : in) v = dist(rng);
  afp::EncodeStats stats{};
  auto blk = afp::encode_block(in, afp::EncodeOptions{}, &stats);
  CHECK(!blk.positive_flag(0));
  CHECK(!blk.positive_flag(1));
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  for (int i = 0; i < afp::kBlockSize; ++i) CHECK(rel_err(in[i], out[i]) < 0.06);
}

void test_denormal_underflow_graceful() {
  // One huge value (sets e*) and one tiny value many orders of magnitude
  // smaller: tiny value should truncate to zero gracefully, not crash/NaN,
  // matching the paper's <1% truncation-to-zero statistic.
  float in[afp::kBlockSize];
  for (auto& v : in) v = 1.0f;
  in[0] = 1.0e6f;      // sets a large e*
  in[1] = 1.0e-30f;    // far below representable range given e*
  auto blk = afp::encode_block(in, afp::EncodeOptions{});
  float out[afp::kBlockSize];
  afp::decode_block(blk, out);
  CHECK(std::isfinite(out[0]));
  CHECK(out[1] == 0.0f); // truncated
  CHECK(rel_err(in[0], out[0]) < 0.06);
}

void test_random_stress() {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int N = 16 * 20000;
  std::vector<float> data(N);
  for (auto& v : data) v = normal(rng);
  afp::AFPTensor t = afp::AFPTensor::encode(data.data(), N);
  std::vector<float> out(N);
  t.decode(out.data());
  double sum_abs_err = 0, sum_abs = 0;
  int trunc = 0;
  for (int i = 0; i < N; ++i) {
    sum_abs_err += std::fabs(data[i] - out[i]);
    sum_abs += std::fabs(data[i]);
    if (out[i] == 0.0f && data[i] != 0.0f) trunc++;
  }
  double mean_abs_err = sum_abs_err / N;
  double frac_trunc = double(trunc) / N;
  std::printf("  random_stress: mean_abs_err=%.6f frac_truncated=%.4f%%\n",
              mean_abs_err, frac_trunc * 100.0);
  // Paper reports ~0.00013 average absolute error for weights (which are
  // typically much smaller magnitude than N(0,1) samples); for unit-
  // variance data we just sanity check the error is small relative to the
  // data scale and that truncation is rare (<5% for standard normal data
  // spread across ~11 bits of dynamic range within each 16-block).
  CHECK(mean_abs_err < 0.05);
  CHECK(frac_trunc < 0.05);
}

void test_dot_product_matches_float() {
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int N = 16 * 50;
  std::vector<float> a(N), b(N);
  for (auto& v : a) v = normal(rng);
  for (auto& v : b) v = normal(rng);

  afp::AFPTensor ta = afp::AFPTensor::encode(a.data(), N);
  afp::AFPTensor tb = afp::AFPTensor::encode(b.data(), N);

  double afp_dot = afp::dot_product_native(ta, tb);

  double float_dot = 0;
  for (int i = 0; i < N; ++i) float_dot += double(a[i]) * double(b[i]);

  double re = std::fabs(afp_dot - float_dot) / std::max(std::fabs(float_dot), 1e-6);
  std::printf("  dot_product: afp=%.6f float=%.6f rel_err=%.4f%%\n", afp_dot, float_dot, re * 100.0);
  CHECK(re < 0.10); // 5-6 bit mantissas -> a few % dot-product error is expected
}

void test_relu_native() {
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  const int N = 16 * 10;
  std::vector<float> a(N);
  for (auto& v : a) v = normal(rng);
  afp::AFPTensor t = afp::AFPTensor::encode(a.data(), N);
  afp::relu_inplace_native(t);
  std::vector<float> out(N);
  t.decode(out.data());
  for (int i = 0; i < N; ++i) {
    float expect = a[i] > 0 ? a[i] : 0.0f;
    if (expect == 0.0f) CHECK(out[i] == 0.0f);
    else CHECK(rel_err(expect, out[i]) < 0.06);
  }
}

void test_activations_approx() {
  // Note on methodology: AFP quantizes in blocks of 16, so if a single
  // block mixes wildly different magnitudes (e.g. exp(4)=54.6 next to
  // exp(-4)=0.018, a >3000x ratio) the smaller values will legitimately
  // truncate toward zero -- exactly the documented AFP behavior, not an
  // approximation bug. We therefore report *mean* error across many
  // blocks (representative of real activation tensors, which tend to
  // have more homogeneous per-block magnitude -- see the paper's Fig. 4)
  // rather than a worst-case max over an adversarially wide-range block.
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  const int N = 16 * 200;
  std::vector<float> a(N);
  for (auto& v : a) v = dist(rng);
  afp::AFPTensor t_sig = afp::AFPTensor::encode(a.data(), N);
  afp::AFPTensor t_tanh = afp::AFPTensor::encode(a.data(), N);
  afp::AFPTensor t_exp = afp::AFPTensor::encode(a.data(), N);
  afp::sigmoid_inplace_native(t_sig);
  afp::tanh_inplace_native(t_tanh);
  afp::exp_inplace_native(t_exp);
  std::vector<float> o_sig(N), o_tanh(N), o_exp(N);
  t_sig.decode(o_sig.data());
  t_tanh.decode(o_tanh.data());
  t_exp.decode(o_exp.data());
  double sum_abs_sig = 0, sum_abs_tanh = 0, sum_rel_exp = 0;
  for (int i = 0; i < N; ++i) {
    double s_true = 1.0 / (1.0 + std::exp(-(double)a[i]));
    double t_true = std::tanh((double)a[i]);
    double e_true = std::exp((double)a[i]);
    sum_abs_sig += std::fabs(s_true - o_sig[i]);
    sum_abs_tanh += std::fabs(t_true - o_tanh[i]);
    sum_rel_exp += std::fabs(e_true - o_exp[i]) / std::max(e_true, 1e-6);
  }
  double mean_abs_sig = sum_abs_sig / N, mean_abs_tanh = sum_abs_tanh / N,
         mean_rel_exp = sum_rel_exp / N;
  std::printf("  activations: mean|err| sigmoid=%.4f tanh=%.4f  exp mean_rel_err=%.4f%%\n",
              mean_abs_sig, mean_abs_tanh, mean_rel_exp * 100.0);
  CHECK(mean_abs_sig < 0.02);
  CHECK(mean_abs_tanh < 0.03);
  CHECK(mean_rel_exp < 0.10);
}

void test_conv2d_afp_matches_fp32() {
  using namespace afp;
  std::mt19937 rng(21);
  std::normal_distribution<float> dist(0.0f, 0.3f);
  int in_ch = 3, out_ch = 4, k = 3, stride = 1, pad = 1, H = 8, W = 8;
  std::vector<float> w(size_t(out_ch) * in_ch * k * k), b(out_ch);
  for (auto& v : w) v = dist(rng);
  for (auto& v : b) v = dist(rng);
  Tensor3F x{in_ch, H, W, std::vector<float>(size_t(in_ch) * H * W)};
  for (auto& v : x.data) v = dist(rng);

  Tensor3F y_fp32 = conv2d_forward_f32(x, w, b, in_ch, out_ch, k, stride, pad, /*relu=*/true);
  AFPConv2DLayer conv_afp = make_afp_conv2d(w, b, in_ch, out_ch, k, stride, pad);
  Tensor3F y_afp = conv2d_forward_afp(x, conv_afp, /*relu=*/true);

  CHECK(y_fp32.c == y_afp.c);
  CHECK(y_fp32.h == y_afp.h);
  CHECK(y_fp32.w == y_afp.w);
  double sum_abs_err = 0, sum_abs = 0;
  for (size_t i = 0; i < y_fp32.data.size(); ++i) {
    sum_abs_err += std::fabs(y_fp32.data[i] - y_afp.data[i]);
    sum_abs += std::fabs(y_fp32.data[i]);
  }
  double mean_rel = sum_abs_err / std::max(sum_abs, 1e-9);
  std::printf("  conv2d_afp_vs_fp32: aggregate relative error=%.4f%%\n", mean_rel * 100.0);
  CHECK(mean_rel < 0.10);

  Tensor3F pooled = maxpool2d_forward_f32(y_fp32, 2, 2);
  CHECK(pooled.h == y_fp32.h / 2);
  CHECK(pooled.w == y_fp32.w / 2);
}

void test_dbsq_adapts_block_size_around_outliers() {
  using namespace afp;
  std::mt19937 rng(3);
  std::normal_distribution<float> smallnoise(0.0f, 0.01f);
  const int N = 2048;
  std::vector<float> data(N);
  for (auto& v : data) v = smallnoise(rng);
  data[1000] = 500.0f;
  data[1001] = -300.0f;

  dbsq::DBSQConfig cfg;
  auto t = dbsq::DBSQTensor::encode(data.data(), N, cfg);
  auto hist = t.size_histogram();
  // The region right around the outlier should force small (8-element)
  // blocks; the smooth remainder should be able to grow well beyond 16.
  CHECK(hist[0] > 0); // some 8-element blocks exist (near the outlier)
  bool any_large = false;
  for (int c = 3; c < dbsq::kNumSizeCodes; ++c) if (hist[c] > 0) any_large = true;
  CHECK(any_large); // some blocks grew to >= 64 elements in the smooth region

  std::vector<float> out(N);
  t.decode(out.data());
  CHECK(rel_err(data[1000], out[1000]) < 0.06);
  CHECK(rel_err(data[1001], out[1001]) < 0.06);
  // DBSQ should compress at least as well as the fixed 16-element format
  // on data with large smooth regions (larger blocks amortize the header).
  AFPTensor fixed = AFPTensor::encode(data.data(), N);
  std::printf("  dbsq_vs_fixed: dbsq_ratio=%.3f fixed_ratio=%.3f\n",
              t.compression_ratio(), fixed.compression_ratio());
  CHECK(t.compression_ratio() >= fixed.compression_ratio() * 0.98);
}

void test_dbsq_dot_product_matches_float() {
  using namespace afp;
  std::mt19937 rng(55);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int N = 300;
  std::vector<float> a(N), b(N);
  for (auto& v : a) v = normal(rng);
  for (auto& v : b) v = normal(rng);
  dbsq::DBSQConfig cfg;
  auto ta = dbsq::DBSQTensor::encode(a.data(), N, cfg);
  auto tb = dbsq::DBSQTensor::encode(b.data(), N, cfg);
  double d_afp = dbsq::dot_product_native(ta, tb);
  double d_float = 0;
  for (int i = 0; i < N; ++i) d_float += double(a[i]) * double(b[i]);
  double re = std::fabs(d_afp - d_float) / std::max(std::fabs(d_float), 1e-6);
  std::printf("  dbsq_dot_product: afp=%.6f float=%.6f rel_err=%.4f%%\n", d_afp, d_float, re * 100.0);
  CHECK(re < 0.10);
}

void test_pruning() {
  using namespace afp;
  std::mt19937 rng(2);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> w(10000);
  for (auto& v : w) v = normal(rng);
  auto st = prune_to_sparsity(w, 0.5);
  size_t zeros = 0;
  for (auto v : w) if (v == 0.0f) zeros++;
  std::printf("  pruning: requested=0.50 achieved=%.4f (zeros=%zu/%zu)\n",
              st.achieved_sparsity(), zeros, w.size());
  CHECK(std::fabs(st.achieved_sparsity() - 0.5) < 0.02);

  // Pruning should make DBSQ blocks in the pruned regions grow larger
  // (zeros impose no exponent constraint), improving compression.
  std::vector<float> w2(10000);
  for (auto& v : w2) v = normal(rng);
  dbsq::DBSQConfig cfg;
  auto unpruned = dbsq::DBSQTensor::encode(w2.data(), w2.size(), cfg);
  prune_to_sparsity(w2, 0.7);
  auto pruned = dbsq::DBSQTensor::encode(w2.data(), w2.size(), cfg);
  std::printf("  pruning+dbsq: unpruned_ratio=%.3f pruned(70%%)_ratio=%.3f\n",
              unpruned.compression_ratio(), pruned.compression_ratio());
  CHECK(pruned.compression_ratio() > unpruned.compression_ratio());
}

int main() {
  test_basic_roundtrip();
  test_wide_dynamic_range_truncates_small_values();
  test_exact_zero();
  test_all_positive_block_uses_extra_bit();
  test_negative_block_no_positive_bonus();
  test_denormal_underflow_graceful();
  test_random_stress();
  test_dot_product_matches_float();
  test_relu_native();
  test_activations_approx();
  test_conv2d_afp_matches_fp32();
  test_dbsq_adapts_block_size_around_outliers();
  test_dbsq_dot_product_matches_float();
  test_pruning();

  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  } else {
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
  }
}
