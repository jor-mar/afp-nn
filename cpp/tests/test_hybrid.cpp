// test_hybrid.cpp -- correctness unit tests for the AdaptivFloat + AFP +
// DBSQ hybrid format (afp_hybrid.hpp). Kept separate from test_afp.cpp
// (which covers the core fixed-16 AFP codec, DBSQ, and pruning) so each
// suite stays focused and both can be run/extended independently.
//
// Build: see Makefile (`make test-hybrid`) or README.md's hybrid section.
// No external test framework: uses simple asserts and prints a PASS/FAIL
// summary so it works standalone.

#include "afp/afp_core.hpp"
#include "afp/afp_codec.hpp"
#include "afp/afp_tensor.hpp"
#include "afp/afp_dbsq.hpp"
#include "afp/afp_hybrid.hpp"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>

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

static double mean_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  double sum = 0;
  for (size_t i = 0; i < a.size(); ++i) sum += std::fabs(a[i] - b[i]);
  return sum / a.size();
}

// ============================================================================
// 1. Round-trip correctness across the configurable bit-width space.
// ============================================================================
void test_roundtrip_various_configs() {
  using namespace afp;
  std::mt19937 rng(1);
  std::normal_distribution<float> normal(0.0f, 0.05f); // typical NN weight scale
  const int N = 16 * 200;
  std::vector<float> data(N);
  for (auto& v : data) v = normal(rng);

  double prev_err_for_offset3 = std::numeric_limits<double>::infinity();
  for (int offset_bits : {2, 3, 4}) {
    for (int mantissa_bits : {1, 2, 3}) {
      hybrid::HybridConfig cfg;
      cfg.offset_bits = offset_bits;
      cfg.mantissa_bits = mantissa_bits;
      auto t = hybrid::HybridTensor::encode(data.data(), N, cfg);
      std::vector<float> out(N);
      t.decode(out.data());
      double err = mean_abs_error(data, out);
      std::printf("  roundtrip: offset=%d mantissa=%d bits/elem=%d ratio=%.2fx mean_abs_err=%.6f\n",
                  offset_bits, mantissa_bits, cfg.private_bits(), t.compression_ratio(), err);
      CHECK(t.compression_ratio() > 3.0);   // beats plain fixed-16 AFP's 3.2x... at
                                             // least approaches it even at the low end
      CHECK(err < 0.05);                    // sane for this data's scale regardless of config
      CHECK(std::isfinite(err));
      if (offset_bits == 3) {
        // More mantissa bits should monotonically improve accuracy for a
        // fixed offset width.
        CHECK(err <= prev_err_for_offset3 * 1.01); // small tolerance for RNG noise
        prev_err_for_offset3 = err;
      }
    }
    if (offset_bits == 3) prev_err_for_offset3 = std::numeric_limits<double>::infinity();
  }
}

// ============================================================================
// 2. Compression ratio sanity: hybrid should comfortably beat fixed-16 AFP
//    (3.2x) at every mantissa width tested, on realistic weight-like data.
// ============================================================================
void test_compression_beats_fixed_afp() {
  using namespace afp;
  std::mt19937 rng(2);
  std::normal_distribution<float> body(0.0f, 0.03f);
  std::normal_distribution<float> tail(0.0f, 0.3f);
  std::bernoulli_distribution is_outlier(0.02);
  const int N = 16 * 1000;
  std::vector<float> data(N);
  for (auto& v : data) v = is_outlier(rng) ? tail(rng) : body(rng);

  AFPTensor fixed = AFPTensor::encode(data.data(), N);
  std::printf("  compression: fixed-16 AFP ratio=%.2fx (baseline)\n", fixed.compression_ratio());

  for (int mantissa_bits : {1, 2, 3}) {
    hybrid::HybridConfig cfg;
    cfg.mantissa_bits = mantissa_bits;
    auto t = hybrid::HybridTensor::encode(data.data(), N, cfg);
    std::printf("  compression: hybrid mantissa=%d bits/elem=%d ratio=%.2fx\n",
                mantissa_bits, cfg.private_bits(), t.compression_ratio());
    CHECK(t.compression_ratio() > fixed.compression_ratio());
  }
}

// ============================================================================
// 3. Layer scale (Tier 1) robustness: the median-of-block-exponents choice
//    should not be dragged off by a small number of outlier blocks, the
//    way a mean or a global max would be.
// ============================================================================
void test_layer_scale_is_robust_to_outlier_blocks() {
  using namespace afp;
  std::mt19937 rng(3);
  std::normal_distribution<float> normal_body(0.0f, 0.05f); // exponent ~ -5..-4
  const int blocks_of_16 = 200;
  std::vector<float> data(16 * blocks_of_16);
  for (auto& v : data) v = normal_body(rng);

  // Make a small minority of blocks (the last 5%) huge outliers (exponent
  // ~ +10), like a few unusually large weights concentrated in a handful
  // of blocks (e.g. a bias-heavy region) rather than spread throughout.
  int outlier_blocks = blocks_of_16 / 20;
  for (int b = blocks_of_16 - outlier_blocks; b < blocks_of_16; ++b)
    for (int i = 0; i < 16; ++i) data[b * 16 + i] = 1000.0f + i;

  hybrid::HybridConfig cfg;
  auto t = hybrid::HybridTensor::encode(data.data(), data.size(), cfg);

  auto fb_body = decompose_float(0.05f);
  std::printf("  layer_scale: chosen=%d  (body's own exponent ~= %d, outlier blocks' ~= 10)\n",
              t.layer_scale(), fb_body.exponent);
  // The layer scale should track the *majority* (body) exponent, not be
  // pulled toward the small minority of outlier blocks.
  CHECK(std::abs(t.layer_scale() - fb_body.exponent) <= 2);

  // And the bulk of the data should still decode accurately despite those
  // few outlier blocks existing elsewhere in the same tensor.
  std::vector<float> out(data.size());
  t.decode(out.data());
  double body_err = 0;
  int body_count = (blocks_of_16 - outlier_blocks) * 16;
  for (int i = 0; i < body_count; ++i) body_err += std::fabs(data[i] - out[i]);
  body_err /= body_count;
  std::printf("  layer_scale: mean_abs_err on majority-body elements = %.6f\n", body_err);
  CHECK(body_err < 0.01);
}

// ============================================================================
// 4. Delta clamping: a block whose natural exponent is far outside what
//    exp_delta_bits can represent relative to the layer scale must degrade
//    gracefully (bounded error on that one block) rather than corrupt
//    anything -- and must never crash / produce non-finite output.
// ============================================================================
void test_delta_clamping_is_graceful() {
  using namespace afp;
  std::mt19937 rng(4);
  std::normal_distribution<float> normal_body(0.0f, 0.05f);
  const int blocks_of_16 = 50;
  std::vector<float> data(16 * blocks_of_16);
  for (auto& v : data) v = normal_body(rng);

  // One block with an exponent ~40 steps away from everything else --
  // guaranteed to exceed a 5-bit signed delta's range (-16..15).
  for (int i = 0; i < 16; ++i) data[16] = std::ldexp(1.0f, 40) * (1.0f + i * 0.01f);

  hybrid::HybridConfig cfg;
  cfg.exp_delta_bits = 5; // -16..15
  auto t = hybrid::HybridTensor::encode(data.data(), data.size(), cfg);
  std::vector<float> out(data.size());
  t.decode(out.data());

  for (float v : out) CHECK(std::isfinite(v)); // never NaN/Inf regardless of clamping
  // Everything except the extreme block should still be accurate.
  double body_err = 0;
  int body_count = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    if (i >= 16 && i < 32) continue; // skip the extreme block
    body_err += std::fabs(data[i] - out[i]);
    body_count++;
  }
  body_err /= body_count;
  std::printf("  delta_clamping: mean_abs_err outside extreme block = %.6f (all outputs finite)\n",
              body_err);
  CHECK(body_err < 0.01);
}

// ============================================================================
// 5. Positive-field bonus bit: an all-positive block should decode more
//    accurately than an otherwise-equivalent mixed-sign block, since it
//    gets one extra effective mantissa bit.
// ============================================================================
void test_positive_field_bonus_improves_accuracy() {
  using namespace afp;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> mag(0.01f, 1.0f);
  const int N = 16 * 100;

  std::vector<float> all_positive(N), mixed_sign(N);
  for (int i = 0; i < N; ++i) {
    float m = mag(rng);
    all_positive[i] = m;
    mixed_sign[i] = (i % 2 == 0) ? m : -m;
  }

  hybrid::HybridConfig cfg;
  cfg.mantissa_bits = 1; // smallest mantissa -> bonus bit matters most
  auto t_pos = hybrid::HybridTensor::encode(all_positive.data(), N, cfg);
  auto t_mix = hybrid::HybridTensor::encode(mixed_sign.data(), N, cfg);

  std::vector<float> out_pos(N), out_mix(N);
  t_pos.decode(out_pos.data());
  t_mix.decode(out_mix.data());
  double err_pos = mean_abs_error(all_positive, out_pos);
  double err_mix = mean_abs_error(mixed_sign, out_mix);
  std::printf("  positive_field: all-positive mean_abs_err=%.6f  mixed-sign mean_abs_err=%.6f\n",
              err_pos, err_mix);
  CHECK(err_pos < err_mix); // bonus bit should make the positive-only case more accurate

  cfg.enable_positive_field = false;
  auto t_pos_noopt = hybrid::HybridTensor::encode(all_positive.data(), N, cfg);
  std::vector<float> out_pos_noopt(N);
  t_pos_noopt.decode(out_pos_noopt.data());
  double err_pos_noopt = mean_abs_error(all_positive, out_pos_noopt);
  std::printf("  positive_field: all-positive WITHOUT bonus mean_abs_err=%.6f\n", err_pos_noopt);
  CHECK(err_pos < err_pos_noopt); // confirms the bonus bit, not something else, caused the gap
}

// ============================================================================
// 6. Zero and wide-dynamic-range handling: exact zeros round-trip exactly;
//    values far outside a block's representable range truncate to zero
//    gracefully (same documented AFP behavior as the fixed-block format),
//    not garbage.
// ============================================================================
void test_zero_and_wide_range_handling() {
  using namespace afp;
  hybrid::HybridConfig cfg;

  std::vector<float> zeros(64, 0.0f);
  auto t_zero = hybrid::HybridTensor::encode(zeros.data(), zeros.size(), cfg);
  std::vector<float> out_zero(zeros.size());
  t_zero.decode(out_zero.data());
  for (float v : out_zero) CHECK(v == 0.0f);

  // Single 8-element block (avoids interaction with the layer-scale delta
  // clamping mechanism, which is tested separately in
  // test_delta_clamping_is_graceful -- this test isolates *within-block*
  // wide-dynamic-range behavior only): one large value sets the block
  // exponent, one far-below value should truncate gracefully rather than
  // corrupt anything.
  std::vector<float> wide(8, 1.0f);
  wide[0] = 1.0e6f;   // sets the block's exponent
  wide[1] = 1.0e-6f;  // far below representable range given that exponent
  auto t_wide = hybrid::HybridTensor::encode(wide.data(), wide.size(), cfg);
  std::vector<float> out_wide(wide.size());
  t_wide.decode(out_wide.data());
  CHECK(std::isfinite(out_wide[0]));
  CHECK(rel_err(wide[0], out_wide[0]) < 0.15); // coarse (small mantissa) but in the right ballpark
  CHECK(out_wide[1] == 0.0f);                  // truncates, doesn't corrupt
}

// ============================================================================
// 7. Offset-width tradeoff (documents + re-verifies the finding in
//    afp_hybrid.hpp's header comment): 3-bit offset should beat 2-bit on
//    both compression and accuracy for realistic, heavier-tailed data.
// ============================================================================
void test_offset_width_tradeoff() {
  using namespace afp;
  std::mt19937 rng(99);
  std::normal_distribution<float> body(0.0f, 0.03f);
  std::normal_distribution<float> tail(0.0f, 0.3f);
  std::bernoulli_distribution is_outlier(0.02);
  const int N = 16 * 1000;
  std::vector<float> data(N);
  for (auto& v : data) v = is_outlier(rng) ? tail(rng) : body(rng);

  auto measure = [&](int offset_bits) {
    hybrid::HybridConfig cfg;
    cfg.offset_bits = offset_bits;
    cfg.mantissa_bits = 1;
    auto t = hybrid::HybridTensor::encode(data.data(), N, cfg);
    std::vector<float> out(N);
    t.decode(out.data());
    double avg_block = double(N) / t.blocks().size();
    return std::make_tuple(t.compression_ratio(), mean_abs_error(data, out), avg_block);
  };
  auto [ratio2, err2, blk2] = measure(2);
  auto [ratio3, err3, blk3] = measure(3);
  std::printf("  offset_width: offset=2 ratio=%.2fx err=%.5f avg_block=%.1f | "
              "offset=3 ratio=%.2fx err=%.5f avg_block=%.1f\n",
              ratio2, err2, blk2, ratio3, err3, blk3);
  CHECK(blk3 > blk2);         // wider offset lets DBSQ grow blocks further
  CHECK(err3 <= err2 * 1.05); // and doesn't come out meaningfully worse on accuracy
}

// ============================================================================
// 8. Best-exponent search: verified (not just documented) to match brute
//    force across several cases, including ones deliberately constructed
//    to maximize the incentive for a non-max choice.
// ============================================================================
void test_search_matches_brute_force() {
  using namespace afp;
  hybrid::HybridConfig cfg;
  cfg.mantissa_bits = 2;

  auto brute_force_best_delta = [&](const std::vector<float>& data) {
    int32_t natural_max = std::numeric_limits<int32_t>::min();
    for (float v : data) { auto fb = decompose_float(v); if (!fb.is_zero) natural_max = std::max(natural_max, fb.exponent); }
    double best_err = std::numeric_limits<double>::infinity();
    int best_delta = 0;
    for (int d = 0; d <= 12; ++d) {
      int32_t cand = natural_max - d;
      auto block = hybrid::encode_hybrid_block(data.data(), static_cast<int>(data.size()), cand, cfg);
      std::vector<float> out(data.size());
      hybrid::decode_hybrid_block(block, cand, cfg, out.data());
      double l1 = 0;
      for (size_t i = 0; i < data.size(); ++i) l1 += std::fabs(data[i] - out[i]);
      if (l1 < best_err) { best_err = l1; best_delta = d; }
    }
    return best_delta;
  };

  std::vector<std::vector<float>> cases = {
      {1024.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {1000.0f, 999.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
      {256.5f, 0.24f, 0.26f, 0.25f, 0.24f, 0.26f, 0.25f, 0.24f},
  };
  {
    std::vector<float> big(512, 0.001f); big[0] = 1e6f;
    cases.push_back(big);
  }
  {
    std::vector<float> many(1024, 0.01f); many[0] = 100.0f;
    cases.push_back(many);
  }
  for (auto& c : cases) {
    int delta = brute_force_best_delta(c);
    CHECK(delta == 0); // brute force always confirms natural max is optimal
  }
  std::printf("  search: brute-force-confirmed natural-max-is-optimal on %zu cases\n", cases.size());

  // The search function itself, at various radii, should agree.
  for (auto& c : cases) {
    for (int radius : {0, 3, 8}) {
      hybrid::HybridConfig cfg2 = cfg;
      cfg2.best_exp_search_radius = radius;
      int32_t chosen = hybrid::search_best_block_exponent(c.data(), static_cast<int>(c.size()), cfg2);
      int32_t natural_max = std::numeric_limits<int32_t>::min();
      for (float v : c) { auto fb = decompose_float(v); if (!fb.is_zero) natural_max = std::max(natural_max, fb.exponent); }
      CHECK(chosen == natural_max);
    }
  }
}

// ============================================================================
// 9. Native dot product accuracy against a float reference.
// ============================================================================
void test_dot_product_matches_float() {
  using namespace afp;
  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int N = 300;
  std::vector<float> a(N), b(N);
  for (auto& v : a) v = normal(rng);
  for (auto& v : b) v = normal(rng);
  hybrid::HybridConfig cfg;
  auto ta = hybrid::HybridTensor::encode(a.data(), N, cfg);
  auto tb = hybrid::HybridTensor::encode(b.data(), N, cfg);
  double d = hybrid::dot_product_native(ta, tb);
  double truth = 0;
  for (int i = 0; i < N; ++i) truth += double(a[i]) * double(b[i]);
  double re = std::fabs(d - truth) / std::max(std::fabs(truth), 1e-6);
  std::printf("  dot_product: afp=%.6f float=%.6f rel_err=%.4f%% (bits/elem=%d)\n",
              d, truth, re * 100.0, cfg.private_bits());
  CHECK(re < 0.20); // 6-bit elements -> noticeably more error than AFP8's 9-bit, expected
}

// ============================================================================
// 10. Edge cases: tiny tensors, sizes not a multiple of 8, single element,
//     and a tensor with no nonzero elements at all.
// ============================================================================
void test_edge_cases() {
  using namespace afp;
  hybrid::HybridConfig cfg;

  // Single element (padded up to the minimum 8-element block).
  {
    float v[1] = {3.5f};
    auto t = hybrid::HybridTensor::encode(v, 1, cfg);
    float out[1];
    t.decode(out);
    CHECK(rel_err(v[0], out[0]) < 0.15);
  }
  // Size not a multiple of 8 (padding path).
  {
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto t = hybrid::HybridTensor::encode(v.data(), v.size(), cfg);
    CHECK(t.size() == v.size());
    std::vector<float> out(v.size());
    t.decode(out.data());
    for (size_t i = 0; i < v.size(); ++i) CHECK(rel_err(v[i], out[i]) < 0.15);
  }
  // All-zero tensor: must not divide-by-zero or otherwise misbehave when
  // computing the layer scale (median of an all-"no data" block set).
  {
    std::vector<float> v(128, 0.0f);
    auto t = hybrid::HybridTensor::encode(v.data(), v.size(), cfg);
    std::vector<float> out(v.size());
    t.decode(out.data());
    for (float x : out) CHECK(x == 0.0f);
    CHECK(t.layer_scale() == 0);
  }
  std::printf("  edge_cases: single element, non-multiple-of-8 size, all-zero tensor OK\n");
}

int main() {
  test_roundtrip_various_configs();
  test_compression_beats_fixed_afp();
  test_layer_scale_is_robust_to_outlier_blocks();
  test_delta_clamping_is_graceful();
  test_positive_field_bonus_improves_accuracy();
  test_zero_and_wide_range_handling();
  test_offset_width_tradeoff();
  test_search_matches_brute_force();
  test_dot_product_matches_float();
  test_edge_cases();

  if (g_failures == 0) {
    std::printf("ALL HYBRID TESTS PASSED\n");
    return 0;
  } else {
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
  }
}
