// quantize_and_benchmark.cpp
//
// Loads a network exported by python/train_mnist.py (FP32 weights/biases +
// the MNIST test set), optionally prunes it, quantizes every weight tensor
// into AFP -- either the fixed-16-element format (afp_codec.hpp, faithful
// to "Be Like Water") or the dynamic block-size DBSQ format
// (afp_dbsq.hpp, inspired by "SmartBlock") -- and reports:
//
//   1. Memory footprint: FP32 vs AFP parameter bytes and compression ratio.
//   2. Inference wall-clock time over the test set, FP32 vs AFP.
//   3. Ground-truth classification accuracy, FP32 vs AFP.
//   4. How closely the AFP model's accuracy matches the FP32 model's.
//
// Usage:
//   ./quantize_and_benchmark <export_dir> [options]
//   ./quantize_and_benchmark --synthetic [options]
//
// Options:
//   --limit N              only evaluate the first N test examples
//   --dbsq                 use dynamic block-size (SmartBlock-style) quantization
//                          instead of the fixed 16-element AFP block format
//   --dbsq-threshold T     DBSQ outlier exponent-range threshold (default 7)
//   --prune S              zero out the smallest-magnitude S fraction of each
//                          weight tensor (global unstructured magnitude
//                          pruning, e.g. --prune 0.5) before quantizing
//   --no-zero-field        disable the zero-field bonus-bit optimization
//   --no-positive-field    disable the positive-field bonus-bit optimization
//
// If <export_dir>/manifest.txt is missing (e.g. because the Python trainer
// hasn't been run -- it needs `pip install torch torchvision`, which this
// sandbox does not have network access to do), pass --synthetic to exercise
// the full pipeline against a randomly-initialized MLP and synthetic data.

#include "afp/afp_core.hpp"
#include "afp/afp_codec.hpp"
#include "afp/afp_tensor.hpp"
#include "afp/afp_ops.hpp"
#include "afp/afp_dbsq.hpp"
#include "afp/pruning.hpp"
#include "afp/network.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <filesystem>

using afp::NetworkGraph;
using afp::LayerSpec;
using afp::LayerType;
using afp::Tensor3F;
using afp::AFPDenseLayer;
using afp::AFPConv2DLayer;
using afp::DBSQDenseLayer;
using afp::DBSQConv2DLayer;

namespace fs = std::filesystem;

enum class QuantMode { Fixed16, DBSQ };

struct Model {
  NetworkGraph graph;
  std::vector<std::vector<float>> W, B; // fp32 (post-pruning) weights/biases, indexed by layer

  std::vector<AFPDenseLayer> afp_dense;
  std::vector<AFPConv2DLayer> afp_conv;
  std::vector<DBSQDenseLayer> dbsq_dense;
  std::vector<DBSQConv2DLayer> dbsq_conv;

  QuantMode mode = QuantMode::Fixed16;
  size_t fp32_param_bytes = 0;
  size_t afp_param_bytes = 0;
  afp::PruneStats prune_stats;
};

static int argmax(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// ---------------------------------------------------------------------------
// Loading + (optional pruning +) quantizing
// ---------------------------------------------------------------------------

Model load_and_quantize(const std::string& dir, QuantMode mode, afp::EncodeOptions opts,
                         const afp::dbsq::DBSQConfig& dbsq_cfg, double prune_sparsity) {
  Model m;
  m.mode = mode;
  m.graph = afp::load_manifest(dir + "/manifest.txt");
  size_t n = m.graph.layers.size();
  m.W.resize(n); m.B.resize(n);
  m.afp_dense.resize(n); m.afp_conv.resize(n);
  m.dbsq_dense.resize(n); m.dbsq_conv.resize(n);

  for (size_t i = 0; i < n; ++i) {
    const auto& L = m.graph.layers[i];
    if (L.type != LayerType::Dense && L.type != LayerType::Conv2D) continue;
    m.W[i] = afp::read_f32_file(dir + "/" + L.weight_file);
    m.B[i] = afp::read_f32_file(dir + "/" + L.bias_file);
    m.fp32_param_bytes += (m.W[i].size() + m.B[i].size()) * sizeof(float);

    if (prune_sparsity > 0.0) {
      auto st = afp::prune_to_sparsity(m.W[i], prune_sparsity);
      m.prune_stats.total += st.total;
      m.prune_stats.pruned += st.pruned;
    }

    if (L.type == LayerType::Dense) {
      if (mode == QuantMode::Fixed16) {
        m.afp_dense[i] = afp::make_afp_dense(m.W[i], m.B[i], L.in, L.out, opts);
        for (auto& row : m.afp_dense[i].rows) m.afp_param_bytes += row.packed_bytes();
      } else {
        m.dbsq_dense[i] = afp::make_dbsq_dense(m.W[i], m.B[i], L.in, L.out, dbsq_cfg, opts);
        for (auto& row : m.dbsq_dense[i].rows) m.afp_param_bytes += row.packed_bytes();
      }
      m.afp_param_bytes += m.B[i].size() * sizeof(float);
    } else {
      if (mode == QuantMode::Fixed16) {
        m.afp_conv[i] = afp::make_afp_conv2d(m.W[i], m.B[i], L.in_ch, L.out_ch, L.k, L.stride,
                                              L.pad, opts);
        for (auto& kern : m.afp_conv[i].kernels) m.afp_param_bytes += kern.packed_bytes();
      } else {
        m.dbsq_conv[i] = afp::make_dbsq_conv2d(m.W[i], m.B[i], L.in_ch, L.out_ch, L.k, L.stride,
                                                L.pad, dbsq_cfg, opts);
        for (auto& kern : m.dbsq_conv[i].kernels) m.afp_param_bytes += kern.packed_bytes();
      }
      m.afp_param_bytes += m.B[i].size() * sizeof(float);
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Forward passes
// ---------------------------------------------------------------------------

std::vector<float> forward_fp32(const Model& m, const Tensor3F& input) {
  Tensor3F cur3 = input;
  std::vector<float> curflat;
  bool flat = false;
  for (const auto& L : m.graph.layers) {
    size_t i = &L - &m.graph.layers[0];
    switch (L.type) {
      case LayerType::Dense: {
        if (!flat) { curflat = cur3.data; flat = true; }
        curflat = afp::dense_forward_f32(curflat, m.W[i], m.B[i], L.in, L.out, L.relu);
        break;
      }
      case LayerType::Conv2D:
        cur3 = afp::conv2d_forward_f32(cur3, m.W[i], m.B[i], L.in_ch, L.out_ch, L.k, L.stride,
                                        L.pad, L.relu);
        break;
      case LayerType::MaxPool2D:
        cur3 = afp::maxpool2d_forward_f32(cur3, L.pool_k, L.pool_stride);
        break;
      case LayerType::Flatten:
        curflat = cur3.data; flat = true;
        break;
    }
  }
  return curflat;
}

std::vector<float> forward_afp(const Model& m, const Tensor3F& input, afp::EncodeOptions opts,
                                const afp::dbsq::DBSQConfig& dbsq_cfg) {
  Tensor3F cur3 = input;
  std::vector<float> curflat;
  bool flat = false;
  for (const auto& L : m.graph.layers) {
    size_t i = &L - &m.graph.layers[0];
    switch (L.type) {
      case LayerType::Dense: {
        if (!flat) { curflat = cur3.data; flat = true; }
        if (m.mode == QuantMode::Fixed16) {
          afp::AFPTensor x = afp::AFPTensor::encode(curflat.data(), curflat.size(), opts);
          curflat = afp::dense_forward_afp(m.afp_dense[i], x, L.relu);
        } else {
          auto x = afp::dbsq::DBSQTensor::encode(curflat.data(), curflat.size(), dbsq_cfg, opts);
          curflat = afp::dense_forward_dbsq(m.dbsq_dense[i], x, L.relu);
        }
        break;
      }
      case LayerType::Conv2D:
        if (m.mode == QuantMode::Fixed16)
          cur3 = afp::conv2d_forward_afp(cur3, m.afp_conv[i], L.relu, opts);
        else
          cur3 = afp::conv2d_forward_dbsq(cur3, m.dbsq_conv[i], L.relu, dbsq_cfg, opts);
        break;
      case LayerType::MaxPool2D:
        cur3 = afp::maxpool2d_forward_f32(cur3, L.pool_k, L.pool_stride);
        break;
      case LayerType::Flatten:
        curflat = cur3.data; flat = true;
        break;
    }
  }
  return curflat;
}

// ---------------------------------------------------------------------------
// Synthetic fallback (no torch / no dataset available in this environment)
// ---------------------------------------------------------------------------

Model make_synthetic_model(int in_dim, int hidden1, int hidden2, int out_dim, unsigned seed,
                            QuantMode mode, afp::EncodeOptions opts,
                            const afp::dbsq::DBSQConfig& dbsq_cfg, double prune_sparsity) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.15f);
  Model m;
  m.mode = mode;
  m.graph.model_name = "synthetic_mlp";
  m.graph.in_c = 1; m.graph.in_h = 1; m.graph.in_w = in_dim;

  auto add_dense = [&](const char* name, int in_f, int out_f, bool relu) {
    LayerSpec spec; spec.type = LayerType::Dense; spec.name = name;
    spec.in = in_f; spec.out = out_f; spec.relu = relu;
    m.graph.layers.push_back(spec);
    std::vector<float> w(size_t(in_f) * out_f), b(out_f);
    for (auto& v : w) v = dist(rng);
    for (auto& v : b) v = dist(rng);
    m.W.push_back(w); m.B.push_back(b);
  };
  add_dense("fc1", in_dim, hidden1, true);
  add_dense("fc2", hidden1, hidden2, true);
  add_dense("fc3", hidden2, out_dim, false);

  size_t n = m.graph.layers.size();
  m.afp_dense.resize(n); m.afp_conv.resize(n);
  m.dbsq_dense.resize(n); m.dbsq_conv.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const auto& L = m.graph.layers[i];
    m.fp32_param_bytes += (m.W[i].size() + m.B[i].size()) * sizeof(float);
    if (prune_sparsity > 0.0) {
      auto st = afp::prune_to_sparsity(m.W[i], prune_sparsity);
      m.prune_stats.total += st.total;
      m.prune_stats.pruned += st.pruned;
    }
    if (mode == QuantMode::Fixed16) {
      m.afp_dense[i] = afp::make_afp_dense(m.W[i], m.B[i], L.in, L.out, opts);
      for (auto& row : m.afp_dense[i].rows) m.afp_param_bytes += row.packed_bytes();
    } else {
      m.dbsq_dense[i] = afp::make_dbsq_dense(m.W[i], m.B[i], L.in, L.out, dbsq_cfg, opts);
      for (auto& row : m.dbsq_dense[i].rows) m.afp_param_bytes += row.packed_bytes();
    }
    m.afp_param_bytes += m.B[i].size() * sizeof(float);
  }
  return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string dir;
  bool synthetic = false;
  int limit = -1;
  afp::EncodeOptions opts;
  QuantMode mode = QuantMode::Fixed16;
  afp::dbsq::DBSQConfig dbsq_cfg;
  double prune_sparsity = 0.0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--synthetic") synthetic = true;
    else if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
    else if (a == "--no-zero-field") opts.enable_zero_field = false;
    else if (a == "--no-positive-field") opts.enable_positive_field = false;
    else if (a == "--dbsq") mode = QuantMode::DBSQ;
    else if (a == "--dbsq-threshold" && i + 1 < argc) dbsq_cfg.outlier_exponent_threshold = std::atoi(argv[++i]);
    else if (a == "--prune" && i + 1 < argc) prune_sparsity = std::atof(argv[++i]);
    else if (dir.empty()) dir = a;
  }

  if (!synthetic && (dir.empty() || !fs::exists(dir + "/manifest.txt"))) {
    std::printf("No manifest found at '%s/manifest.txt'.\n", dir.c_str());
    std::printf("Run python/train_mnist.py first, or pass --synthetic to smoke-test the\n");
    std::printf("pipeline with a randomly-initialized network.\n");
    return synthetic ? 0 : 1;
  }

  Model model;
  std::vector<float> images; std::vector<int32_t> labels;
  int n_examples = 0;
  double reported_fp32_acc = -1;

  if (synthetic && dir.empty()) {
    std::printf("=== Running in --synthetic mode (no trained model / dataset found) ===\n\n");
    model = make_synthetic_model(64, 32, 16, 10, /*seed=*/42, mode, opts, dbsq_cfg, prune_sparsity);
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    n_examples = limit > 0 ? limit : 500;
    images.resize(size_t(n_examples) * 64);
    for (auto& v : images) v = dist(rng);
    labels.resize(n_examples);
    for (int i = 0; i < n_examples; ++i) {
      Tensor3F in{1, 1, 64, std::vector<float>(images.begin() + i * 64, images.begin() + (i + 1) * 64)};
      labels[i] = argmax(forward_fp32(model, in));
    }
  } else {
    model = load_and_quantize(dir, mode, opts, dbsq_cfg, prune_sparsity);
    images = afp::read_f32_file(dir + "/test_images.bin");
    auto labels32 = afp::read_i32_file(dir + "/test_labels.bin");
    labels = labels32;
    size_t per_example = size_t(model.graph.in_c) * model.graph.in_h * model.graph.in_w;
    n_examples = static_cast<int>(images.size() / per_example);
    if (limit > 0) n_examples = std::min(n_examples, limit);
    std::ifstream meta(dir + "/test_meta.txt");
    if (meta) {
      std::string tok; double val;
      while (meta >> tok >> val) if (tok == "fp32_accuracy") reported_fp32_acc = val;
    }
    std::printf("=== Loaded model '%s' from %s (%d test examples%s) ===\n\n",
                model.graph.model_name.c_str(), dir.c_str(), n_examples,
                limit > 0 ? " [limited]" : "");
  }

  std::printf("Quantization mode: %s\n",
              mode == QuantMode::Fixed16 ? "fixed 16-element AFP blocks" : "DBSQ (dynamic block size)");
  if (mode == QuantMode::DBSQ)
    std::printf("  DBSQ outlier exponent-range threshold: %d\n", dbsq_cfg.outlier_exponent_threshold);
  if (prune_sparsity > 0.0)
    std::printf("Pruning: requested global sparsity=%.2f, achieved=%.4f (%zu/%zu weights zeroed)\n",
                prune_sparsity, model.prune_stats.achieved_sparsity(), model.prune_stats.pruned,
                model.prune_stats.total);
  std::printf("\n");

  // ---- Size comparison -----------------------------------------------------
  std::printf("---- Parameter memory footprint ----\n");
  std::printf("FP32 parameter bytes : %10zu (%.2f KB)\n", model.fp32_param_bytes,
              model.fp32_param_bytes / 1024.0);
  std::printf("AFP  parameter bytes : %10zu (%.2f KB)\n", model.afp_param_bytes,
              model.afp_param_bytes / 1024.0);
  std::printf("Compression ratio    : %.2fx\n\n",
              double(model.fp32_param_bytes) / double(model.afp_param_bytes));

  // ---- Inference: FP32 vs AFP ------------------------------------------------
  size_t per_example = size_t(model.graph.in_c) * model.graph.in_h * model.graph.in_w;
  std::vector<int> pred_fp32(n_examples), pred_afp(n_examples);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_examples; ++i) {
    Tensor3F in{model.graph.in_c, model.graph.in_h, model.graph.in_w,
                std::vector<float>(images.begin() + size_t(i) * per_example,
                                    images.begin() + size_t(i + 1) * per_example)};
    pred_fp32[i] = argmax(forward_fp32(model, in));
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < n_examples; ++i) {
    Tensor3F in{model.graph.in_c, model.graph.in_h, model.graph.in_w,
                std::vector<float>(images.begin() + size_t(i) * per_example,
                                    images.begin() + size_t(i + 1) * per_example)};
    pred_afp[i] = argmax(forward_afp(model, in, opts, dbsq_cfg));
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  double fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double afp_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

  int correct_fp32 = 0, correct_afp = 0, agree = 0;
  for (int i = 0; i < n_examples; ++i) {
    correct_fp32 += (pred_fp32[i] == labels[i]);
    correct_afp += (pred_afp[i] == labels[i]);
    agree += (pred_fp32[i] == pred_afp[i]);
  }
  double acc_fp32 = double(correct_fp32) / n_examples;
  double acc_afp = double(correct_afp) / n_examples;
  double agreement = double(agree) / n_examples;

  std::printf("---- Inference time over %d examples ----\n", n_examples);
  std::printf("FP32 total: %8.2f ms  (%.4f ms/example)\n", fp32_ms, fp32_ms / n_examples);
  std::printf("AFP  total: %8.2f ms  (%.4f ms/example)   [reference software codec; not\n",
              afp_ms, afp_ms / n_examples);
  std::printf("            representative of a hardware AFP MAC pipeline -- see cpp/rtl/\n");
  std::printf("            for a cycle-level FP32-vs-AFP hardware comparison, and README]\n\n");

  std::printf("---- Accuracy ----\n");
  if (reported_fp32_acc >= 0)
    std::printf("FP32 accuracy (reported by trainer, full test set): %.4f\n", reported_fp32_acc);
  std::printf("FP32 accuracy (this harness, %d examples): %.4f\n", n_examples, acc_fp32);
  std::printf("AFP  accuracy (this harness, %d examples): %.4f\n", n_examples, acc_afp);
  std::printf("AFP / FP32 accuracy ratio                : %.4f  (paper's target: >= 0.99)\n",
              acc_fp32 > 0 ? acc_afp / acc_fp32 : 0.0);
  std::printf("Per-example prediction agreement (AFP vs FP32): %.4f\n", agreement);

  return 0;
}
