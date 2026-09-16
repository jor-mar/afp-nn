// network.hpp
//
// A tiny, dependency-free sequential network runtime that can execute the
// same graph two ways:
//   - forward_fp32(): plain IEEE-754 float32 arithmetic (the baseline).
//   - forward_afp():  weights are AFP-encoded once (offline "quantization"),
//                     and *activations* are also round-tripped through AFP
//                     between every layer -- mirroring the paper's own
//                     evaluation methodology (Section 4: "all the weights
//                     were rounded when instantiating the model and all
//                     layer outputs were rounded between every layer,
//                     before being input into the next layer"). The actual
//                     matmul/convolution reduction in the AFP path uses
//                     afp::dot_product_native (Section 3's shift/integer
//                     dot-product formula), not a float fallback.
//
// Supported layers: dense (fully connected), conv2d, maxpool2d, flatten --
// enough to run both the MLP and the small CNN produced by
// python/train_mnist.py. The manifest text format is documented in that
// script's docstring and re-summarized above load_manifest() below.

#pragma once

#include "afp_core.hpp"
#include "afp_codec.hpp"
#include "afp_tensor.hpp"
#include "afp_ops.hpp"
#include "afp_dbsq.hpp"
#include "pruning.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace afp {

// ============================================================================
// Manifest / layer graph
// ============================================================================

enum class LayerType { Dense, Conv2D, MaxPool2D, Flatten };

struct LayerSpec {
  LayerType type;
  std::string name;
  // dense
  int in = 0, out = 0;
  // conv2d
  int in_ch = 0, out_ch = 0, k = 0, stride = 1, pad = 0;
  // maxpool2d
  int pool_k = 2, pool_stride = 2;
  std::string weight_file, bias_file;
  bool relu = false;
};

struct NetworkGraph {
  std::string model_name;
  int in_c = 1, in_h = 28, in_w = 28;
  std::vector<LayerSpec> layers;
};

inline std::vector<float> read_f32_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open file: " + path);
  f.seekg(0, std::ios::end);
  size_t bytes = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> data(bytes / sizeof(float));
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

inline std::vector<int32_t> read_i32_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open file: " + path);
  f.seekg(0, std::ios::end);
  size_t bytes = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<int32_t> data(bytes / sizeof(int32_t));
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

inline NetworkGraph load_manifest(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open manifest: " + path);
  NetworkGraph g;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string tok;
    iss >> tok;
    if (tok == "model") {
      iss >> g.model_name;
    } else if (tok == "input_shape") {
      iss >> g.in_c >> g.in_h >> g.in_w;
    } else if (tok == "layer") {
      std::string kind;
      iss >> kind;
      LayerSpec spec;
      if (kind == "dense") {
        spec.type = LayerType::Dense;
        std::string act;
        iss >> spec.name >> spec.in >> spec.out >> spec.weight_file >> spec.bias_file >> act;
        spec.relu = (act == "relu");
      } else if (kind == "conv2d") {
        spec.type = LayerType::Conv2D;
        std::string act;
        iss >> spec.name >> spec.in_ch >> spec.out_ch >> spec.k >> spec.stride >> spec.pad
            >> spec.weight_file >> spec.bias_file >> act;
        spec.relu = (act == "relu");
      } else if (kind == "maxpool2d") {
        spec.type = LayerType::MaxPool2D;
        iss >> spec.name >> spec.pool_k >> spec.pool_stride;
      } else if (kind == "flatten") {
        spec.type = LayerType::Flatten;
        iss >> spec.name;
      } else {
        throw std::runtime_error("Unknown layer kind in manifest: " + kind);
      }
      g.layers.push_back(std::move(spec));
    }
  }
  return g;
}

// A CHW-shaped float buffer (channels-first, matching PyTorch's layout).
struct Tensor3F {
  int c = 0, h = 0, w = 0;
  std::vector<float> data; // size c*h*w
  size_t size() const { return data.size(); }
};

// ============================================================================
// FP32 baseline layer kernels
// ============================================================================

inline void relu_inplace_f32(std::vector<float>& v) {
  for (auto& x : v) if (x < 0.0f) x = 0.0f;
}

inline std::vector<float> dense_forward_f32(const std::vector<float>& x,
                                             const std::vector<float>& W,
                                             const std::vector<float>& b,
                                             int in_f, int out_f, bool relu) {
  std::vector<float> y(out_f, 0.0f);
  for (int o = 0; o < out_f; ++o) {
    const float* row = &W[static_cast<size_t>(o) * in_f];
    double acc = 0.0;
    for (int i = 0; i < in_f; ++i) acc += double(row[i]) * double(x[i]);
    y[o] = static_cast<float>(acc) + b[o];
  }
  if (relu) relu_inplace_f32(y);
  return y;
}

inline Tensor3F conv2d_forward_f32(const Tensor3F& x, const std::vector<float>& W,
                                    const std::vector<float>& b, int in_ch, int out_ch,
                                    int k, int stride, int pad, bool relu) {
  int out_h = (x.h + 2 * pad - k) / stride + 1;
  int out_w = (x.w + 2 * pad - k) / stride + 1;
  Tensor3F y{out_ch, out_h, out_w, std::vector<float>(size_t(out_ch) * out_h * out_w, 0.0f)};
  for (int oc = 0; oc < out_ch; ++oc) {
    const float* kernel_oc = &W[static_cast<size_t>(oc) * in_ch * k * k];
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        double acc = 0.0;
        for (int ic = 0; ic < in_ch; ++ic) {
          const float* kernel_c = kernel_oc + size_t(ic) * k * k;
          const float* xin = &x.data[size_t(ic) * x.h * x.w];
          for (int ky = 0; ky < k; ++ky) {
            int iy = oy * stride - pad + ky;
            if (iy < 0 || iy >= x.h) continue;
            for (int kx = 0; kx < k; ++kx) {
              int ix = ox * stride - pad + kx;
              if (ix < 0 || ix >= x.w) continue;
              acc += double(kernel_c[ky * k + kx]) * double(xin[iy * x.w + ix]);
            }
          }
        }
        float val = static_cast<float>(acc) + b[oc];
        if (relu && val < 0.0f) val = 0.0f;
        y.data[(size_t(oc) * out_h + oy) * out_w + ox] = val;
      }
    }
  }
  return y;
}

inline Tensor3F maxpool2d_forward_f32(const Tensor3F& x, int k, int stride) {
  int out_h = (x.h - k) / stride + 1;
  int out_w = (x.w - k) / stride + 1;
  Tensor3F y{x.c, out_h, out_w, std::vector<float>(size_t(x.c) * out_h * out_w, 0.0f)};
  for (int c = 0; c < x.c; ++c) {
    const float* xin = &x.data[size_t(c) * x.h * x.w];
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float m = -std::numeric_limits<float>::infinity();
        for (int ky = 0; ky < k; ++ky)
          for (int kx = 0; kx < k; ++kx)
            m = std::max(m, xin[(oy * stride + ky) * x.w + (ox * stride + kx)]);
        y.data[(size_t(c) * out_h + oy) * out_w + ox] = m;
      }
    }
  }
  return y;
}

// ============================================================================
// AFP-native layer kernels
// ============================================================================

// A dense layer's weights encoded once as one AFPTensor per output row.
inline AFPDenseLayer make_afp_dense(const std::vector<float>& W, const std::vector<float>& b,
                                     int in_f, int out_f, EncodeOptions opts = {}) {
  AFPDenseLayer layer;
  layer.rows.resize(out_f);
  layer.bias = b;
  for (int o = 0; o < out_f; ++o) {
    layer.rows[o] = AFPTensor::encode(&W[static_cast<size_t>(o) * in_f], in_f, opts);
  }
  return layer;
}

inline std::vector<float> dense_forward_afp(const AFPDenseLayer& layer, const AFPTensor& x,
                                             bool relu) {
  std::vector<float> y(layer.num_out());
  layer.forward_native(x, y.data());
  if (relu) relu_inplace_f32(y); // scalar clamp; see relu_inplace_native for the
                                  // fully bit-native block variant used on AFPTensors.
  return y;
}

// A conv2d layer's weights encoded once as one AFPTensor per output
// channel (flattened [in_ch*k*k] kernel).
struct AFPConv2DLayer {
  std::vector<AFPTensor> kernels; // size out_ch, each length in_ch*k*k
  std::vector<float> bias;
  int in_ch, out_ch, k, stride, pad;
};

inline AFPConv2DLayer make_afp_conv2d(const std::vector<float>& W, const std::vector<float>& b,
                                       int in_ch, int out_ch, int k, int stride, int pad,
                                       EncodeOptions opts = {}) {
  AFPConv2DLayer layer;
  layer.in_ch = in_ch; layer.out_ch = out_ch; layer.k = k; layer.stride = stride; layer.pad = pad;
  layer.bias = b;
  layer.kernels.resize(out_ch);
  size_t ksize = size_t(in_ch) * k * k;
  for (int oc = 0; oc < out_ch; ++oc) {
    layer.kernels[oc] = AFPTensor::encode(&W[oc * ksize], ksize, opts);
  }
  return layer;
}

// Convolution executed by extracting each output pixel's receptive-field
// patch, AFP-encoding that patch (this is the "layer outputs / activations
// are rounded before being input into the next layer" step from the
// paper), and then reducing via the AFP-native dot product against each
// pre-encoded kernel row.
inline Tensor3F conv2d_forward_afp(const Tensor3F& x, const AFPConv2DLayer& layer, bool relu,
                                    EncodeOptions opts = {}) {
  int out_h = (x.h + 2 * layer.pad - layer.k) / layer.stride + 1;
  int out_w = (x.w + 2 * layer.pad - layer.k) / layer.stride + 1;
  Tensor3F y{layer.out_ch, out_h, out_w,
             std::vector<float>(size_t(layer.out_ch) * out_h * out_w, 0.0f)};
  size_t patch_len = size_t(layer.in_ch) * layer.k * layer.k;
  std::vector<float> patch(patch_len);
  for (int oy = 0; oy < out_h; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      size_t idx = 0;
      for (int ic = 0; ic < layer.in_ch; ++ic) {
        const float* xin = &x.data[size_t(ic) * x.h * x.w];
        for (int ky = 0; ky < layer.k; ++ky) {
          int iy = oy * layer.stride - layer.pad + ky;
          for (int kx = 0; kx < layer.k; ++kx) {
            int ix = ox * layer.stride - layer.pad + kx;
            patch[idx++] = (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) ? 0.0f
                                                                          : xin[iy * x.w + ix];
          }
        }
      }
      AFPTensor patch_afp = AFPTensor::encode(patch.data(), patch_len, opts);
      for (int oc = 0; oc < layer.out_ch; ++oc) {
        double d = dot_product_native(layer.kernels[oc], patch_afp);
        float val = static_cast<float>(d) + layer.bias[oc];
        if (relu && val < 0.0f) val = 0.0f;
        y.data[(size_t(oc) * out_h + oy) * out_w + ox] = val;
      }
    }
  }
  return y;
}

} // namespace afp

// ============================================================================
// DBSQ (dynamic block-size) variants of the dense/conv2d layer kernels
// ============================================================================
namespace afp {

struct DBSQDenseLayer {
  std::vector<dbsq::DBSQTensor> rows;
  std::vector<float> bias;
  size_t num_out() const { return rows.size(); }
  void forward_native(const dbsq::DBSQTensor& x, float* out) const {
    for (size_t o = 0; o < rows.size(); ++o) {
      double d = dbsq::dot_product_native(rows[o], x);
      out[o] = static_cast<float>(d) + (o < bias.size() ? bias[o] : 0.0f);
    }
  }
};

inline DBSQDenseLayer make_dbsq_dense(const std::vector<float>& W, const std::vector<float>& b,
                                       int in_f, int out_f, const dbsq::DBSQConfig& cfg = {},
                                       EncodeOptions opts = {}) {
  DBSQDenseLayer layer;
  layer.rows.resize(out_f);
  layer.bias = b;
  for (int o = 0; o < out_f; ++o)
    layer.rows[o] = dbsq::DBSQTensor::encode(&W[static_cast<size_t>(o) * in_f], in_f, cfg, opts);
  return layer;
}

inline std::vector<float> dense_forward_dbsq(const DBSQDenseLayer& layer,
                                              const dbsq::DBSQTensor& x, bool relu) {
  std::vector<float> y(layer.num_out());
  layer.forward_native(x, y.data());
  if (relu) relu_inplace_f32(y);
  return y;
}

struct DBSQConv2DLayer {
  std::vector<dbsq::DBSQTensor> kernels;
  std::vector<float> bias;
  int in_ch, out_ch, k, stride, pad;
};

inline DBSQConv2DLayer make_dbsq_conv2d(const std::vector<float>& W, const std::vector<float>& b,
                                         int in_ch, int out_ch, int k, int stride, int pad,
                                         const dbsq::DBSQConfig& cfg = {}, EncodeOptions opts = {}) {
  DBSQConv2DLayer layer;
  layer.in_ch = in_ch; layer.out_ch = out_ch; layer.k = k; layer.stride = stride; layer.pad = pad;
  layer.bias = b;
  layer.kernels.resize(out_ch);
  size_t ksize = size_t(in_ch) * k * k;
  for (int oc = 0; oc < out_ch; ++oc)
    layer.kernels[oc] = dbsq::DBSQTensor::encode(&W[oc * ksize], ksize, cfg, opts);
  return layer;
}

inline Tensor3F conv2d_forward_dbsq(const Tensor3F& x, const DBSQConv2DLayer& layer, bool relu,
                                     const dbsq::DBSQConfig& cfg = {}, EncodeOptions opts = {}) {
  int out_h = (x.h + 2 * layer.pad - layer.k) / layer.stride + 1;
  int out_w = (x.w + 2 * layer.pad - layer.k) / layer.stride + 1;
  Tensor3F y{layer.out_ch, out_h, out_w,
             std::vector<float>(size_t(layer.out_ch) * out_h * out_w, 0.0f)};
  size_t patch_len = size_t(layer.in_ch) * layer.k * layer.k;
  std::vector<float> patch(patch_len);
  for (int oy = 0; oy < out_h; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      size_t idx = 0;
      for (int ic = 0; ic < layer.in_ch; ++ic) {
        const float* xin = &x.data[size_t(ic) * x.h * x.w];
        for (int ky = 0; ky < layer.k; ++ky) {
          int iy = oy * layer.stride - layer.pad + ky;
          for (int kx = 0; kx < layer.k; ++kx) {
            int ix = ox * layer.stride - layer.pad + kx;
            patch[idx++] = (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) ? 0.0f
                                                                          : xin[iy * x.w + ix];
          }
        }
      }
      dbsq::DBSQTensor patch_afp = dbsq::DBSQTensor::encode(patch.data(), patch_len, cfg, opts);
      for (int oc = 0; oc < layer.out_ch; ++oc) {
        double d = dbsq::dot_product_native(layer.kernels[oc], patch_afp);
        float val = static_cast<float>(d) + layer.bias[oc];
        if (relu && val < 0.0f) val = 0.0f;
        y.data[(size_t(oc) * out_h + oy) * out_w + ox] = val;
      }
    }
  }
  return y;
}

} // namespace afp
