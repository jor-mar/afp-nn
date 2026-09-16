// pruning.hpp
//
// Simple, well-understood unstructured magnitude pruning, usable as an
// optional pre-quantization step (mirrors the equivalent option added to
// python/train_mnist.py's --prune-sparsity flag, and independently
// available here for quantizing arbitrary pre-trained / model-zoo
// checkpoints that were not pruned during training).
//
// Pruning composes naturally with AFP/DBSQ: zeroed weights encode for
// free under AFP's canonical zero representation (offset=7, mantissa=0),
// and -- more importantly for DBSQ -- long runs of exact zero collapse a
// region's exponent range to nothing (GroupStat::has_data == false is
// treated as "no constraint" by dbsq::plan_block_sizes), so pruned
// regions are exactly the regions DBSQ can grow to its largest candidate
// block size, compounding the two techniques' memory savings.

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

namespace afp {

struct PruneStats {
  size_t total = 0;
  size_t pruned = 0;
  double achieved_sparsity() const { return total == 0 ? 0.0 : double(pruned) / double(total); }
};

// Zero out every element whose magnitude is < threshold. O(n).
inline PruneStats prune_threshold(std::vector<float>& w, float threshold) {
  PruneStats st; st.total = w.size();
  for (auto& v : w) {
    if (std::fabs(v) < threshold) { if (v != 0.0f) st.pruned++; v = 0.0f; }
  }
  return st;
}

// Zero out the smallest-magnitude `sparsity` fraction of elements (global,
// unstructured magnitude pruning -- the standard baseline pruning
// criterion, e.g. Han et al. 2015). O(n log n) via nth_element.
inline PruneStats prune_to_sparsity(std::vector<float>& w, double sparsity) {
  PruneStats st; st.total = w.size();
  if (sparsity <= 0.0 || w.empty()) return st;
  sparsity = std::min(sparsity, 1.0);
  size_t k = static_cast<size_t>(sparsity * w.size());
  if (k == 0) return st;
  std::vector<float> mags(w.size());
  for (size_t i = 0; i < w.size(); ++i) mags[i] = std::fabs(w[i]);
  std::vector<float> tmp = mags;
  std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end());
  float threshold = tmp[k - 1];
  for (size_t i = 0; i < w.size(); ++i) {
    if (mags[i] <= threshold && w[i] != 0.0f) {
      w[i] = 0.0f;
      st.pruned++;
      if (st.pruned >= k) {} // allow ties to slightly overshoot k, harmless
    }
  }
  return st;
}

} // namespace afp
