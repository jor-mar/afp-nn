# AFP: Adaptive Floating Point (paper-faithful C++ implementation)

A from-scratch C++20 implementation of the **Adaptive Floating Point (AFP)**
numerical format from:

> Thomas Y. Yeh, Maxwell R. Sterner, Zerlina Lai, Brandon Y. Chuang,
> Alexander Ihler. **"Be Like Water: Adaptive Floating Point for Machine
> Learning."** ICML 2022, PMLR 162:25490-25500.
> https://proceedings.mlr.press/v162/yeh22a/yeh22a.pdf

plus a PyTorch MNIST trainer and a C++ harness that quantizes the trained
FP32 weights into AFP and compares size, speed, and accuracy against the
FP32 baseline.

```
afp_project/
  cpp/
    include/afp/
      afp_core.hpp     bit-packing primitives, AFPBlock layout, float<->bits tricks
      afp_codec.hpp     encode_block()/decode_block(): the actual AFP codec (fixed 16-block)
      afp_tensor.hpp     AFPTensor: block-structured storage for arbitrary-length arrays
      afp_dbsq.hpp     DBSQ: dynamic/adaptive block-size AFP (SmartBlock-inspired)
      pruning.hpp     magnitude pruning (global-sparsity and fixed-threshold)
      afp_ops.hpp     AFP-native math: ReLU (bit-level), dot product (shift/int), activations
      fastmath.hpp     libm-free fast exp/sigmoid/tanh (bit-trick 2^x construction)
      network.hpp     Dense/Conv2D/MaxPool2D/Flatten network (fixed-block AND DBSQ variants) + manifest loader
    src/quantize_and_benchmark.cpp   main benchmark/quantization harness (--dbsq, --prune, ...)
    tests/test_afp.cpp     unit tests (codec, DBSQ, pruning, dot product, activations, conv2d)
    rtl/     Verilog AFP-vs-FP32 dot-product hardware comparison (see rtl/README.md)
    Makefile
  python/
    train_mnist.py     PyTorch MNIST trainer (MLP, CNN, or any torchvision zoo model) + exporter,
                        with optional magnitude pruning
    afp_sim.py     NumPy-vectorized AFP quantization simulator (bit-exact-verified against the
                        C++ codec) for arbitrary PyTorch tensors/models
    quantize_model_zoo.py     quantize+evaluate any torchvision model-zoo classifier via forward hooks
  README.md (this file)
```

## Quick start

### 1. Build and test the C++ library

```bash
cd cpp
make test        # unit tests
make bench        # end-to-end pipeline smoke test on a random synthetic model
```

`make test` exercises: encode/decode round-trip accuracy, the positive-field
and zero-field bonus-bit optimizations, denormal/graceful-underflow
behavior, the AFP-native integer dot product against a float reference, the
bit-native ReLU, the fast activation approximations, an AFP-quantized
Conv2D layer against its FP32 reference, DBSQ's outlier-driven block-size
adaptation and its native dot product, and pruning (including its
compounding effect on DBSQ's compression ratio).

### 2. Train a real model and quantize it

```bash
cd ../python
pip install torch torchvision numpy
python3 train_mnist.py --model mlp --epochs 5 --out ../cpp/export/mlp
python3 train_mnist.py --model cnn --epochs 5 --prune-sparsity 0.5 --prune-finetune-epochs 2 --out ../cpp/export/cnn_pruned

cd ../cpp
make all
./build/quantize_and_benchmark export/mlp                       # fixed-block AFP
./build/quantize_and_benchmark export/mlp --dbsq                # dynamic block-size (DBSQ)
./build/quantize_and_benchmark export/cnn_pruned --dbsq --prune 0   # already pruned on disk
```

This sandbox does not have network/PyTorch access, so `train_mnist.py`
could not actually be executed here -- it is written against stable,
well-documented `torch`/`torchvision` APIs and its export format was
validated end-to-end by feeding the C++ harness a synthetic model of the
same shape (`./build/quantize_and_benchmark --synthetic`).

### 3. No PyTorch available? Run the synthetic smoke test

```bash
./build/quantize_and_benchmark --synthetic --limit 2000                       # fixed AFP
./build/quantize_and_benchmark --synthetic --limit 2000 --dbsq                # DBSQ
./build/quantize_and_benchmark --synthetic --limit 2000 --dbsq --prune 0.5    # DBSQ + pruning
```

This builds a randomly-initialized MLP directly in C++, quantizes it to
AFP, and evaluates both paths against synthetic inputs (labels = the FP32
model's own argmax, so accuracy measures pure quantization fidelity).
Pruning and DBSQ compound (pruned/zeroed regions impose no exponent
constraint, so DBSQ can grow larger blocks through them), which the test
suite (`test_pruning`) checks directly.

### 4. Any PyTorch model-zoo classifier

```bash
cd python
python3 quantize_model_zoo.py --model resnet18                                     # size/compression report only
python3 quantize_model_zoo.py --model resnet18 --prune-sparsity 0.3                # + pruning
python3 quantize_model_zoo.py --model resnet18 --dataset /path/to/imagefolder/val --limit 500  # + FP32-vs-AFP accuracy
```

Any name accepted by `torchvision.models.get_model` works (resnet*,
mobilenet_v2/v3, efficientnet_b*, vgg*, densenet*, vit_b_16, ...). See
"PyTorch model zoo compatibility" below for exactly what this path does
and does not cover, and why.

### 5. RTL: FP32-vs-AFP hardware cycle comparison

```bash
cd cpp/rtl
./run.sh          # generates test vectors + runs the Icarus Verilog testbench, if installed
```

See `cpp/rtl/README.md` -- **this could not be simulated in the sandbox**
that produced this repo (no network access to install `iverilog`); the
RTL is a carefully-written but simulator-unverified first draft, and that
README explains exactly what was and wasn't possible to check here.


## Quick start

### 1. Build and test the C++ library

```bash
cd cpp
make test        # unit tests
make bench        # end-to-end pipeline smoke test on a random synthetic model
```

`make test` exercises: encode/decode round-trip accuracy, the positive-field
and zero-field bonus-bit optimizations, denormal/graceful-underflow
behavior, the AFP-native integer dot product against a float reference, the
bit-native ReLU, the fast activation approximations, and an AFP-quantized
Conv2D layer against its FP32 reference.

### 2. Train a real model and quantize it

```bash
cd python
pip install torch torchvision numpy
python3 train_mnist.py --model mlp --epochs 5 --out ../cpp/export/mlp
# or: python3 train_mnist.py --model cnn --epochs 5 --out ../cpp/export/cnn

cd ../cpp
make all
./build/quantize_and_benchmark export/mlp
```

This sandbox does not have network/PyTorch access, so `train_mnist.py`
could not actually be executed here -- it is written against stable,
well-documented `torch`/`torchvision` APIs and its export format was
validated end-to-end by feeding the C++ harness a synthetic model of the
same shape (`./build/quantize_and_benchmark --synthetic`, see below).

### 3. No PyTorch available? Run the synthetic smoke test

```bash
./build/quantize_and_benchmark --synthetic --limit 2000
```

This builds a randomly-initialized MLP directly in C++, quantizes it to
AFP, and evaluates both paths against synthetic inputs (labels = the FP32
model's own argmax, so accuracy measures pure quantization fidelity).
Typical output:

```
---- Parameter memory footprint ----
FP32 parameter bytes :      11112 (10.85 KB)
AFP  parameter bytes :       3632 (3.55 KB)
Compression ratio    : 3.06x  (paper reports 3.2x vs FP32 ...)

---- Accuracy ----
FP32 accuracy (this harness, 2000 examples): 1.0000
AFP  accuracy (this harness, 2000 examples): 0.9900
AFP / FP32 accuracy ratio                : 0.9900  (paper's target: >= 0.99)
```

The 3.2x figure in the paper is for *weight tensors alone*; our harness's
`fp32_param_bytes`/`afp_param_bytes` also include biases (kept in FP32 on
both sides, since they are a tiny fraction of parameter count), which is
why the measured ratio is a bit below 3.2x. Pass `--no-positive-field` /
`--no-zero-field` to see the effect of disabling either Section-3.3
optimization.

## Format summary and where it deviates from the paper

The paper specifies the bit layout precisely (Section 3.2/3.3) but leaves a
handful of implementation details unstated. Every place this code had to
make a judgment call is documented with a comment at the point of use in
`afp_core.hpp`/`afp_codec.hpp`; the two worth calling out up front:

* **Denormal (offset = 7) reference exponent.** The paper says offset-7
  values drop the implicit leading one but doesn't spell out the exact
  reconstruction formula. We use the same convention as IEEE-754
  subnormals: denormals share the minimum *normalized* exponent
  (`e* - 6`, since offset 0..6 are normalized) rather than `e* - 7`, so the
  denormal range is contiguous with the smallest normalized value instead
  of leaving a gap or overlap. This was verified against hand-computed
  examples in the test suite (`test_wide_dynamic_range_truncates_small_values`,
  and the boundary case in `test_activations_approx`/`test_basic_roundtrip`).
* **Rounding-carry at the block maximum.** Round-to-nearest can carry a
  mantissa up to the next power of two. Normally that's handled by
  decrementing the offset; but if the value being rounded is already at
  offset 0 (the block's maximum exponent), there is no room to decrement,
  so we clamp to the largest representable mantissa at offset 0 instead
  (losing at most one ULP), rather than incorrectly wrapping to zero.

Bit layout actually implemented (matches Section 3.2/3.3 exactly otherwise):

* 16-element blocks (`kBlockSize`).
* Shared header, 2 bytes/block: 8-bit shared exponent `e*` (IEEE-754 bias
  127) + 8-bit characterization byte (2 positive-half bits, 4 zero-field
  bits, 2 reserved).
* Private fields, 9 bits/value: 1 sign + 3 offset + 5 mantissa, nominally.
* **Auto Focus**: per-value offset `t` in [0,7] from the block's shared max
  exponent; values with `t <= 6` are normalized (implicit leading one);
  `t == 7` is denormal (no implicit one, graceful underflow); exact zero is
  `t == 7, mantissa == 0`.
* **Positive-field optimization** (Section 3.3.1): when all 8 values in a
  half-block are non-negative, the sign bit position is repurposed as a
  6th mantissa bit for every value in that half -- same 9-bit footprint,
  one more bit of precision.
* **Zero-field optimization** (Section 3.3.2): for a half-block's offset-0
  or offset-1 group, if the top fractional mantissa bit is provably zero
  across every member, that bit is dropped from storage and the freed slot
  extends the mantissa by one bit on the low end -- again, same footprint,
  more precision. Only applied when the positive-field bonus isn't already
  active for that half (matching the paper: "the zero bits are ignored
  when the all positive bit is on").
* Total: `16 + 16*9 = 160` bits = **20 bytes / 16-element block = 10
  bits/value average** -> `32 / 10 = 3.2x` memory density vs FP32, matching
  the paper's headline number exactly.

## Dynamic block-size quantization (DBSQ, `afp_dbsq.hpp`)

Inspired by Xiao Ju et al., **"SmartBlock: Adaptive Block Floating Point
Quantization for Efficient DNN Acceleration"** (ICPP 2025,
https://dl.acm.org/doi/full/10.1145/3754598.3754660). That paper's ACM
page is largely paywalled beyond the abstract/introduction, so unlike the
"Be Like Water" implementation above (fully public, followed bit-for-bit),
`afp_dbsq.hpp` implements a **documented, good-faith interpretation** of
its stated design goal: *"assign larger block sizes ... to regions
without outliers to minimize hardware overhead, and smaller block sizes
... around outliers to reduce quantization error,"* with *"a
hardware-friendly block size parameter encoding scheme."* Concretely:

- Candidate block sizes are powers of two times AFP's 8-element group
  granularity: `{8, 16, 32, 64, 128, 256, 512, 1024}` -- exactly 8 options,
  so the "hardware-friendly encoding scheme" is a 3-bit per-block size
  code, the same width as AFP's own private offset field.
- An outlier-aware greedy partitioner (`plan_block_sizes`) scans a
  tensor's per-8-element exponent range and **doubles** the candidate
  block size as long as the merged exponent range stays within a
  threshold (default: `kMaxOffset`, i.e. "would growing the block start
  truncating values a smaller block would have preserved?"). Outlier
  regions stop doubling early and fall back toward 8 elements; smooth
  regions grow to the largest candidate, amortizing the per-block header
  over far more values.
- Each variable-length block reuses the *exact same* per-element codec
  (`afp::detail::plan_element`, sign/offset/mantissa, the positive-field
  and zero-field bonus bits) as the fixed-16 format, generalized from "2
  halves of 8" to "N/8 groups of 8".

`test_dbsq_adapts_block_size_around_outliers` confirms DBSQ picks 8-element
blocks around an injected outlier and much larger blocks in the smooth
remainder, and that its overall compression ratio is at least as good as
the fixed-16 format. `test_pruning`'s second half shows DBSQ and pruning
compounding: pruned (exact-zero) regions impose no exponent constraint on
`plan_block_sizes`, so they're exactly the regions DBSQ grows to its
largest candidate block size.

Two operands (e.g. a static weight row and a dynamic activation vector)
are planned independently and are not guaranteed to share block
boundaries, so `dbsq::dot_product_native` decodes both to a flat
per-element `(sign, significand, exponent)` array and does the same
integer-multiply/shift-accumulate arithmetic as the fixed-block dot
product, just indexed per element rather than per aligned block pair (see
the comment on `DBSQTensor::decode_scalars_flat()`).

## Pruning (`pruning.hpp`, and the Python trainer's `--prune-sparsity`)

Standard global unstructured magnitude pruning (Han et al. 2015 style):
zero out the smallest-magnitude `S` fraction of every weight tensor.
Available in three places:

- **C++, pre-quantization**: `afp::prune_to_sparsity(weights, S)` /
  `afp::prune_threshold(weights, T)` in `pruning.hpp`, wired into
  `quantize_and_benchmark`'s `--prune S` flag -- useful for pruning an
  arbitrary already-trained FP32 checkpoint at quantization time,
  independent of whether it was pruned during training.
- **Python, during training**: `train_mnist.py --prune-sparsity S
  --prune-finetune-epochs K` prunes after the main training loop (via
  `torch.nn.utils.prune.global_unstructured` + `prune.remove` to bake the
  zeros in permanently), then optionally fine-tunes the surviving weights
  for `K` epochs (re-zeroing pruned weights after each epoch, the standard
  "prune, then fine-tune the survivors" recipe) to recover accuracy.
- **Python, model zoo**: `quantize_model_zoo.py --prune-sparsity S` applies
  the same pruning to any torchvision classifier before analysis.

Pruning composes with both AFP formats: zeroed weights encode for free
under AFP's canonical zero representation, and -- more importantly for
DBSQ -- long runs of exact zero let DBSQ grow to much larger blocks
through them (see above), so the two techniques' memory savings compound
rather than just add.

## PyTorch model zoo compatibility

"Model zoo compatible" spans two related but distinct things this repo
does, and it's worth being precise about which is which:

1. **Quantize, prune, and measure the size/compression of any
   torchvision classifier's weights.** `train_mnist.py --model
   zoo:<name>` (optionally fine-tuning on your own `ImageFolder` dataset
   first) and `quantize_model_zoo.py --model <name>` both work for
   *any* architecture `torchvision.models.get_model` accepts --
   resnet18/50, mobilenet_v2/v3, efficientnet_b0..b7, vgg16, densenet121,
   vit_b_16, etc. -- because this direction only touches each
   `nn.Linear`/`nn.Conv2d` module's weight tensor directly; it doesn't
   need to understand the model's control flow at all.
2. **Run FP32-vs-AFP inference end-to-end and compare accuracy.** Here
   there are two independent paths with different scope:
   - `cpp/src/quantize_and_benchmark.cpp` only understands simple
     sequential `Dense -> Conv2D -> MaxPool2D -> Flatten` graphs (i.e.
     architectures shaped like `train_mnist.py`'s hand-built MLP/CNN). It
     cannot execute a ResNet's skip connections or similar -- writing a
     general computational-graph interpreter in C++ capable of arbitrary
     PyTorch control flow is a project unto itself and out of scope here.
   - `python/afp_sim.py` + `quantize_model_zoo.py` instead quantize
     weights *and* activations (via `register_forward_pre_hook` on every
     `nn.Linear`/`nn.Conv2d`) and let the **real PyTorch model** run its
     own real forward pass -- so this path works for *any* architecture,
     including ones with residual/skip connections, entirely by piggy-
     backing on PyTorch's own graph execution instead of reimplementing
     it. This is the practical way to get "does AFP preserve ResNet's
     ImageNet accuracy" style answers without a C++ ResNet interpreter.

`afp_sim.py`'s quantization grid was cross-validated bit-for-bit against
the C++ codec on 32,000 random values (`afp_quantize(x)` compared to
`AFPTensor::encode(x).decode()` with the zero-field optimization disabled
on the C++ side, since `afp_sim.py` -- documented in its own header
comment -- omits that one optimization as it doesn't vectorize as cleanly
in NumPy; that omission can only make its accuracy figures a little more
pessimistic than real AFP hardware, never more optimistic). Two real bugs
turned up and were fixed during that cross-validation: a rounding-carry
edge case that wasn't resetting the mantissa on the "decrement offset"
branch, and a positive-field flag computed over the whole 16-element block
instead of each 8-element half; after both fixes, 0 of 32,000 values
mismatched the C++ reference.

## AFP-native math (`afp_ops.hpp`)

* **`relu_inplace_native`**: operates directly on the packed bit stream --
  reads only the sign bit (or skips entirely if a half is already flagged
  all-positive) and overwrites negative entries with the canonical zero
  encoding. Zero float conversions.
* **`dot_product_native`**: implements the paper's Section 3 formula

  ```
  a . b = 2^(ea*+eb*) * sum_i (-1)^(sa_i xor sb_i) 2^-(ta_i+tb_i) ma_i mb_i
  ```

  using integer significand multiplies and a 64-bit fixed-point
  accumulator per block pair (mirroring Figure 10's "decode -> shift by
  offset sum -> integer MAC -> scale by shared exponent" pipeline), with a
  single `ldexp` (a pure exponent-field bit-set, not a real FP multiply)
  to place each block's fixed-point partial sum at the right overall
  scale before summing across blocks.
* **`sigmoid_inplace_native` / `tanh_inplace_native` / `exp_inplace_native`**:
  the paper doesn't specify a native AFP algorithm for transcendental
  functions (its own hardware design, Figure 10, is a decode -> compute ->
  encode pipeline for fused ops). We follow that same structure: decode a
  block via a genuine bit-level trick (constructing the IEEE-754 bit
  pattern directly by shifting the AFP mantissa/exponent into place, no
  `ldexp`/`frexp` calls), evaluate a libm-free approximation
  (`fastmath.hpp`: Schraudolph-style `2^x` via exponent-field bit
  construction + a degree-5 minimax polynomial for the fractional part,
  <1e-6 relative error), then re-encode. This is the practical, "fast in
  the sense that matters for an ML kernel" reading of "AFP-native math"
  the task description explicitly allows for.

## RTL: an actual AFP ALU, for the compute-density claim

The software honesty note below explains why a CPU-side software codec
can't demonstrate AFP's *compute-density* advantage -- doing that
properly needs actual AFP hardware. `cpp/rtl/` provides exactly that: a
synthesizable AFP block dot-product datapath (decode -> shift-align ->
16x parallel 6x6-bit integer multiply -> adder tree -> exponent-scale,
following Figure 10 of the paper) plus a behavioral FP32 scalar reference,
with a testbench that runs both over the same data and reports both
computed values and clock-cycle counts. See `cpp/rtl/README.md` for the
full design writeup, its deliberately-scoped-down "baseline AFP8 layout"
(no positive/zero-field bonus bits, since modeling their variable-width
reinterpretation needs a bespoke radix-4 multiplier design that's out of
scope here), and -- importantly -- an explicit note that **this RTL could
not be run through a simulator in the sandbox that wrote it** (no network
access to install Icarus Verilog), so treat it as a carefully-written,
self-consistent first draft rather than a validated design until you've
run it locally.

## Honesty about the speed benchmark

`quantize_and_benchmark`'s reported AFP inference time is for this
*reference software codec* running on a general-purpose CPU, where it is
slower than plain FP32 -- encode/decode overhead (bit-packing 16 elements
at a time) dominates and there's no hardware to exploit AFP's higher
per-MAC compute density. The paper's 4x/12x compute-density and
1.6x/3.2x memory-density claims are about a **custom AFP accelerator**
(Section 3.8: dedicated shifters, a radix-4 multiplier sized for 6-7 bit
operands, fixed-point adder trees) compared against other *accelerator*
number formats, not about beating a CPU's native `float` FMA throughput
with a software emulation layer. The honest, reproducible wins this
codebase demonstrates on a CPU are: (a) the **3.2x (fixed-block) or
better (DBSQ, especially combined with pruning) memory-density**
reduction (directly measured, no hardware needed), and (b) **>=99% of
FP32 accuracy** preserved after quantization (directly measured). The
compute-density claim needs actual AFP hardware -- see the RTL section
above for a first attempt at that (unverified in this sandbox; run it
locally to check).

## Testing methodology notes

Some AFP behavior that looks like a bug at first glance is actually the
documented, intentional AFP tradeoff, and the test suite calls this out
explicitly rather than silently working around it:

* Elements more than 7 exponent-steps below their block's maximum
  legitimately truncate to zero (Section 3.2.2: the paper itself reports
  <1% of values do this in real models). `test_wide_dynamic_range_truncates_small_values`
  pins this as expected behavior instead of treating it as an error.
* A block that mixes values across a very wide dynamic range (e.g.
  `exp(4) ~= 54.6` next to `exp(-4) ~= 0.018` in the same 16-element
  block) will show large *relative* error on the smaller values for the
  same reason. `test_activations_approx` therefore reports mean error
  across many blocks (representative of real activation tensors) rather
  than a worst-case max over an adversarially wide-range block.
