# AFP RTL: FP32 vs AFP dot-product hardware comparison

Synthesizable(-ish) Verilog implementing the AFP block dot-product
datapath described in "Be Like Water" Section 3 / Figure 10 (decode ->
shift-by-offset-sum -> integer multiply-accumulate -> scale by shared
exponent), plus a behavioral FP32 reference and a testbench that runs
both over the same data and compares results and clock-cycle counts.

**This could not be simulated in the sandbox that produced it.** The
sandbox has no network access, so `apt-get install iverilog` (confirmed
to otherwise work -- apt resolved the package and only failed at the
final download step with `403 Forbidden`) could not complete, and no
other Verilog simulator (Verilator, a vendor tool, etc.) was preinstalled.
Every module below was written and manually re-reviewed for balanced
`begin`/`end`, port-width consistency, and the bit-ordering conventions
documented at the top of each file, and the C++ test-vector generator
(`gen_testvectors.cpp`, which *does* build and run in this sandbox, using
the same `afp_codec.hpp` as the rest of the repo) was used to produce and
sanity-check the `.hex` files this RTL consumes -- but the RTL itself is
**unverified against a real simulator** and should be treated as a
carefully-written first draft, not a validated design. Please run the
steps below locally and treat any simulator errors as expected first-pass
debugging, not evidence the overall architecture is wrong.

## Files

| File | Purpose |
|---|---|
| `afp_element_decode.v` | Decodes one 9-bit AFP private field (sign/offset/mantissa) into (sign, 6-bit significand, shift, is_zero). Synthesizable. |
| `afp_block_dot.v` | `afp_block_dot16`: one 16-element AFP block-pair dot product -- 16x parallel decode, shift-align, 6x6 signed multiply, 16-way adder tree. Synthesizable. |
| `afp_dot_product_top.v` | FSM streaming `NUM_BLOCKS` block pairs through `afp_block_dot16`, cross-block accumulation, cycle counting. Synthesizable. |
| `fp32_mac_behavioral.v` | Non-synthesizable behavioral FP32 scalar MAC (uses Verilog `real`), 1 element/cycle, for a cycle-count baseline. Simulation only. |
| `tb_afp_vs_fp32.v` | Testbench: loads test vectors, runs both pipelines, reports values + cycle counts + a pass/fail check against the expected value. |
| `gen_testvectors.cpp` | C++ generator (builds against the same `include/afp/` used everywhere else) producing the `.hex` test vectors and the expected AFP-quantized dot product. |

## Scope: why the RTL uses the *baseline* AFP8 layout

The rest of this repo's AFP codec (`afp_core.hpp`/`afp_codec.hpp`) defaults
to the positive-field and zero-field bonus-bit optimizations (Section
3.3), which reinterpret a value's 9-bit private field as either
`{offset[2:0], mantissa[5:0]}` (no sign bit, positive-only half) or
`{sign, offset[2:0], mantissa[4:0]}` with an implicit zero MSB restored,
*depending on a per-half/per-group flag*. Modeling that reinterpretation
in hardware is exactly what the paper says requires "a radix-4 multiplier
... designed to support the extra bit with minimal overhead" (Section
3.3.1) -- i.e. a bespoke variable-width multiplier, which is a
substantially larger RTL design than fits this task's scope. This RTL
therefore implements the **baseline fixed layout** (always
`{sign, offset[2:0], mantissa[4:0]}`, 9 bits/element, no bonus bits) --
still the paper-faithful core "Auto Focus" shift/offset dot-product
datapath from Figure 10, just without the two Section 3.3 refinements.
`gen_testvectors.cpp` encodes its test vectors with
`EncodeOptions{enable_positive_field=false, enable_zero_field=false}` so
the hardware and the test data agree on the format.

## Bit-ordering convention

Within a block's 144-bit `fields_i` bus, **element `i`'s 9-bit field
occupies bits `[9*i+8 : 9*i]`** -- element 0 in the least significant 9
bits, element 15 in the most significant. This is an arbitrary but fixed
choice (unrelated to `afp_codec.hpp`'s own MSB-first serial byte-packing,
which is a private storage-format detail of the software codec);
`gen_testvectors.cpp` builds its hex output in this exact layout. If you
extend the RTL, keep this convention or update both the Verilog and the
generator together.

## Known issue found and fixed (v1 -> v2)

A first version of `afp_dot_product_top.v` was reported to produce garbage
on real hardware simulation: `mantissa=-2305843009214349390` (~ -2^61) for
an 8-block test vector whose expected result was ~-20. Root cause: v1
picked the *first* block's exponent as a single fixed reference for the
whole vector and aligned every other block onto it with an **unclamped**
left shift. Two 16-element blocks of Gaussian-ish data can legitimately
land 30-40+ exponent steps apart by chance (e.g. one block's 16 samples
all happening to be unusually small), and shifting a ~40-bit mantissa
left by that much overflows a 64-bit accumulator and silently wraps to
garbage -- exactly the reported symptom.

v2 replaces that with a small custom floating-point-style accumulator:
the running total is kept as a normalized `(mantissa, exponent)` pair,
each new block is folded in via an align-and-add with a **clamped**
shift (a block whose magnitude is far below the running total's current
precision correctly rounds to a negligible contribution, exactly as it
would in any floating-point sum -- this is not a hack, it's what
"negligible" means), followed by renormalization so the mantissa can
never grow unbounded regardless of block count or exponent spread. This
mirrors `afp::dot_product_native`'s own two-tier design in
`afp_ops.hpp`: fixed-point accumulation *within* a block (bounded range,
safe), floating accumulation *across* blocks (unbounded range, needs
floating-style handling) -- v1's bug used fixed-point-style accumulation
for the *across-blocks* step too, where it isn't safe.

The new logic was hand-verified in Python (mirroring the exact
align/clamp/renormalize algorithm the Verilog implements) against an
adversarial 8-block case with exponents spanning -55 to +56 (111 steps):
no overflow, and the result matched a double-precision reference to
~1e-11 relative error. That's the strongest verification possible
without a Verilog simulator in this environment -- **please still run
the actual RTL locally and report back if anything looks off**; the
Verilog port of this algorithm has not been simulator-verified, only the
underlying algorithm has been.

## How to build and run (locally, not in this sandbox)

```bash
sudo apt-get install iverilog gtkwave   # or: brew install icarus-verilog
cd cpp/rtl

# 1. Build the C++ test-vector generator against the shared afp/ headers
g++ -std=c++20 -O2 -I../include gen_testvectors.cpp -o gen_testvectors

# 2. Generate test vectors (8 blocks = 128 elements by default)
./gen_testvectors 8 42

# 3. Compile and run the RTL simulation
iverilog -g2012 -o sim afp_element_decode.v afp_block_dot.v afp_dot_product_top.v fp32_mac_behavioral.v tb_afp_vs_fp32.v
vvp sim

# 4. (optional) inspect waveforms
gtkwave tb_afp_vs_fp32.vcd
```

Expected console output shape (illustrative -- exact numbers depend on
the random seed):

```
========================================================
AFP  vs FP32 dot-product RTL comparison (128 elements)
========================================================
Expected (software, float64) : -20.002354
AFP  RTL result               : -20.002354  (mantissa=... exp=...)
FP32 RTL result (behavioral)  : -20.074711
--------------------------------------------------------
AFP  cycles : 24  (5 elements/cycle throughput)
FP32 cycles : 128  (1 elements/cycle throughput)
Speedup (FP32 cycles / AFP cycles): 5.33x
--------------------------------------------------------
PASS: AFP RTL result matches expected value within tolerance.
```

Notes on that comparison:
- The **AFP RTL result** should match the *expected* value (computed in
  software from the same AFP-quantized, baseline-layout data) closely --
  any large discrepancy indicates an RTL bug to chase down, not expected
  quantization error.
- The **FP32 behavioral result** differs from both by ordinary AFP
  quantization error (this is expected -- it's the unquantized ground
  truth, same role as the C++ harness's `forward_fp32` path).
- The **cycle-count speedup** is this RTL's demonstration of AFP's
  compute-density advantage: a block of 16 elements moves through
  `afp_block_dot16`'s parallel integer-multiplier array in a fixed ~3
  cycles regardless of block size, while the behavioral FP32 reference is
  charged 1 cycle per scalar multiply-accumulate (already a generous,
  favorable-to-FP32 assumption -- see `fp32_mac_behavioral.v`'s header
  comment). Treat the resulting ~5x as illustrative of the *mechanism*
  the paper describes (parallel low-bit-width integer MACs vs. sequential
  full-width FP MACs at the same clock), not as a citable number: a real
  comparison needs a synthesizable FP32 FPU with real pipeline latency and
  a real technology/clock target, which is out of scope here (see the
  C++ harness's own printed caveat about software-emulation speed for the
  same reasoning).

## Extending

- **Positive-field/zero-field support**: add a per-lane 2-bit "mode"
  input (derived from the block's characterization byte) to
  `afp_element_decode.v`'s field interpretation, and widen the multiplier
  operands to 7 bits to accommodate the occasional 6-bit-mantissa lane.
- **Pipelining**: `afp_block_dot16` is purely combinational; for a higher
  target clock, pipeline the decode/shift-align/multiply/adder-tree
  stages (`afp_dot_product_top.v`'s `S_COMPUTE` state already reserves a
  cycle boundary where a pipeline register could be inserted).
- **DBSQ (variable block size)**: this RTL only models the fixed
  16-element block; a DBSQ-aware version would need the block's 3-bit
  size code decoded first to determine how many `afp_element_decode`
  instances to enable for that block (a runtime-configurable-width
  datapath), which is a meaningfully larger design.
