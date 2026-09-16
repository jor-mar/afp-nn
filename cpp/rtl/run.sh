#!/usr/bin/env bash
# run.sh -- build test vectors and run the AFP-vs-FP32 RTL testbench.
# See README.md: this requires Icarus Verilog (`iverilog`/`vvp`) installed
# locally; it could not be installed/run in the sandbox this repo was
# authored in (no network access).
set -euo pipefail
cd "$(dirname "$0")"

NUM_BLOCKS="${1:-8}"
SEED="${2:-42}"

echo "==> Building gen_testvectors"
g++ -std=c++20 -O2 -I../include gen_testvectors.cpp -o gen_testvectors

echo "==> Generating test vectors (${NUM_BLOCKS} blocks, seed=${SEED})"
./gen_testvectors "${NUM_BLOCKS}" "${SEED}"

if ! command -v iverilog >/dev/null 2>&1; then
  echo "iverilog not found. Install it (e.g. 'sudo apt-get install iverilog') and re-run this script." >&2
  exit 1
fi

echo "==> Compiling RTL"
iverilog -g2012 -o sim afp_element_decode.v afp_block_dot.v \
    afp_dot_product_top.v fp32_mac_behavioral.v tb_afp_vs_fp32.v

echo "==> Running simulation"
vvp sim
