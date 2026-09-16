"""
afp_sim.py
==========

A NumPy-vectorized simulator of the AFP quantization grid (the same
fixed-16-element-block format implemented bit-exactly in C++ at
cpp/include/afp/afp_core.hpp / afp_codec.hpp), usable as a drop-in
"round-trip through AFP" function on arbitrary NumPy/PyTorch tensors.

Why this file exists: the C++ side (network.hpp / quantize_and_benchmark)
can only *run* inference for the simple sequential Dense/Conv2D/MaxPool2D/
Flatten graphs it knows how to interpret (i.e. models shaped like the MLP
or small CNN python/train_mnist.py produces). A full PyTorch **model zoo**
classifier -- ResNet, MobileNet, EfficientNet, ViT, ... -- has arbitrary
control flow (residual/skip connections, attention, etc.) that a small
hand-rolled C++ interpreter has no business trying to re-implement.
Instead, this module quantizes weights *and* activations to the AFP grid
and is wired in via PyTorch forward hooks (see quantize_model_zoo.py), so
the *real* model (with its real control flow) does the forward pass, and
every nn.Linear/nn.Conv2d module's input/weight is snapped to the AFP grid
first -- giving a faithful "what would this model's accuracy be if its
compute used AFP-quantized values" measurement for literally any
torchvision architecture.

Known simplification vs. the bit-exact C++ codec: the zero-field bonus-bit
optimization (Section 3.3.2 of the AFP paper) is not implemented here, as
it requires a per-offset-group, cross-element scan within each block that
doesn't vectorize cleanly in NumPy the way the positive-field optimization
does. That omission can only make afp_sim's reported accuracy a little
*more* pessimistic than real AFP hardware (or the C++ codec) would
achieve, never more optimistic -- afp_core.hpp/afp_codec.hpp remain the
authoritative, complete bit-exact reference.

Everything here operates on the *last axis* by default (flattened first,
per array, matching how the C++ AFPTensor flattens an arbitrary tensor).
"""

import numpy as np

BLOCK_SIZE = 16
EXP_BIAS = 127
MAX_OFFSET = 7          # offset field is 3 bits: 0..7
DENORM_REF_SHIFT = MAX_OFFSET - 1  # 6; see afp_codec.hpp's denormal-reference-exponent note
MANTISSA_BITS = 5


def _decompose(x: np.ndarray):
    """float32 array -> (sign in {0,1}, unbiased exponent, 23-bit raw
    mantissa, is_zero) as int32/uint32/bool arrays, matching
    afp::decompose_float in afp_core.hpp."""
    bits = x.astype(np.float32).view(np.uint32)
    sign = (bits >> 31) & 1
    raw_exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    is_zero = raw_exp == 0  # true zero and FP32 subnormals both treated as zero
    exponent = raw_exp.astype(np.int32) - EXP_BIAS
    return sign.astype(np.int32), exponent, mantissa.astype(np.uint32), is_zero


def afp_quantize(x, block_size: int = BLOCK_SIZE, enable_positive_field: bool = True):
    """Round `x` (any shape, will be flattened/padded/reshaped internally)
    through the AFP quantization grid and return an array of the same
    shape/dtype float32, with values snapped to what AFP would represent.

    Accepts either a NumPy array or a PyTorch tensor (returns the same
    type it was given, to make this convenient inside `torch.no_grad()`
    forward hooks without an explicit conversion at every call site).
    """
    is_torch = hasattr(x, "detach")
    if is_torch:
        import torch
        device, dtype = x.device, x.dtype
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    orig_shape = arr.shape
    flat = arr.reshape(-1).astype(np.float32)
    n = flat.shape[0]
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, block_size)

    sign, exponent, mantissa, is_zero = _decompose(blocks)

    exp_masked = np.where(is_zero, np.int32(-2**30), exponent)
    e_star = exp_masked.max(axis=1, keepdims=True)
    e_star = np.where(e_star < -2**29, np.int32(0), e_star)  # all-zero block

    raw_offset = np.clip(e_star - exponent, 0, None).astype(np.int32)

    # Positive-field flag is per 8-element *half*, not per whole 16-element
    # block (matching afp_core.hpp's two half-block positive bits).
    half = blocks.shape[1] // 2
    sign_h = sign.reshape(-1, 2, half)
    zero_h = is_zero.reshape(-1, 2, half)
    positive_half = np.all((sign_h == 0) | zero_h, axis=2)  # (num_blocks, 2)
    if not enable_positive_field:
        positive_half = np.zeros_like(positive_half, dtype=bool)
    positive_block = np.repeat(positive_half, half, axis=1)  # (num_blocks, 16) bool
    width = np.where(positive_block, MANTISSA_BITS + 1, MANTISSA_BITS).astype(np.int32)

    normalized = raw_offset <= (MAX_OFFSET - 1)

    # ---- normalized path: round the 23-bit fraction to `width` bits ----
    shift_n = (23 - width).astype(np.int64)
    half_ulp_n = (np.int64(1) << (shift_n - 1)).astype(np.uint32)
    rounded_n = (mantissa + half_ulp_n) >> shift_n.astype(np.uint32)
    carry = rounded_n == (np.uint32(1) << width.astype(np.uint32))
    off_after_carry = np.where(raw_offset > 0, raw_offset - 1, raw_offset)
    mant_after_carry = np.where(raw_offset > 0, np.int64(0), (np.int64(1) << width) - 1)
    offset_n = np.where(carry, off_after_carry, raw_offset)
    mantissa_n = np.where(carry, mant_after_carry, rounded_n.astype(np.int64))

    # ---- denormal path (t == 7): rebase onto e*-6, no implicit leading 1 ----
    shift_extra = np.clip(raw_offset - DENORM_REF_SHIFT, 0, 30).astype(np.int64)
    full_q23 = (np.uint64(1) << 23) | mantissa.astype(np.uint64)
    shifted = full_q23 >> shift_extra.astype(np.uint64)
    shifted = np.minimum(shifted, np.uint64(0xFFFFFFFF)).astype(np.uint32)
    max_val_d = (np.int64(1) << width) - 1
    rounded_d = (shifted.astype(np.uint32) + half_ulp_n) >> shift_n.astype(np.uint32)
    mantissa_d = np.minimum(rounded_d.astype(np.int64), max_val_d)
    offset_d = np.full_like(offset_n, MAX_OFFSET)

    offset = np.where(normalized, offset_n, offset_d)
    out_mantissa = np.where(normalized, mantissa_n, mantissa_d)
    out_mantissa = np.where(is_zero, np.int64(0), out_mantissa)
    offset = np.where(is_zero, np.int32(MAX_OFFSET), offset)

    denorm_final = offset == MAX_OFFSET
    recon_exp = np.where(denorm_final, e_star - DENORM_REF_SHIFT, e_star - offset)
    implicit = np.where(denorm_final, np.int64(0), np.int64(1))
    significand = (implicit << width) + out_mantissa

    value = significand.astype(np.float64) * np.exp2((recon_exp - width).astype(np.float64))
    value = np.where(sign == 1, -value, value)
    value = np.where((out_mantissa == 0) & denorm_final, 0.0, value)

    result = value.astype(np.float32).reshape(-1)[:n].reshape(orig_shape)

    if is_torch:
        import torch
        return torch.from_numpy(result).to(device=device, dtype=dtype)
    return result


def compression_ratio(num_elements: int, block_size: int = BLOCK_SIZE) -> float:
    """Matches the C++ AFPTensor::compression_ratio() for a tensor of this
    size (10 bits/value average at block_size=16, i.e. 32/10 = 3.2x)."""
    num_blocks = (num_elements + block_size - 1) // block_size
    packed_bits = num_blocks * (16 + block_size * 9)  # 16-bit header + 9 bits/value
    return (num_elements * 32) / packed_bits if packed_bits else 0.0


if __name__ == "__main__":
    # Quick self-check against hand-computed expectations (mirrors
    # cpp/tests/test_afp.cpp's basic_roundtrip / wide_dynamic_range cases).
    x = np.array([1.0, -1.0, 0.5, -0.25, 3.14159, -2.71828, 100.0, -100.0,
                  0.9, 0.0, 1.5, -1.5, 42.0, -42.0, 7.5, -7.5], dtype=np.float32)
    y = afp_quantize(x)
    for a, b in zip(x, y):
        rel = abs(a - b) / max(abs(a), 1e-30) if a != 0 else abs(b)
        status = "OK" if rel < 0.06 else "FAIL"
        print(f"{status:4s} in={a:>10.5f} out={b:>10.5f} rel_err={rel:.4f}")
    print(f"compression_ratio(16) = {compression_ratio(16):.3f}x (expect 3.20x)")
