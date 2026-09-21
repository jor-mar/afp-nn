#!/usr/bin/env python3
"""
quantize_model_zoo.py
======================

Loads *any* torchvision model-zoo classifier, optionally prunes it, then
quantizes both its weights and its inter-layer activations onto the AFP
grid (via afp_sim.py, cross-validated bit-exact against the C++ codec --
see the header comment in afp_sim.py) using PyTorch forward hooks. Because
this runs inside the real model, it works for any architecture, including
ones with residual/skip connections that cpp/src/quantize_and_benchmark.cpp
cannot execute (that C++ tool only understands simple sequential
Dense/Conv2D/MaxPool2D/Flatten graphs, e.g. the MLP/CNN train_mnist.py
produces).

Usage
-----
    pip install torch torchvision numpy pillow

    # Weight/size analysis only (no accuracy numbers, no dataset needed):
    python3 quantize_model_zoo.py --model resnet18

    # With pruning:
    python3 quantize_model_zoo.py --model resnet18 --prune-sparsity 0.3

    # Full FP32-vs-AFP accuracy comparison on a labeled image folder
    # (ImageFolder layout: <dataset>/<class_name>/*.jpg):
    python3 quantize_model_zoo.py --model resnet18 --dataset /path/to/val --limit 500

Any name accepted by `torchvision.models.get_model` works, e.g. resnet18,
resnet50, mobilenet_v2, mobilenet_v3_small, efficientnet_b0, vgg16,
densenet121, vit_b_16, ...
"""

import argparse
import sys
import time

from python.afp_sim import afp_quantize, compression_ratio


def analyze_and_quantize(model_name, prune_sparsity=0.0):
    import torch.nn as nn
    import torchvision.models as tvm

    if not hasattr(tvm, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")
    model = tvm.get_model(model_name, weights="DEFAULT")
    model.eval()

    if prune_sparsity > 0.0:
        from python.train_mnist import apply_global_magnitude_pruning, measure_sparsity
        apply_global_magnitude_pruning(model, prune_sparsity)
        print(f"Pruned to sparsity={measure_sparsity(model):.4f} (requested {prune_sparsity})")

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    print(f"\n{model_name}: {len(layers)} quantizable layers "
          f"({sum(m.weight.numel() for _, m in layers):,} weight elements)")

    total_fp32_bytes, total_afp_bytes = 0, 0
    print(f"{'layer':40s} {'shape':>20s} {'fp32 KB':>10s} {'afp KB':>10s} {'ratio':>7s}")
    for name, module in layers:
        n = module.weight.numel()
        fp32_bytes = n * 4
        ratio = compression_ratio(n)
        afp_bytes = fp32_bytes / ratio
        total_fp32_bytes += fp32_bytes
        total_afp_bytes += afp_bytes
        shape = "x".join(str(s) for s in module.weight.shape)
        print(f"{name:40s} {shape:>20s} {fp32_bytes/1024:10.2f} {afp_bytes/1024:10.2f} {ratio:6.2f}x")

    print(f"\n{'TOTAL':40s} {'':>20s} {total_fp32_bytes/1024:10.2f} {total_afp_bytes/1024:10.2f} "
          f"{total_fp32_bytes/total_afp_bytes:6.2f}x")
    return model, layers


def quantize_weights_inplace(model):
    """Replaces every Linear/Conv2d weight with its AFP round-trip (in
    place). Returns the original FP32 weights so the caller can restore
    them afterward (to also measure the FP32 baseline with the same
    model instance)."""
    import torch.nn as nn
    originals = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            originals[name] = module.weight.data.clone()
            module.weight.data = afp_quantize(module.weight.data)
    return originals


def restore_weights(model, originals):
    import torch.nn as nn
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = originals[name]


def install_activation_quant_hooks(model):
    """Registers forward-pre-hooks on every Linear/Conv2d that snap the
    *input* activation to the AFP grid before the layer runs -- the
    "activations are rounded between every layer" step from the AFP
    paper's own evaluation methodology (see network.hpp's docstring for
    the same design applied to the C++ harness)."""
    import torch.nn as nn

    handles = []

    def hook(module, inputs):
        x = inputs[0]
        return (afp_quantize(x),) + tuple(inputs[1:])

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_pre_hook(hook))
    return handles


def evaluate(model, dataset_dir, limit, batch_size=32):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(dataset_dir, transform=transform)
    if limit and limit < len(ds):
        ds = torch.utils.data.Subset(ds, list(range(limit)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    elapsed = time.time() - t0
    return correct / total if total else 0.0, elapsed, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="torchvision.models name, e.g. resnet18")
    ap.add_argument("--dataset", default=None, help="ImageFolder root for accuracy comparison")
    ap.add_argument("--limit", type=int, default=500, help="max images to evaluate")
    ap.add_argument("--prune-sparsity", type=float, default=0.0)
    args = ap.parse_args()

    try:
        model, layers = analyze_and_quantize(args.model, args.prune_sparsity)
    except ImportError as e:
        print(f"Missing dependency: {e}\nInstall with: pip install torch torchvision numpy pillow",
              file=sys.stderr)
        sys.exit(1)

    if not args.dataset:
        print("\nNo --dataset given: skipping accuracy comparison (size/compression report only).")
        print("Pass --dataset <ImageFolder root> to also compare FP32 vs AFP top-1 accuracy.")
        return

    print(f"\nEvaluating FP32 baseline on up to {args.limit} images from {args.dataset} ...")
    acc_fp32, t_fp32, n = evaluate(model, args.dataset, args.limit)
    print(f"FP32 accuracy: {acc_fp32:.4f}  ({n} images, {t_fp32:.2f}s, {t_fp32/max(n,1)*1000:.2f} ms/img)")

    originals = quantize_weights_inplace(model)
    hook_handles = install_activation_quant_hooks(model)
    print(f"\nEvaluating AFP-simulated model on the same {n} images ...")
    acc_afp, t_afp, _ = evaluate(model, args.dataset, args.limit)
    print(f"AFP  accuracy: {acc_afp:.4f}  ({n} images, {t_afp:.2f}s, {t_afp/max(n,1)*1000:.2f} ms/img)")
    print(f"AFP / FP32 accuracy ratio: {acc_afp/acc_fp32 if acc_fp32 else 0:.4f}")

    for h in hook_handles:
        h.remove()
    restore_weights(model, originals)


if __name__ == "__main__":
    main()
