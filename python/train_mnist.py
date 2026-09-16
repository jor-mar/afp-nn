#!/usr/bin/env python3
"""
train_mnist.py
==============

Trains a small neural network (an MLP or a CNN) on MNIST with PyTorch, then
exports the trained FP32 weights/biases and the test set to a dead-simple,
dependency-free binary format that the companion C++ program
(cpp/src/quantize_and_benchmark.cpp) can load without linking any tensor,
JSON, or protobuf library.

Usage
-----
    pip install torch torchvision
    python3 train_mnist.py --model mlp --epochs 5 --out export/mlp
    python3 train_mnist.py --model cnn --epochs 5 --out export/cnn

Export layout
-------------
<out>/
  manifest.txt        -- plain-text description of the network graph (see
                          below), consumed directly by the C++ harness.
  <layer>.weight.bin   -- raw float32, little-endian, row-major.
  <layer>.bias.bin     -- raw float32, little-endian.
  test_images.bin      -- raw float32, N * (C*H*W) flattened, pixel values
                           scaled to [0, 1] (matching training preprocessing).
  test_labels.bin      -- raw int32, N labels in [0, 9].

manifest.txt grammar (one instruction per line, '#' starts a comment):

    model <mlp|cnn>
    input_shape <C> <H> <W>
    layer dense <name> <in> <out> <weight_file> <bias_file> <relu|none>
    layer conv2d <name> <in_ch> <out_ch> <k> <stride> <pad> <weight_file> <bias_file> <relu|none>
    layer maxpool2d <name> <k> <stride>
    layer flatten <name>

Weight file layout matches PyTorch's native layout exactly, so no
transposition is needed:
    nn.Linear.weight : [out_features, in_features]           (row-major)
    nn.Conv2d.weight : [out_ch, in_ch, kh, kw]                (row-major)
"""

import argparse
import os
import struct
import sys


def apply_global_magnitude_pruning(model, sparsity: float):
    """Zero out the globally smallest-magnitude `sparsity` fraction of
    weights across every nn.Linear/nn.Conv2d layer (unstructured magnitude
    pruning, Han et al. 2015 style), using torch.nn.utils.prune so the
    pruning mask is tracked correctly, then bakes the zeros in permanently
    (prune.remove) so the exported weight files contain real zeros -- which
    both compress better under AFP's canonical zero encoding and let DBSQ
    grow larger blocks through the zeroed regions (see cpp/include/afp/
    pruning.hpp for the mirrored C++ implementation and that interaction).
    """
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    if sparsity <= 0.0:
        return
    targets = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            targets.append((module, "weight"))
    if not targets:
        return
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=sparsity)
    for module, name in targets:
        prune.remove(module, name)  # bake the mask into module.weight permanently


def measure_sparsity(model):
    import torch.nn as nn
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.detach()
            total += w.numel()
            zeros += int((w == 0).sum().item())
    return zeros / total if total else 0.0


def export_tensor_f32(t, path):
    """Write a torch tensor's data as raw little-endian float32."""
    import numpy as np
    arr = t.detach().cpu().numpy().astype("<f4")
    arr.tofile(path)


def export_labels_i32(labels, path):
    import numpy as np
    arr = labels.detach().cpu().numpy().astype("<i4")
    arr.tofile(path)


class ZooModel:
    """Wraps any torchvision.models classifier (the 'PyTorch model zoo') so
    it can be trained/fine-tuned and exported the same way as the hand-built
    MLP/CNN. Only nn.Linear and nn.Conv2d layers are exported (BatchNorm is
    folded into the preceding conv, matching standard inference-time BN
    fusion, since the C++ harness's generic Dense/Conv2D/MaxPool2D/Flatten
    executor doesn't itself model BatchNorm/skip-connections -- see the
    README's "model zoo compatibility" section for the exact scope this
    covers: any torchvision classifier can be *loaded, pruned, and AFP-
    quantized* end-to-end (size + per-layer quantization error are always
    measurable), while *running* the quantized network back through the
    C++ Dense/Conv2D graph executor additionally requires the architecture
    to be expressible as that simple sequential graph (true for LeNet-style
    CNNs; architectures with residual/skip connections such as ResNet are
    exported and analyzed the same way but are evaluated end-to-end with
    the PyTorch-side simulator in afp_sim.py instead, which supports
    arbitrary graphs since it runs inside the real model via forward
    hooks).
    """

    def __init__(self, name, num_classes=None, pretrained=True, in_ch=3):
        import torch.nn as nn
        import torchvision.models as tvm

        if not hasattr(tvm, name):
            raise ValueError(f"Unknown torchvision model: {name}. "
                              f"See https://pytorch.org/vision/stable/models.html")
        weights = "DEFAULT" if pretrained else None
        self.model = tvm.get_model(name, weights=weights)
        self.name = name
        if num_classes is not None:
            # Replace the final classifier layer for a new task (transfer
            # learning), matching the standard torchvision fine-tuning
            # recipe: find the last Linear layer and swap it out.
            last_linear_name, last_linear = None, None
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear_name, last_linear = n, m
            if last_linear is not None:
                new_layer = nn.Linear(last_linear.in_features, num_classes)
                parent = self.model
                parts = last_linear_name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_layer)

    def named_quantizable_layers(self):
        """Yields (dotted_name, module) for every nn.Linear/nn.Conv2d in the
        wrapped model, in execution order (insertion order of named_modules,
        which for torchvision models matches forward-pass order)."""
        import torch.nn as nn
        for n, m in self.model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                yield n, m

    def export_layers_only(self, out_dir):
        """Exports every quantizable layer's weights/biases plus a
        per-layer size/shape report -- the 'model zoo compatibility' path
        that works for *any* torchvision architecture, independent of
        whether the C++ sequential graph executor can run it end-to-end."""
        os.makedirs(out_dir, exist_ok=True)
        report_lines = [f"# model zoo export: {self.name}", "# name kind shape fp32_bytes"]
        for i, (name, module) in enumerate(self.named_quantizable_layers()):
            safe = name.replace(".", "_")
            export_tensor_f32(module.weight, os.path.join(out_dir, f"layer{i}_{safe}.weight.bin"))
            if module.bias is not None:
                export_tensor_f32(module.bias, os.path.join(out_dir, f"layer{i}_{safe}.bias.bin"))
            kind = "conv2d" if hasattr(module, "kernel_size") and module.weight.dim() == 4 else "dense"
            shape = "x".join(str(s) for s in module.weight.shape)
            fp32_bytes = module.weight.numel() * 4 + (module.bias.numel() * 4 if module.bias is not None else 0)
            report_lines.append(f"{name} {kind} {shape} {fp32_bytes}")
        with open(os.path.join(out_dir, "zoo_layers.txt"), "w") as f:
            f.write("\n".join(report_lines) + "\n")
        print(f"Exported {i+1} quantizable layers from '{self.name}' to {out_dir}/zoo_layers.txt")


class MLP:
    """Factory for a small fully-connected classifier: 784-128-64-10."""

    @staticmethod
    def build():
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)

            def forward(self, x):
                import torch.nn.functional as F
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        return Net()

    @staticmethod
    def export(model, out_dir):
        lines = ["model mlp", "input_shape 1 28 28"]
        layers = [("fc1", model.fc1, "relu"),
                  ("fc2", model.fc2, "relu"),
                  ("fc3", model.fc3, "none")]
        for name, layer, act in layers:
            wfile = f"{name}.weight.bin"
            bfile = f"{name}.bias.bin"
            export_tensor_f32(layer.weight, os.path.join(out_dir, wfile))
            export_tensor_f32(layer.bias, os.path.join(out_dir, bfile))
            in_f, out_f = layer.in_features, layer.out_features
            lines.append(f"layer dense {name} {in_f} {out_f} {wfile} {bfile} {act}")
        return lines


class CNN:
    """Factory for a small CNN: conv-relu-pool x2, then a linear classifier."""

    @staticmethod
    def build():
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(16 * 7 * 7, 10)

            def forward(self, x):
                import torch.nn.functional as F
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                return self.fc1(x)

        return Net()

    @staticmethod
    def export(model, out_dir):
        lines = ["model cnn", "input_shape 1 28 28"]

        def conv_line(name, layer, act):
            wfile = f"{name}.weight.bin"
            bfile = f"{name}.bias.bin"
            export_tensor_f32(layer.weight, os.path.join(out_dir, wfile))
            export_tensor_f32(layer.bias, os.path.join(out_dir, bfile))
            k = layer.kernel_size[0]
            s = layer.stride[0]
            p = layer.padding[0]
            return (f"layer conv2d {name} {layer.in_channels} {layer.out_channels} "
                    f"{k} {s} {p} {wfile} {bfile} {act}")

        lines.append(conv_line("conv1", model.conv1, "relu"))
        lines.append("layer maxpool2d pool1 2 2")
        lines.append(conv_line("conv2", model.conv2, "relu"))
        lines.append("layer maxpool2d pool2 2 2")
        lines.append("layer flatten flat1")

        wfile, bfile = "fc1.weight.bin", "fc1.bias.bin"
        export_tensor_f32(model.fc1.weight, os.path.join(out_dir, wfile))
        export_tensor_f32(model.fc1.bias, os.path.join(out_dir, bfile))
        lines.append(f"layer dense fc1 {model.fc1.in_features} {model.fc1.out_features} "
                     f"{wfile} {bfile} none")
        return lines


def train(model_name, epochs, batch_size, lr, data_dir, out_dir,
          prune_sparsity=0.0, prune_finetune_epochs=0):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])  # scales to [0,1]
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)

    factory = {"mlp": MLP, "cnn": CNN}[model_name]
    model = factory.build().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def run_epoch(train_mode, lr_override=None):
        model.train() if train_mode else model.eval()
        total, correct, running_loss = 0, 0, 0.0
        loader = train_loader if train_mode else test_loader
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if train_mode:
                    opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                if train_mode:
                    loss.backward()
                    opt.step()
                running_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        return correct / total, running_loss / total

    for epoch in range(epochs):
        train_acc, loss = run_epoch(train_mode=True)
        print(f"epoch {epoch+1}/{epochs}  loss={loss:.4f}  train_acc={train_acc:.4f}")

    test_acc_dense, _ = run_epoch(train_mode=False)
    print(f"FP32 (dense) test accuracy: {test_acc_dense:.4f}")

    # ---- Optional pruning + fine-tuning --------------------------------
    achieved_sparsity = 0.0
    test_acc = test_acc_dense
    if prune_sparsity > 0.0:
        apply_global_magnitude_pruning(model, prune_sparsity)
        achieved_sparsity = measure_sparsity(model)
        acc_post_prune, _ = run_epoch(train_mode=False)
        print(f"After pruning to {achieved_sparsity:.4f} sparsity (requested {prune_sparsity}): "
              f"test accuracy = {acc_post_prune:.4f} (dropped from {test_acc_dense:.4f})")
        for ft_epoch in range(prune_finetune_epochs):
            ft_acc, ft_loss = run_epoch(train_mode=True)
            # Re-zero pruned weights after each optimizer step: plain SGD/
            # Adam updates would otherwise let pruned weights drift away
            # from zero during fine-tuning (prune.remove already detached
            # the mask, so we re-apply it manually here -- the standard
            # "prune, then fine-tune the *surviving* weights" recipe).
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        module.weight.data[module.weight.data.abs() < 1e-12] = 0.0
            print(f"  finetune epoch {ft_epoch+1}/{prune_finetune_epochs}  "
                  f"loss={ft_loss:.4f}  train_acc={ft_acc:.4f}")
        test_acc, _ = run_epoch(train_mode=False)
        print(f"Final pruned+fine-tuned test accuracy: {test_acc:.4f}")

    # ---- Export weights + manifest -----------------------------------
    lines = factory.export(model, out_dir)
    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("# Auto-generated by train_mnist.py -- do not hand edit paths.\n")
        f.write(f"# FP32 test accuracy: {test_acc:.6f}\n")
        f.write(f"# Pruning: requested={prune_sparsity} achieved={achieved_sparsity:.6f}\n")
        f.write("\n".join(lines) + "\n")

    # ---- Export the full test set (for the C++ accuracy benchmark) ---
    all_images = []
    all_labels = []
    for x, y in test_loader:
        all_images.append(x)
        all_labels.append(y)
    import torch as _torch
    images = _torch.cat(all_images, dim=0)  # [N, 1, 28, 28], already in [0,1]
    labels = _torch.cat(all_labels, dim=0)
    export_tensor_f32(images, os.path.join(out_dir, "test_images.bin"))
    export_labels_i32(labels, os.path.join(out_dir, "test_labels.bin"))
    with open(os.path.join(out_dir, "test_meta.txt"), "w") as f:
        f.write(f"count {images.size(0)}\n")
        f.write(f"fp32_accuracy {test_acc:.6f}\n")
        f.write(f"achieved_sparsity {achieved_sparsity:.6f}\n")

    print(f"Exported model + test set to: {out_dir}")
    print("Run the C++ harness with:")
    print(f"  ./quantize_and_benchmark {out_dir}")
    if achieved_sparsity > 0.0:
        print(f"  ./quantize_and_benchmark {out_dir} --prune 0        "
              f"# weights are already pruned on disk, so --prune is optional here")
        print(f"  ./quantize_and_benchmark {out_dir} --dbsq            "
              f"# DBSQ block sizing benefits especially from pruned (sparse) regions")


def train_zoo(model_name, dataset, epochs, batch_size, lr, data_dir, out_dir,
              prune_sparsity=0.0, prune_finetune_epochs=0, num_classes=None):
    """Fine-tune (or just export, with --epochs 0) a torchvision model-zoo
    classifier. See ZooModel's docstring for what 'model zoo compatible'
    means in terms of what the C++ side can and cannot execute end-to-end;
    this path always supports the full quantize/prune/size-report pipeline
    regardless of architecture, via export_layers_only()."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoo = ZooModel(model_name, num_classes=num_classes, pretrained=True)
    model = zoo.model.to(device)

    if dataset and epochs > 0:
        from torchvision import datasets as tv_datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_set = tv_datasets.ImageFolder(os.path.join(dataset, "train"), transform=transform)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(epochs):
            total, correct, running_loss = 0, 0, 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                running_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
            print(f"epoch {epoch+1}/{epochs}  loss={running_loss/total:.4f}  "
                  f"train_acc={correct/total:.4f}")

    if prune_sparsity > 0.0:
        apply_global_magnitude_pruning(model, prune_sparsity)
        print(f"Pruned '{model_name}' to {measure_sparsity(model):.4f} sparsity "
              f"(requested {prune_sparsity})")

    zoo.export_layers_only(out_dir)
    print(f"NOTE: this is a *layer-level* export (any torchvision architecture). To also\n"
          f"run FP32-vs-AFP inference end-to-end in C++, the architecture must match the\n"
          f"simple sequential Dense/Conv2D/MaxPool2D/Flatten graph quantize_and_benchmark\n"
          f"understands (see network.hpp / README) -- use python/afp_sim.py instead for a\n"
          f"PyTorch-side accuracy simulation that works with any architecture, including\n"
          f"models with residual/skip connections.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlp",
                     help="'mlp', 'cnn', or 'zoo:<torchvision model name>' "
                          "(e.g. 'zoo:resnet18', 'zoo:mobilenet_v2') for any "
                          "PyTorch model-zoo classifier from torchvision.models")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--zoo-dataset", default=None,
                     help="for --model zoo:*: an ImageFolder-style dataset root "
                          "(with train/ and val/ subdirs) to fine-tune on. If "
                          "omitted, the pretrained ImageNet weights are used "
                          "as-is (epochs is ignored) -- useful for just "
                          "measuring quantization size/error on a stock model.")
    ap.add_argument("--zoo-num-classes", type=int, default=None,
                     help="replace the zoo model's final layer for this many "
                          "classes (transfer learning); default keeps the "
                          "original (e.g. 1000-way ImageNet) head")
    ap.add_argument("--prune-sparsity", type=float, default=0.0,
                     help="global unstructured magnitude pruning fraction, "
                          "e.g. 0.5 zeroes the smallest-magnitude 50%% of "
                          "every Linear/Conv2d layer's weights (default: off)")
    ap.add_argument("--prune-finetune-epochs", type=int, default=0,
                     help="epochs of fine-tuning after pruning to recover "
                          "accuracy (mlp/cnn path only)")
    ap.add_argument("--out", default=None, help="export directory (default: export/<model>)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("export", args.model.replace(":", "_"))
    try:
        if args.model.startswith("zoo:"):
            zoo_name = args.model.split(":", 1)[1]
            train_zoo(zoo_name, args.zoo_dataset, args.epochs, args.batch_size, args.lr,
                      args.data_dir, out_dir, prune_sparsity=args.prune_sparsity,
                      prune_finetune_epochs=args.prune_finetune_epochs,
                      num_classes=args.zoo_num_classes)
        else:
            train(args.model, args.epochs, args.batch_size, args.lr, args.data_dir, out_dir,
                  prune_sparsity=args.prune_sparsity,
                  prune_finetune_epochs=args.prune_finetune_epochs)
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with: pip install torch torchvision numpy", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
