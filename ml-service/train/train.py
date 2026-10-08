"""Train a leaf disease classifier on PlantVillage (or any ImageFolder dataset)
with transfer learning, then export it to ONNX for the Flask service.

Dataset layout (one folder per class, e.g. the PlantVillage "color" split):
    data_dir/
        Apple___Apple_scab/*.jpg
        Apple___healthy/*.jpg
        ...

Usage:
    pip install -r train/requirements.txt
    python train/train.py --data-dir /path/to/plantvillage --epochs 5
Outputs models/leaf_model.onnx and models/labels.json.
"""
import argparse
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

IMAGE_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch, num_classes, weights_path=None):
    """Build a classifier with ImageNet-pretrained weights.

    `weights_path` loads backbone weights from a local file instead of downloading
    them (useful offline or behind a firewall). `arch` may also be
    `timm:<model_name>` to use any timm model (requires `pip install timm`).
    """
    if arch.startswith("timm:"):
        import timm

        model = timm.create_model(arch[5:], pretrained=weights_path is None, num_classes=num_classes)
        if weights_path:
            _load_backbone(model, weights_path)
        return model
    model = _torchvision_model(arch, num_classes, pretrained=weights_path is None)
    if weights_path:
        _load_backbone(model, weights_path)
    return model


def _load_backbone(model, weights_path):
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("state_dict", state)
    own = model.state_dict()
    # Skip the ImageNet classifier head (shape mismatch with our class count).
    kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
    print(f"Loaded {len(kept)}/{len(own)} tensors from {weights_path}")


def _torchvision_model(arch, num_classes, pretrained=True):
    if arch == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch {arch}")
    return model


def loaders(data_dir, batch_size, val_split, workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    base = datasets.ImageFolder(data_dir)
    n_val = int(len(base) * val_split)
    gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(base)), [len(base) - n_val, n_val], generator=gen)
    train_ds = torch.utils.data.Subset(datasets.ImageFolder(data_dir, train_tf), train_idx.indices)
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(data_dir, val_tf), val_idx.indices)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        base.classes,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def export_onnx(model, out_dir):
    """Export to a single self-contained ONNX file (weights embedded)."""
    import onnx

    onnx_path = os.path.join(out_dir, "leaf_model.onnx")
    torch.onnx.export(
        model, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE), onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18,
    )
    # Newer exporters may write weights to a side-car .data file; fold them back in.
    onnx.save(onnx.load(onnx_path), onnx_path, save_as_external_data=False)
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    return onnx_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    p.add_argument("--arch", default="mobilenet_v3",
                   help="mobilenet_v3 | efficientnet_b0 | resnet50 | timm:<model_name>")
    p.add_argument("--weights", help="Local pretrained weights file (skips downloading)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_dl, val_dl, classes = loaders(args.data_dir, args.batch_size, args.val_split, args.workers)
    print(f"{len(classes)} classes, {len(train_dl.dataset)} train / {len(val_dl.dataset)} val images on {device}")

    model = build_model(args.arch, len(classes), args.weights).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "best.pt")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        print(f"epoch {epoch}/{args.epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  "
              f"val {va_loss:.3f}/{va_acc:.3f}  ({time.time() - t0:.0f}s)")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().cpu()
    onnx_path = export_onnx(model, args.out_dir)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": classes, "image_size": IMAGE_SIZE, "arch": args.arch,
                   "val_accuracy": round(best_acc, 4)}, f, indent=2)
    print(f"Best val accuracy {best_acc:.4f}. Exported {onnx_path}")


if __name__ == "__main__":
    main()
