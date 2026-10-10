"""Train a leaf disease classifier with transfer learning and export it to ONNX
for the Flask service.

Real-world accuracy needs real-world photos: train on PlantDoc (field photos)
together with PlantVillage (lab photos), selecting the best epoch on held-out
PlantDoc images. `train/prepare_data.py` builds that layout:

    python train/prepare_data.py --plantdoc PlantDoc-Dataset --plantvillage plantvillage/color --out data
    python train/train.py \\
        --train-dir data/train_pd:2 --train-dir data/train_pv:1:bg \\
        --val-dir data/val_pd --test-dir data/test_pd \\
        --arch timm:efficientnet_b2 --epochs 10

`--train-dir PATH[:WEIGHT][:bg]` may be repeated. WEIGHT sets how often images
from that folder are sampled relative to the others; `bg` pastes the leaf onto
random backgrounds (for lab photos shot on plain backgrounds), which teaches the
model to ignore the background. Each folder has one sub-folder per class
(PlantVillage class names).

The older single-folder form still works: `--data-dir DIR` (20 % held out).
Outputs models/leaf_model.onnx and models/labels.json.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMAGE_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Predictions at or above the threshold should be right at least this often.
TARGET_PRECISION = 0.88


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


# --------------------------------------------------------------------------- data

def scan(folder):
    """[(path, class_name)] for an ImageFolder-style directory."""
    items = []
    for cls in sorted(os.listdir(folder)):
        d = os.path.join(folder, cls)
        if os.path.isdir(d):
            items += [(os.path.join(d, f), cls) for f in sorted(os.listdir(d)) if f.lower().endswith(IMAGE_EXT)]
    return items


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def leaf_mask(im):
    """Leaf vs the plain lab background.

    The background colour is estimated from the image border; pixels whose
    colour (chromaticity) differs from it, or that are strongly saturated, are
    leaf. Shadows are darker but keep the background's chromaticity, so they
    stay background. Holes enclosed by the leaf (shine, dark lesions) are filled.
    """
    from PIL import ImageDraw

    rgb = np.asarray(im, dtype=np.float32)
    border = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
                             rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    chroma = rgb / (rgb.sum(-1, keepdims=True) + 1e-3)
    bg_chroma = bg / (bg.sum() + 1e-3)
    sat = np.asarray(im.convert("HSV"), dtype=np.uint8)[..., 1]
    leaf = (np.linalg.norm(chroma - bg_chroma, axis=-1) > 0.045) | (sat > 90)
    mask = Image.fromarray(leaf.astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # drop speckles
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))  # close thin gaps
    # Fill holes: flood the background from the border; whatever isn't reached is leaf.
    w, h = mask.size
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:
        if mask.getpixel(xy) == 0:
            ImageDraw.floodfill(mask, xy, 128)
    filled = Image.fromarray(((np.asarray(mask) != 128) * 255).astype(np.uint8))
    return filled.filter(ImageFilter.GaussianBlur(1.2))


def random_background(size, pool, rng):
    w, h = size
    kind = rng.random()
    if kind < 0.45 and pool:
        # A real photo blurred beyond recognition: realistic colours, no leaf shapes.
        # Blurring at quarter size looks the same and is much cheaper.
        small = load_rgb(rng.choice(pool)).resize((max(1, w // 4), max(1, h // 4)))
        return small.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.5))).resize((w, h), Image.BILINEAR)
    if kind < 0.75:
        # Soil / foliage / mulch-like texture: one luminance noise field tinted
        # with a natural colour (independent per-channel noise looks like confetti).
        palette = [(110, 80, 55), (140, 110, 80), (70, 55, 40), (90, 120, 60), (60, 90, 45),
                   (150, 140, 120), (190, 180, 160), (120, 130, 140)]
        base = np.array(rng.choice(palette), dtype=np.float32) * rng.uniform(0.7, 1.3)
        gen = np.random.default_rng(rng.randrange(1 << 30))
        cell = rng.choice([4, 8, 16])
        lum = gen.normal(0, rng.uniform(0.1, 0.35), (h // cell + 1, w // cell + 1, 1))
        tint = gen.normal(0, 6, (h // cell + 1, w // cell + 1, 3))
        tex = np.clip(base * (1 + lum) + tint, 0, 255).astype(np.uint8)
        tex = Image.fromarray(tex).resize((w, h), Image.BICUBIC)
        return tex.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))
    # Smooth two-colour gradient (table, wall, sky, hand-held paper...).
    a, b = np.array([rng.uniform(30, 240) for _ in range(3)]), np.array([rng.uniform(30, 240) for _ in range(3)])
    t = np.linspace(0, 1, h)[:, None, None]
    return Image.fromarray(np.broadcast_to(a * (1 - t) + b * t, (h, w, 3)).astype(np.uint8))


def replace_background(im, pool, rng, mask=None):
    mask = mask if mask is not None else leaf_mask(im)
    scale = rng.uniform(0.55, 1.0)  # leaf may fill only part of the new photo
    w, h = im.size
    canvas = random_background((w, h), pool, rng)
    lw, lh = max(1, int(w * scale)), max(1, int(h * scale))
    x, y = rng.randint(0, w - lw), rng.randint(0, h - lh)
    canvas.paste(im.resize((lw, lh)), (x, y), mask.resize((lw, lh)))
    return canvas


class MixedDataset(Dataset):
    def __init__(self, sources, class_to_idx, transform, bg_pool=(), seed=0):
        self.items, self.weights = [], []
        for folder, weight, bg in sources:
            found = scan(folder)
            per_item = weight / max(1, len(found))
            self.items += [(p, class_to_idx[c], bg) for p, c in found]
            self.weights += [per_item] * len(found)
        self.transform, self.bg_pool = transform, list(bg_pool)
        self.rng = random.Random(seed)
        self.masks = {}  # per-worker cache of lab-photo leaf masks

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, bg = self.items[i]
        im = load_rgb(path)
        if bg and self.rng.random() < 0.6:
            if path not in self.masks:
                self.masks[path] = leaf_mask(im).resize((128, 128))
            im = replace_background(im, self.bg_pool, self.rng, self.masks[path].resize(im.size, Image.BILINEAR))
        return self.transform(im), label


def worker_init(worker_id):
    torch.set_num_threads(1)  # the training process already uses every core
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(torch.initial_seed() % (1 << 31) + worker_id)


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.3, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.35, 0.04),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])
# Matches the service's preprocessing (whole image squashed to 224×224).
EVAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ------------------------------------------------------------------ train / eval

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_probs(model, folder, class_to_idx, device, batch_size=64):
    """Softmax probabilities averaged over the image and its mirror image
    (the same test-time augmentation the service uses)."""
    model.eval()
    items = [(p, class_to_idx[c]) for p, c in scan(folder) if c in class_to_idx]
    probs, labels = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        x = torch.stack([EVAL_TF(load_rgb(p)) for p, _ in chunk]).to(device)
        logits = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        probs.append((logits / 2).cpu())
        labels += [y for _, y in chunk]
    return torch.cat(probs), torch.tensor(labels)


def metrics(probs, labels):
    top = probs.topk(min(3, probs.shape[1]), dim=1).indices
    return {
        "top1": round((top[:, 0] == labels).float().mean().item(), 4),
        "top3": round((top == labels[:, None]).any(1).float().mean().item(), 4),
        "n": len(labels),
    }


def calibrate(probs, labels, target=TARGET_PRECISION):
    """Lowest confidence threshold at which accepted predictions are right at
    least `target` of the time; below it the service reports 'not sure'."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    table = []
    for t in [i / 20 for i in range(1, 20)]:
        keep = conf >= t
        if keep.sum() == 0:
            break
        table.append({"threshold": t, "coverage": round(keep.float().mean().item(), 3),
                      "precision": round(correct[keep].mean().item(), 3)})
    chosen = next((r["threshold"] for r in table if r["precision"] >= target), None)
    if chosen is None:
        # Never fall back to a lenient value: use the most precise threshold seen.
        chosen = max(table, key=lambda r: (r["precision"], r["threshold"]))["threshold"] if table else 0.95
        print(f"WARNING: no threshold reaches {target:.0%} precision; using the strictest-performing {chosen}")
    return chosen, table


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


def parse_source(spec):
    """PATH[:WEIGHT][:bg] -> (path, weight, background_replacement)."""
    parts = spec.split(":")
    # Windows drive letters ("C:\\data") contain a colon; rejoin them.
    if len(parts) > 1 and len(parts[0]) == 1 and parts[1].startswith(("\\", "/")):
        parts = [parts[0] + ":" + parts[1]] + parts[2:]
    folder, flags = parts[0], [p for p in parts[1:] if p]
    weight = 1.0
    for f in flags:
        try:
            weight = float(f)
        except ValueError:
            pass
    return folder, weight, "bg" in flags


def split_single_dir(data_dir, out_root, val_split):
    """Back-compat for --data-dir: hold out val_split of each class, using links
    (or copies where links aren't possible) in a temporary folder."""
    import shutil

    rng = random.Random(42)
    for path, cls in scan(data_dir):
        split = "val" if rng.random() < val_split else "train"
        d = os.path.join(out_root, split, cls)
        os.makedirs(d, exist_ok=True)
        target = os.path.join(d, os.path.basename(path))
        try:
            os.link(path, target)
        except OSError:
            shutil.copy(path, target)
    return os.path.join(out_root, "train"), os.path.join(out_root, "val")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", action="append", default=[], help="PATH[:WEIGHT][:bg], repeatable")
    p.add_argument("--val-dir", action="append", default=[], help="selects the best epoch (first one) + calibration")
    p.add_argument("--test-dir", action="append", default=[], help="reported only, never used for selection")
    p.add_argument("--data-dir", help="single ImageFolder dir (older form); 20%% is held out for validation")
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    p.add_argument("--arch", default="mobilenet_v3", help="mobilenet_v3 | efficientnet_b0 | resnet50 | timm:<model_name>")
    p.add_argument("--weights", help="Local pretrained weights file (skips downloading)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--epoch-size", type=int, default=0, help="images sampled per epoch (default: all)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.data_dir:
        import atexit
        import shutil
        import tempfile

        split_root = tempfile.mkdtemp(prefix="leafcare-split-")
        atexit.register(shutil.rmtree, split_root, True)
        tr, va = split_single_dir(args.data_dir, split_root, args.val_split)
        args.train_dir, args.val_dir = [tr], [va]
    if not args.train_dir or not args.val_dir:
        p.error("give --train-dir and --val-dir (or --data-dir)")

    sources = [parse_source(s) for s in args.train_dir]
    classes = sorted({c for folder, _, _ in sources for _, c in scan(folder)})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    bg_pool = [path for folder, _, bg in sources if not bg for path, _ in scan(folder)]
    train_ds = MixedDataset(sources, class_to_idx, TRAIN_TF, bg_pool)
    epoch_size = args.epoch_size or len(train_ds)
    sampler = WeightedRandomSampler(train_ds.weights, epoch_size, replacement=True)
    batch_size = min(args.batch_size, epoch_size)  # tiny datasets: one batch per epoch
    train_dl = DataLoader(train_ds, batch_size, sampler=sampler, num_workers=args.workers,
                          worker_init_fn=worker_init if args.workers > 0 else None,
                          persistent_workers=args.workers > 0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{len(classes)} classes; {len(train_ds)} training images, {epoch_size} sampled per epoch; device {device}")
    for folder, weight, bg in sources:
        print(f"  {folder}  weight={weight}{'  background-replacement' if bg else ''}")

    model = build_model(args.arch, len(classes), args.weights).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, args.epochs * (epoch_size // batch_size))
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    ckpt = os.path.join(args.out_dir, "best.pt")
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        val = {os.path.basename(os.path.normpath(d)): metrics(*predict_probs(model, d, class_to_idx, device))
               for d in args.val_dir}
        score = next(iter(val.values()))["top1"]
        history.append({"epoch": epoch, "train_acc": round(tr_acc, 4), **val})
        print(f"epoch {epoch}/{args.epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  "
              + "  ".join(f"{k} top1 {v['top1']:.3f} top3 {v['top3']:.3f}" for k, v in val.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    threshold, table = calibrate(*predict_probs(model, args.val_dir[0], class_to_idx, device))
    report = {name: metrics(*predict_probs(model, d, class_to_idx, device))
              for name, d in [(os.path.basename(os.path.normpath(d)), d) for d in args.val_dir + args.test_dir]}
    for name, m in report.items():
        print(f"best model on {name}: top1 {m['top1']:.3f}  top3 {m['top3']:.3f}  (n={m['n']})")
    print(f"confidence threshold {threshold} (≥{TARGET_PRECISION:.0%} precision on {args.val_dir[0]})")

    model.eval().cpu()
    onnx_path = export_onnx(model, args.out_dir)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({
            "labels": classes, "image_size": IMAGE_SIZE, "arch": args.arch,
            "confidence_threshold": threshold, "calibration": table,
            "metrics": report, "history": history,
            "train_dirs": [f"{os.path.basename(os.path.normpath(d))}:{w}{':bg' if bg else ''}" for d, w, bg in sources],
        }, f, indent=2)
    print(f"Exported {onnx_path}")


if __name__ == "__main__":
    main()
