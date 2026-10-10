"""Evaluate the exported ONNX model on a held-out ImageFolder-style directory,
using exactly the same inference path as the Flask service (EXIF rotation,
test-time augmentation, calibrated "not sure" threshold).

Usage:
    python train/evaluate.py --data-dir /path/to/test [--model-dir models]

Reports accuracy without and with the crop hint (what users get when they pick
the crop), how often the right crop is identified, and how the confidence
threshold splits predictions into "confident" and "not sure".
"""
import argparse
import collections
import os
import sys

from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from predictor import CROP_SUPPORT_MIN, OnnxPredictor, _crop_key  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    args = p.parse_args()

    predictor = OnnxPredictor(os.path.join(args.model_dir, "leaf_model.onnx"),
                              os.path.join(args.model_dir, "labels.json"))
    known = set(predictor.labels)
    per_class = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    n = top1 = top3 = crop_ok = hinted = 0
    confident = confident_ok = hint_confident = hint_confident_ok = 0

    for label in sorted(os.listdir(args.data_dir)):
        folder = os.path.join(args.data_dir, label)
        if not os.path.isdir(folder) or label not in known:
            continue
        crop = _crop_key(label)
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(IMAGE_EXT):
                continue
            with Image.open(os.path.join(folder, name)) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            top, _ = predictor.predict(img)
            hint_top, _ = predictor.predict(img, crop=crop)
            n += 1
            per_class[label][1] += 1
            right = top[0]["label"] == label
            if right:
                top1 += 1
                per_class[label][0] += 1
            else:
                confusions[(label, top[0]["label"])] += 1
            top3 += any(t["label"] == label for t in top)
            crop_ok += _crop_key(top[0]["label"]) == crop
            hinted += hint_top[0]["label"] == label
            if top[0]["confidence"] >= predictor.threshold:
                confident += 1
                confident_ok += right
            # Same rule as the service when a crop is selected.
            if hint_top[0]["confidence"] >= predictor.threshold and hint_top[0]["crop_support"] >= CROP_SUPPORT_MIN:
                hint_confident += 1
                hint_confident_ok += hint_top[0]["label"] == label

    print(f"{n} images")
    print(f"Top-1 accuracy:            {top1 / n:.3f}")
    print(f"Top-3 accuracy:            {top3 / n:.3f}")
    print(f"Crop identified correctly: {crop_ok / n:.3f}")
    print(f"Top-1 with crop selected:  {hinted / n:.3f}")
    print(f"Confidence threshold {predictor.threshold}: {confident / n:.1%} of photos get a confident answer, "
          f"{(confident_ok / confident if confident else 0):.1%} of those are right; "
          f"{1 - confident / n:.1%} are reported as 'not sure'")
    print(f"  with the crop selected: {hint_confident / n:.1%} confident, "
          f"{(hint_confident_ok / hint_confident if hint_confident else 0):.1%} of those right")
    print("\nLowest per-class accuracy:")
    for label, (c, k) in sorted(per_class.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]:
        print(f"  {c / k:6.1%}  {label} ({c}/{k})")
    if confusions:
        print("\nMost common confusions (true -> predicted):")
        for (t, pr), k in confusions.most_common(8):
            print(f"  {k:3d}  {t} -> {pr}")


if __name__ == "__main__":
    main()
