"""Evaluate the exported ONNX model on a held-out ImageFolder-style directory,
using exactly the same inference path as the Flask service.

Usage:
    python train/evaluate.py --data-dir /path/to/test [--model-dir models]
"""
import argparse
import collections
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from predictor import OnnxPredictor  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    args = p.parse_args()

    predictor = OnnxPredictor(os.path.join(args.model_dir, "leaf_model.onnx"),
                              os.path.join(args.model_dir, "labels.json"))
    per_class = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    top3_hits = 0

    for label in sorted(os.listdir(args.data_dir)):
        folder = os.path.join(args.data_dir, label)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            with Image.open(os.path.join(folder, name)) as img:
                top, _ = predictor.predict(img)
            per_class[label][1] += 1
            if top[0]["label"] == label:
                per_class[label][0] += 1
            else:
                confusions[(label, top[0]["label"])] += 1
            top3_hits += any(t["label"] == label for t in top)

    correct = sum(c for c, _ in per_class.values())
    total = sum(n for _, n in per_class.values())
    print(f"Top-1 accuracy: {correct / total:.4f} ({correct}/{total})")
    print(f"Top-3 accuracy: {top3_hits / total:.4f}")
    print("\nLowest per-class accuracy:")
    for label, (c, n) in sorted(per_class.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]:
        print(f"  {c / n:6.1%}  {label} ({c}/{n})")
    if confusions:
        print("\nMost common confusions (true -> predicted):")
        for (t, pr), n in confusions.most_common(8):
            print(f"  {n:3d}  {t} -> {pr}")


if __name__ == "__main__":
    main()
