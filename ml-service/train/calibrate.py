"""(Re)calibrate the "not sure" confidence threshold of an exported model, using
the same inference path as the Flask service, and store it in labels.json.

Usage:
    python train/calibrate.py --val-dir data/val_pd [--model-dir models] [--target 0.88]

Picks the lowest threshold at which confident answers are right at least
`--target` of the time on the validation photos. Use photos the model was not
trained on (and not the final test set).
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from predictor import CROP_SUPPORT_MIN, OnnxPredictor, _crop_key  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", required=True)
    p.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    p.add_argument("--target", type=float, default=0.88)
    args = p.parse_args()

    labels_path = os.path.join(args.model_dir, "labels.json")
    predictor = OnnxPredictor(os.path.join(args.model_dir, "leaf_model.onnx"), labels_path)
    rows = []  # (confidence, correct, hinted confidence, hinted correct, crop support)
    for label in sorted(os.listdir(args.val_dir)):
        folder = os.path.join(args.val_dir, label)
        if not os.path.isdir(folder) or label not in predictor.labels:
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(IMAGE_EXT):
                continue
            with Image.open(os.path.join(folder, name)) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
            top, _ = predictor.predict(img)
            hint, _ = predictor.predict(img, crop=_crop_key(label))
            rows.append((top[0]["confidence"], top[0]["label"] == label,
                         hint[0]["confidence"], hint[0]["label"] == label, hint[0]["crop_support"]))

    def stats(t, hinted=False):
        if hinted:
            keep = [r for r in rows if r[2] >= t and r[4] >= CROP_SUPPORT_MIN]
            ok = sum(r[3] for r in keep)
        else:
            keep = [r for r in rows if r[0] >= t]
            ok = sum(r[1] for r in keep)
        return len(keep) / len(rows), (ok / len(keep) if keep else 0.0)

    table = []
    for t in [i / 20 for i in range(1, 20)]:
        cov, prec = stats(t)
        hcov, hprec = stats(t, hinted=True)
        table.append({"threshold": t, "coverage": round(cov, 3), "precision": round(prec, 3),
                      "coverage_with_crop": round(hcov, 3), "precision_with_crop": round(hprec, 3)})
    reaching = [r for r in table if r["precision"] >= args.target and r["coverage"] > 0]
    if reaching:
        chosen = reaching[0]["threshold"]
    else:
        chosen = max(table, key=lambda r: (r["precision"], r["threshold"]))["threshold"]
        print(f"WARNING: no threshold reaches {args.target:.0%} precision; using {chosen}")

    with open(labels_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta.update({"confidence_threshold": chosen, "calibration": table,
                 "calibration_set": os.path.basename(os.path.normpath(args.val_dir)),
                 "calibration_target": args.target})
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    row = next(r for r in table if r["threshold"] == chosen)
    print(f"{len(rows)} validation photos. Threshold {chosen}: {row['coverage']:.0%} get a confident answer "
          f"({row['precision']:.0%} right); with the crop selected {row['coverage_with_crop']:.0%} "
          f"({row['precision_with_crop']:.0%} right). Saved to {labels_path}")


if __name__ == "__main__":
    main()
