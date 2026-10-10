"""Leaf disease classifiers.

`OnnxPredictor` runs a CNN trained with `train/train.py` (exported to ONNX).
`HeuristicPredictor` is a colour-feature fallback so the system is usable
before a model has been trained; it only distinguishes generic symptom groups.
"""
import json
import logging
import os
import platform
import sys

import numpy as np
from PIL import Image

from image_analysis import analyze_leaf

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# When the user picks a crop, the model must itself put at least this much
# probability on that crop's classes for the answer to count as confident.
CROP_SUPPORT_MIN = 0.2


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _top_k(labels, probs, k):
    order = np.argsort(probs)[::-1][:k]
    top = [{"label": labels[i], "confidence": float(probs[i])} for i in order]
    # Drop near-zero alternatives (e.g. other crops' classes after a crop filter).
    return top[:1] + [t for t in top[1:] if t["confidence"] >= 0.005]


class OnnxPredictor:
    mode = "model"
    # Used when labels.json has no calibrated value.
    DEFAULT_THRESHOLD = 0.5

    def __init__(self, model_path, labels_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        with open(labels_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta = meta if isinstance(meta, dict) else {"labels": meta}
        self.labels = meta["labels"]
        self.image_size = meta.get("image_size", 224)
        # Below this confidence a prediction is reported as "not sure"; calibrated
        # on held-out real-world photos by train/train.py.
        self.threshold = float(meta.get("confidence_threshold", self.DEFAULT_THRESHOLD))
        # Older exported models have a fixed batch size of 1.
        batch_dim = self.session.get_inputs()[0].shape[0]
        self.batched = not isinstance(batch_dim, int) or batch_dim != 1

    def _preprocess(self, image):
        img = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1).astype(np.float32)

    def predict(self, image, crop=None, top_k=3):
        x = self._preprocess(image)
        # Test-time augmentation: average over the image and its mirror image,
        # which makes predictions steadier on real photos.
        views = np.stack([x, x[:, :, ::-1]])
        if self.batched:
            logits = self.session.run(None, {self.input_name: np.ascontiguousarray(views)})[0]
        else:
            logits = np.concatenate([self.session.run(None, {self.input_name: np.ascontiguousarray(v[None])})[0]
                                     for v in views])
        probs = np.mean([_softmax(l.astype(np.float64)) for l in logits], axis=0)
        support = 1.0
        if crop:
            # Restrict to the user-selected crop when it is known to the model.
            mask = np.array([_crop_key(l) == crop.lower() for l in self.labels])
            if mask.any():
                support = float(probs[mask].sum())
                probs = np.where(mask, probs, 0.0)
                # Re-normalise only when the crop has several classes; for a
                # single-class crop that would make every photo 100 % certain.
                if mask.sum() > 1:
                    probs = probs / max(support, 1e-12)
        top = _top_k(self.labels, probs, top_k)
        top[0]["crop_support"] = support
        return top, analyze_leaf(image)


class HeuristicPredictor:
    mode = "heuristic"
    fallback_reason = None
    # The colour heuristic is rough, so most of its answers count as "not sure"
    # (which also keeps its treatment plans non-chemical).
    threshold = 0.6
    labels = [
        "Generic___healthy",
        "Generic___Leaf_spot_or_blight",
        "Generic___Powdery_mildew",
        "Generic___Rust",
        "Generic___Chlorosis_or_viral_yellowing",
    ]

    def predict(self, image, crop=None, top_k=3):
        a = analyze_leaf(image)
        lesion = a.lesion_fraction
        scores = np.array([
            6.0 * (a.green - 0.85) + 4.0 * (0.06 - lesion) * 10,
            12.0 * (a.brown + a.dark) + 2.0 * lesion,
            16.0 * a.white,
            18.0 * a.orange,
            10.0 * a.yellow - 4.0 * (a.brown + a.dark),
        ])
        probs = _softmax(scores)
        return _top_k(self.labels, probs, top_k), a


def _crop_key(label):
    raw = label.split("___")[0].lower()
    aliases = {"cherry_(including_sour)": "cherry", "corn_(maize)": "corn", "pepper,_bell": "bell pepper"}
    return aliases.get(raw, raw.replace("_", " "))


def load_predictor():
    model_path = os.path.join(MODEL_DIR, "leaf_model.onnx")
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    if not (os.path.exists(model_path) and os.path.exists(labels_path)):
        return _fallback(f"No trained model found in {MODEL_DIR}.")
    try:
        return OnnxPredictor(model_path, labels_path)
    except (ImportError, OSError) as exc:
        # Typically a missing native runtime, e.g. the Microsoft Visual C++
        # Redistributable on Windows. Keep the service usable instead of crashing.
        hint = ""
        if sys.platform == "win32":
            arch = "arm64" if platform.machine().lower() == "arm64" else "x64"
            hint = (" On Windows, install the Microsoft Visual C++ Redistributable "
                    f"(https://aka.ms/vs/17/release/vc_redist.{arch}.exe) and restart the ML service.")
        return _fallback(f"Could not load onnxruntime ({exc}).{hint}")
    except Exception as exc:  # corrupt model file, unsupported ONNX version, ...
        return _fallback(f"Could not load the model {model_path} ({exc}).")


def _fallback(reason):
    logging.getLogger(__name__).warning("Using heuristic fallback: %s", reason)
    predictor = HeuristicPredictor()
    predictor.fallback_reason = reason
    return predictor
