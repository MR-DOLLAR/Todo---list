"""Colour-based leaf segmentation used to estimate how much of a leaf is diseased.

Works on HSV pixel classes so it needs nothing beyond numpy + Pillow. The output
feeds both the severity estimate and the heuristic fallback classifier.
"""
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

ANALYSIS_SIZE = 256


@dataclass
class LeafAnalysis:
    leaf_fraction: float      # share of the image covered by leaf tissue
    green: float              # shares below are fractions of leaf tissue
    yellow: float
    brown: float
    dark: float
    orange: float
    white: float
    lesion_fraction: float    # everything on the leaf that is not healthy green

    def to_dict(self):
        return {k: round(v, 4) for k, v in asdict(self).items()}


def _hsv(image):
    arr = np.asarray(image.convert("RGB").resize((ANALYSIS_SIZE, ANALYSIS_SIZE)).convert("HSV"), dtype=np.float32)
    h = arr[..., 0] * (360.0 / 255.0)
    return h, arr[..., 1], arr[..., 2]


def analyze_leaf(image: Image.Image) -> LeafAnalysis:
    h, s, v = _hsv(image)

    green = (h >= 65) & (h <= 170) & (s > 45) & (v > 40)
    yellow = (h >= 40) & (h < 65) & (s > 60) & (v > 110)
    orange = (h >= 15) & (h < 40) & (s > 140) & (v > 140)
    brown = ((h < 40) | (h > 330)) & (s > 50) & (v > 35) & ~orange
    plant = green | yellow | orange | brown

    # Dark necrotic tissue and white powdery growth are low-saturation, so they
    # only count when they sit inside the leaf's bounding box.
    rows, cols = np.where(plant)
    inside = np.zeros_like(plant)
    if rows.size:
        inside[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    dark = inside & (v <= 45) & ~plant
    white = inside & (s < 45) & (v > 185) & ~plant
    # Only count white pixels that touch leaf tissue, so a bright background
    # around the leaf is not mistaken for mildew.
    white &= _dilate(plant, 3)
    dark &= _dilate(plant, 3)

    leaf = plant | dark | white
    leaf_px = max(int(leaf.sum()), 1)

    def frac(mask):
        return float(mask.sum()) / leaf_px

    lesion = yellow | orange | brown | dark | white
    return LeafAnalysis(
        leaf_fraction=float(leaf.sum()) / leaf.size,
        green=frac(green),
        yellow=frac(yellow),
        brown=frac(brown),
        dark=frac(dark),
        orange=frac(orange),
        white=frac(white),
        lesion_fraction=frac(lesion),
    )


def _dilate(mask, radius):
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def severity_from_analysis(analysis: LeafAnalysis, healthy: bool):
    """Map affected-area share to a severity level and 0-100 score."""
    affected = 0.0 if healthy else analysis.lesion_fraction
    pct = round(min(affected, 1.0) * 100, 1)
    if healthy or pct < 3:
        level = "none" if healthy else "mild"
    elif pct < 15:
        level = "mild"
    elif pct < 35:
        level = "moderate"
    else:
        level = "severe"
    return {"level": level, "affected_area_pct": pct}
