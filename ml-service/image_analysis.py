"""Colour-based leaf segmentation used to estimate how much of a leaf is diseased.

Works on HSV pixel classes so it needs nothing beyond numpy + Pillow. The output
feeds both the severity estimate and the heuristic fallback classifier.
"""
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

ANALYSIS_SIZE = 256
# Below this Laplacian variance (at ANALYSIS_SIZE) a photo is noticeably blurry.
BLUR_THRESHOLD = 12.0


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
    sharpness: float = 0.0    # Laplacian variance; low = blurry photo

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def analyze_leaf(image: Image.Image) -> LeafAnalysis:
    small = image.convert("RGB").resize((ANALYSIS_SIZE, ANALYSIS_SIZE))
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * (360.0 / 255.0), hsv[..., 1], hsv[..., 2]
    rgb = np.asarray(small, dtype=np.float32)

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

    # In field photos soil, mulch and dead material are brown too. Lesions grow out
    # of leaf tissue, so only count discoloured pixels connected to green tissue;
    # separate brown regions (soil, a table seen between leaves) don't count. With
    # essentially no green left (a dead leaf) there is nothing to anchor to, so
    # everything counts.
    if green.sum() >= 0.005 * green.size:
        keep = _reconstruct(green, plant | dark | white)
        yellow, orange, brown = yellow & keep, orange & keep, brown & keep
        dark, white = dark & keep, white & keep
        plant = green | yellow | orange | brown

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
        sharpness=_sharpness(rgb),
    )


def _dilate(mask, radius):
    """Square dilation via an integral image: O(pixels) for any radius."""
    k = 2 * radius + 1
    padded = np.pad(mask.astype(np.int32), radius)
    ii = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    box = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return box > 0


def _reconstruct(seed, mask, step=2, max_iter=200):
    """Pixels of `mask` connected to `seed` (morphological reconstruction)."""
    cur = seed & mask
    for _ in range(max_iter):
        nxt = _dilate(cur, step) & mask
        if (nxt == cur).all():
            break
        cur = nxt
    return cur | seed


def _sharpness(rgb):
    """Variance of the Laplacian of the grey image: a standard focus measure."""
    g = rgb.mean(axis=-1)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def photo_warnings(image: Image.Image, analysis: LeafAnalysis):
    """Plain-language hints about photo problems that hurt accuracy."""
    warnings = []
    if min(image.size) < 160:
        warnings.append("The photo is very small. Use a larger, closer photo of the leaf.")
    if analysis.sharpness < BLUR_THRESHOLD:
        warnings.append("The photo looks blurry. Hold the camera steady and tap the leaf to focus.")
    if analysis.leaf_fraction < 0.15:
        warnings.append("The leaf fills only a small part of the photo. Move closer so one leaf fills most of the frame.")
    return warnings


def severity_from_analysis(analysis: LeafAnalysis, healthy: bool):
    """Map affected-area share to a severity level and 0-100 score.

    `measurable` is False when a disease was predicted but no discoloured tissue
    was found (e.g. viral mottling, or the photo doesn't show the symptoms).
    """
    affected = 0.0 if healthy else analysis.lesion_fraction
    pct = round(min(affected, 1.0) * 100, 1)
    measurable = healthy or pct >= 1
    if healthy or pct < 3:
        level = "none" if healthy else "mild"
    elif pct < 15:
        level = "mild"
    elif pct < 35:
        level = "moderate"
    else:
        level = "severe"
    return {"level": level, "affected_area_pct": pct, "measurable": measurable}
