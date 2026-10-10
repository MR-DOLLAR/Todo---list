"""Prepare training data: PlantDoc (real-world photos) + PlantVillage (lab photos).

PlantDoc folder names are mapped to PlantVillage class names so both datasets
share one label set. Images are EXIF-rotated, converted to RGB and resized
(shorter side 320 px) so training reads small files. Training images that are
exact or near duplicates of a PlantDoc test image are dropped to keep the test
score honest.

Usage:
    python train/prepare_data.py --plantdoc /path/to/PlantDoc-Dataset \\
        --plantvillage /path/to/plantvillage/color --out data/prepared [--pv-per-class 200]

Output: <out>/{train_pv,train_pd,val_pd,test_pd}/<PlantVillage class>/*.jpg
"""
import argparse
import hashlib
import os
import random

from PIL import Image, ImageOps

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
SHORT_SIDE = 320
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def resized(im):
    w, h = im.size
    scale = SHORT_SIDE / min(w, h)
    if scale < 1:
        im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    return im


def dhash(im, size=8):
    """Difference hash: robust to resizing/recompression, used for near-duplicates."""
    g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def save(im, out_dir, label, name):
    d = os.path.join(out_dir, label)
    os.makedirs(d, exist_ok=True)
    base = hashlib.md5(name.encode()).hexdigest()[:16]
    im.save(os.path.join(d, base + ".jpg"), quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantdoc", required=True, help="PlantDoc-Dataset folder (with train/ and test/)")
    ap.add_argument("--plantvillage", help="PlantVillage 'color' folder (one sub-folder per class)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pv-per-class", type=int, default=200)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # Test set first, so training near-duplicates of test images can be dropped.
    test_hashes, test_md5 = [], set()
    n_test = 0
    for folder, label in PLANTDOC_TO_PLANTVILLAGE.items():
        src = os.path.join(args.plantdoc, "test", folder)
        if not os.path.isdir(src):
            continue
        for f in files(src):
            path = os.path.join(src, f)
            test_md5.add(hashlib.md5(open(path, "rb").read()).hexdigest())
            im = load_rgb(path)
            test_hashes.append(dhash(im))
            save(resized(im), os.path.join(args.out, "test_pd"), label, folder + f)
            n_test += 1

    def near_test(h):
        return any(bin(h ^ t).count("1") <= 4 for t in test_hashes)

    dropped = n_train = n_val = 0
    for folder, label in PLANTDOC_TO_PLANTVILLAGE.items():
        src = os.path.join(args.plantdoc, "train", folder)
        if not os.path.isdir(src):
            continue
        items = files(src)
        rng.shuffle(items)
        n_val_here = max(1, round(len(items) * args.val_fraction))
        for i, f in enumerate(items):
            path = os.path.join(src, f)
            if hashlib.md5(open(path, "rb").read()).hexdigest() in test_md5:
                dropped += 1
                continue
            try:
                im = load_rgb(path)
            except OSError:
                continue
            if near_test(dhash(im)):
                dropped += 1
                continue
            split = "val_pd" if i < n_val_here else "train_pd"
            save(resized(im), os.path.join(args.out, split), label, folder + f)
            n_val += split == "val_pd"
            n_train += split == "train_pd"

    n_pv = 0
    if args.plantvillage:
        for label in sorted(os.listdir(args.plantvillage)):
            src = os.path.join(args.plantvillage, label)
            if not os.path.isdir(src):
                continue
            items = files(src)
            rng.shuffle(items)
            for f in items[: args.pv_per_class]:
                save(resized(load_rgb(os.path.join(src, f))), os.path.join(args.out, "train_pv"), label, f)
                n_pv += 1

    print(f"PlantDoc: {n_train} train, {n_val} val, {n_test} test ({dropped} train duplicates of test dropped)")
    print(f"PlantVillage: {n_pv} train")


if __name__ == "__main__":
    main()
