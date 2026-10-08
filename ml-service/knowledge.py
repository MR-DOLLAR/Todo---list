"""Disease knowledge base: symptoms, causes and treatments per class label."""
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diseases.json")

# PlantVillage class labels (38 classes) in the canonical alphabetical order
# used by torchvision.datasets.ImageFolder.
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_CROP_NAMES = {
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Pepper,_bell": "Bell Pepper",
    "Generic": "Unknown",
}


def _crop_from_label(label):
    raw = label.split("___")[0]
    return _CROP_NAMES.get(raw, raw.replace("_", " "))


def _healthy_entry(label):
    crop = _crop_from_label(label)
    return {
        "crop": crop,
        "name": "Healthy",
        "type": "none",
        "pathogen": None,
        "description": f"The {crop.lower() if crop != 'Unknown' else 'plant'} leaf shows no visible signs of disease.",
        "symptoms": [],
        "causes": [],
        "treatment": {"organic": [], "chemical": [], "cultural": []},
        "prevention": [
            "Keep monitoring leaves weekly, especially after wet weather",
            "Water at the base of the plant in the morning",
            "Maintain balanced fertilisation and good airflow",
            "Rotate crops and remove plant debris at season end",
        ],
        "spread_risk": "none",
        "recovery_days": 0,
    }


class KnowledgeBase:
    def __init__(self, path=DATA_PATH):
        with open(path, encoding="utf-8") as f:
            self._data = json.load(f)
        for label in PLANTVILLAGE_CLASSES + ["Generic___healthy"]:
            if label.endswith("___healthy"):
                self._data.setdefault(label, _healthy_entry(label))
        for label, entry in self._data.items():
            entry["id"] = label
            entry["healthy"] = label.endswith("___healthy")

    def get(self, label):
        return self._data.get(label)

    def all(self):
        return list(self._data.values())

    def search(self, crop=None, query=None, include_generic=False):
        results = []
        for entry in self._data.values():
            if not include_generic and entry["id"].startswith("Generic___"):
                continue
            if crop and entry["crop"].lower() != crop.lower():
                continue
            if query:
                haystack = " ".join(
                    [entry["name"], entry["crop"], entry.get("pathogen") or "", entry["description"]]
                    + entry["symptoms"]
                ).lower()
                if query.lower() not in haystack:
                    continue
            results.append(entry)
        return sorted(results, key=lambda e: (e["crop"], e["healthy"], e["name"]))

    def crops(self):
        return sorted({e["crop"] for e in self._data.values() if e["crop"] != "Unknown"})
