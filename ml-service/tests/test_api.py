import io
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app  # noqa: E402
from knowledge import PLANTVILLAGE_CLASSES, KnowledgeBase  # noqa: E402
from predictor import HeuristicPredictor  # noqa: E402


def leaf_image(spots=None, spot_count=0, powder=False):
    """Draw a green leaf on a grey background with optional coloured lesions."""
    img = Image.new("RGB", (256, 256), (120, 120, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((30, 50, 226, 206), fill=(50, 140, 45))
    rng = np.random.default_rng(0)
    for _ in range(spot_count):
        x, y = rng.integers(70, 180), rng.integers(80, 170)
        r = rng.integers(8, 16)
        d.ellipse((x - r, y - r, x + r, y + r), fill=spots)
    if powder:
        for _ in range(25):
            x, y = rng.integers(60, 190), rng.integers(80, 170)
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(235, 235, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture()
def client():
    return create_app(predictor=HeuristicPredictor()).test_client()


def post(client, buf, **form):
    return client.post("/predict", data={"image": (buf, "leaf.png"), **form}, content_type="multipart/form-data")


def test_knowledge_base_covers_all_plantvillage_classes():
    kb = KnowledgeBase()
    missing = [c for c in PLANTVILLAGE_CLASSES + HeuristicPredictor.labels if kb.get(c) is None]
    assert missing == []


def test_health(client):
    assert client.get("/health").get_json()["mode"] == "heuristic"


def test_healthy_leaf(client):
    body = post(client, leaf_image()).get_json()
    assert body["prediction"]["healthy"] is True
    assert body["severity"]["level"] == "none"
    assert body["treatment_plan"]["approach"] == "preventive"


def test_brown_spots_detected_as_leaf_spot(client):
    body = post(client, leaf_image(spots=(110, 60, 25), spot_count=14)).get_json()
    assert body["prediction"]["label"] == "Generic___Leaf_spot_or_blight"
    assert body["severity"]["affected_area_pct"] > 5
    assert body["treatment_plan"]["steps"]


def test_powdery_mildew(client):
    body = post(client, leaf_image(powder=True)).get_json()
    assert body["prediction"]["label"] == "Generic___Powdery_mildew"


def test_rust(client):
    body = post(client, leaf_image(spots=(235, 120, 20), spot_count=14)).get_json()
    assert body["prediction"]["label"] == "Generic___Rust"


def test_rejects_missing_and_invalid_images(client):
    assert client.post("/predict").status_code == 400
    assert post(client, io.BytesIO(b"not an image")).status_code == 400


def test_disease_library(client):
    tomato = client.get("/diseases?crop=Tomato").get_json()
    assert len(tomato) == 10
    assert client.get("/diseases/Potato___Late_blight").get_json()["pathogen"] == "Phytophthora infestans"
    assert client.get("/diseases/nope").status_code == 404
    assert "Tomato" in client.get("/crops").get_json()


def test_incurable_disease_plan():
    from treatment import build_plan

    kb = KnowledgeBase()
    plan = build_plan(kb.get("Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
                      {"level": "mild", "affected_area_pct": 4}, 0.9)
    assert plan["curable"] is False
    assert plan["approach"] == "containment"
    assert plan["urgency"] in ("high", "critical")


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "leaf_model.onnx")), reason="no trained model")
def test_bundled_model_loads_and_predicts_known_labels():
    from predictor import load_predictor

    predictor = load_predictor()
    assert predictor.mode == "model"
    kb = KnowledgeBase()
    assert all(kb.get(label) for label in predictor.labels)
    top, _ = predictor.predict(Image.open(leaf_image(spots=(110, 60, 25), spot_count=10)), crop="Tomato")
    assert top[0]["label"].startswith("Tomato___")
    assert 0 < top[0]["confidence"] <= 1
    assert top[0]["confidence"] >= top[-1]["confidence"]


def test_falls_back_to_heuristic_when_onnxruntime_cannot_load(monkeypatch):
    import predictor

    def broken(*_args, **_kwargs):
        raise ImportError("DLL load failed while importing onnxruntime_pybind11_state")

    monkeypatch.setattr(predictor, "OnnxPredictor", broken)
    monkeypatch.setattr(predictor.os.path, "exists", lambda _p: True)
    fallback = predictor.load_predictor()
    assert fallback.mode == "heuristic"
    assert "DLL load failed" in fallback.fallback_reason

    health = create_app(predictor=fallback).test_client().get("/health").get_json()
    assert health["mode"] == "heuristic"
    assert "DLL load failed" in health["fallback_reason"]


def test_falls_back_to_heuristic_when_model_file_is_corrupt(tmp_path, monkeypatch):
    import predictor

    (tmp_path / "leaf_model.onnx").write_bytes(b"not an onnx model")
    (tmp_path / "labels.json").write_text('{"labels": ["Tomato___healthy"]}')
    monkeypatch.setattr(predictor, "MODEL_DIR", str(tmp_path))
    fallback = predictor.load_predictor()
    assert fallback.mode == "heuristic"
    assert "Could not load the model" in fallback.fallback_reason


class StubPredictor:
    """Returns a fixed prediction and records the image it was given."""
    mode = "model"
    labels = ["Tomato___Early_blight", "Tomato___healthy", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

    def __init__(self, label="Tomato___Early_blight", confidence=0.9, threshold=0.6):
        self.label, self.confidence, self.threshold = label, confidence, threshold
        self.seen = None

    def predict(self, image, crop=None, top_k=3):
        from image_analysis import analyze_leaf

        self.seen = image
        rest = (1 - self.confidence) / 2
        others = [lab for lab in self.labels if lab != self.label][:2]
        top = [{"label": self.label, "confidence": self.confidence}] + [{"label": o, "confidence": rest} for o in others]
        return top, analyze_leaf(image)


def stub_client(**kwargs):
    stub = StubPredictor(**kwargs)
    return create_app(predictor=stub).test_client(), stub


def test_confident_prediction_has_status_and_full_plan():
    client, _ = stub_client(confidence=0.9)
    body = post(client, leaf_image(spots=(110, 60, 25), spot_count=10)).get_json()
    assert body["status"] == "confident"
    assert body["message"] is None
    assert body["treatment_plan"]["approach"] in ("organic", "integrated")
    assert "Tomato" in body["supported_crops"]


def test_low_confidence_is_reported_as_uncertain_with_non_chemical_plan():
    client, _ = stub_client(confidence=0.4)
    body = post(client, leaf_image(spots=(110, 60, 25), spot_count=10), preference="chemical").get_json()
    assert body["status"] == "uncertain"
    assert "not sure" in body["message"]
    plan = body["treatment_plan"]
    assert plan["approach"] == "confirm first"
    assert plan["summary"].startswith("Possibly Early Blight")
    assert plan["urgency"] in ("low", "medium")
    assert plan["prognosis"]["recovery_chance"] is None
    assert any("not sure" in w for w in plan["warnings"])


def test_uncertain_incurable_disease_never_says_destroy_plants():
    client, _ = stub_client(label="Tomato___Tomato_Yellow_Leaf_Curl_Virus", confidence=0.3)
    plan = post(client, leaf_image(spots=(200, 200, 60), spot_count=8)).get_json()["treatment_plan"]
    assert plan["approach"] == "confirm first"
    actions = " ".join(a for step in plan["steps"] for a in step["actions"]).lower()
    assert "destroy" not in actions and "remove and" not in actions


def test_photo_without_a_leaf_is_reported():
    client, _ = stub_client(confidence=0.3)
    img = Image.new("RGB", (256, 256), (40, 90, 200))  # blue sky, no plant colours
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    body = post(client, buf).get_json()
    assert body["status"] == "no_leaf"
    assert "No leaf" in body["message"]


def test_exif_rotation_is_applied_before_prediction():
    client, stub = stub_client()
    img = Image.new("RGB", (300, 100), (50, 140, 45))
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways: rotate 90° clockwise to view
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    assert post(client, buf).status_code == 200
    assert stub.seen.size == (100, 300)


def test_photo_warnings_flag_blurry_and_tiny_images():
    from PIL import ImageFilter

    client, _ = stub_client()
    img = Image.open(leaf_image(spots=(110, 60, 25), spot_count=10)).resize((120, 120)).filter(ImageFilter.GaussianBlur(4))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    warnings = post(client, buf).get_json()["photo_warnings"]
    assert any("small" in w for w in warnings)
    assert any("blurry" in w for w in warnings)


def test_uncertain_plans_never_suggest_products_or_removing_plants():
    import re
    from treatment import build_plan

    unsafe = re.compile(r"copper|sulfur|fungicid|insecticid|spray|\boil\b|neem|destroy|kill|burn|whole plant|"
                        r"remov\w*.*\b(plants?|trees?)\b", re.IGNORECASE)
    kb = KnowledgeBase()
    for disease in kb.all():
        if disease["healthy"]:
            continue
        for level in ("mild", "moderate", "severe"):
            for pref in ("organic", "chemical", "integrated"):
                plan = build_plan(disease, {"level": level, "affected_area_pct": 50}, 0.3, pref, uncertain=True)
                for step in plan["steps"]:
                    for action in step["actions"]:
                        if action.startswith(("Confirm the diagnosis", "Remove only clearly spotted leaves")):
                            continue
                        assert not unsafe.search(action), (disease["id"], action)


def test_choosing_a_single_class_crop_does_not_force_full_confidence():
    import predictor

    p = predictor.load_predictor()
    if p.mode != "model":
        pytest.skip("no trained model")
    sky = Image.new("RGB", (256, 256), (40, 90, 200))
    top, _ = p.predict(sky, crop="Orange")
    assert top[0]["confidence"] < 0.9          # not re-normalised to 100 %
    assert top[0]["crop_support"] < predictor.CROP_SUPPORT_MIN
    assert all(t["confidence"] >= 0.005 for t in top[1:])


def test_screenshot_like_image_without_plant_colours_is_no_leaf_even_if_model_is_sure():
    client, _ = stub_client(confidence=0.95)
    img = Image.new("RGB", (400, 300), (250, 250, 250))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for y in range(20, 280, 18):
        d.line((20, y, 380, y), fill=(30, 30, 30), width=3)  # lines of "text"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    body = post(client, buf).get_json()
    assert body["status"] == "no_leaf"
    assert body["treatment_plan"]["approach"] == "confirm first"


@pytest.mark.parametrize("share,level", [(0.3, "moderate"), (0.5, "severe"), (0.7, "severe")])
def test_large_necrotic_areas_are_not_underestimated(share, level):
    from image_analysis import analyze_leaf, severity_from_analysis

    img = Image.new("RGB", (256, 256), (120, 120, 120))
    d = ImageDraw.Draw(img)
    d.rectangle((16, 16, 240, 240), fill=(50, 140, 45))                       # leaf filling the frame
    d.rectangle((16, 16, 16 + int(224 * share), 240), fill=(110, 60, 25))      # brown dead tissue
    sev = severity_from_analysis(analyze_leaf(img), healthy=False)
    assert sev["level"] == level
    assert abs(sev["affected_area_pct"] - share * 100) < 8


def test_brown_soil_away_from_the_leaf_is_not_counted_as_disease():
    from image_analysis import analyze_leaf

    img = Image.new("RGB", (256, 256), (105, 75, 45))                          # soil everywhere
    d = ImageDraw.Draw(img)
    d.rectangle((100, 100, 156, 156), fill=(120, 120, 120))                    # gap between leaf and soil
    d.ellipse((110, 110, 146, 146), fill=(50, 140, 45))                        # small healthy leaf
    assert analyze_leaf(img).lesion_fraction < 0.1
