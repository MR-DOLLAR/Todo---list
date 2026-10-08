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
