"""Flask ML service: leaf disease classification + treatment prediction."""
import io
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from image_analysis import severity_from_analysis
from knowledge import KnowledgeBase
from predictor import load_predictor
from treatment import build_plan

MAX_UPLOAD_MB = 10
PREFERENCES = {"integrated", "organic", "chemical"}


def create_app(predictor=None, kb=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    CORS(app)
    app.predictor = predictor or load_predictor()
    app.kb = kb or KnowledgeBase()

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            mode=app.predictor.mode,
            classes=len(app.predictor.labels),
            fallback_reason=getattr(app.predictor, "fallback_reason", None),
        )

    @app.post("/predict")
    def predict():
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify(error="No image uploaded (expected form field 'image')"), 400
        try:
            image = Image.open(io.BytesIO(file.read()))
            image.load()
        except (UnidentifiedImageError, OSError):
            return jsonify(error="File is not a valid image"), 400

        crop = (request.form.get("crop") or "").strip() or None
        preference = request.form.get("preference", "integrated")
        if preference not in PREFERENCES:
            preference = "integrated"

        top, analysis = app.predictor.predict(image, crop=crop)
        best = top[0]
        disease = app.kb.get(best["label"])
        if disease is None:
            return jsonify(error=f"No knowledge-base entry for label {best['label']}"), 500

        severity = severity_from_analysis(analysis, disease["healthy"])
        plan = build_plan(disease, severity, best["confidence"], preference)

        return jsonify(
            mode=app.predictor.mode,
            prediction={
                "label": best["label"],
                "confidence": round(best["confidence"], 4),
                "crop": disease["crop"],
                "disease": disease["name"],
                "healthy": disease["healthy"],
            },
            alternatives=[
                {
                    "label": t["label"],
                    "confidence": round(t["confidence"], 4),
                    "name": (app.kb.get(t["label"]) or {}).get("name", t["label"]),
                    "crop": (app.kb.get(t["label"]) or {}).get("crop"),
                }
                for t in top[1:]
            ],
            severity=severity,
            analysis=analysis.to_dict(),
            disease=disease,
            treatment_plan=plan,
        )

    @app.get("/diseases")
    def diseases():
        return jsonify(app.kb.search(crop=request.args.get("crop"), query=request.args.get("q")))

    @app.get("/diseases/<path:disease_id>")
    def disease_detail(disease_id):
        entry = app.kb.get(disease_id)
        if entry is None:
            return jsonify(error="Disease not found"), 404
        return jsonify(entry)

    @app.get("/crops")
    def crops():
        return jsonify(app.kb.crops())

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"Image too large (max {MAX_UPLOAD_MB} MB)"), 413

    return app


if __name__ == "__main__":
    # 5001 rather than Flask's usual 5000, which macOS AirPlay Receiver occupies.
    # Only the Node API talks to this service, so listen on localhost by default.
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("LEAFCARE_HOST", "").strip() or "127.0.0.1"
    app = create_app()
    print(f"LeafCare ML service on http://{host}:{port} (mode: {app.predictor.mode})", flush=True)
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
