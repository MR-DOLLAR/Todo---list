# 🌿 LeafCare — Leaf Disease Detection & Treatment Prediction

A full-stack system that diagnoses plant leaf diseases from a photo, estimates how severe the infection is,
and predicts a treatment plan with a recovery prognosis.

```
 React (Vite)  ──►  Node.js / Express API  ──►  Python / Flask ML service
  client/           server/                     ml-service/
  upload, camera,   uploads, history store,     CNN (ONNX) classifier,
  results, history, stats, proxy to ML,         leaf segmentation & severity,
  disease library   serves built client         knowledge base, treatment planner
```

## Features

- **Disease detection** – CNN (MobileNetV3 / EfficientNet / ResNet50 transfer learning) trained on the
  [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset: 38 classes across 14 crops
  (apple, corn, grape, potato, tomato, …). An optional crop hint narrows predictions to that crop.
- **Severity estimation** – HSV colour segmentation measures the share of leaf tissue that is necrotic, yellowing,
  rusty or covered in powdery growth → `none / mild / moderate / severe`.
- **Treatment / cure prediction** – combines disease, severity, spread risk and the user's preference
  (organic / chemical / integrated) into a phased plan (Today → Days 1-7 → Weeks 2-3 → Ongoing),
  with urgency, curability, recovery chance and estimated recovery time. Incurable diseases (viruses, citrus
  greening, esca) get a containment plan instead.
- **Knowledge base** – symptoms, causes, pathogen, organic/chemical/cultural treatments and prevention for every class.
- **History & stats** – every diagnosis is stored with its image; dashboard of healthy vs diseased, severity, most common diseases.
- **Disease library** – searchable by crop, name, symptom or pathogen.
- **Pre-trained model included** – `ml-service/models/leaf_model.onnx` (MobileNetV3, 17 MB) scores
  **97.6 % top-1 / 99.7 % top-3** on 1,132 held-out PlantVillage images (see [Model](#model)).
- **Fallback mode** – if the model file is removed, the ML service falls back to a colour-heuristic classifier
  that recognises generic symptom groups (leaf spot/blight, powdery mildew, rust, chlorosis, healthy) and the UI
  shows a "demo mode" banner.

## Quick start (local)

Requirements: Python 3.10+, Node.js 20+.

```bash
# 1. ML service (port 5000)
cd ml-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py

# 2. API server (port 4000)
cd server
npm install
npm run dev

# 3. React client (port 5173, proxies /api to :4000)
cd client
npm install
npm run dev
```

Open http://localhost:5173.

For a production-style run, run `npm run build` in `client/`; the Node server then serves `client/dist` on port 4000.

### Docker

```bash
cd client && npm install && npm run build && cd ..
docker compose up --build
```

Open http://localhost:4000.

## Model

The bundled model is timm's `mobilenetv3_large_100` (ImageNet-pretrained) fine-tuned for 4 epochs on CPU
(~12 min) on a 200-images-per-class subset of PlantVillage `color`:

| Split | Images | Result |
| --- | --- | --- |
| Train / validation | 5,457 / 963 | 98.0 % val accuracy |
| Held-out test (never seen in training) | 1,132 | **97.6 % top-1, 99.7 % top-3** |
| End-to-end through the Node API | 114 | 97.4 %, ~22 ms per request on CPU |

The remaining errors are mostly between look-alike diseases (corn gray leaf spot ↔ northern leaf blight).
PlantVillage photos are single leaves on plain backgrounds; real field photos are harder, so expect lower accuracy
in the wild and consider fine-tuning on your own photos.

### Training your own

1. Download PlantVillage and point at the `color` folder (one sub-folder per class).
2. Train and export:

```bash
cd ml-service
pip install -r train/requirements.txt
python train/train.py --data-dir /path/to/plantvillage/color --epochs 5 --arch mobilenet_v3
```

Behind a firewall that blocks `download.pytorch.org` / Hugging Face, pass local pretrained weights instead, e.g.
the timm MobileNetV3 weights from GitHub releases:

```bash
curl -LO https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
python train/train.py --data-dir /path/to/train --arch timm:mobilenetv3_large_100 \
    --weights mobilenetv3_large_100_ra-f55367f5.pth --epochs 4
```

This writes `models/leaf_model.onnx` and `models/labels.json`; restart the ML service to pick them up
(`GET /health` reports `"mode": "model"`). Evaluate on a held-out folder with the same inference code the service uses:

```bash
python train/evaluate.py --data-dir /path/to/test
```

Inference needs only `onnxruntime`, `numpy` and `Pillow`; PyTorch is only required for training.

## API

### Node API (`:4000/api`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | API + ML service status |
| POST | `/diagnose` | multipart: `image` (required), `crop`, `preference` (`integrated`/`organic`/`chemical`), `notes` → stored diagnosis record |
| GET | `/history` | Recent diagnoses (summary) |
| GET / DELETE | `/history/:id` | Full record / delete |
| GET | `/stats` | Totals, severity breakdown, top diseases |
| GET | `/diseases?crop=&q=` | Disease library |
| GET | `/diseases/:id` | One disease |
| GET | `/crops` | Supported crops |

### Flask ML service (`:5000`)

`GET /health`, `POST /predict` (same form fields), `GET /diseases`, `GET /diseases/<id>`, `GET /crops`.

Example `/predict` response (abridged):

```json
{
  "mode": "model",
  "prediction": { "label": "Tomato___Early_blight", "crop": "Tomato", "disease": "Early Blight", "confidence": 0.97, "healthy": false },
  "alternatives": [{ "label": "Tomato___Target_Spot", "name": "Target Spot", "confidence": 0.02 }],
  "severity": { "level": "moderate", "affected_area_pct": 21.4 },
  "treatment_plan": {
    "urgency": "medium", "curable": true, "approach": "integrated",
    "summary": "Early Blight on tomato detected. ...",
    "steps": [{ "phase": "Today", "actions": ["Remove visibly infected leaves ..."] }],
    "prognosis": { "recovery_chance": 0.75, "estimated_recovery_days": 25 },
    "warnings": []
  },
  "disease": { "symptoms": [], "causes": [], "treatment": { "organic": [], "chemical": [], "cultural": [] }, "prevention": [] }
}
```

## Configuration

| Variable | Service | Default |
| --- | --- | --- |
| `PORT` | ml-service / server | `5000` / `4000` |
| `MODEL_DIR` | ml-service | `ml-service/models` |
| `ML_SERVICE_URL` | server | `http://localhost:5000` |
| `DATA_DIR` | server | `server/data` (history JSON + uploaded images) |
| `CLIENT_DIST` | server | `client/dist` |

## Tests

```bash
cd ml-service && pytest -q     # Flask API, classifier, severity, treatment planner
cd server && npm test          # Express API against a mock ML service
```

## Disclaimer

Recommendations are guidance only. Always follow pesticide label instructions and local regulations, and confirm
serious or unusual cases with an agricultural extension service.
