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

## How to run

Pick **one** of the options below. Option 1 needs only Docker; option 2 needs Node.js and Python.

| Port | Service |
| --- | --- |
| 5173 | Web app in development mode (option 2 `npm run dev`) |
| 4000 | API server; also serves the web app in options 1 and `npm start` |
| 5001 | Python ML service (internal; only the API server talks to it) |

### Get the code

```bash
git clone https://github.com/MR-DOLLAR/Todo---list.git
cd Todo---list
```

No git? On GitHub click **Code → Download ZIP**, unzip it and open a terminal in the unzipped folder.
All commands below run from this folder (the one containing `docker-compose.yml`).

### Option 1 — Docker (simplest: nothing else to install)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine with Compose v2 (Linux).

```bash
docker compose up --build
```

Wait for the build (a few minutes the first time), then open **http://localhost:4000**.
Stop with `Ctrl+C`; start again later with `docker compose up` (after updating the code, use
`docker compose up --build` again so the images are rebuilt). Diagnosis history is kept in a Docker volume
(`docker compose down -v` deletes it).

Port 4000 already taken? Create a file named `.env` next to `docker-compose.yml` containing the line
`LEAFCARE_PORT=4300` (any free port), run `docker compose up --build` again and open http://localhost:4300.
Compose reads `.env` every time, so the setting sticks.

### Option 2 — Node.js + Python, one command

Requirements:

| | Version | Install |
| --- | --- | --- |
| Node.js | 22 LTS recommended (20+) | [nodejs.org](https://nodejs.org) · Windows `winget install OpenJS.NodeJS.LTS` · macOS `brew install node@22` · Ubuntu: use nodejs.org/nvm (apt's version is too old) |
| Python | 3.12 or 3.13 recommended (3.10–3.14) | [python.org](https://www.python.org/downloads/) · Windows `winget install Python.Python.3.13` · macOS `brew install python@3.13` · Ubuntu `sudo apt install python3 python3-venv` |

Python 3.15 is not supported yet (no onnxruntime build); on Intel Macs use 3.10–3.13, on macOS 12 use 3.10–3.12,
on Windows ARM use 3.11+. `npm run setup` checks this for you.

```bash
npm run setup
npm run dev
```

`npm run setup` is needed only once: it creates `ml-service/.venv` and installs all Python and npm packages
(about 1–3 minutes). `npm run dev` starts the ML service, API server and web app together; wait for
*“✔ LeafCare is running”*, then open **http://localhost:5173**. Press `Ctrl+C` to stop everything.
Next time, just run `npm run dev`.

| Command | What it does |
| --- | --- |
| `npm run setup` | One-time install (safe to re-run) |
| `npm run dev` | Development mode → http://localhost:5173 |
| `npm start` | Builds the web app and serves everything from one server → http://localhost:4000 |
| `npm test` | Runs the Python and Node test suites |

**Windows PowerShell:** if you see *"running scripts is disabled on this system"*, use **Command Prompt** instead,
type `npm.cmd` instead of `npm` (e.g. `npm.cmd run dev`), or allow scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Option 3 — start each service by hand (three terminals)

Useful when you want to see or restart one service at a time. Open three terminals, each in the project folder.

**Terminal 1 — ML service** (port 5001)

macOS / Linux:

```bash
cd ml-service
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Windows (Command Prompt or PowerShell):

```bat
cd ml-service
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

(Use whichever supported version you have, e.g. `py -3.12`. Next time only the last line is needed.)

**Terminal 2 — API server** (port 4000)

```bash
cd server
npm install
npm run dev
```

**Terminal 3 — web app** (port 5173)

```bash
cd client
npm install
npm run dev
```

Open **http://localhost:5173**. Keep all three terminals running.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| Red **“API offline”** banner | The API server (port 4000) isn't running — start it (option 3, terminal 2) or use `npm run dev`. |
| Red **“ML offline”** banner / *“ML service is unavailable”* | The ML service (port 5001) isn't running — start it (option 3, terminal 1). |
| Yellow **demo mode** banner on Windows mentioning onnxruntime | Install the Microsoft Visual C++ Redistributable ([x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) · [ARM64](https://aka.ms/vs/17/release/vc_redist.arm64.exe), or `winget install Microsoft.VCRedist.2015+.x64`) and restart. |
| `python: command not found` (macOS/Linux) | Use `python3`. |
| *“ensurepip is not available”* (Ubuntu/Debian) | `sudo apt install python3.X-venv` (X = the version setup printed, e.g. `python3.12-venv`), then run setup again. |
| *“No matching distribution found for onnxruntime”* | Your Python version, OS version or CPU type has no onnxruntime build (e.g. Python 3.15; 32-bit Python 3.13+; Python 3.14 on an Intel Mac; Python 3.13 on macOS 12 → use 3.12). Install a supported 64-bit Python, delete `ml-service/.venv`, run setup again. `npm run setup` checks this and tells you which version to use. |
| Setup or `npm run dev` says packages are missing | Run `npm run setup` again (it is safe to re-run). |
| *“Port … is already in use”* / Docker *“port is already allocated”* | LeafCare (or another program) is already running on that port — stop it, or choose other ports (see [Configuration](#configuration); Docker: `LEAFCARE_PORT`). |
| http://localhost:4000 says *“web interface has not been built”* | Expected in development mode: use http://localhost:5173, or run `npm start` / `npm run build` in `client/`. |
| `npm install` prints audit warnings | Don't run `npm audit fix --force` (it makes breaking upgrades). |
| Red *“This is a development server”* line from Flask | Expected for local use; Docker runs the ML service under gunicorn. |
| Opening the app from your phone doesn't work | By default it only listens on this computer. Use Docker, or start with `LEAFCARE_HOST=0.0.0.0 npm start` (PowerShell: `$env:LEAFCARE_HOST='0.0.0.0'; npm start` · Command Prompt: `set LEAFCARE_HOST=0.0.0.0&& npm start`), then open `http://<your-computer's-IP>:4000` on the phone. |

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
.venv/bin/python -m pip install -r train/requirements.txt
.venv/bin/python train/train.py --data-dir /path/to/plantvillage/color --epochs 5 --arch mobilenet_v3
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

PyTorch wheels exist for Python 3.10–3.14 (Intel Macs: up to 3.12). Training on a CPU works but is slow on the
full dataset; a GPU is recommended.

Behind a firewall that blocks `download.pytorch.org` / Hugging Face, pass local pretrained weights instead, e.g.
the timm MobileNetV3 weights from GitHub releases:

```bash
curl -LO https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
.venv/bin/python train/train.py --data-dir /path/to/train --arch timm:mobilenetv3_large_100 --weights mobilenetv3_large_100_ra-f55367f5.pth --epochs 4
```

(Windows PowerShell: use `curl.exe` instead of `curl`.)

This writes `models/leaf_model.onnx` and `models/labels.json`; restart the ML service to pick them up
(`GET /health` reports `"mode": "model"`). Evaluate on a held-out folder with the same inference code the service uses:

```bash
.venv/bin/python train/evaluate.py --data-dir /path/to/test
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

### Flask ML service (`:5001`)

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
| `PORT` | ml-service / server | `5001` / `4000` |
| `LEAFCARE_HOST` | ml-service (`python app.py`) / server | `127.0.0.1` (Docker: `0.0.0.0`). Set it to `0.0.0.0` for the server to open the app from a phone on your network |
| `MODEL_DIR` | ml-service | `ml-service/models` |
| `ML_SERVICE_URL` | server | `http://127.0.0.1:5001` |
| `DATA_DIR` | server | `server/data` (history JSON + uploaded images) |
| `CLIENT_DIST` | server | `client/dist` |
| `LEAFCARE_API_URL` | client dev server (Vite proxy) | `http://127.0.0.1:4000` |
| `LEAFCARE_PORT` | docker compose (host port) | `4000` |

With `npm run dev` / `npm start`, set `ML_PORT` and `PORT` to change the ML and API ports, e.g.
`PORT=4010 npm run dev` (PowerShell: `$env:PORT=4010; npm run dev` · Command Prompt: `set PORT=4010&& npm run dev`).
The dev web server always uses port 5173.

## Tests

After `npm run setup`, run both test suites with:

```bash
npm test
```

Or individually, from the project folder. The Python suite covers the Flask API, classifier, severity and
treatment planner; the Node suite tests the Express API against a mock ML service. On Windows use
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.

```bash
cd ml-service
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
cd ../server
npm test
```

## Disclaimer

Recommendations are guidance only. Always follow pesticide label instructions and local regulations, and confirm
serious or unusual cases with an agricultural extension service.
