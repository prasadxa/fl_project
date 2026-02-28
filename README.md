# Tecnomate — Federated Learning Medical Image Classifier

A privacy-preserving, continuously-learning federated learning (FL) system using **Flower** and **PyTorch** that classifies medical images across two datasets — Brain Tumor MRI scans and Pneumonia Chest X-Rays — without any patient data ever leaving the local device.

The system ships two clinical interfaces:

| Interface | Stack | Launch |
|---|---|---|
| **Web Dashboard** *(primary)* | FastAPI + Vanilla SPA | `run_api.bat` → `http://127.0.0.1:8000` |
| **Streamlit UI** *(legacy)* | Streamlit | `run_edge.bat` → `http://localhost:8501` |

Both UIs support doctor-in-the-loop feedback, automatic EXIF anonymisation, and a continual learning pipeline that incorporates new doctor-verified images into the next federated training round.

---

## Project Overview

| Property | Value |
|---|---|
| **Task** | 6-class medical image classification |
| **FL Framework** | Flower (flwr) 1.26.1 |
| **Deep Learning** | PyTorch 2.10.0 (CPU) |
| **Algorithm** | FedAvg (Federated Averaging) |
| **Clients** | 3 simulated FL clients |
| **Rounds** | 5 initial FL rounds + 3 continuation rounds via `train_sim.py` |
| **Total Images** | 13,056 |
| **Final Accuracy** | **91.73%** |

---

## Datasets

### 1. Brain Tumor MRI Classification
- **Source**: Kaggle — Brain Tumor MRI Dataset
- **Folder**: `data/raw/brain_tumer_classification/`
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Images**: ~7,200

### 2. Pneumonia Chest X-Ray Classification
- **Source**: Kaggle — Chest X-Ray Images (Pneumonia)
- **Folder**: `data/raw/pneomonia_classification/`
- **Classes**: Normal, Pneumonia
- **Images**: ~5,856

### Combined 6 Classes

| # | Class Key | Display Name | Scan Type |
|---|---|---|---|
| 0 | `glioma` | Glioma (Brain Tumor) | Brain MRI |
| 1 | `meningioma` | Meningioma (Brain Tumor) | Brain MRI |
| 2 | `notumor` | No Tumor Detected | Brain MRI |
| 3 | `pituitary` | Pituitary Tumor | Brain MRI |
| 4 | `normal` | Normal / Healthy (CXR) | Chest X-Ray |
| 5 | `pneumonia` | Pneumonia Detected (CXR) | Chest X-Ray |

---

## Project Structure

```
fl_project/
├── frontend/                           # Web Dashboard (SPA)
│   ├── index.html                      # Single-page application shell
│   ├── style.css                       # Medical-grade UI (navy + white + blue)
│   └── app.js                          # Inference, chart, OCR, doctor flow, report
│
├── src/
│   ├── api.py                          # FastAPI backend — serves SPA + all API routes
│   ├── model.py                        # MedicalCNN architecture + train/eval helpers
│   ├── dataset.py                      # PyTorch Dataset, DataLoader, CLASS_NAMES
│   ├── anonymizer.py                   # EXIF/metadata stripper (Pillow pixel rebuild)
│   ├── ocr_reader.py                   # RapidOCR wrapper (PP-OCRv3, optional)
│   ├── preprocess.py                   # Phase 1 — data loading and partitioning
│   ├── client.py                       # Flower FL client + continual learning
│   ├── server.py                       # Flower FL server (continuous / always-on)
│   ├── app.py                          # Legacy Streamlit clinical UI (dual-mode)
│   ├── evaluate_global.py              # Final evaluation and metrics
│   ├── train_sim.py                    # Standalone FedAvg simulation (no network)
│   └── utils.py                        # Shared utilities (deprecation filter)
│
├── data/
│   ├── raw/
│   │   ├── brain_tumer_classification/
│   │   │   └── Training/Testing/{glioma,meningioma,notumor,pituitary}/
│   │   └── pneomonia_classification/
│   │       └── train/val/test/{NORMAL,PNEUMONIA}/
│   ├── partitions/
│   │   ├── client_1/                   # ~3,698 images (6 class subfolders)
│   │   ├── client_2/                   # ~3,699 images (6 class subfolders)
│   │   ├── client_3/                   # ~3,700 images (6 class subfolders)
│   │   └── global_test/                # ~1,959 images (held-out evaluation set)
│   ├── new_collected_data/             # Doctor-verified images queued for next FL round
│   │   └── {glioma,meningioma,notumor,pituitary,normal,pneumonia}/
│   └── archived_data/                  # Processed images moved here after each FL round
│
├── models/
│   ├── global_model.pth                # Latest aggregated global model (always overwritten)
│   ├── global_model_round_N_<ts>.pth   # Timestamped checkpoint per round
│   └── confusion_matrix.png            # Confusion matrix heatmap
│
├── logs/
│   ├── server.log
│   ├── client1.log
│   ├── client2.log
│   └── client3.log
│
├── run_api.bat                         # ★ Launch FastAPI web dashboard (recommended)
├── run_edge.bat                        # Launch full FL edge system + Streamlit UI
├── run.bat                             # Launch FL server + 3 clients only (no UI)
├── requirements.txt                    # Pinned Python dependencies
└── README.md
```

---

## Web Dashboard (FastAPI + SPA)

### Architecture

```
Browser  ←──────────────────────────────────────────────────────►  FastAPI (src/api.py)
 │                                                                          │
 │  GET /          → serves frontend/index.html                            │
 │  GET /style.css → serves frontend/style.css                             │
 │  GET /app.js    → serves frontend/app.js                                │
 │                                                                          │
 │  GET  /api/health        → model + OCR liveness check                   │
 │  GET  /api/model-info    → class registry, scan modes, colours          │
 │  POST /api/predict       → image inference (multipart/form-data)        │
 │  POST /api/feedback      → doctor confirmation / override               │
 │  GET  /api/queue         → last 50 feedback entries                     │
 │  GET  /api/report        → plain-text diagnostic report (download)      │
 │                                                                          │
 │  Docs: /api/docs  (Swagger UI)                                           │
 │  Docs: /api/redoc (ReDoc)                                                │
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check — returns `model_loaded`, `ocr_available`, timestamp |
| `GET` | `/api/model-info` | Class names, scan-mode configs, risk colours |
| `POST` | `/api/predict` | Upload image + scan type → probabilities, top class, OCR lines |
| `POST` | `/api/feedback` | Submit doctor confirmation / override; saves anonymised image |
| `GET` | `/api/queue` | Fetch last 50 confirmed/overridden feedback entries |
| `GET` | `/api/report` | Download a plain-text diagnostic report for a session |

#### `POST /api/predict` — Request / Response

**Request** (`multipart/form-data`):

| Field | Type | Description |
|---|---|---|
| `image` | file | JPEG or PNG medical scan (max 20 MB) |
| `scan_type` | string | `"Brain MRI"` or `"Chest X-Ray"` |

**Response** (JSON):

```json
{
  "session_id": "uuid4-string",
  "filename": "abc123.jpg",
  "scan_type": "Brain MRI",
  "mode_predicted_class": "Glioma (Brain Tumor)",
  "mode_predicted_key": "glioma",
  "mode_confidence": 0.9412,
  "probabilities": {
    "glioma": 0.9412,
    "meningioma": 0.0301,
    "notumor": 0.0187,
    "pituitary": 0.0100,
    "normal": 0.0000,
    "pneumonia": 0.0000
  },
  "ocr_text": "Patient ID: ...",
  "ocr_lines": [
    { "text": "Patient ID: 00421", "confidence": 0.97 }
  ],
  "ocr_elapsed_ms": 142
}
```

### Frontend Features

| Feature | Detail |
|---|---|
| **Scan type selector** | Sidebar toggle between 🧠 Brain MRI and 🫁 Chest X-Ray |
| **Drag-and-drop upload** | JPEG / PNG, up to 20 MB; shimmer animation on hover |
| **AI inference** | Runs against `/api/predict`; shows top class + risk badge |
| **Probability chart** | Horizontal bar chart (Chart.js) — scan-type filtered classes only |
| **OCR panel** | Extracted text lines with per-line confidence badges (requires `rapidocr-onnxruntime`) |
| **Clinician review** | Doctor confirms or overrides the AI prediction via a dropdown |
| **AI Override badge** | Warning badge appears when doctor selection differs from AI |
| **Session queue** | Sidebar counters track confirmed / overridden cases this session |
| **Report download** | Plain-text diagnostic report including probabilities and OCR text |
| **Status pills** | Live header indicators for model readiness and OCR availability |
| **Responsive layout** | Collapses gracefully at 900 px and 680 px breakpoints |

---

## Model Architecture — `MedicalCNN`

**Input**: 1-channel grayscale image, **128 × 128 pixels**

```
Block 1:  Conv2d(1→32)   + BatchNorm + ReLU + MaxPool  →  64×64
Block 2:  Conv2d(32→64)  + BatchNorm + ReLU + MaxPool  →  32×32
Block 3:  Conv2d(64→128) + BatchNorm + ReLU + MaxPool  →  16×16
          AdaptiveAvgPool(4×4)                          →   4×4
          Flatten                                       →  2048
          Dropout(0.5) → Linear(2048→256) → ReLU
          Dropout(0.3) → Linear(256→6)
```

| Property | Value |
|---|---|
| **Parameters** | 618,982 |
| **Loss** | CrossEntropyLoss |
| **Optimizer** | Adam (lr = 1e-3) |
| **LR Scheduler** | StepLR (γ = 0.9 per round) |
| **Preprocessing** | RGB → BGR → Grayscale → resize 128×128 (cv2 INTER_AREA) → Normalize(0.5, 0.5) |

---

## Federated Learning Setup

```
                ┌────────────────────────────────┐
                │         FL Server (FedAvg)      │
                │  - Aggregates model weights     │
                │  - Saves checkpoint each round  │
                │  - Runs continuously (no limit) │
                └───────┬──────────┬──────────────┘
                        │          │
          ┌─────────────┘          └─────────────┐
          │                                       │
   ┌──────▼──────┐  ┌──────────────┐  ┌──────────▼────┐
   │  Client 1   │  │   Client 2   │  │   Client 3    │
   │ ~3,698 imgs │  │ ~3,699 imgs  │  │ ~3,700 imgs   │
   │ local train │  │ local train  │  │ local train   │
   └─────────────┘  └──────────────┘  └───────────────┘
```

| Property | Value |
|---|---|
| **Algorithm** | FedAvg — weights averaged after each round |
| **Local epochs** | 2 per round |
| **Batch size** | 32 |
| **Data transmitted** | Model weights only — raw images never leave the device |
| **Data split** | 85% train / 15% test, stratified by class |
| **Server mode** | Continuous (`num_rounds=9999`, `min_fit_clients=2`) |

---

## Data Partitioning

| Split | Images | Role |
|---|---|---|
| client_1 | 3,698 | Federated training |
| client_2 | 3,699 | Federated training |
| client_3 | 3,700 | Federated training |
| global_test | 1,959 | Final evaluation only |
| **Total** | **13,056** | |

---

## Continuous Edge Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                  TECNOMATE CLINICAL EDGE PIPELINE                │
│                                                                  │
│  Clinician selects Scan Type (Brain MRI / Chest X-Ray)          │
│  Clinician uploads image                                         │
│       │                                                          │
│       ▼                                                          │
│  FastAPI  /api/predict                                           │
│  ├─ AI inference → restricted to valid classes for scan type    │
│  ├─ Probabilities returned for scan-type classes only           │
│  ├─ OCR extracts text annotations (if rapidocr installed)       │
│  └─ Temp image stored server-side, keyed by session_id          │
│       │                                                          │
│       ▼  (doctor confirms or overrides)                          │
│  FastAPI  /api/feedback                                          │
│  └─ anonymizer.strip_metadata_and_save()                         │
│       Saves as {class}_{uuid4}.jpg — no EXIF, no timestamps     │
│       Image stored in data/new_collected_data/{class}/          │
│       │                                                          │
│       ▼  (next FL round triggered)                               │
│  client.py  fit()                                                │
│  ├─ ImageFolder loads new_collected_data/ as a Dataset          │
│  ├─ ConcatDataset merges with base partition                    │
│  ├─ Trains on combined data for 2 local epochs                  │
│  └─ Archives processed images → data/archived_data/            │
│       │                                                          │
│       ▼                                                          │
│  server.py  (continuous, min 2 clients, UTC timestamps)         │
│  └─ FedAvg aggregation → overwrites models/global_model.pth    │
└──────────────────────────────────────────────────────────────────┘
```

### Dual-Mode Scan Type Selector

Both UIs enforce a scan-type gate that prevents cross-modality false positives:

| Mode | Inference Classes | Override Options |
|---|---|---|
| 🧠 Brain MRI | Glioma, Meningioma, No Tumor, Pituitary | Brain classes only |
| 🫁 Chest X-Ray | Normal / Healthy, Pneumonia Detected | CXR classes only |

---

## How to Run

### Prerequisites

```
Python >= 3.10
```

Install all dependencies:

```bash
pip install fastapi "uvicorn[standard]" python-multipart torch torchvision \
            flwr opencv-python scikit-learn pillow matplotlib seaborn numpy \
            streamlit pandas
```

Optional — enable OCR:

```bash
pip install rapidocr-onnxruntime
```

---

### Option A — Web Dashboard (Recommended)

```bat
run_api.bat
```

This script:
1. Verifies the Python installation
2. Auto-installs `fastapi`, `uvicorn`, and `python-multipart` if missing
3. Starts the FastAPI server on `http://127.0.0.1:8000`
4. Opens the browser automatically after 3 seconds

The SPA frontend is served directly by FastAPI — no separate frontend server needed.

To start manually:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
Set-Location "path\to\fl_project"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API docs are available at:
- Swagger UI → `http://127.0.0.1:8000/api/docs`
- ReDoc → `http://127.0.0.1:8000/api/redoc`

---

### Option B — Full FL Edge System + Streamlit UI

```bat
run_edge.bat
```

Opens 5 separate windows: FL Server, Client 1, Client 2, Client 3, Streamlit UI.

| Component | URL / Port |
|---|---|
| Streamlit clinical UI | `http://localhost:8501` |
| FL Server (gRPC) | `localhost:8080` |

---

### Option C — FL Training Only (No UI)

```bat
run.bat
```

Starts the FL server and 3 clients in background processes; logs written to `logs/`.
Press any key in the console to stop all processes.

---

### Manual Step-by-Step

#### Step 1 — Preprocess Data *(only needed once)*

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python src/preprocess.py
```

Reads all raw images, resizes to 128×128 grayscale (cv2 INTER_AREA), and partitions into `data/partitions/`.

#### Step 2 — Start FL Server

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python src/server.py                     # default: min 2 clients, 9999 rounds
python src/server.py --min_clients 3     # require all 3 clients
```

#### Step 3 — Start FL Clients *(3 separate terminals)*

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python src/client.py --client_id 1 --batch_size 32 --epochs 2
python src/client.py --client_id 2 --batch_size 32 --epochs 2
python src/client.py --client_id 3 --batch_size 32 --epochs 2
```

> Start the server first — it waits for at least 2 clients before beginning Round 1.

#### Step 4 — Evaluate Global Model

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python src/evaluate_global.py
```

Outputs per-class precision / recall / F1 and saves `models/confusion_matrix.png`.

#### Step 5 — Continue Training (Single-Process, No Network)

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python src/train_sim.py
```

Runs standalone FedAvg simulation — identical math to the real FL stack but without gRPC or Ray. Ideal for continued training on a single machine.

#### Monitor Logs

```powershell
# Live tail of server log
Get-Content logs/server.log -Wait | Select-Object -Last 20

# Stop all Python processes
taskkill /F /IM python.exe /T
```

---

## Results

### Training Loss per Round

| Round | Train Loss | Test Accuracy | Method |
|---|---|---|---|
| 1 | 0.6916 | — | FL (gRPC, 3 clients) |
| 2 | 0.3779 | — | FL |
| 3 | 0.3322 | — | FL |
| 4 | 0.2996 | — | FL |
| 5 | 0.2653 | 89.99% | FL — best gRPC checkpoint |
| 6 | 0.3269 | 90.45% | `train_sim.py` continuation |
| 7 | 0.2961 | 91.12% | `train_sim.py` |
| 8 | 0.2889 | **91.73%** | `train_sim.py` — **current best** |

### Final Evaluation — Global Test Set (1,959 images)

| Metric | Score |
|---|---|
| **Overall Accuracy** | **91.73%** |
| **Precision (weighted)** | **91.65%** |
| **Recall (weighted)** | **91.73%** |
| **F1-Score (weighted)** | **91.63%** |

### Per-Class Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Glioma | 0.89 | 0.82 | 0.85 | 270 |
| Meningioma | 0.84 | 0.79 | 0.81 | 270 |
| No Tumor | 0.90 | 0.97 | 0.93 | 270 |
| Pituitary | 0.92 | 0.97 | 0.94 | 270 |
| Normal (CXR) | 0.93 | 0.89 | 0.91 | 238 |
| Pneumonia | 0.96 | 0.98 | 0.97 | 641 |

Confusion matrix saved at: `models/confusion_matrix.png`

---

## Key Files

| File | Purpose |
|---|---|
| `src/api.py` | FastAPI app — mounts SPA at `/`, exposes all `/api/*` routes, loads model on startup |
| `frontend/index.html` | SPA shell — header, sidebar, upload card, results grid, OCR panel, doctor panel |
| `frontend/style.css` | Medical UI — CSS custom properties, fixed chart height (no `flex:1`), sticky sidebar |
| `frontend/app.js` | Client logic — fetch wrappers, Chart.js bar chart, OCR render, feedback flow, report download |
| `src/model.py` | `MedicalCNN` definition + `train_one_epoch` / `evaluate` helpers |
| `src/dataset.py` | `MedicalDataset`, `CLASS_NAMES`, `NUM_CLASSES`, `get_transforms` |
| `src/anonymizer.py` | `strip_metadata_and_save(pil_img, path)` — pixel rebuild strips 100% of EXIF/XMP/GPS |
| `src/ocr_reader.py` | `extract_text()`, `filter_medical_text()`, `is_ocr_available()` — RapidOCR PP-OCRv3 wrapper |
| `src/client.py` | Flower `NumPyClient` — scans `new_collected_data/` before each `fit()`, archives after |
| `src/server.py` | Always-on Flower server — `num_rounds=9999`, per-round `.pth` checkpoints, UTC log timestamps |
| `src/train_sim.py` | Self-contained FedAvg simulation — no Flower network stack, no Ray |
| `src/evaluate_global.py` | Loads `global_model.pth`, runs inference on `global_test/`, prints report + saves confusion matrix |
| `run_api.bat` | One-click FastAPI + SPA launcher; auto-installs deps, opens browser |
| `run_edge.bat` | Launches FL server, 3 clients, and Streamlit UI in separate windows |
| `run.bat` | Launches FL server + 3 clients only (background, logs to `logs/`) |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| FastAPI + Vanilla SPA | Zero build step — edit HTML/CSS/JS and refresh; no Node.js required |
| Grayscale 128×128 input | Unified representation for both MRI and CXR modalities |
| cv2 INTER_AREA downscaling | Best quality for image size reduction; identical in preprocessing and inference |
| Stratified partitioning | Class proportions balanced across all 3 clients and global test set |
| Federated privacy | Raw images never leave client devices — only model weights are shared |
| UUID filenames for saved images | No timestamp = no patient metadata leakage through filename |
| Dual-mode scan type selector | Prevents cross-modality false positives at inference time |
| Chart height set before Chart.js init | Avoids the ResizeObserver feedback loop that caused endless page scrolling |
| `overflow: hidden` + explicit height on chart container | Prevents `flex: 1` from inflating the card beyond the viewport |
| Continuous server mode | Enables real-world always-on deployment without restarting |
| Per-round `.pth` checkpoints | Rollback possible if a round degrades accuracy |
| `train_sim.py` simulation mode | Single-process FedAvg — identical math, no network/Ray; ideal for continued training |
| RapidOCR (PP-OCRv3, ONNX) | Lightweight OCR without full PaddlePaddle install; gracefully degrades if absent |
| Server checkpoint resume | `get_model()` loads `global_model.pth` on API startup — restarting never resets training |

---

## Notes

- Training runs on **CPU only** — no GPU required; each FL round takes ~7–10 minutes
- `PYTHONIOENCODING=utf-8` is required on Windows to avoid cp1252 encoding errors with Flower's log output
- The pneumonia dataset folder has a typo (`pneomonia_classification`) — intentional; matches the extracted folder name from Kaggle
- The best known checkpoint is `global_model.pth` at **91.73%** accuracy, produced by 3 continuation rounds of `train_sim.py` from the Round 5 FL base
- OCR is fully optional — the dashboard degrades gracefully and shows an install hint when `rapidocr-onnxruntime` is not present