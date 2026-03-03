# Tecnomate — Federated Learning Medical Image Classifier

A privacy-preserving, continuously-learning federated learning (FL) system using **Flower** and **PyTorch** that classifies medical images across two datasets — Brain Tumor MRI scans and Pneumonia Chest X-Rays — without any patient data ever leaving the local device.

The system ships a **React + Vite** web dashboard backed by **FastAPI**, plus a legacy Streamlit UI for FL-edge workflows.

| Interface | Stack | Launch |
|---|---|---|
| **Web Dashboard** *(primary)* | FastAPI + React/Vite SPA | see [Option A](#option-a--web-dashboard-recommended) → `http://127.0.0.1:8000` |
| **Streamlit UI** *(legacy)* | Streamlit | see [Option B](#option-b--full-fl-edge-system--streamlit-ui) → `http://localhost:8501` |

Both UIs support doctor-in-the-loop feedback, automatic EXIF anonymisation, a layered upload-validation gate (ScanGate ML + visual heuristics + OCR), and a continual learning pipeline that incorporates new doctor-verified images into the next federated training round.

---

## Project Overview

| Property | Value |
|---|---|
| **Task** | 6-class medical image classification |
| **FL Framework** | Flower (flwr) 1.26.1 |
| **Deep Learning** | PyTorch ≥ 2.7.0 |
| **Algorithm** | FedAvg (Federated Averaging) |
| **Clients** | 3 simulated FL clients |
| **Rounds** | 5 initial FL rounds + continuation rounds via `train_sim.py` |
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
├── frontend/                           # React + Vite SPA (web dashboard)
│   ├── src/
│   │   ├── pages/
│   │   │   └── Classify.jsx            # Upload, scan-type selector, results UI
│   │   └── utils/
│   │       └── api.js                  # Fetch wrappers, canonicalizeScanType()
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── api.py                          # FastAPI app — all /api/* routes + SPA mount
│   ├── model.py                        # MedicalCNN architecture + train/eval helpers
│   ├── dataset.py                      # PyTorch Dataset, DataLoader, CLASS_NAMES
│   ├── anonymizer.py                   # EXIF/metadata stripper (Pillow pixel rebuild)
│   ├── ocr_reader.py                   # RapidOCR wrapper (PP-OCRv3, optional)
│   ├── scan_classifier.py              # ★ ScanGate — EfficientNet-B0 4-class gatekeeper
│   ├── train_gate.py                   # ★ Training script for scan_gate.pth
│   ├── preprocess.py                   # Phase 1 — data loading and partitioning
│   ├── client.py                       # Flower FL client + continual learning
│   ├── app.py                          # Legacy Streamlit clinical UI (dual-mode)
│   ├── evaluate_global.py              # Final evaluation and metrics
│   ├── train_sim.py                    # Standalone FedAvg simulation (no network)
│   ├── db.py                           # SQLite persistence (sessions + feedback)
│   ├── report_generator.py             # Plain-text diagnostic report builder
│   ├── latex_report.py                 # PDF report (ReportLab)
│   └── utils.py                        # Shared utilities
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
│   ├── gate_data/
│   │   ├── ct_scan/                    # Optional CT images for ScanGate training
│   │   └── non_medical/                # Optional extra non-medical images
│   ├── cifar10_cache/                  # Auto-downloaded by train_gate.py
│   ├── new_collected_data/             # Doctor-verified images → next FL round
│   │   └── {glioma,meningioma,notumor,pituitary,normal,pneumonia}/
│   ├── archived_data/                  # Processed images moved here after FL round
│   └── audit.log                       # Gate pass/reject audit log (rotating, 5 MB)
│
├── models/
│   ├── global_model.pth                # Latest aggregated global model (MedicalCNN)
│   ├── global_model_round_N_<ts>.pth   # Timestamped checkpoint per round
│   ├── scan_gate.pth                   # ★ ScanGate EfficientNet-B0 weights
│   └── confusion_matrix.png            # Confusion matrix heatmap
│
├── logs/
│   ├── server.log
│   ├── client1.log / client2.log / client3.log
│
├── run_api.bat                         # Windows: launch FastAPI + serve SPA
├── run_edge.bat                        # Windows: full FL edge system + Streamlit
├── run.bat                             # Windows: FL server + 3 clients only
├── requirements.txt                    # Pinned Python dependencies
└── README.md
```

---

## Upload Validation Gate (ScanGate)

Every image upload passes through a **three-layer gate** before reaching the main clinical model. This prevents wrong-modality uploads and non-medical images from being classified.

```
Upload
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — ScanGate ML  (backend/scan_classifier.py)    │
│  EfficientNet-B0 4-class classifier                     │
│  Classes: chest_xray · brain_mri · ct_scan · non_medical│
│  Threshold: 0.65 confidence                             │
│  → REJECT if predicted class ≠ expected OR conf < 0.65  │
└─────────────────────────┬───────────────────────────────┘
                          │ PASS
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 — Visual Heuristic  (_is_likely_medical_scan)  │
│  Pixel-level grayscale ratio, saturation, histogram     │
│  Also runs browser-side (canvas API) on file select     │
│  → REJECT if image is clearly non-medical / colour photo│
└─────────────────────────┬───────────────────────────────┘
                          │ PASS
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 — OCR Gate  (backend/ocr_reader.py)            │
│  RapidOCR PP-OCRv3 scans for modality text markers      │
│  • OCR confirms correct modality  → ALLOW               │
│  • OCR confirms wrong modality    → REJECT               │
│  • OCR inconclusive (no text)     → defer to ScanGate   │
│    – ScanGate conf ≥ 0.65         → ALLOW               │
│    – ScanGate conf < 0.65         → REJECT               │
└─────────────────────────┬───────────────────────────────┘
                          │ PASS
                          ▼
                  Main CNN Inference
                  (MedicalCNN global_model.pth)
```

### ScanGate Classes & Routing

| Gate Label | Accepted For |
|---|---|
| `chest_xray` | Chest X-Ray mode |
| `brain_mri` | Brain MRI mode |
| `ct_scan` | Rejected (not a supported inference mode) |
| `non_medical` | Always rejected |

### Audit Log Events

All gate decisions are written to `data/audit.log`:

| Event | Meaning |
|---|---|
| `SCAN_GATE_PASS` | ScanGate confirmed expected modality |
| `SCAN_GATE_REJECT` | ScanGate identified wrong modality or non-medical |
| `OCR_INCONCLUSIVE_GATE_PASS` | No OCR text found; ScanGate confidence ≥ 0.65 — allowed |
| `OCR_INCONCLUSIVE_GATE_REJECT` | No OCR text found; ScanGate confidence < 0.65 — rejected |

### Test Results (local partition sweep)

| Test | Result |
|---|---|
| Chest X-Rays in Chest X-Ray mode | 690 / 690 passed |
| Brain MRIs in Brain MRI mode | 1,799 / 1,800 passed |
| Brain MRI uploaded as Chest X-Ray | Blocked |
| Chest X-Ray uploaded as Brain MRI | Blocked |
| Non-medical / CIFAR-10 images | Blocked |

---

## Web Dashboard (FastAPI + React SPA)

### Architecture

```
Browser (React/Vite)  ◄──────────────────────►  FastAPI (backend/api.py)
 │                                                        │
 │  GET  /                  → serves built SPA            │
 │  GET  /api/health        → model + OCR liveness        │
 │  GET  /api/model-info    → class registry, scan modes  │
 │  POST /api/ocr-check     → standalone OCR pre-check    │
 │  POST /api/predict       → full gate + inference       │
 │  POST /api/feedback      → doctor confirm / override   │
 │  GET  /api/queue         → last 50 feedback entries    │
 │  GET  /api/report        → plain-text report download  │
 │  POST /api/pdf-report    → PDF report generation       │
 │  GET  /api/admin/stats           → aggregate stats     │
 │  GET  /api/admin/feedback        → paginated log       │
 │  GET  /api/admin/export-csv      → full CSV export     │
 │  GET  /api/admin/export-excel    → Excel workbook      │
 │  GET  /api/admin/sessions        → paginated sessions  │
 │                                                        │
 │  Docs: /api/docs  (Swagger UI)                         │
 │  Docs: /api/redoc (ReDoc)                              │
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness — returns `model_loaded`, `ocr_available`, `scan_gate_loaded` |
| `GET` | `/api/model-info` | Class names, scan-mode configs, risk colours |
| `POST` | `/api/ocr-check` | Standalone OCR pre-check (called by frontend before predict) |
| `POST` | `/api/predict` | Upload image + scan type → full gate + probabilities + OCR lines |
| `POST` | `/api/feedback` | Doctor confirmation / override; saves anonymised image |
| `GET` | `/api/queue` | Fetch last 50 confirmed/overridden feedback entries |
| `GET` | `/api/report` | Download plain-text diagnostic report for a session |
| `POST` | `/api/pdf-report` | Generate and download PDF diagnostic report |
| `GET` | `/api/admin/stats` | Aggregate statistics from SQLite |
| `GET` | `/api/admin/feedback` | Paginated feedback log |
| `GET` | `/api/admin/export-csv` | Download full feedback CSV |
| `GET` | `/api/admin/export-excel` | Download full admin report as Excel workbook |
| `GET` | `/api/admin/sessions` | Paginated prediction sessions |

#### `POST /api/predict` — Request / Response

**Request** (`multipart/form-data`):

| Field | Type | Description |
|---|---|---|
| `image` | file | JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, HEIC, DICOM — max 30 MB |
| `scan_type` | string | `"Brain MRI"` or `"Chest X-Ray"` |

**Response** (JSON):

```json
{
  "session_id": "uuid4-string",
  "filename": "abc123.jpg",
  "scan_type": "Chest X-Ray",
  "mode_predicted_class": "Pneumonia Detected",
  "mode_predicted_key": "pneumonia",
  "mode_confidence": 0.9723,
  "probabilities": {
    "glioma": 0.0000,
    "meningioma": 0.0000,
    "notumor": 0.0000,
    "pituitary": 0.0000,
    "normal": 0.0277,
    "pneumonia": 0.9723
  },
  "ocr_text": "AP CHEST",
  "ocr_lines": [
    { "text": "AP CHEST", "confidence": 0.94 }
  ],
  "ocr_elapsed_ms": 138,
  "gate_label": "chest_xray",
  "gate_confidence": 0.9712
}
```

**Gate rejection response** (HTTP 400):

```json
{
  "detail": "ScanGate rejected: predicted 'brain_mri' (conf 0.91) — expected 'chest_xray'. Upload a Chest X-Ray image."
}
```

### Frontend Features

| Feature | Detail |
|---|---|
| **Scan type selector** | Toggle between 🧠 Brain MRI and 🫁 Chest X-Ray |
| **Browser-side visual check** | Canvas analysis on file select — rejects obvious non-medical images before upload |
| **Classify button guard** | Button disabled while browser visual check is running |
| **Drag-and-drop upload** | JPEG / PNG / WebP up to 30 MB; shimmer animation on hover |
| **AI inference** | Runs against `/api/predict` through full gate; shows top class + risk badge |
| **Probability chart** | Horizontal bar chart — scan-type filtered classes only |
| **OCR panel** | Extracted text lines with per-line confidence badges |
| **Gate status message** | Shows whether ScanGate or OCR (or both) confirmed the image |
| **Clinician review** | Doctor confirms or overrides the AI prediction via a dropdown |
| **AI Override badge** | Warning badge when doctor selection differs from AI |
| **Session queue** | Sidebar counters: confirmed / overridden cases this session |
| **Report download** | Plain-text or PDF diagnostic report |
| **Status pills** | Live header indicators for model readiness, OCR, and ScanGate |
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
| **Preprocessing** | RGB → Grayscale → resize 128×128 (cv2 INTER_AREA) → Normalize(0.5, 0.5) |

---

## ScanGate Architecture — `EfficientNet-B0`

**Input**: RGB image, **224 × 224 pixels** (ImageNet normalisation)

| Property | Value |
|---|---|
| **Base model** | EfficientNet-B0 (pretrained ImageNet) |
| **Output classes** | 4 (`chest_xray`, `brain_mri`, `ct_scan`, `non_medical`) |
| **Gate threshold** | 0.65 confidence (empirically set; real X-rays score ≥ 0.70) |
| **Weights** | `models/scan_gate.pth` |
| **Fallback** | Pure pixel heuristics if `scan_gate.pth` is missing |
| **Training data** | `data/partitions/global_test/` (medical) + CIFAR-10 (non-medical) |
| **Non-medical CIFAR classes** | cat, dog, horse, deer, ship, automobile, bird, frog |

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
│       ▼  (browser canvas pre-check — blocks obvious non-medical) │
│  FastAPI  /api/ocr-check  (OCR pre-flight)                       │
│       │                                                          │
│       ▼                                                          │
│  FastAPI  /api/predict                                           │
│  ├─ Layer 1: ScanGate ML (EfficientNet-B0)                      │
│  ├─ Layer 2: Visual heuristic (_is_likely_medical_scan)         │
│  ├─ Layer 3: OCR gate (modality text markers)                   │
│  ├─ Main CNN inference (MedicalCNN — restricted to scan mode)   │
│  └─ Probabilities + gate metadata returned                      │
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

| Mode | Inference Classes | Gate Label Required |
|---|---|---|
| 🧠 Brain MRI | Glioma, Meningioma, No Tumor, Pituitary | `brain_mri` |
| 🫁 Chest X-Ray | Normal / Healthy, Pneumonia Detected | `chest_xray` |

---

## How to Run

### Quick Start *(new laptop / first run)*

Everything you need to go from a fresh clone to a running app — copy and paste in order.

#### macOS / Linux

```bash
# 1. Clone and enter the project
git clone https://github.com/prasadxa/fl_project
cd fl_project

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install and build the frontend
cd frontend && npm install && npm run build && cd ..

# 4. Start the backend (serves the app at http://127.0.0.1:8000)
PYTHONIOENCODING=utf-8 uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

#### Windows (PowerShell)

```powershell
# 1. Clone and enter the project
git clone https://github.com/prasadxa/fl_project
cd fl_project

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install and build the frontend
cd frontend; npm install; npm run build; cd ..

# 4. Start the backend (serves the app at http://127.0.0.1:8000)
$env:PYTHONIOENCODING = 'utf-8'
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Then open **http://127.0.0.1:8000** in your browser. That's it — no retraining needed, model weights are included in the repo.

> **Note:** Steps 1–3 are one-time setup. After that, only step 4 is needed to start the app.

---

### Development Mode *(hot-reload for frontend changes)*

Run the backend and frontend in two separate terminals for live UI editing:

#### macOS / Linux

```bash
# Terminal 1 — Backend (API)
cd fl_project
PYTHONIOENCODING=utf-8 uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend (Vite dev server with hot-reload)
cd fl_project/frontend
npm run dev
```

#### Windows (PowerShell)

```powershell
# Terminal 1 — Backend (API)
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Frontend (Vite dev server with hot-reload)
cd fl_project/frontend
npm run dev
```

| Service | URL |
|---|---|
| Frontend (Vite, hot-reload) | http://localhost:5173 |
| Backend API | http://127.0.0.1:8000 |
| Swagger UI | http://127.0.0.1:8000/api/docs |

---

### Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18 (for the React frontend)
- Datasets downloaded from Kaggle into `data/raw/` (see [Datasets](#datasets))

---

### Step 0 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

Install individually (if you prefer):

```bash
pip install torch torchvision fastapi "uvicorn[standard]" python-multipart \
    opencv-python Pillow numpy scikit-learn pandas openpyxl \
    matplotlib seaborn reportlab flwr streamlit
```

Optional — enable OCR:

```bash
pip install rapidocr-onnxruntime
```

Optional — enable DICOM support:

```bash
pip install pydicom
```

---

### Step 0b — Build the Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

> The built files land in `frontend/dist/`. FastAPI serves them automatically — no separate frontend server is needed in production.

For live frontend development (hot-reload):

```bash
cd frontend
npm run dev
```

Then start the backend separately (see Option A below) and the Vite dev server will proxy API calls.

---

### Option A — Web Dashboard (Recommended)

Start the FastAPI backend (serves the built SPA at `/`):

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

**Windows (PowerShell):**

```powershell
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

**Windows (CMD / `.bat`):**

```bat
run_api.bat
```

Then open: **http://127.0.0.1:8000**

Interactive API docs:
- Swagger UI → **http://127.0.0.1:8000/api/docs**
- ReDoc → **http://127.0.0.1:8000/api/redoc**

---

### Option B — Full FL Edge System + Streamlit UI

**Windows:**

```bat
run_edge.bat
```

**macOS / Linux** (5 separate terminal tabs):

```bash
# Terminal 1 — FL Server
cd fl_project
PYTHONIOENCODING=utf-8 python backend/app.py --mode server

# Terminal 2 — FL Client 1
cd fl_project
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 1

# Terminal 3 — FL Client 2
cd fl_project
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 2

# Terminal 4 — FL Client 3
cd fl_project
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 3

# Terminal 5 — Streamlit UI
cd fl_project
streamlit run backend/app.py
```

| Component | URL / Port |
|---|---|
| Streamlit clinical UI | `http://localhost:8501` |
| FL Server (gRPC) | `localhost:8080` |

---

### Option C — FL Training Only (No UI)

**Windows:**

```bat
run.bat
```

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 1 &
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 2 &
PYTHONIOENCODING=utf-8 python backend/client.py --client_id 3 &
```

Logs are written to `logs/client1.log`, `logs/client2.log`, `logs/client3.log`.

---

### Manual Step-by-Step

#### Step 1 — Preprocess Data *(only needed once)*

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 python backend/preprocess.py
```

**Windows (PowerShell):**

```powershell
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
python backend/preprocess.py
```

Reads all raw images, resizes to 128×128 grayscale (cv2 INTER_AREA), and partitions into `data/partitions/`.

---

#### Step 2 — Train the ScanGate *(only needed once, or to retrain)*

**macOS / Linux:**

```bash
cd fl_project
python backend/train_gate.py
```

With options:

```bash
# Custom hyperparameters
python backend/train_gate.py --epochs 20 --batch-size 64 --lr 3e-4

# Skip CIFAR-10 download (use only local non-medical images)
python backend/train_gate.py --no-cifar

# Force CPU even if GPU available
python backend/train_gate.py --device cpu
```

**Windows (PowerShell):**

```powershell
cd fl_project
python backend/train_gate.py
python backend/train_gate.py --epochs 20 --batch-size 64 --lr 3e-4
```

Output saved to `models/scan_gate.pth`. The script downloads CIFAR-10 automatically into `data/cifar10_cache/` on first run (requires internet).

---

#### Step 3 — Start FL Server

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 python backend/app.py --mode server
# or with options:
PYTHONIOENCODING=utf-8 python backend/app.py --mode server --min_clients 3
```

**Windows (PowerShell):**

```powershell
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
python backend/app.py --mode server
python backend/app.py --mode server --min_clients 3
```

---

#### Step 4 — Start FL Clients *(3 separate terminals)*

**macOS / Linux:**

```bash
# Terminal 1
cd fl_project && PYTHONIOENCODING=utf-8 python backend/client.py --client_id 1 --batch_size 32 --epochs 2

# Terminal 2
cd fl_project && PYTHONIOENCODING=utf-8 python backend/client.py --client_id 2 --batch_size 32 --epochs 2

# Terminal 3
cd fl_project && PYTHONIOENCODING=utf-8 python backend/client.py --client_id 3 --batch_size 32 --epochs 2
```

**Windows (PowerShell):**

```powershell
# Terminal 1
$env:PYTHONIOENCODING = 'utf-8'; python backend/client.py --client_id 1 --batch_size 32 --epochs 2

# Terminal 2
$env:PYTHONIOENCODING = 'utf-8'; python backend/client.py --client_id 2 --batch_size 32 --epochs 2

# Terminal 3
$env:PYTHONIOENCODING = 'utf-8'; python backend/client.py --client_id 3 --batch_size 32 --epochs 2
```

> Start the server first — it waits for at least 2 clients before beginning Round 1.

---

#### Step 5 — Evaluate Global Model

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 python backend/evaluate_global.py
```

**Windows (PowerShell):**

```powershell
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
python backend/evaluate_global.py
```

Outputs per-class precision / recall / F1 and saves `models/confusion_matrix.png`.

---

#### Step 6 — Continue Training (Single-Process, No Network)

**macOS / Linux:**

```bash
cd fl_project
PYTHONIOENCODING=utf-8 python backend/train_sim.py
```

**Windows (PowerShell):**

```powershell
cd fl_project
$env:PYTHONIOENCODING = 'utf-8'
python backend/train_sim.py
```

Runs standalone FedAvg simulation — identical math to the real FL stack but without gRPC. Ideal for continued training on a single machine.

---

### Monitor & Tail Logs

**macOS / Linux:**

```bash
# Live tail — server log
tail -f fl_project/logs/server.log

# Live tail — all client logs at once
tail -f fl_project/logs/client1.log fl_project/logs/client2.log fl_project/logs/client3.log

# Tail audit gate log
tail -f fl_project/data/audit.log

# Search for gate rejections
grep "SCAN_GATE_REJECT" fl_project/data/audit.log

# Search for OCR inconclusive passes
grep "OCR_INCONCLUSIVE_GATE_PASS" fl_project/data/audit.log
```

**Windows (PowerShell):**

```powershell
# Live tail — server log
Get-Content logs/server.log -Wait | Select-Object -Last 30

# Live tail — audit log
Get-Content data/audit.log -Wait | Select-Object -Last 30

# Search gate rejections
Select-String "SCAN_GATE_REJECT" data/audit.log

# Stop all Python processes
taskkill /F /IM python.exe /T
```

---

### Smoke-Test API Endpoints

After starting the backend, verify it is healthy:

```bash
# Health check
curl http://127.0.0.1:8000/api/health

# Model info
curl http://127.0.0.1:8000/api/model-info

# Upload a test image (Chest X-Ray)
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "image=@/path/to/chest_xray.jpg" \
  -F "scan_type=Chest X-Ray"

# Upload a test image (Brain MRI)
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "image=@/path/to/brain_mri.jpg" \
  -F "scan_type=Brain MRI"

# Run standalone OCR pre-check
curl -X POST http://127.0.0.1:8000/api/ocr-check \
  -F "image=@/path/to/chest_xray.jpg" \
  -F "scan_type=Chest X-Ray"
```

**Windows (PowerShell — using Invoke-RestMethod):**

```powershell
# Health check
Invoke-RestMethod http://127.0.0.1:8000/api/health

# Upload a test image
$form = @{
    image     = Get-Item "C:\path\to\chest_xray.jpg"
    scan_type = "Chest X-Ray"
}
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/predict -Method Post -Form $form
```

---

### Deployment Checklist

After pulling new code or retraining the ScanGate:

```bash
# 1. Pull latest code
cd fl_project
git pull

# 2. Install any new dependencies
pip install -r requirements.txt

# 3. Rebuild the frontend (if frontend files changed)
cd frontend && npm install && npm run build && cd ..

# 4. Retrain ScanGate (if scan_gate.pth is missing or you want to update it)
python backend/train_gate.py

# 5. Restart the backend
pkill -f "uvicorn backend.api"
PYTHONIOENCODING=utf-8 uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

# 6. Confirm ScanGate is loaded
curl http://127.0.0.1:8000/api/health | python3 -m json.tool
# → "scan_gate_loaded": true
```

**Windows (PowerShell):**

```powershell
# 1. Pull latest code
cd fl_project
git pull

# 2. Install any new dependencies
pip install -r requirements.txt

# 3. Rebuild the frontend
cd frontend; npm install; npm run build; cd ..

# 4. Retrain ScanGate
python backend/train_gate.py

# 5. Restart the backend (kill old process first)
taskkill /F /IM python.exe /T
$env:PYTHONIOENCODING = 'utf-8'
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

# 6. Confirm ScanGate is loaded
Invoke-RestMethod http://127.0.0.1:8000/api/health
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

### ScanGate Gate Confidence (local partition test)

| Modality | Mean Confidence | Min Confidence | Pass Rate |
|---|---|---|---|
| Chest X-Ray | 96.7% | 69.7% | 690/690 |
| Brain MRI | ~95% | ~71% | 1,799/1,800 |

---

## Key Files

| File | Purpose |
|---|---|
| `backend/api.py` | FastAPI app — mounts SPA at `/`, exposes all `/api/*` routes, loads models on startup |
| `backend/scan_classifier.py` | **ScanGate** — EfficientNet-B0 4-class gatekeeper; heuristic fallback if weights missing |
| `backend/train_gate.py` | Training script for `scan_gate.pth`; downloads CIFAR-10 automatically |
| `backend/model.py` | `MedicalCNN` definition + `train_one_epoch` / `evaluate` helpers |
| `backend/dataset.py` | `MedicalDataset`, `CLASS_NAMES`, `NUM_CLASSES`, `get_transforms` |
| `backend/anonymizer.py` | `strip_metadata_and_save()` — pixel rebuild strips 100% of EXIF/XMP/GPS |
| `backend/ocr_reader.py` | `extract_text()`, `filter_medical_text()`, `is_ocr_available()` — RapidOCR wrapper |
| `backend/client.py` | Flower `NumPyClient` — scans `new_collected_data/` before each `fit()`, archives after |
| `backend/train_sim.py` | Self-contained FedAvg simulation — no Flower network stack |
| `backend/evaluate_global.py` | Loads `global_model.pth`, runs inference on `global_test/`, prints report + confusion matrix |
| `frontend/src/utils/api.js` | Fetch wrappers + `canonicalizeScanType()` (prevents string-mismatch bugs) |
| `frontend/src/pages/Classify.jsx` | Main upload UI, canvas pre-check, scan-type selector, results render |
| `models/global_model.pth` | Latest MedicalCNN weights (6-class classifier) |
| `models/scan_gate.pth` | ScanGate EfficientNet-B0 weights |
| `data/audit.log` | Rotating audit log for all gate pass/reject events |
| `run_api.bat` | Windows: one-click FastAPI launcher |
| `run_edge.bat` | Windows: launches FL server, 3 clients, and Streamlit UI |
| `run.bat` | Windows: FL server + 3 clients only (background, logs to `logs/`) |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Three-layer upload gate (ScanGate → visual → OCR) | OCR alone is brittle — clean clinical images often have no text; layered approach handles the full range of real-world inputs |
| EfficientNet-B0 for ScanGate | Lightweight (~5M params), pretrained on ImageNet, excellent transfer to medical modality detection |
| Confidence threshold 0.65 | Real X-rays score ≥ 0.70 (min observed); non-medical images score ≤ 0.58 — safe margin with no false rejects in the test set |
| OCR inconclusive → defer to ScanGate | Many real clinical images have no overlay text; requiring OCR confirmation would block 100% of clean scans |
| `canonicalizeScanType()` on frontend + backend | Prevents silent routing bugs from whitespace variants like `"Brain  MRI"` (double space) |
| Browser canvas pre-check | Rejects obvious non-medical images before the network round-trip; improves UX latency |
| FastAPI + React/Vite SPA | Vite provides fast HMR in development; production build served directly by FastAPI — no separate server |
| Grayscale 128×128 for MedicalCNN | Unified representation for both MRI and CXR modalities |
| cv2 INTER_AREA downscaling | Best quality for image size reduction; identical in preprocessing and inference |
| Stratified partitioning | Class proportions balanced across all 3 clients and global test set |
| Federated privacy | Raw images never leave client devices — only model weights are shared |
| UUID filenames for saved images | No timestamp = no patient metadata leakage through filename |
| Audit log for gate events | Full observability: every SCAN_GATE_PASS / REJECT event is persisted with confidence and timestamp |
| Heuristic fallback in ScanGate | Server never hard-crashes on a missing `scan_gate.pth`; degrades gracefully to pixel heuristics |
| Per-round `.pth` checkpoints | Rollback possible if a round degrades accuracy |
| `train_sim.py` simulation mode | Single-process FedAvg — identical math, no network/Ray; ideal for continued training |
| RapidOCR (PP-OCRv3, ONNX) | Lightweight OCR without full PaddlePaddle install; gracefully degrades if absent |
| CIFAR-10 for non-medical training data | Auto-downloaded; provides diverse everyday objects; deliberately excludes airplane/truck (ambiguous with radiology screenshots) |

---

## Notes

- Training runs on **CPU only** by default — no GPU required; each FL round takes ~7–10 minutes
- `PYTHONIOENCODING=utf-8` is required on Windows to avoid cp1252 encoding errors with Flower's log output
- The pneumonia dataset folder has a typo (`pneomonia_classification`) — intentional; matches the extracted folder name from Kaggle
- The best known checkpoint is `global_model.pth` at **91.73%** accuracy, produced by 3 continuation rounds of `train_sim.py` from the Round 5 FL base
- OCR is fully optional — the dashboard degrades gracefully and shows an install hint when `rapidocr-onnxruntime` is not present
- `scan_gate.pth` is required for full gate functionality but is not strictly required to start the server — if missing, the backend logs a warning and falls back to heuristics
- CIFAR-10 is downloaded automatically on the first `train_gate.py` run (~163 MB); cached in `data/cifar10_cache/` for subsequent runs
- The ScanGate keeps RGB colour information at inference time — this is intentional; colour is the primary signal used to reject non-medical photos