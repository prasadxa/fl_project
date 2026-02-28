# Federated Learning for Medical Image Classification

A privacy-preserving federated learning (FL) system using **Flower** and **PyTorch** that classifies medical images across two datasets — Brain Tumor MRI scans and Pneumonia Chest X-Rays — without any client data ever leaving the local device.

---

## Project Overview

| Property | Value |
|---|---|
| **Task** | 6-class medical image classification |
| **FL Framework** | Flower (flwr) 1.26.1 |
| **Deep Learning** | PyTorch 2.10.0 (CPU) |
| **Algorithm** | FedAvg (Federated Averaging) |
| **Clients** | 3 simulated FL clients |
| **Rounds** | 5 federation rounds |
| **Total Images** | 13,056 |
| **Final Accuracy** | **89.99%** |

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

| # | Class | Dataset |
|---|---|---|
| 0 | Glioma | Brain Tumor MRI |
| 1 | Meningioma | Brain Tumor MRI |
| 2 | No Tumor | Brain Tumor MRI |
| 3 | Pituitary | Brain Tumor MRI |
| 4 | Normal | Pneumonia CXR |
| 5 | Pneumonia | Pneumonia CXR |

---

## Project Structure

```
fl_project/
├── data/
│   ├── raw/
│   │   ├── brain_tumer_classification/
│   │   │   └── Training/Testing/{glioma,meningioma,notumor,pituitary}/
│   │   └── pneomonia_classification/
│   │       └── train/val/test/{NORMAL,PNEUMONIA}/
│   └── partitions/
│       ├── client_1/          # ~3,698 images (6 class subfolders)
│       ├── client_2/          # ~3,699 images (6 class subfolders)
│       ├── client_3/          # ~3,700 images (6 class subfolders)
│       └── global_test/       # ~1,959 images (6 class subfolders)
├── src/
│   ├── preprocess.py          # Phase 1 – data loading and partitioning
│   ├── dataset.py             # Phase 2 – PyTorch Dataset and DataLoader
│   ├── model.py               # Phase 3 – MedicalCNN architecture
│   ├── client.py              # Phase 4 – Flower FL client
│   ├── server.py              # Phase 5 – Flower FL server (FedAvg)
│   └── evaluate_global.py     # Phase 6 – final evaluation and metrics
├── models/
│   ├── global_model.pth           # Best global model (latest round)
│   ├── global_model_round_1.pth   # Checkpoint after round 1
│   ├── global_model_round_2.pth   # Checkpoint after round 2
│   ├── global_model_round_3.pth   # Checkpoint after round 3
│   ├── global_model_round_4.pth   # Checkpoint after round 4
│   ├── global_model_round_5.pth   # Checkpoint after round 5
│   └── confusion_matrix.png       # Confusion matrix (saved by evaluate_global.py)
├── logs/
│   ├── server.log
│   ├── client1.log
│   ├── client2.log
│   └── client3.log
├── run.bat                    # One-click Windows launcher
└── README.md
```

---

## Model Architecture — `MedicalCNN`

Input: **1-channel grayscale image, 128×128 pixels**

```
Block 1:  Conv2d(1→32)   + BatchNorm + ReLU + MaxPool  →  64×64
Block 2:  Conv2d(32→64)  + BatchNorm + ReLU + MaxPool  →  32×32
Block 3:  Conv2d(64→128) + BatchNorm + ReLU + MaxPool  →  16×16
          AdaptiveAvgPool(4×4)                          →   4×4
          Flatten                                       →  2048
          Dropout(0.5) → Linear(2048→256) → ReLU
          Dropout(0.3) → Linear(256→6)
```

- **Total parameters**: 618,982
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-3)
- **LR Scheduler**: StepLR (gamma=0.9 per round)

---

## Federated Learning Setup

```
                ┌────────────────────────────────┐
                │         FL Server (FedAvg)      │
                │  - Aggregates model weights     │
                │  - Saves checkpoint each round  │
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

- Algorithm: **FedAvg** — model weights averaged after each round
- Each client trains locally for **2 epochs** with **batch size 32**
- Only model weights (not data) are sent to the server
- Data split: **85% train / 15% test**, stratified by class

---

## Data Partitioning

| Split | Images | Role |
|---|---|---|
| client_1 | 3,698 | Federated training |
| client_2 | 3,699 | Federated training |
| client_3 | 3,700 | Federated training |
| global_test | 1,959 | Final evaluation only |
| **Total** | **13,056** | **100% of all images** |

---

## Requirements

```
Python >= 3.10
torch >= 2.0.0
flwr >= 1.0.0
torchvision
opencv-python
scikit-learn
Pillow
matplotlib
seaborn
numpy
```

Install all dependencies:
```bash
pip install torch torchvision flwr opencv-python scikit-learn pillow matplotlib seaborn numpy
```

---

## How to Run

### Step 1 — Preprocess Data

```powershell
$env:PYTHONIOENCODING='utf-8'
Set-Location "path\to\fl_project\fl_project"
python src/preprocess.py
```

This reads all raw images, resizes to 128×128 grayscale, and partitions them into `data/partitions/`.

### Step 2 — Start Federated Training

Open **4 separate PowerShell terminals**, all from `fl_project\fl_project\`:

**Terminal 1 — Server:**
```powershell
$env:PYTHONIOENCODING='utf-8'
python src/server.py *> logs/server.log
```

**Terminal 2 — Client 1:**
```powershell
$env:PYTHONIOENCODING='utf-8'
python src/client.py --client_id 1 --batch_size 32 --epochs 2 *> logs/client1.log
```

**Terminal 3 — Client 2:**
```powershell
$env:PYTHONIOENCODING='utf-8'
python src/client.py --client_id 2 --batch_size 32 --epochs 2 *> logs/client2.log
```

**Terminal 4 — Client 3:**
```powershell
$env:PYTHONIOENCODING='utf-8'
python src/client.py --client_id 3 --batch_size 32 --epochs 2 *> logs/client3.log
```

> **Note**: Start the server first, then the 3 clients. The server waits until all 3 clients connect before beginning Round 1.

### Step 3 — Evaluate Global Model

```powershell
$env:PYTHONIOENCODING='utf-8'
python src/evaluate_global.py
```

Outputs per-class metrics and saves `models/confusion_matrix.png`.

### One-Click (Windows)

```bat
run.bat
```

---

## Results

### Training Loss per Round

| Round | Distributed Loss |
|---|---|
| 1 | 0.6916 |
| 2 | 0.3779 |
| 3 | 0.3322 |
| 4 | 0.2996 |
| 5 | 0.2653 |

### Final Evaluation on Global Test Set (1,959 images)

| Metric | Score |
|---|---|
| **Accuracy** | **89.99%** |
| **Precision (weighted)** | **89.98%** |
| **Recall (weighted)** | **89.99%** |
| **F1-Score (weighted)** | **89.82%** |

### Per-Class Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Glioma | 0.81 | 0.83 | 0.82 | 270 |
| Meningioma | 0.84 | 0.69 | 0.76 | 270 |
| No Tumor | 0.91 | 0.96 | 0.93 | 270 |
| Pituitary | 0.88 | 0.97 | 0.92 | 270 |
| Normal (CXR) | 0.88 | 0.94 | 0.91 | 238 |
| Pneumonia | 0.98 | 0.95 | 0.96 | 641 |

**Confusion matrix**: `models/confusion_matrix.png`

---

## Key Design Decisions

- **Grayscale input**: Both MRI and X-Ray images converted to single-channel 128×128 for a unified model
- **Stratified splits**: Class proportions preserved across all 3 clients and test set to prevent data imbalance
- **Federated privacy**: Raw images never leave client devices — only model gradients/weights are shared
- **Weighted metrics**: Precision/Recall/F1 are weighted to handle class imbalance (pneumonia has 2.4× more samples)
- **Per-round checkpoints**: Model saved after every round for rollback or comparison

---

## Notes

- Training was done on **CPU only** (no GPU) — each round takes ~8–10 minutes
- The pneumonia dataset folder has a typo (`pneomonia_classification`) — this is intentional and matches the actual extracted folder name
- `PYTHONIOENCODING=utf-8` must be set on Windows to avoid cp1252 encoding errors
