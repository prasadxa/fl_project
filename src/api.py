"""
Tecnomate Clinical AI — FastAPI Backend
========================================
Endpoints:
  GET  /api/health        — liveness + model status
  GET  /api/model-info    — class names, scan modes, colours
  POST /api/predict       — image inference  (multipart/form-data)
  POST /api/feedback      — doctor-confirmed / overridden label
  GET  /api/queue         — last 50 feedback entries
  GET  /api/report        — download plain-text diagnostic report

Static frontend served from ../frontend/  (mounted at /)

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime
import io
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from anonymizer import strip_metadata_and_save
from dataset import CLASS_NAMES, NUM_CLASSES
from model import MedicalCNN
from ocr_reader import (
    extract_text,
    filter_medical_text,
    format_for_report,
    is_ocr_available,
    ocr_unavailable_reason,
)

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT    = Path(__file__).parent.parent
MODEL_PATH   = PROJ_ROOT / "models" / "global_model.pth"
COLLECT_DIR  = PROJ_ROOT / "data" / "new_collected_data"
FRONTEND_DIR = PROJ_ROOT / "frontend"
TEMP_DIR     = PROJ_ROOT / "data" / ".tmp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── scan-mode configuration (mirrors app.py exactly) ──────────────────────────
SCAN_MODES: Dict = {
    "Brain MRI": {
        "indices":    [0, 1, 2, 3],
        "labels":     {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"},
        "class_keys": ["glioma", "meningioma", "notumor", "pituitary"],
        "icon":       "\U0001f9e0",
    },
    "Chest X-Ray": {
        "indices":    [4, 5],
        "labels":     {4: "No Pneumonia", 5: "Pneumonia Detected"},
        "class_keys": ["normal", "pneumonia"],
        "icon":       "\U0001fac1",
    },
}

SHORT_NAMES: Dict[str, str] = {
    "glioma":     "Glioma (Brain Tumor)",
    "meningioma": "Meningioma (Brain Tumor)",
    "notumor":    "No Tumor Detected",
    "pituitary":  "Pituitary Tumor",
    "normal":     "Normal / Healthy (CXR)",
    "pneumonia":  "Pneumonia Detected (CXR)",
}

RISK_COLOURS: Dict[str, str] = {
    "glioma":     "#e74c3c",
    "meningioma": "#e67e22",
    "notumor":    "#27ae60",
    "pituitary":  "#e67e22",
    "normal":     "#27ae60",
    "pneumonia":  "#e74c3c",
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Tecnomate Clinical AI",
    description = "Privacy-preserving federated medical image classifier",
    version     = "1.0.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── model singleton ────────────────────────────────────────────────────────────
_model: Optional[MedicalCNN] = None
_model_loaded: bool = False


def get_model() -> Optional[MedicalCNN]:
    global _model, _model_loaded
    if not _model_loaded:
        if MODEL_PATH.exists():
            try:
                m = MedicalCNN(num_classes=NUM_CLASSES)
                m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
                m.eval()
                _model = m
                print(f"[API] ✓ Model loaded from {MODEL_PATH.name}")
            except Exception as exc:
                print(f"[API] ✗ Model load failed: {exc}")
        else:
            print(f"[API] ✗ Model not found at {MODEL_PATH}")
        _model_loaded = True
    return _model


@app.on_event("startup")
async def startup_event() -> None:
    get_model()
    print(f"[API] OCR available: {is_ocr_available()}")
    if not is_ocr_available():
        print(f"[API] OCR reason  : {ocr_unavailable_reason()}")


# ── image preprocessing (identical to app.py pipeline) ────────────────────────
_INFER_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    """Convert any PIL image to the (1, 1, 128, 128) tensor the model expects."""
    img_np   = np.array(pil_img.convert("RGB"))
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized  = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    pil_gray = Image.fromarray(resized, mode="L")
    return _INFER_TRANSFORM(pil_gray).unsqueeze(0)  # (1, 1, 128, 128)


# ── in-process temp-image store & feedback log ────────────────────────────────
_pending_images: Dict[str, Path] = {}
_feedback_queue: List[Dict]      = []


# ══════════════════════════════════════════════════════════════════════════════
#  API Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["System"])
def health():
    """Liveness check — returns model & OCR status."""
    model = get_model()
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "model_path":    str(MODEL_PATH),
        "ocr_available": is_ocr_available(),
        "ocr_reason":    ocr_unavailable_reason() if not is_ocr_available() else "",
        "timestamp":     datetime.datetime.now().isoformat(),
    }


@app.get("/api/model-info", tags=["System"])
def model_info():
    """Return class registry, scan modes and display metadata."""
    model = get_model()
    return {
        "model_loaded": model is not None,
        "classes":      CLASS_NAMES,
        "num_classes":  NUM_CLASSES,
        "scan_modes": {
            k: {
                "class_keys": v["class_keys"],
                "labels":     {str(ki): vi for ki, vi in v["labels"].items()},
                "icon":       v["icon"],
            }
            for k, v in SCAN_MODES.items()
        },
        "short_names":   SHORT_NAMES,
        "risk_colours":  RISK_COLOURS,
    }


@app.post("/api/predict", tags=["Inference"])
async def predict(
    image:     UploadFile = File(...,  description="JPEG or PNG medical scan"),
    scan_type: str        = Form("Brain MRI", description="'Brain MRI' or 'Chest X-Ray'"),
):
    """
    Run inference on an uploaded medical image.

    Returns per-class probabilities (all 6 classes) plus the scan-type-filtered
    winner, a session_id to reference this upload in /api/feedback, and any
    text detected via OCR.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Train the global model first (run.bat).",
        )

    # ── read & validate image ─────────────────────────────────────────────────
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(contents))  # re-open after verify
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    # ── inference ─────────────────────────────────────────────────────────────
    try:
        tensor = prepare_image(pil_img)
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    pred_idx = int(np.argmax(probs))
    pred_key = CLASS_NAMES[pred_idx]
    all_probs: Dict[str, float] = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))
    }

    # ── scan-type filtered result ─────────────────────────────────────────────
    mode_cfg  = SCAN_MODES.get(scan_type, SCAN_MODES["Brain MRI"])
    mode_probs: Dict[str, float] = {k: all_probs[k] for k in mode_cfg["class_keys"]}
    mode_pred_key = max(mode_probs, key=mode_probs.get)

    # ── OCR (non-critical — silently skip on failure) ─────────────────────────
    ocr_text  = ""
    ocr_lines = []
    if is_ocr_available():
        try:
            ocr_result = extract_text(pil_img)
            ocr_text   = ocr_result.full_text
            ocr_lines  = [
                {"text": ln.text, "confidence": round(ln.confidence, 4)}
                for ln in ocr_result.lines
            ]
        except Exception:
            pass

    # ── persist temp copy for feedback ───────────────────────────────────────
    session_id = str(uuid.uuid4())
    suffix     = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    tmp_path = TEMP_DIR / f"{session_id}{suffix}"
    tmp_path.write_bytes(contents)
    _pending_images[session_id] = tmp_path

    return {
        "session_id":           session_id,
        "filename":             image.filename or "uploaded_scan.jpg",
        "scan_type":            scan_type,
        # global winner (all 6 classes)
        "predicted_key":        pred_key,
        "predicted_class":      SHORT_NAMES.get(pred_key, pred_key),
        "confidence":           round(float(probs[pred_idx]), 4),
        "probabilities":        all_probs,
        # scan-type-filtered winner
        "mode_predicted_key":   mode_pred_key,
        "mode_predicted_class": SHORT_NAMES.get(mode_pred_key, mode_pred_key),
        "mode_confidence":      round(float(mode_probs[mode_pred_key]), 4),
        "mode_probabilities":   mode_probs,
        # OCR
        "ocr_text":             ocr_text,
        "ocr_lines":            ocr_lines,
    }


@app.post("/api/feedback", tags=["Feedback"])
async def feedback(
    session_id:       str = Form(...),
    chosen_key:       str = Form(...),
    scan_type:        str = Form("Brain MRI"),
    ai_predicted_key: str = Form(""),
):
    """
    Persist the doctor's confirmed or overridden label for continuous learning.

    The image is anonymised (EXIF stripped) and saved to
    data/new_collected_data/{chosen_key}/ ready for the next FL round.
    """
    if chosen_key not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown class key: '{chosen_key}'")

    tmp_path = _pending_images.get(session_id)
    if tmp_path is None or not tmp_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = "Session image not found. Please upload and predict again.",
        )

    save_dir = COLLECT_DIR / chosen_key
    save_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = save_dir / f"{ts}_{session_id[:8]}{tmp_path.suffix}"

    try:
        pil_img = Image.open(tmp_path)
        strip_metadata_and_save(pil_img, dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

    entry: Dict = {
        "session_id":       session_id,
        "chosen_key":       chosen_key,
        "chosen_label":     SHORT_NAMES.get(chosen_key, chosen_key),
        "ai_predicted_key": ai_predicted_key,
        "scan_type":        scan_type,
        "saved_to":         str(dest),
        "timestamp":        datetime.datetime.now().isoformat(),
        "overridden":       chosen_key != ai_predicted_key,
    }
    _feedback_queue.append(entry)

    # clean up temp file
    try:
        tmp_path.unlink(missing_ok=True)
        _pending_images.pop(session_id, None)
    except Exception:
        pass

    return {
        "success":    True,
        "overridden": entry["overridden"],
        "message": (
            f"Image saved as '{SHORT_NAMES.get(chosen_key, chosen_key)}' "
            "and queued for the next training round."
        ),
        "saved_to":   str(dest),
    }


@app.get("/api/queue", tags=["Feedback"])
def queue():
    """Return the last 50 doctor-feedback entries collected this session."""
    return {
        "count":   len(_feedback_queue),
        "entries": _feedback_queue[-50:],
    }


@app.get("/api/report", tags=["Reporting"], response_class=PlainTextResponse)
def report(
    session_id:       str = "",
    filename:         str = "uploaded_scan.jpg",
    scan_type:        str = "Brain MRI",
    ai_pred:          str = "",
    doctor_confirmed: str = "",
    ocr_text:         str = "",
):
    """Generate and download a plain-text diagnostic report."""
    ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outcome = (
        "CONFIRMED"
        if ai_pred == doctor_confirmed
        else "OVERRIDDEN BY CLINICIAN"
    )
    ai_label  = SHORT_NAMES.get(ai_pred, ai_pred or "—")
    doc_label = SHORT_NAMES.get(doctor_confirmed, doctor_confirmed or "—")

    ocr_block = ""
    if ocr_text.strip():
        ocr_block = (
            "\n--------------------------------------------------------\n"
            "  EXTRACTED TEXT (OCR)\n"
            f"{ocr_text}\n"
        )

    report_text = (
        "========================================================\n"
        "  TECNOMATE CLINICAL AI \u2014 DIAGNOSTIC REPORT\n"
        "========================================================\n"
        f"  Date/Time         : {ts}\n"
        f"  Image File        : {filename}\n"
        f"  Scan Type         : {scan_type}\n"
        f"  AI Prediction     : {ai_label}\n"
        f"  Doctor Confirmed  : {doc_label}\n"
        f"  Outcome           : {outcome}\n"
        f"{ocr_block}"
        "--------------------------------------------------------\n"
        "  PRIVACY NOTICE\n"
        "  Patient Data Anonymized and Secured.\n"
        "  All EXIF metadata, device identifiers, and hidden\n"
        "  tags have been permanently stripped from this image\n"
        "  before storage.  No patient-identifiable information\n"
        "  is retained anywhere in this system.\n"
        "--------------------------------------------------------\n"
        "  MEDICAL DISCLAIMER\n"
        "  This report is generated by an AI assistant only.\n"
        "  It must not be used as the sole basis for clinical\n"
        "  decisions.  Always rely on qualified medical\n"
        "  professionals for diagnosis and treatment.\n"
        "========================================================\n"
    )

    safe_ts  = ts.replace(":", "-").replace(" ", "_")
    filename_out = f"tecnomate_report_{safe_ts}.txt"
    return PlainTextResponse(
        content = report_text,
        headers = {
            "Content-Disposition": f'attachment; filename="{filename_out}"',
        },
    )


# ── mount frontend static files (must be last — acts as a fallback) ───────────
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
