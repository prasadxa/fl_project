"""
Tecnomate Clinical AI — FastAPI Backend
========================================
Endpoints:
  GET  /api/health            — liveness + model status
  GET  /api/model-info        — class names, scan modes, colours
  POST /api/predict           — image inference  (multipart/form-data)
  POST /api/feedback          — doctor-confirmed / overridden label
  GET  /api/queue             — last 50 feedback entries (legacy)
  GET  /api/report            — download plain-text diagnostic report (legacy)
  POST /api/pdf-report        — generate & download professional PDF report
  GET  /api/admin/stats       — aggregate statistics from SQLite
  GET  /api/admin/feedback    — paginated feedback log
  GET  /api/admin/export-csv  — download full feedback CSV
  GET  /api/admin/sessions    — paginated prediction sessions

Static frontend served from ../frontend/  (mounted at /)

Supported image formats  : JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, HEIC/HEIF
                           DICOM (.dcm) via pydicom (optional)
Maximum upload size      : 30 MB

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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from anonymizer import strip_metadata_and_save
from dataset import CLASS_NAMES, NUM_CLASSES
from db import get_db
from model import MedicalCNN
from ocr_reader import (
    extract_text,
    is_ocr_available,
    ocr_unavailable_reason,
)
from report_generator import (
    PatientInfo,
    ReportRequest,
    UncertaintyInfo,
    build_pdf_report,
    compute_gradcam,
    compute_mc_uncertainty,
)

# ── paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJ_ROOT / "models" / "global_model.pth"
COLLECT_DIR = PROJ_ROOT / "data" / "new_collected_data"
FRONTEND_DIR = PROJ_ROOT / "frontend"
TEMP_DIR = PROJ_ROOT / "data" / ".tmp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── upload constraints ─────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES: int = 30 * 1024 * 1024  # 30 MB hard limit

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",  # JPEG
        ".png",  # PNG
        ".webp",  # WebP (lossy & lossless)
        ".bmp",  # BMP / DIB
        ".tif",
        ".tiff",  # TIFF (common DICOM-export format)
        ".gif",  # GIF  (takes first frame)
        ".avif",  # AVIF / AV1
        ".heic",
        ".heif",  # HEIC/HEIF
        ".dcm",  # DICOM (pydicom, optional)
    }
)

ALLOWED_MIME_PREFIXES: tuple[str, ...] = (
    "image/",
    "application/octet-stream",
    "application/dicom",
)

# ── scan-mode configuration ────────────────────────────────────────────────────
SCAN_MODES: Dict = {
    "Brain MRI": {
        "indices": [0, 1, 2, 3],
        "labels": {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary Tumor"},
        "class_keys": ["glioma", "meningioma", "notumor", "pituitary"],
        "icon": "\U0001f9e0",
    },
    "Chest X-Ray": {
        "indices": [4, 5],
        "labels": {4: "No Pneumonia", 5: "Pneumonia Detected"},
        "class_keys": ["normal", "pneumonia"],
        "icon": "\U0001fac1",
    },
}

SHORT_NAMES: Dict[str, str] = {
    "glioma": "Glioma (Brain Tumor)",
    "meningioma": "Meningioma (Brain Tumor)",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
    "normal": "Normal / Healthy (CXR)",
    "pneumonia": "Pneumonia Detected (CXR)",
}

RISK_COLOURS: Dict[str, str] = {
    "glioma": "#e74c3c",
    "meningioma": "#e67e22",
    "notumor": "#27ae60",
    "pituitary": "#e67e22",
    "normal": "#27ae60",
    "pneumonia": "#e74c3c",
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Tecnomate Clinical AI",
    description="Privacy-preserving federated medical image classifier",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    # Initialise the DB singleton so the file is created at startup.
    get_db()
    print(f"[API] OCR available: {is_ocr_available()}")
    if not is_ocr_available():
        print(f"[API] OCR reason  : {ocr_unavailable_reason()}")


# ── image preprocessing ────────────────────────────────────────────────────────
_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def _load_dicom(contents: bytes) -> Image.Image:
    """
    Decode a DICOM file using pydicom and return a PIL greyscale image.
    Raises ImportError if pydicom is not installed.
    Raises ValueError on decode failure.
    """
    try:
        import pydicom  # type: ignore[import]

        try:
            from pydicom.pixel_data_handlers.util import (  # type: ignore[import]
                apply_voi_lut,
            )
        except ImportError:
            apply_voi_lut = None  # type: ignore[assignment]
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to process DICOM files. "
            "Install it with: pip install pydicom"
        ) from exc

    try:
        ds = pydicom.dcmread(io.BytesIO(contents))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply VOI LUT / windowing when available (improves contrast)
        try:
            if apply_voi_lut is not None:
                pixel_array = apply_voi_lut(pixel_array, ds).astype(np.float32)
        except Exception:
            pass  # Not all DICOM files carry windowing info

        # Normalise to 0-255 uint8
        lo, hi = pixel_array.min(), pixel_array.max()
        if hi > lo:
            pixel_array = (pixel_array - lo) / (hi - lo) * 255.0
        pixel_array = pixel_array.clip(0, 255).astype(np.uint8)

        # Handle multi-frame (cine / 3-D series) → first frame
        if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
            pixel_array = pixel_array[0]

        if pixel_array.ndim == 3:
            # RGB DICOM (rare)
            return Image.fromarray(pixel_array, mode="RGB")
        else:
            return Image.fromarray(pixel_array, mode="L")

    except Exception as exc:
        raise ValueError(f"Failed to decode DICOM file: {exc}") from exc


def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert any PIL image to the (1, 1, 128, 128) float tensor the model expects.
    """
    try:
        pil_img.seek(0)
    except (EOFError, AttributeError):
        pass

    mode = pil_img.mode
    if mode == "L":
        grey_pil = pil_img.copy()
    elif mode in ("LA", "PA"):
        grey_pil = pil_img.convert("L")
    elif mode in ("I", "I;16", "I;16B"):
        arr = np.array(pil_img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        grey_pil = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif mode == "F":
        arr = np.array(pil_img, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        grey_pil = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        rgb_pil = pil_img.convert("RGB")
        img_np = np.array(rgb_pil, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        grey_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        grey_pil = Image.fromarray(grey_arr, mode="L")

    grey_arr = np.array(grey_pil, dtype=np.uint8)
    resized = cv2.resize(grey_arr, (128, 128), interpolation=cv2.INTER_AREA)
    pil_out = Image.fromarray(resized, mode="L")
    return _INFER_TRANSFORM(pil_out).unsqueeze(0)  # type: ignore[union-attr]


def _validate_upload(contents: bytes, filename: str) -> None:
    """Raise HTTP 413 / 415 for invalid uploads."""
    if len(contents) > MAX_UPLOAD_BYTES:
        mb = len(contents) / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({mb:.1f} MB).  "
                f"Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )
    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file extension '{ext}'.  "
                    "Accepted: JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, HEIC, DICOM."
                ),
            )


# ── in-process temp-image store  ──────────────────────────────────────────────
_pending_images: Dict[str, Path] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  API Routes
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/health", tags=["System"])
def health():
    """Liveness check — returns model, OCR and DB status."""
    model = get_model()
    db = get_db()
    counts = db.count_feedback()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "ocr_available": is_ocr_available(),
        "ocr_reason": ocr_unavailable_reason() if not is_ocr_available() else "",
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "feedback_total": counts["total"],
        "feedback_overridden": counts["overridden"],
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/api/model-info", tags=["System"])
def model_info():
    """Return class registry, scan modes and display metadata."""
    model = get_model()
    return {
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "scan_modes": {
            k: {
                "class_keys": v["class_keys"],
                "labels": {str(ki): vi for ki, vi in v["labels"].items()},
                "icon": v["icon"],
            }
            for k, v in SCAN_MODES.items()
        },
        "short_names": SHORT_NAMES,
        "risk_colours": RISK_COLOURS,
    }


# ── predict ────────────────────────────────────────────────────────────────────


@app.post("/api/predict", tags=["Inference"])
async def predict(
    image: UploadFile = File(
        ..., description="Medical scan — JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, DICOM"
    ),
    scan_type: str = Form("Brain MRI", description="'Brain MRI' or 'Chest X-Ray'"),
    gradcam: bool = Form(False, description="Compute Grad-CAM (slower)"),
    mc_dropout: bool = Form(False, description="Compute MC-Dropout uncertainty"),
    mc_samples: int = Form(20, description="Number of MC samples (5-50)"),
):
    """
    Run inference on an uploaded medical image.

    Extra options (all optional):
      gradcam    — if true, a Grad-CAM heatmap is computed and the path
                   is stored with the session for use in PDF reports.
      mc_dropout — if true, runs MC-Dropout uncertainty estimation and
                   returns mean_entropy + std_confidence in the response.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the global model first (run.bat).",
        )

    # ── read + validate upload ────────────────────────────────────────────────
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file received.")
    _validate_upload(contents, image.filename or "")

    # ── decide whether file is DICOM ─────────────────────────────────────────
    orig_ext = Path(image.filename).suffix.lower() if image.filename else ""
    is_dicom = orig_ext == ".dcm"

    pil_img: Image.Image
    detected_format = "unknown"

    if is_dicom:
        try:
            pil_img = _load_dicom(contents)
            detected_format = "DICOM"
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    else:
        # Standard Pillow decoding
        try:
            pil_img = Image.open(io.BytesIO(contents))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(contents))
            pil_img.load()
            detected_format = pil_img.format or "unknown"
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode image: {exc}.",
            )

    # ── inference ─────────────────────────────────────────────────────────────
    try:
        tensor = prepare_image(pil_img)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    pred_idx = int(np.argmax(probs))
    pred_key = CLASS_NAMES[pred_idx]
    all_probs: Dict[str, float] = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))
    }

    # ── scan-type filtered result ─────────────────────────────────────────────
    mode_cfg = SCAN_MODES.get(scan_type, SCAN_MODES["Brain MRI"])
    mode_probs = {k: all_probs[k] for k in mode_cfg["class_keys"]}
    mode_pred_key = max(mode_probs, key=mode_probs.get)  # type: ignore[arg-type]
    mode_pred_idx = CLASS_NAMES.index(mode_pred_key)

    # ── scan-type mismatch detection ──────────────────────────────────────────
    # The global winner (pred_key) comes from all 6 classes.
    # If it belongs to the OTHER scan mode's class set, the user most likely
    # uploaded the wrong image type (e.g. a brain MRI on the Chest X-Ray tab).
    #
    # We also compute the total probability mass each scan mode holds to give
    # a confidence-weighted mismatch score.
    OTHER_MODE = "Chest X-Ray" if scan_type == "Brain MRI" else "Brain MRI"
    other_cfg = SCAN_MODES[OTHER_MODE]
    other_probs = {k: all_probs[k] for k in other_cfg["class_keys"]}

    # Sum of probabilities belonging to selected mode vs other mode
    selected_mass = sum(mode_probs.values())
    other_mass = sum(other_probs.values())

    # Mismatch when the global winner is from the other mode AND
    # the other mode holds more than 60 % of total probability mass
    global_winner_in_other = pred_key in other_cfg["class_keys"]
    scan_type_mismatch = global_winner_in_other and other_mass > 0.60

    # Suggested scan type and its best class
    if scan_type_mismatch:
        suggested_scan_type = OTHER_MODE
        other_pred_key = max(other_probs, key=other_probs.get)  # type: ignore[arg-type]
        suggested_class = SHORT_NAMES.get(other_pred_key, other_pred_key)
        suggested_confidence = round(float(other_probs[other_pred_key]), 4)
        mismatch_detail = (
            f"The image appears to be a {OTHER_MODE} scan "
            f"(model assigns {other_mass * 100:.1f}% probability mass to "
            f"{OTHER_MODE} classes, vs {selected_mass * 100:.1f}% to "
            f"{scan_type} classes). "
            f"Suggested result under '{OTHER_MODE}': "
            f"{suggested_class} ({suggested_confidence * 100:.1f}% confidence)."
        )
    else:
        suggested_scan_type = scan_type
        suggested_class = ""
        suggested_confidence = 0.0
        mismatch_detail = ""

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_text = ""
    ocr_lines: List[Dict] = []
    if is_ocr_available():
        try:
            ocr_result = extract_text(pil_img)
            ocr_text = ocr_result.full_text
            ocr_lines = [
                {"text": ln.text, "confidence": round(ln.confidence, 4)}
                for ln in ocr_result.lines
            ]
        except Exception:
            pass

    # ── MC-Dropout uncertainty ────────────────────────────────────────────────
    uncertainty_dict: Dict = {}
    if mc_dropout:
        mc_n = max(5, min(50, mc_samples))
        unc = compute_mc_uncertainty(model, tensor, n_samples=mc_n)
        model.eval()  # restore eval mode after MC pass
        uncertainty_dict = {
            "mean_entropy": unc.mean_entropy,
            "std_confidence": unc.std_confidence,
            "mc_samples": unc.mc_samples,
            "uncertainty_label": unc.uncertainty_label,
        }

    # ── persist temp copy ────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    _FORMAT_TO_EXT: dict[str, str] = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "TIFF": ".tiff",
        "GIF": ".gif",
        "AVIF": ".avif",
        "DICOM": ".dcm",
    }
    if orig_ext in ALLOWED_EXTENSIONS:
        suffix = orig_ext
    else:
        suffix = _FORMAT_TO_EXT.get(detected_format.upper(), ".jpg")

    tmp_path = TEMP_DIR / f"{session_id}{suffix}"
    tmp_path.write_bytes(contents)
    _pending_images[session_id] = tmp_path

    # ── Grad-CAM (optional, stored alongside temp image) ─────────────────────
    gradcam_path: Optional[Path] = None
    if gradcam:
        try:
            cam_overlay = compute_gradcam(model, tensor, mode_pred_idx, pil_img)
            cam_pil = Image.fromarray(cam_overlay, mode="RGB")
            gradcam_path = TEMP_DIR / f"{session_id}_gradcam.png"
            cam_pil.save(str(gradcam_path))
        except Exception as exc:
            print(f"[API] Grad-CAM failed: {exc}")

    # ── persist session to DB ────────────────────────────────────────────────
    predict_response: Dict = {
        "session_id": session_id,
        "filename": image.filename or f"uploaded_scan{suffix}",
        "detected_format": detected_format,
        "scan_type": scan_type,
        "predicted_key": pred_key,
        "predicted_class": SHORT_NAMES.get(pred_key, pred_key),
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": all_probs,
        "mode_predicted_key": mode_pred_key,
        "mode_predicted_class": SHORT_NAMES.get(mode_pred_key, mode_pred_key),
        "mode_confidence": round(float(mode_probs[mode_pred_key]), 4),
        "mode_probabilities": mode_probs,
        # probability mass each mode holds (0–1)
        "selected_mode_mass": round(selected_mass, 4),
        "other_mode_mass": round(other_mass, 4),
        # mismatch fields
        "scan_type_mismatch": scan_type_mismatch,
        "suggested_scan_type": suggested_scan_type,
        "suggested_class": suggested_class,
        "suggested_confidence": suggested_confidence,
        "mismatch_detail": mismatch_detail,
        "ocr_text": ocr_text,
        "ocr_lines": ocr_lines,
        "file_size_bytes": len(contents),
        "image_dimensions": list(pil_img.size),
        "gradcam_available": gradcam_path is not None,
    }
    if uncertainty_dict:
        predict_response["uncertainty"] = uncertainty_dict

    try:
        get_db().add_session(predict_response)
    except Exception as exc:
        print(f"[API] DB session save failed: {exc}")

    return predict_response


# ── feedback ───────────────────────────────────────────────────────────────────


@app.post("/api/feedback", tags=["Feedback"])
async def feedback(
    session_id: str = Form(...),
    chosen_key: str = Form(...),
    scan_type: str = Form("Brain MRI"),
    ai_predicted_key: str = Form(""),
    clinician_name: str = Form(""),
    clinician_id: str = Form(""),
    notes: str = Form(""),
):
    """
    Persist the doctor's confirmed / overridden label for continuous learning.
    The image is anonymised and saved for the next FL round.
    """
    if chosen_key not in CLASS_NAMES:
        raise HTTPException(
            status_code=400, detail=f"Unknown class key: '{chosen_key}'"
        )

    tmp_path = _pending_images.get(session_id)
    if tmp_path is None or not tmp_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Session image not found. Please upload and predict again.",
        )

    save_dir = COLLECT_DIR / chosen_key
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = save_dir / f"{ts}_{session_id[:8]}{tmp_path.suffix}"

    try:
        pil_img = Image.open(tmp_path)
        strip_metadata_and_save(pil_img, dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {exc}")

    entry: Dict = {
        "session_id": session_id,
        "chosen_key": chosen_key,
        "chosen_label": SHORT_NAMES.get(chosen_key, chosen_key),
        "ai_predicted_key": ai_predicted_key,
        "scan_type": scan_type,
        "saved_to": str(dest),
        "timestamp": datetime.datetime.now().isoformat(),
        "overridden": chosen_key != ai_predicted_key,
        "clinician_name": clinician_name,
        "clinician_id": clinician_id,
        "notes": notes,
    }

    try:
        get_db().add_feedback(entry)
    except Exception as exc:
        print(f"[API] DB feedback save failed: {exc}")

    # clean up temp scan (keep Grad-CAM temp until PDF is generated or timeout)
    try:
        tmp_path.unlink(missing_ok=True)
        _pending_images.pop(session_id, None)
    except Exception:
        pass

    return {
        "success": True,
        "overridden": entry["overridden"],
        "message": (
            f"Image saved as '{SHORT_NAMES.get(chosen_key, chosen_key)}' "
            "and queued for the next training round."
        ),
        "saved_to": str(dest),
    }


# ── legacy queue endpoint (kept for backwards compatibility) ──────────────────


@app.get("/api/queue", tags=["Feedback"])
def queue(limit: int = 50):
    """Return the last N doctor-feedback entries."""
    db = get_db()
    rows = db.list_feedback(limit=min(limit, 200))
    counts = db.count_feedback()
    return {
        "count": counts["total"],
        "entries": rows,
    }


# ── legacy plain-text report (kept for backwards compatibility) ───────────────


@app.get("/api/report", tags=["Reporting"], response_class=PlainTextResponse)
def report(
    session_id: str = "",
    filename: str = "uploaded_scan.jpg",
    scan_type: str = "Brain MRI",
    ai_pred: str = "",
    doctor_confirmed: str = "",
    ocr_text: str = "",
):
    """Generate and download a plain-text diagnostic report (legacy format)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outcome = "CONFIRMED" if ai_pred == doctor_confirmed else "OVERRIDDEN BY CLINICIAN"
    ai_label = SHORT_NAMES.get(ai_pred, ai_pred or "\u2014")
    doc_label = SHORT_NAMES.get(doctor_confirmed, doctor_confirmed or "\u2014")

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

    safe_ts = ts.replace(":", "-").replace(" ", "_")
    filename_out = f"tecnomate_report_{safe_ts}.txt"
    return PlainTextResponse(
        content=report_text,
        headers={"Content-Disposition": f'attachment; filename="{filename_out}"'},
    )


# ── PDF report ─────────────────────────────────────────────────────────────────


@app.post("/api/pdf-report", tags=["Reporting"])
async def pdf_report(
    # ── core identification (required) ────────────────────────────────────────
    session_id: str = Form(...),
    scan_type: str = Form("Brain MRI"),
    ai_pred_key: str = Form(...),
    ai_confidence: float = Form(0.0),
    doctor_choice_key: str = Form(...),
    # ── probabilities (JSON string) ───────────────────────────────────────────
    probabilities_json: str = Form("{}"),
    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_text: str = Form(""),
    ocr_lines_json: str = Form("[]"),
    # ── patient / visit fields (all optional) ─────────────────────────────────
    patient_name: str = Form("Anonymous / De-identified"),
    patient_id: str = Form("N/A"),
    date_of_birth: str = Form("N/A"),
    gender: str = Form("N/A"),
    referring_doctor: str = Form("N/A"),
    institution: str = Form("Tecnomate Health Network"),
    visit_date: str = Form(""),
    clinical_notes: str = Form(""),
    # ── clinician identity (optional) ─────────────────────────────────────────
    clinician_name: str = Form(""),
    clinician_id: str = Form(""),
    # ── model metadata ────────────────────────────────────────────────────────
    fl_round: int = Form(0),
    model_version: str = Form("global_model.pth"),
    # ── uncertainty (optional — populated by /api/predict if mc_dropout=True) ─
    mc_entropy: float = Form(0.0),
    mc_std_conf: float = Form(0.0),
    mc_samples: int = Form(0),
    mc_label: str = Form(""),
):
    """
    Generate a professional clinical PDF report and return it for download.

    Accepts all prediction and patient metadata as form fields so the
    frontend can submit a single POST after the clinician has filled in
    the optional patient fields.

    The report includes:
      - Cover band with report timestamp and session ID
      - Patient & visit information table
      - Original scan thumbnail + Grad-CAM heatmap (if available)
      - AI prediction banner with risk level and ICD-10 code
      - Per-class probability bar chart
      - MC-Dropout uncertainty (if mc_samples > 0)
      - Clinician review / final diagnosis + signature line
      - OCR-extracted text block (if any)
      - Audit trail
      - Privacy notice, FL notice, medical disclaimer
    """
    import json as _json

    # ── parse JSON blobs ──────────────────────────────────────────────────────
    try:
        probabilities: Dict[str, float] = _json.loads(probabilities_json)
    except Exception:
        probabilities = {}

    try:
        ocr_lines: List[Dict] = _json.loads(ocr_lines_json)
    except Exception:
        ocr_lines = []

    # ── look up scan image from temp dir ─────────────────────────────────────
    scan_image_path: Optional[Path] = _pending_images.get(session_id)
    if scan_image_path is None or not scan_image_path.exists():
        # Try to find leftover temp file by session prefix
        candidates = list(TEMP_DIR.glob(f"{session_id}.*"))
        non_gradcam = [c for c in candidates if "_gradcam" not in c.name]
        scan_image_path = non_gradcam[0] if non_gradcam else None

    # ── look up Grad-CAM overlay ──────────────────────────────────────────────
    gradcam_np: Optional[np.ndarray] = None
    gradcam_temp = TEMP_DIR / f"{session_id}_gradcam.png"
    if gradcam_temp.exists():
        try:
            gradcam_np = np.array(Image.open(gradcam_temp).convert("RGB"))
        except Exception:
            pass
    elif scan_image_path and scan_image_path.exists():
        # Compute Grad-CAM on the fly if not pre-computed
        model = get_model()
        if model is not None:
            try:
                pil_scan = Image.open(scan_image_path)
                infer_tensor = prepare_image(pil_scan)
                target_idx = (
                    CLASS_NAMES.index(ai_pred_key) if ai_pred_key in CLASS_NAMES else 0
                )
                gradcam_np = compute_gradcam(model, infer_tensor, target_idx, pil_scan)
            except Exception as exc:
                print(f"[API/PDF] on-demand Grad-CAM failed: {exc}")

    # ── assemble ReportRequest ────────────────────────────────────────────────
    req = ReportRequest(
        session_id=session_id,
        filename=(
            _pending_images.get(session_id, Path(f"scan_{session_id[:8]}.jpg")).name
            if scan_image_path is None
            else scan_image_path.name
        ),
        scan_type=scan_type,
        ai_pred_key=ai_pred_key,
        ai_confidence=ai_confidence,
        probabilities=probabilities,
        doctor_choice_key=doctor_choice_key,
        ocr_text=ocr_text,
        ocr_lines=ocr_lines,
        scan_image_path=scan_image_path,
        gradcam_image=gradcam_np,
        patient=PatientInfo(
            patient_name=patient_name,
            patient_id=patient_id,
            date_of_birth=date_of_birth,
            gender=gender,
            referring_doctor=referring_doctor,
            institution=institution,
            visit_date=visit_date,
            clinical_notes=clinical_notes,
        ),
        uncertainty=UncertaintyInfo(
            mean_entropy=mc_entropy,
            std_confidence=mc_std_conf,
            mc_samples=mc_samples,
            uncertainty_label=mc_label or ("N/A" if mc_samples == 0 else ""),
        ),
        fl_round=fl_round,
        model_version=model_version,
        server_timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        clinician_name=clinician_name,
        clinician_id=clinician_id,
    )

    # ── build PDF ─────────────────────────────────────────────────────────────
    try:
        pdf_bytes = build_pdf_report(req)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {exc}",
        )

    # ── clean up Grad-CAM temp file after embedding it ────────────────────────
    try:
        gradcam_temp.unlink(missing_ok=True)
    except Exception:
        pass

    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"tecnomate_report_{ts_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── admin endpoints ────────────────────────────────────────────────────────────


@app.get("/api/admin/stats", tags=["Admin"])
def admin_stats():
    """Return aggregate statistics from the SQLite database."""
    return get_db().stats()


@app.get("/api/admin/feedback", tags=["Admin"])
def admin_feedback(
    limit: int = 50,
    offset: int = 0,
    overridden_only: bool = False,
    scan_type: str = "",
):
    """Return paginated feedback records."""
    db = get_db()
    rows = db.list_feedback(
        limit=min(limit, 500),
        offset=offset,
        overridden_only=overridden_only,
        scan_type=scan_type or None,
    )
    counts = db.count_feedback(scan_type=scan_type or None)
    return {
        "total": counts["total"],
        "confirmed": counts["confirmed"],
        "overridden": counts["overridden"],
        "rows": rows,
    }


@app.get("/api/admin/export-csv", tags=["Admin"])
def admin_export_csv():
    """Download all feedback data as a CSV file."""
    csv_data = get_db().export_feedback_csv()
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"tecnomate_feedback_{ts_str}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/admin/sessions", tags=["Admin"])
def admin_sessions(limit: int = 50, offset: int = 0):
    """Return paginated prediction sessions."""
    return {"sessions": get_db().list_sessions(limit=min(limit, 500), offset=offset)}


# ── mount frontend static files (must be last — acts as a fallback) ───────────
if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
