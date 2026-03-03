"""
scan_classifier.py — Scan-Type Gatekeeper
==========================================
A lightweight EfficientNet-B0 classifier that acts as the FIRST line of
defence before OCR and inference.  It answers one question:

    "Is this image a Chest X-Ray, Brain MRI, CT Scan, or Non-Medical?"

Classes (label → index):
    0  chest_xray
    1  brain_mri
    2  ct_scan
    3  non_medical

Pipeline position (called from api.py):
    Upload → [ScanGate] → OCR check → Main CNN inference

If the model file is absent the gate falls back to pure heuristics so the
server never hard-crashes on a missing weight file.

Public API
----------
    gate = ScanGate()                   # loads model once at import time
    result = gate.check(pil_img, expected_scan_type)
    # result.allowed        bool
    # result.label          str   e.g. "chest_xray"
    # result.confidence     float 0-1
    # result.probabilities  dict  {label: prob}
    # result.rejection_reason str
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("tecnomate.scan_gate")

# ── paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent
_PROJ_ROOT = _BACKEND_DIR.parent
GATE_MODEL_PATH = _PROJ_ROOT / "models" / "scan_gate.pth"

# ── class definitions ──────────────────────────────────────────────────────────
GATE_CLASSES = ["chest_xray", "brain_mri", "ct_scan", "non_medical"]
GATE_CLASS_IDX = {c: i for i, c in enumerate(GATE_CLASSES)}
NUM_GATE_CLASSES = len(GATE_CLASSES)

# Expected scan_type strings from the API → gate class label
_SCAN_TYPE_TO_GATE: Dict[str, str] = {
    "Chest X-Ray": "chest_xray",
    "Brain MRI": "brain_mri",
}

# Minimum confidence to pass the gate (else reject as "uncertain")
CONFIDENCE_THRESHOLD = 0.80

# ── image transform (matches training) ────────────────────────────────────────
# EfficientNet-B0 was pre-trained on ImageNet (RGB 224×224).
# We keep RGB here because the gate needs colour information to distinguish
# non-medical photos from grayscale medical images.
GATE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── model builder ──────────────────────────────────────────────────────────────
def build_gate_model(
    num_classes: int = NUM_GATE_CLASSES, pretrained: bool = False
) -> nn.Module:
    """
    Build an EfficientNet-B0 with a replaced classifier head.

    Args:
        num_classes: number of output classes (default 4).
        pretrained:  load ImageNet weights (used during training only).
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return net


# ── heuristic fallback ─────────────────────────────────────────────────────────
def _heuristic_check(img: Image.Image) -> Dict[str, float]:
    """
    Pure-pixel heuristics that approximate the model when weights are absent.

    Returns a probability-like dict {label: score} that sums to ~1.
    The scores are rough estimates, not calibrated probabilities.

    Heuristics:
        1. Grayscale ratio      — medical images are near-grayscale
        2. Mean saturation      — colour photos have high saturation
        3. Intensity histogram  — X-rays have characteristic bimodal distribution
        4. Unique colour count  — non-medical photos have many distinct colours
        5. Image aspect ratio   — most X-rays/MRI are roughly square
    """
    rgb = img.convert("RGB").resize((128, 128))
    arr = np.array(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # ── 1. Grayscale ratio ─────────────────────────────────────────────────────
    diff_max = np.maximum(np.abs(r - g), np.maximum(np.abs(r - b), np.abs(g - b)))
    gray_ratio = float((diff_max < 18).mean())

    # ── 2. Mean saturation ─────────────────────────────────────────────────────
    max_c = np.maximum(np.maximum(r, g), b) / 255.0
    min_c = np.minimum(np.minimum(r, g), b) / 255.0
    l_val = (max_c + min_c) / 2.0
    denom = np.where(l_val > 0.5, 2.0 - max_c - min_c, max_c + min_c)
    sat = np.where(denom > 1e-6, (max_c - min_c) / (denom + 1e-6), 0.0)
    mean_sat = float(sat.mean())
    high_sat_ratio = float((sat > 0.25).mean())

    # ── 3. Intensity histogram spread (grayscale) ──────────────────────────────
    gray_arr = 0.299 * r + 0.587 * g + 0.114 * b
    hist, _ = np.histogram(gray_arr.ravel(), bins=32, range=(0, 255))
    hist_norm = hist / (hist.sum() + 1e-8)
    # X-rays tend to have a large dark peak (background) + spread bright pixels
    dark_mass = float(hist_norm[:8].sum())  # pixels 0-63
    bright_mass = float(hist_norm[24:].sum())  # pixels 192-255
    mid_mass = 1.0 - dark_mass - bright_mass

    # ── 4. Unique quantised colour count ──────────────────────────────────────
    small = img.convert("RGB").resize((64, 64))
    q = (np.array(small) // 16).reshape(-1, 3)
    unique_colors = len(set(map(tuple, q.tolist())))

    # ── 5. Aspect ratio ────────────────────────────────────────────────────────
    w, h = img.size
    aspect = max(w, h) / (min(w, h) + 1e-6)
    square_ish = aspect < 1.6

    # ── Scoring ────────────────────────────────────────────────────────────────
    # non_medical: colourful, many unique colours, high saturation
    p_non_medical = 0.0
    if gray_ratio < 0.70:
        p_non_medical += 0.5
    if mean_sat > 0.15:
        p_non_medical += 0.3
    if high_sat_ratio > 0.20:
        p_non_medical += 0.2
    if unique_colors > 900:
        p_non_medical += 0.2
    p_non_medical = min(p_non_medical, 0.97)

    # If clearly non-medical, assign most probability there
    if p_non_medical > 0.6:
        leftover = 1.0 - p_non_medical
        return {
            "chest_xray": leftover * 0.4,
            "brain_mri": leftover * 0.35,
            "ct_scan": leftover * 0.25,
            "non_medical": p_non_medical,
        }

    # Medical image — distinguish modality by histogram and structure
    # Chest X-Rays: large dark background (lungs), spread mid-tones, square-ish
    # Brain MRI:    bright centre blob, dark border, varies by sequence
    # CT:           similar to X-ray but denser mid-range

    p_xray = 0.0
    p_mri = 0.0
    p_ct = 0.0

    # X-ray signature: bimodal (dark + bright), square
    if dark_mass > 0.35 and bright_mass > 0.10 and square_ish:
        p_xray += 0.5
    if dark_mass > 0.45:
        p_xray += 0.2
    if bright_mass > 0.15:
        p_xray += 0.1

    # MRI signature: mid-range heavy, bright centre
    if mid_mass > 0.50:
        p_mri += 0.45
    if dark_mass > 0.30 and bright_mass < 0.15:
        p_mri += 0.15

    # CT signature: dense mid-range, lower dark mass than X-ray
    if mid_mass > 0.45 and dark_mass < 0.35:
        p_ct += 0.4
    if unique_colors < 300 and gray_ratio > 0.85:
        p_ct += 0.15

    # Normalise
    total = p_xray + p_mri + p_ct + p_non_medical + 1e-9
    return {
        "chest_xray": p_xray / total,
        "brain_mri": p_mri / total,
        "ct_scan": p_ct / total,
        "non_medical": p_non_medical / total,
    }


# ── result dataclass ───────────────────────────────────────────────────────────
@dataclass
class GateResult:
    allowed: bool
    label: str  # predicted class label
    confidence: float  # max probability
    probabilities: Dict[str, float] = field(default_factory=dict)
    rejection_reason: str = ""
    used_heuristics: bool = False  # True if model was absent
    elapsed_ms: float = 0.0


# ── main gate class ────────────────────────────────────────────────────────────
class ScanGate:
    """
    Singleton-style gatekeeper.  Instantiate once and call .check() per request.

    The model is loaded lazily on first .check() call so the server starts fast.
    If the weights file is missing, all checks fall back to heuristics.
    """

    def __init__(
        self,
        model_path: Path = GATE_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self._model_path = model_path
        self._threshold = confidence_threshold
        self._device = device or self._pick_device()
        self._model: Optional[nn.Module] = None
        self._model_available = False
        self._load_attempted = False

    # ── device selection ───────────────────────────────────────────────────────
    @staticmethod
    def _pick_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ── lazy model loader ──────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._load_attempted and (
            self._model_available or not self._model_path.exists()
        ):
            return
        self._load_attempted = True
        if not self._model_path.exists():
            logger.warning(
                "ScanGate: model weights not found at %s — falling back to heuristics.",
                self._model_path,
            )
            return
        try:
            net = build_gate_model(pretrained=False)
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            # Move to target device AFTER loading weights on CPU to avoid MPS issues
            net.to(self._device)
            net.eval()
            self._model = net
            self._model_available = True
            logger.info(
                "ScanGate: model loaded from %s (device=%s, params=%d)",
                self._model_path,
                self._device,
                sum(p.numel() for p in net.parameters()),
            )
        except Exception as exc:
            # Reset so the next request retries loading (transient errors)
            self._load_attempted = False
            self._model_available = False
            self._model = None
            logger.error(
                "ScanGate: failed to load model from %s — %s. Falling back to heuristics.",
                self._model_path,
                exc,
                exc_info=True,
            )

    # ── inference ──────────────────────────────────────────────────────────────
    def _predict(self, img: Image.Image) -> Dict[str, float]:
        """Run model inference. Returns {label: probability}."""
        tensor = GATE_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, 4)
            probs = F.softmax(logits, dim=1)[0]  # (4,)
        return {cls: float(probs[i]) for i, cls in enumerate(GATE_CLASSES)}

    # ── public check ───────────────────────────────────────────────────────────
    def check(
        self,
        img: Image.Image,
        expected_scan_type: str,
    ) -> GateResult:
        """
        Gate an uploaded image against the expected scan type.

        Args:
            img:               PIL Image (any mode — converted internally).
            expected_scan_type: canonical string from the API,
                               e.g. "Chest X-Ray" or "Brain MRI".

        Returns:
            GateResult with allowed=True only when:
              1. The top predicted class matches the expected scan type AND
              2. Confidence ≥ threshold.
        """
        t0 = time.perf_counter()
        self._ensure_loaded()

        used_heuristics = not self._model_available

        # ── get probabilities ──────────────────────────────────────────────────
        try:
            if self._model_available:
                probs = self._predict(img)
            else:
                probs = _heuristic_check(img)
        except Exception as exc:
            logger.error("ScanGate.check error: %s — failing open.", exc)
            # Fail-open: don't block the user on gate errors
            elapsed = (time.perf_counter() - t0) * 1000
            return GateResult(
                allowed=True,
                label="unknown",
                confidence=0.0,
                probabilities={c: 0.0 for c in GATE_CLASSES},
                rejection_reason="",
                used_heuristics=True,
                elapsed_ms=elapsed,
            )

        # ── top prediction ─────────────────────────────────────────────────────
        top_label = max(probs, key=lambda k: probs[k])
        top_conf = probs[top_label]

        # ── map expected scan type to gate class ───────────────────────────────
        expected_gate_class = _SCAN_TYPE_TO_GATE.get(expected_scan_type)
        # If we don't know what was expected (new modality), fail-open
        if expected_gate_class is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return GateResult(
                allowed=True,
                label=top_label,
                confidence=top_conf,
                probabilities=probs,
                rejection_reason="",
                used_heuristics=used_heuristics,
                elapsed_ms=elapsed,
            )

        elapsed = (time.perf_counter() - t0) * 1000

        # ── gate decision ──────────────────────────────────────────────────────
        # Case 1: Definitely non-medical
        if top_label == "non_medical" and top_conf >= self._threshold:
            return GateResult(
                allowed=False,
                label=top_label,
                confidence=top_conf,
                probabilities=probs,
                rejection_reason=(
                    f"This image does not appear to be a medical scan "
                    f"(confidence {top_conf:.0%}). "
                    "Please upload an original medical imaging file."
                ),
                used_heuristics=used_heuristics,
                elapsed_ms=elapsed,
            )

        # Case 2: Non-medical but lower confidence — still reject
        if top_label == "non_medical":
            non_med_p = probs["non_medical"]
            if non_med_p > 0.5:
                return GateResult(
                    allowed=False,
                    label=top_label,
                    confidence=non_med_p,
                    probabilities=probs,
                    rejection_reason=(
                        "This image appears to be a non-medical photograph. "
                        "Please upload a genuine medical scan."
                    ),
                    used_heuristics=used_heuristics,
                    elapsed_ms=elapsed,
                )

        # Case 3: Wrong modality with high confidence
        if (
            top_label != expected_gate_class
            and top_label != "non_medical"
            and top_conf >= self._threshold
        ):
            friendly = {
                "chest_xray": "a Chest X-Ray",
                "brain_mri": "a Brain MRI",
                "ct_scan": "a CT Scan",
            }
            detected_str = friendly.get(top_label, top_label)
            expected_str = friendly.get(expected_gate_class, expected_gate_class)
            return GateResult(
                allowed=False,
                label=top_label,
                confidence=top_conf,
                probabilities=probs,
                rejection_reason=(
                    f"This image appears to be {detected_str} "
                    f"({top_conf:.0%} confidence), but you selected "
                    f"{expected_str} mode. "
                    "Please upload the correct scan type or switch the scan mode."
                ),
                used_heuristics=used_heuristics,
                elapsed_ms=elapsed,
            )

        # Case 4: Low confidence — cannot confirm
        if top_conf < self._threshold:
            # Check if the expected class has at least some support
            expected_prob = probs.get(expected_gate_class, 0.0)
            if expected_prob < 0.40:
                return GateResult(
                    allowed=False,
                    label=top_label,
                    confidence=top_conf,
                    probabilities=probs,
                    rejection_reason=(
                        f"Cannot verify this image as a valid medical scan "
                        f"(max confidence {top_conf:.0%}, threshold "
                        f"{self._threshold:.0%}). "
                        "Please upload a clear, unobstructed medical image."
                    ),
                    used_heuristics=used_heuristics,
                    elapsed_ms=elapsed,
                )
            # Expected class has some probability — allow with a note
            return GateResult(
                allowed=True,
                label=expected_gate_class,
                confidence=expected_prob,
                probabilities=probs,
                rejection_reason="",
                used_heuristics=used_heuristics,
                elapsed_ms=elapsed,
            )

        # Case 5: Top label matches expected — allow
        return GateResult(
            allowed=True,
            label=top_label,
            confidence=top_conf,
            probabilities=probs,
            rejection_reason="",
            used_heuristics=used_heuristics,
            elapsed_ms=elapsed,
        )

    # ── convenience ───────────────────────────────────────────────────────────
    @property
    def model_loaded(self) -> bool:
        self._ensure_loaded()
        return self._model_available

    def __repr__(self) -> str:
        status = "loaded" if self._model_available else "heuristics-only"
        return (
            f"ScanGate(threshold={self._threshold}, "
            f"device={self._device}, status={status})"
        )


# ── module-level singleton (imported by api.py) ────────────────────────────────
_gate_instance: Optional[ScanGate] = None


def get_scan_gate() -> ScanGate:
    """Return the module-level ScanGate singleton (created on first call)."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = ScanGate()
    return _gate_instance
