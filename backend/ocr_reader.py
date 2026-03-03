"""
OCR Reader — ocr_reader.py
===========================
Extracts text from uploaded medical scan images using RapidOCR
(PP-OCRv3 engine, ONNX runtime — no PaddlePaddle required).

Recommended by the Tecnomate pipeline for reading annotations,
labels, measurements, and report text embedded in or alongside
medical X-ray and MRI images.

Install the runtime once:
    pip install rapidocr-onnxruntime

For systems with very limited RAM (< 2 GB free), the nano variant:
    pip install rapidocr-onnxruntime    (same package, auto-selects model)

Usage:
    from ocr_reader import extract_text, is_ocr_available

    if is_ocr_available():
        result = extract_text(pil_image)
        for item in result.lines:
            print(item.text, item.confidence)
    else:
        print("Install rapidocr-onnxruntime to enable text extraction.")
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# ── RapidOCR optional import ──────────────────────────────────────────────────
_OCR_ENGINE = None  # singleton, initialised lazily on first call
_OCR_AVAILABLE = False
_OCR_ERROR = ""

try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_ERROR = (
        "rapidocr-onnxruntime is not installed.\n"
        "Run:  pip install rapidocr-onnxruntime\n"
        "then restart the app."
    )
except Exception as _e:
    _OCR_ERROR = f"RapidOCR failed to load: {_e}"


# ─────────────────────────────────────────────────────────────────────────────
# Public result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TextLine:
    """A single line of text detected in the image."""

    text: str
    confidence: float  # 0.0 – 1.0
    bbox: Optional[List[List[int]]] = None  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence * 100:.1f}%"

    def __str__(self) -> str:
        return self.text


@dataclass
class OCRResult:
    """
    Full result returned by extract_text().

    Attributes
    ----------
    lines       : list of TextLine objects (one per detected text region)
    full_text   : all detected text joined with newlines (convenience accessor)
    available   : False if rapidocr-onnxruntime is not installed
    error       : non-empty string when something went wrong
    elapsed_ms  : approximate inference time in milliseconds
    """

    lines: List[TextLine] = field(default_factory=list)
    available: bool = True
    error: str = ""
    elapsed_ms: float = 0.0

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def found_text(self) -> bool:
        """True if at least one text region was detected."""
        return len(self.lines) > 0

    @property
    def high_confidence_lines(self) -> List[TextLine]:
        """Only lines with confidence >= 0.80."""
        return [ln for ln in self.lines if ln.confidence >= 0.80]

    def summary(self) -> str:
        """Human-readable one-liner for UI display."""
        if not self.available:
            return f"OCR unavailable: {self.error}"
        if self.error:
            return f"OCR error: {self.error}"
        if not self.found_text:
            return "No text detected in this image."
        n = len(self.lines)
        avg_conf = sum(l.confidence for l in self.lines) / n
        return f"{n} text region(s) detected  |  avg confidence {avg_conf * 100:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_engine() -> Optional["_RapidOCR"]:
    """Return the singleton RapidOCR engine, initialising it on first call."""
    global _OCR_ENGINE
    if not _OCR_AVAILABLE:
        return None
    if _OCR_ENGINE is None:
        # Use default PP-OCRv3 det + rec models.
        # print_verbose=False suppresses the model download progress spam.
        try:
            _OCR_ENGINE = _RapidOCR()
        except Exception as exc:
            warnings.warn(f"[OCR] Engine init failed: {exc}")
            return None
    return _OCR_ENGINE


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a uint8 BGR numpy array — the format RapidOCR expects.
    Handles RGB, RGBA, L (grayscale), and palette modes.
    """
    img = pil_img

    # Flatten alpha / palette to RGB first
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode == "L":
        # Grayscale → RGB (RapidOCR needs 3 channels for its detection model)
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    arr_rgb = np.array(img, dtype=np.uint8)
    # PIL gives RGB; OpenCV / RapidOCR expect BGR
    arr_bgr = arr_rgb[:, :, ::-1].copy()
    return arr_bgr


def _parse_rapidocr_output(
    raw_result,
    elapsed: float,
) -> OCRResult:
    """
    Parse the raw tuple returned by RapidOCR into a clean OCRResult.

    RapidOCR returns:  (result, elapse)
    where result is a list of [bbox, text, score]  or None.
    """
    lines: List[TextLine] = []

    if raw_result is not None:
        for item in raw_result:
            try:
                # item layout: [bbox, text, confidence]
                # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] as floats
                bbox_raw, text, confidence = item[0], item[1], item[2]

                # Sanitise
                text = str(text).strip()
                if not text:
                    continue

                confidence = float(confidence) if confidence is not None else 0.0
                confidence = max(0.0, min(1.0, confidence))

                # Convert bbox coordinates to int
                bbox = (
                    [[int(pt[0]), int(pt[1])] for pt in bbox_raw]
                    if bbox_raw is not None
                    else None
                )

                lines.append(TextLine(text=text, confidence=confidence, bbox=bbox))

            except (IndexError, TypeError, ValueError):
                # Malformed result item — skip silently
                continue

    # Sort top-to-bottom by the y-coordinate of the top-left bbox corner
    lines.sort(key=lambda ln: ln.bbox[0][1] if ln.bbox else 0)

    return OCRResult(
        lines=lines,
        available=True,
        error="",
        elapsed_ms=elapsed * 1000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# X-ray / scan-type detection via OCR text analysis
# ─────────────────────────────────────────────────────────────────────────────

# Text markers typically present on real X-ray PACS / radiograph overlays.
# Only include tokens that are SPECIFIC to X-ray and unlikely to appear on
# non-X-ray content.  Short ambiguous tokens (ap, pa, lat, kv) are matched
# with word boundaries in is_xray_scan() to avoid false positives.
_XRAY_MARKERS: frozenset = frozenset(
    {
        # Unambiguous multi-char / phrase markers
        "kvp",
        "mas",
        "cxr",
        "chest x-ray",
        "chest xray",
        "chest radiograph",
        "x-ray",
        "xray",
        "radiograph",
        "fluoroscopy",
        "radiography",
        "diaphragm",
        "mediastinum",
        "thorax",
        "pleura",
        "pneumothorax",
        "cardiomegaly",
        "atelectasis",
        "effusion",
        "pneumonia",
        "consolidation",
        "aec",
        "dap",
        # Short tokens that need word-boundary checks — kept here so the
        # caller can still list them; matching is done with _xray_word_match()
        "chest",
        "lungs",
        "ribs",
        "supine",
        "erect",
        "portable",
        "projection",
        "anterior",
        "posterior",
    }
)

# Short X-ray tokens that must be matched as whole words only (not substrings)
# to avoid false matches inside longer words (e.g. "lateral" matching "lat").
_XRAY_WORD_TOKENS: frozenset = frozenset(
    {
        "ap",
        "pa",
        "lat",
        "kv",
        "film",
        "grid",
        "dose",
    }
)

# Text markers typically present on MRI scanner overlays (not X-ray).
# ONLY include tokens that are UNAMBIGUOUS and would NOT normally appear on a
# real chest X-ray PACS overlay.
# Removed: "tr", "te", "t1", "t2", "adc", "gre", "axial", "coronal", "sagittal"
# — these are too short / common and appear in dates, patient IDs, and CXR
# DICOM overlays, causing false MRI detection on real X-rays.
_MRI_MARKERS: frozenset = frozenset(
    {
        "flair",
        "dwi",
        "mri",
        "tesla",
        "dti",
        "bold",
        "stir",
        "tse",
        "flip angle",
        "localizer",
        "inversion recovery",
        "magnetic resonance",
        "diffusion weighted",
        "echo planar",
        "gradient echo",
        "spin echo",
    }
)

# Text markers typical of CT scanner overlays.
# REMOVED "window" and "level" — these are ordinary English words that also
# appear in PACS overlays for X-rays (Window/Level = brightness/contrast
# controls) and would cause valid chest X-rays to be rejected as CT scans.
# REMOVED "level" for the same reason.
# Kept only terms that are CT-specific and would not appear on a chest X-ray.
_CT_MARKERS: frozenset = frozenset(
    {
        "hounsfield",
        " hu ",
        "ctdi",
        "dlp",
        "computed tomography",
        "ct scan",
        "slice thickness",
        "reconstruction",
        "pitch",
        "gantry",
    }
)

# Patterns that strongly suggest non-medical / non-radiograph content
_NON_MEDICAL_MARKERS: frozenset = frozenset(
    {
        "instagram",
        "twitter",
        "facebook",
        "tiktok",
        "youtube",
        "http://",
        "https://",
        "www.",
        ".com",
        ".org",
        ".net",
        "follow us",
        "subscribe",
        "photo by",
        "taken by",
        "copyright",
        "©",
        "screenshot",
        "shutterstock",
        "getty images",
        "alamy",
        "dreamstime",
        "istock",
        "stock photo",
        "royalty free",
        "123rf",
        "depositphotos",
        "bigstock",
        "gettyimages",
        "image id:",
        "imageid:",
        "credit:",
    }
)

# Pre-compiled word-boundary patterns for short X-ray tokens
_XRAY_WORD_PATTERNS: dict = {
    tok: re.compile(r"\b" + re.escape(tok) + r"\b") for tok in _XRAY_WORD_TOKENS
}


@dataclass
class XrayScanResult:
    """
    Result of OCR-based X-ray / scan-type detection.

    Attributes
    ----------
    is_xray          : True=X-ray confirmed, False=confirmed NOT an X-ray,
                       None=inconclusive (no or ambiguous text found).
    scan_type_detected : "xray" | "mri" | "ct" | "non_medical" | "unknown"
    confidence       : 0.0–1.0 — how confident the OCR analysis is.
    keywords_found   : list of matched marker strings that drove the decision.
    rejection_reason : human-readable rejection message (non-empty when
                       is_xray is False and the caller should reject).
    ocr_available    : False when rapidocr-onnxruntime is not installed.
    ocr_ran          : False when the image contained no detectable text.
    """

    is_xray: Optional[bool]
    scan_type_detected: str
    confidence: float
    keywords_found: List[str]
    rejection_reason: str
    ocr_available: bool
    ocr_ran: bool


# Stock-photo watermark keywords — subset of _NON_MEDICAL_MARKERS
_WATERMARK_KEYWORDS: frozenset = frozenset(
    {
        "alamy",
        "shutterstock",
        "istock",
        "getty images",
        "dreamstime",
        "stock photo",
        "royalty free",
        "123rf",
        "depositphotos",
        "bigstock",
    }
)


def _build_nonmed_rejection(nonmed_hits: list) -> str:
    """
    Return a human-readable rejection reason for non-medical content,
    specifically calling out stock-photo watermarks when detected.
    """
    hits_lower = [h.lower() for h in nonmed_hits]

    # Check whether the detected markers are stock-photo watermarks
    watermark_hits = [h for h in hits_lower if h in _WATERMARK_KEYWORDS]

    # Also catch partial matches like 'www.alamy.com' already split into tokens
    is_watermarked = bool(watermark_hits) or any(
        w in " ".join(hits_lower) for w in _WATERMARK_KEYWORDS
    )

    if is_watermarked:
        sources = [h for h in nonmed_hits if h.lower() in _WATERMARK_KEYWORDS]
        label = ", ".join(sources[:3]) if sources else ", ".join(nonmed_hits[:3])
        return (
            "This image contains a stock-photo watermark "
            f"({label}) and cannot be used for clinical analysis. "
            "Please upload an original, unwatermarked chest X-ray from a "
            "radiology system or PACS export (JPEG / PNG / DICOM)."
        )

    return (
        "This image does not appear to be a medical scan. "
        f"Non-medical content was detected: {', '.join(nonmed_hits[:3])}. "
        "Please upload an original, unedited chest X-ray or MRI image."
    )


def is_xray_scan(image: Image.Image) -> XrayScanResult:
    """
    Use OCR text analysis to determine whether an uploaded image is an X-ray.

    Strategy
    --------
    Many real X-ray images from clinical datasets have *no* embedded text —
    those return ``is_xray=None`` (inconclusive) and are **never rejected**.

    The check only returns ``is_xray=False`` when text *positively* and
    unambiguously identifies the image as something other than an X-ray:
      • Non-medical content (URLs, social-media handles, watermarks)  → reject
      • MRI-specific overlay markers (FLAIR, DWI, MRI, Tesla …)        → reject
      • CT-specific overlay markers (Hounsfield, CTDI, DLP, CT scan …) → reject

    X-ray confirmation uses a generous rule: any X-ray marker is sufficient
    to confirm UNLESS a stronger opposing signal (MRI/CT specific markers)
    is also present.

    Returns
    -------
    XrayScanResult
        Callers should check ``is_xray is False`` (not just falsy) to decide
        whether to reject.  ``is_xray=None`` means "cannot tell from OCR".

    Notes
    -----
    • OCR is run at ``min_confidence=0.40`` to catch faint scanner overlays.
    • Short tokens (ap, pa, lat, kv …) are matched as whole words only to
      avoid false matches inside longer words.
    • Results are deterministic: same image → same outcome.
    • If ``rapidocr-onnxruntime`` is not installed the function returns
      ``ocr_available=False`` and never rejects.
    """
    _inconclusive = XrayScanResult(
        is_xray=None,
        scan_type_detected="unknown",
        confidence=0.0,
        keywords_found=[],
        rejection_reason="",
        ocr_available=True,
        ocr_ran=False,
    )

    if not _OCR_AVAILABLE:
        return XrayScanResult(
            is_xray=None,
            scan_type_detected="unknown",
            confidence=0.0,
            keywords_found=[],
            rejection_reason="",
            ocr_available=False,
            ocr_ran=False,
        )

    ocr_result = extract_text(image, min_confidence=0.40)
    if not ocr_result.found_text:
        return _inconclusive

    full_lower = ocr_result.full_text.lower()

    # Substring matches for multi-char markers
    xray_hits = [kw for kw in _XRAY_MARKERS if kw in full_lower]
    mri_hits = [kw for kw in _MRI_MARKERS if kw in full_lower]
    ct_hits = [kw for kw in _CT_MARKERS if kw in full_lower]
    nonmed_hits = [kw for kw in _NON_MEDICAL_MARKERS if kw in full_lower]

    # Word-boundary matches for short / ambiguous X-ray tokens
    word_xray_hits = [
        tok for tok, pat in _XRAY_WORD_PATTERNS.items() if pat.search(full_lower)
    ]
    all_xray_hits = list(dict.fromkeys(xray_hits + word_xray_hits))  # deduplicated

    # ── 1. Non-medical content (highest priority) ─────────────────────────────
    if nonmed_hits:
        return XrayScanResult(
            is_xray=False,
            scan_type_detected="non_medical",
            confidence=min(0.70 + len(nonmed_hits) * 0.10, 0.97),
            keywords_found=nonmed_hits,
            rejection_reason=_build_nonmed_rejection(nonmed_hits),
            ocr_available=True,
            ocr_ran=True,
        )

    # ── 2. X-ray confirmed ────────────────────────────────────────────────────
    # Rule: any X-ray hit confirms the image as an X-ray, UNLESS the opposing
    # signal from MRI or CT is both present AND strictly stronger (more unique
    # unambiguous markers).  This prevents false rejection when a PACS X-ray
    # viewer happens to show overlay labels that also exist as generic words.
    if all_xray_hits:
        # Only override X-ray confirmation if MRI/CT has MORE hits and those
        # hits are unambiguous multi-word phrases (len > 4 chars).
        strong_mri = [h for h in mri_hits if len(h) > 4]
        strong_ct = [h for h in ct_hits if len(h) > 4]
        n_xray = len(all_xray_hits)
        n_opposing = max(len(strong_mri), len(strong_ct))

        if n_opposing <= n_xray:
            # X-ray wins — confirmed
            return XrayScanResult(
                is_xray=True,
                scan_type_detected="xray",
                confidence=min(0.55 + n_xray * 0.10, 0.95),
                keywords_found=all_xray_hits,
                rejection_reason="",
                ocr_available=True,
                ocr_ran=True,
            )
        # else: fall through to check MRI/CT below

    # ── 3. MRI detected ───────────────────────────────────────────────────────
    # Require at least 2 strong MRI markers to avoid single-token false positives.
    strong_mri_hits = [h for h in mri_hits if len(h) > 3]
    if strong_mri_hits and len(strong_mri_hits) >= 2:
        return XrayScanResult(
            is_xray=False,
            scan_type_detected="mri",
            confidence=min(0.55 + len(strong_mri_hits) * 0.12, 0.95),
            keywords_found=strong_mri_hits,
            rejection_reason=(
                "This image appears to be an MRI scan, not an X-ray. "
                f"MRI markers detected: {', '.join(strong_mri_hits[:3])}. "
                "Please select 'Brain MRI' scan type, or upload a chest X-ray."
            ),
            ocr_available=True,
            ocr_ran=True,
        )

    # ── 4. CT detected ────────────────────────────────────────────────────────
    # Require at least 2 strong CT markers OR 1 highly specific CT marker
    # (hounsfield, ctdi, dlp, computed tomography) to avoid false positives.
    _HIGH_SPECIFICITY_CT = {
        "hounsfield",
        "ctdi",
        "dlp",
        "computed tomography",
        "ct scan",
    }
    strong_ct_hits = [h for h in ct_hits if h.strip() in _HIGH_SPECIFICITY_CT]
    if strong_ct_hits:
        return XrayScanResult(
            is_xray=False,
            scan_type_detected="ct",
            confidence=min(0.60 + len(strong_ct_hits) * 0.12, 0.92),
            keywords_found=strong_ct_hits,
            rejection_reason=(
                "This image appears to be a CT scan. "
                "This system only accepts X-ray and MRI images — "
                "CT scans are not supported."
            ),
            ocr_available=True,
            ocr_ran=True,
        )

    # ── 5. Text found but no conclusive markers ───────────────────────────────
    return XrayScanResult(
        is_xray=None,
        scan_type_detected="unknown",
        confidence=0.25,
        keywords_found=[],
        rejection_reason="",
        ocr_available=True,
        ocr_ran=True,
    )


def is_ocr_available() -> bool:
    """
    Return True if rapidocr-onnxruntime is installed and the engine
    can be loaded successfully.
    """
    return _OCR_AVAILABLE and _get_engine() is not None


def ocr_unavailable_reason() -> str:
    """Return the human-readable reason why OCR is not available, or ''."""
    if _OCR_AVAILABLE and _get_engine() is not None:
        return ""
    return _OCR_ERROR or "RapidOCR engine could not be initialised."


def extract_text(
    image: Image.Image,
    min_confidence: float = 0.50,
) -> OCRResult:
    """
    Extract all text from a PIL Image using RapidOCR (PP-OCRv3).

    Parameters
    ----------
    image          : PIL.Image.Image — the uploaded scan (any mode/size)
    min_confidence : discard detections with confidence below this threshold
                     (default 0.50 — keeps most detections for medical use)

    Returns
    -------
    OCRResult  — always returned; check .available and .error for failure modes.

    Notes
    -----
    • The image is NOT resized before OCR — RapidOCR handles arbitrary
      resolutions internally.  Uploading the original high-res scan gives
      better text detection than the 128×128 model input tensor.
    • Grayscale MRI/CXR images are converted to RGB for the detection model.
    • Results are sorted top-to-bottom by bounding-box position.
    """
    if not _OCR_AVAILABLE:
        return OCRResult(available=False, error=_OCR_ERROR)

    engine = _get_engine()
    if engine is None:
        return OCRResult(
            available=False,
            error="RapidOCR engine failed to initialise. "
            "Try reinstalling: pip install --force-reinstall rapidocr-onnxruntime",
        )

    try:
        bgr = _pil_to_bgr(image)
        raw_result, elapsed = engine(bgr)
        result = _parse_rapidocr_output(raw_result, elapsed)

        # Apply confidence threshold
        if min_confidence > 0:
            result.lines = [
                ln for ln in result.lines if ln.confidence >= min_confidence
            ]

        return result

    except Exception as exc:
        return OCRResult(
            available=True,
            error=f"OCR inference failed: {exc}",
            elapsed_ms=0.0,
        )


def extract_text_from_array(
    bgr_array: np.ndarray,
    min_confidence: float = 0.50,
) -> OCRResult:
    """
    Convenience wrapper: accepts a uint8 BGR numpy array directly
    (e.g. from cv2.imread) instead of a PIL Image.
    """
    if not _OCR_AVAILABLE:
        return OCRResult(available=False, error=_OCR_ERROR)

    engine = _get_engine()
    if engine is None:
        return OCRResult(available=False, error="RapidOCR engine init failed.")

    try:
        raw_result, elapsed = engine(bgr_array)
        result = _parse_rapidocr_output(raw_result, elapsed)
        if min_confidence > 0:
            result.lines = [
                ln for ln in result.lines if ln.confidence >= min_confidence
            ]
        return result
    except Exception as exc:
        return OCRResult(available=True, error=f"OCR inference failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Medical-specific helpers
# ─────────────────────────────────────────────────────────────────────────────

# Keywords commonly found on medical scans / X-ray overlays
_MEDICAL_KEYWORDS = {
    # Patient / study info
    "patient",
    "name",
    "id",
    "dob",
    "age",
    "sex",
    "male",
    "female",
    "date",
    "time",
    "study",
    "exam",
    "accession",
    # Imaging parameters
    "kv",
    "kVp",
    "mas",
    "mAs",
    "dose",
    "kvp",
    "cm",
    "mm",
    "ap",
    "pa",
    "lat",
    "lateral",
    "anterior",
    "posterior",
    "left",
    "right",
    "l",
    "r",
    # Radiology findings
    "impression",
    "findings",
    "normal",
    "abnormal",
    "no acute",
    "opacity",
    "consolidation",
    "effusion",
    "pneumonia",
    "tumor",
    "mass",
    "lesion",
    "nodule",
    "cardiomegaly",
    # Brain MRI markers
    "t1",
    "t2",
    "flair",
    "dwi",
    "adc",
    "gre",
    "tr",
    "te",
    "axial",
    "coronal",
    "sagittal",
    "mri",
    "ct",
    # Institution
    "hospital",
    "clinic",
    "radiology",
    "dr",
    "dr.",
    "md",
}


def filter_medical_text(result: OCRResult) -> OCRResult:
    """
    Return a copy of the OCRResult containing only lines that contain
    at least one recognised medical keyword.

    Useful for filtering out scanner artefacts and non-informative
    text overlays (e.g. scale bars, manufacturer watermarks).
    """
    filtered_lines = []
    for line in result.lines:
        lower = line.text.lower()
        if any(kw in lower for kw in _MEDICAL_KEYWORDS):
            filtered_lines.append(line)

    return OCRResult(
        lines=filtered_lines,
        available=result.available,
        error=result.error,
        elapsed_ms=result.elapsed_ms,
    )


def format_for_report(result: OCRResult, header: str = "Extracted Text") -> str:
    """
    Format an OCRResult as a plain-text block suitable for inclusion
    in the downloadable diagnostic report generated by app.py.

    Parameters
    ----------
    result : OCRResult from extract_text()
    header : section heading to use in the report block

    Returns
    -------
    A formatted multi-line string ready to append to the report.
    """
    if not result.available:
        return f"\n{header}\n{'─' * 40}\n  [OCR not available — install rapidocr-onnxruntime]\n"

    if result.error:
        return f"\n{header}\n{'─' * 40}\n  [OCR error: {result.error}]\n"

    if not result.found_text:
        return f"\n{header}\n{'─' * 40}\n  [No text detected in this image]\n"

    lines_block = "\n".join(
        f"  {ln.text}  ({ln.confidence_pct})" for ln in result.lines
    )
    return (
        f"\n{header}\n"
        f"{'─' * 40}\n"
        f"{lines_block}\n"
        f"  [{len(result.lines)} region(s), {result.elapsed_ms:.0f}ms]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  Tecnomate OCR Reader — smoke test")
    print("=" * 55)
    print(f"  RapidOCR available : {is_ocr_available()}")
    if not is_ocr_available():
        print(f"  Reason             : {ocr_unavailable_reason()}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("\n  Usage: python src/ocr_reader.py <image_path>")
        print(
            "  Example: python src/ocr_reader.py data/partitions/global_test/pneumonia/some.jpg"
        )
        sys.exit(0)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"  [ERROR] File not found: {img_path}")
        sys.exit(1)

    print(f"\n  Reading: {img_path.name}")
    pil_img = Image.open(img_path)
    print(f"  Image size : {pil_img.size}  mode={pil_img.mode}")

    result = extract_text(pil_img)
    print(f"\n  {result.summary()}")
    print(f"  Elapsed    : {result.elapsed_ms:.0f} ms")

    if result.found_text:
        print("\n  Detected text:")
        for ln in result.lines:
            print(f"    [{ln.confidence_pct:>6}]  {ln.text}")

        medical = filter_medical_text(result)
        if medical.found_text:
            print(f"\n  Medical keywords found ({len(medical.lines)} line(s)):")
            for ln in medical.lines:
                print(f"    [{ln.confidence_pct:>6}]  {ln.text}")

    print("\n" + format_for_report(result))
