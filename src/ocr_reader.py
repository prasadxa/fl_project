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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# ── RapidOCR optional import ──────────────────────────────────────────────────
_OCR_ENGINE   = None      # singleton, initialised lazily on first call
_OCR_AVAILABLE = False
_OCR_ERROR    = ""

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
    text:       str
    confidence: float                   # 0.0 – 1.0
    bbox:       Optional[List[List[int]]] = None   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

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
    lines:      List[TextLine] = field(default_factory=list)
    available:  bool           = True
    error:      str            = ""
    elapsed_ms: float          = 0.0

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
        return f"{n} text region(s) detected  |  avg confidence {avg_conf*100:.1f}%"


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
                bbox = [[int(pt[0]), int(pt[1])] for pt in bbox_raw] \
                       if bbox_raw is not None else None

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
    "patient", "name", "id", "dob", "age", "sex", "male", "female",
    "date", "time", "study", "exam", "accession",
    # Imaging parameters
    "kv", "kVp", "mas", "mAs", "dose", "kvp", "cm", "mm",
    "ap", "pa", "lat", "lateral", "anterior", "posterior",
    "left", "right", "l", "r",
    # Radiology findings
    "impression", "findings", "normal", "abnormal", "no acute",
    "opacity", "consolidation", "effusion", "pneumonia", "tumor",
    "mass", "lesion", "nodule", "cardiomegaly",
    # Brain MRI markers
    "t1", "t2", "flair", "dwi", "adc", "gre", "tr", "te",
    "axial", "coronal", "sagittal", "mri", "ct",
    # Institution
    "hospital", "clinic", "radiology", "dr", "dr.", "md",
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
        return f"\n{header}\n{'─'*40}\n  [OCR not available — install rapidocr-onnxruntime]\n"

    if result.error:
        return f"\n{header}\n{'─'*40}\n  [OCR error: {result.error}]\n"

    if not result.found_text:
        return f"\n{header}\n{'─'*40}\n  [No text detected in this image]\n"

    lines_block = "\n".join(
        f"  {ln.text}  ({ln.confidence_pct})"
        for ln in result.lines
    )
    return (
        f"\n{header}\n"
        f"{'─'*40}\n"
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
        print("  Example: python src/ocr_reader.py data/partitions/global_test/pneumonia/some.jpg")
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
