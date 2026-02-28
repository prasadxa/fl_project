"""
Tecnomate Clinical AI — PDF Report Generator
=============================================
Generates a professional clinical-grade PDF diagnostic report using
ReportLab (pure-Python, zero LaTeX installation required).

The report includes:
  - Patient / visit metadata (optional — entered by clinician)
  - Scan image thumbnail + Grad-CAM heatmap overlay
  - AI prediction + confidence + uncertainty (MC-Dropout)
  - Per-class probability bar chart
  - Doctor confirmed / overridden label + ICD-10 code
  - OCR-extracted text block
  - Audit trail (session ID, timestamps, FL round)
  - Privacy notice + medical disclaimer
  - Clinician signature line

Dependencies (add to requirements.txt):
    reportlab>=4.0.0
    matplotlib>=3.7.0  (already present)

Usage (called from api.py):
    from report_generator import build_pdf_report, ReportRequest
    pdf_bytes = build_pdf_report(req)
"""

from __future__ import annotations

import datetime
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

# ── palette ───────────────────────────────────────────────────────────────────
_BRAND_BLUE = colors.HexColor("#1a6ed8")
_BRAND_DARK = colors.HexColor("#0f172a")
_BRAND_MID = colors.HexColor("#1e293b")
_BRAND_LIGHT = colors.HexColor("#f1f5f9")
_BRAND_BORDER = colors.HexColor("#cbd5e1")
_GREEN = colors.HexColor("#16a34a")
_AMBER = colors.HexColor("#d97706")
_RED = colors.HexColor("#dc2626")
_TEXT_MUTED = colors.HexColor("#64748b")
_TEXT_BODY = colors.HexColor("#1e293b")
_WHITE = colors.white
_HEATMAP_ALPHA = 0.45  # overlay transparency when blending Grad-CAM

# ── ICD-10 code map ───────────────────────────────────────────────────────────
ICD10_MAP: Dict[str, Tuple[str, str]] = {
    "glioma": ("C71.9", "Malignant neoplasm of brain, unspecified"),
    "meningioma": ("D32.9", "Benign neoplasm of meninges, unspecified"),
    "notumor": (
        "Z03.89",
        "Encounter for observation for other suspected diseases — no diagnosis",
    ),
    "pituitary": ("D35.2", "Benign neoplasm of pituitary gland"),
    "normal": (
        "Z03.89",
        "Chest X-Ray within normal limits — no acute cardiopulmonary disease",
    ),
    "pneumonia": ("J18.9", "Pneumonia, unspecified organism"),
}

RISK_LEVEL: Dict[str, str] = {
    "glioma": "HIGH",
    "meningioma": "MEDIUM",
    "notumor": "LOW",
    "pituitary": "MEDIUM",
    "normal": "LOW",
    "pneumonia": "HIGH",
}

SHORT_NAMES: Dict[str, str] = {
    "glioma": "Glioma (Malignant Brain Tumor)",
    "meningioma": "Meningioma (Brain Tumor)",
    "notumor": "No Tumor Detected",
    "pituitary": "Pituitary Tumor",
    "normal": "Normal / Healthy (CXR)",
    "pneumonia": "Pneumonia Detected (CXR)",
}

RISK_COLOR: Dict[str, object] = {
    "HIGH": _RED,
    "MEDIUM": _AMBER,
    "LOW": _GREEN,
}


# ═════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PatientInfo:
    """Optional patient / visit fields — never stored, only in the PDF."""

    patient_name: str = "Anonymous / De-identified"
    patient_id: str = "N/A"
    date_of_birth: str = "N/A"
    gender: str = "N/A"
    referring_doctor: str = "N/A"
    institution: str = "Tecnomate Health Network"
    visit_date: str = ""  # defaults to today if empty
    clinical_notes: str = ""


@dataclass
class UncertaintyInfo:
    """MC-Dropout uncertainty estimates."""

    mean_entropy: float = 0.0
    std_confidence: float = 0.0
    mc_samples: int = 0
    uncertainty_label: str = "N/A"  # Low / Moderate / High


@dataclass
class ReportRequest:
    """All data needed to build one PDF report."""

    # core prediction
    session_id: str
    filename: str
    scan_type: str  # "Brain MRI" | "Chest X-Ray"
    ai_pred_key: str
    ai_confidence: float
    probabilities: Dict[str, float]  # key → 0-1
    doctor_choice_key: str
    ocr_text: str = ""
    ocr_lines: List[Dict] = field(default_factory=list)

    # images
    scan_image_path: Optional[Path] = None  # original upload (temp copy)
    gradcam_image: Optional[np.ndarray] = None  # H×W×3 uint8 overlay

    # optional metadata
    patient: PatientInfo = field(default_factory=PatientInfo)
    uncertainty: UncertaintyInfo = field(default_factory=UncertaintyInfo)
    fl_round: int = 0
    model_version: str = "global_model.pth"
    server_timestamp: str = ""
    clinician_name: str = ""
    clinician_id: str = ""


# ═════════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═════════════════════════════════════════════════════════════════════════════


def compute_gradcam(
    model,  # MedicalCNN instance
    tensor,  # (1, 1, 128, 128) float tensor
    target_class_idx: int,
    original_pil: Image.Image,
) -> np.ndarray:
    """
    Compute Grad-CAM for `target_class_idx` over the last conv layer
    (`model.features[-1]`) and return an H×W×3 uint8 heatmap-on-image overlay.

    Returns a zero-alpha blank (original image only) if anything fails.
    """
    import torch
    import torch.nn.functional as F_nn

    try:
        # Storage hooks
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(module, inp, out):
            activations.clear()
            activations.append(out.detach())

        def bwd_hook(module, grad_in, grad_out):
            gradients.clear()
            gradients.append(grad_out[0].detach())

        last_conv = model.features[-1]
        h1 = last_conv.register_forward_hook(fwd_hook)
        h2 = last_conv.register_full_backward_hook(bwd_hook)

        model.eval()
        inp = tensor.clone().requires_grad_(True)
        logits = model(inp)

        model.zero_grad()
        score = logits[0, target_class_idx]
        score.backward()

        h1.remove()
        h2.remove()

        act = activations[0].squeeze(0)  # (C, H, W)
        grad = gradients[0].squeeze(0)  # (C, H, W)

        # Global average pool gradients → channel weights
        weights = grad.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(act.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = F_nn.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_np = cam.numpy()

        # Resize cam to original image size
        import cv2

        orig_w, orig_h = original_pil.size
        cam_resized = cv2.resize(
            cam_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # Colourmap (jet) → RGB uint8
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend with original image (convert to RGB first)
        orig_rgb = np.array(original_pil.convert("RGB"), dtype=np.uint8)
        # Resize heatmap to match if sizes differ (safety)
        if heatmap_rgb.shape[:2] != orig_rgb.shape[:2]:
            heatmap_rgb = cv2.resize(
                heatmap_rgb, (orig_rgb.shape[1], orig_rgb.shape[0])
            )

        overlay = (
            (
                (1 - _HEATMAP_ALPHA) * orig_rgb.astype(np.float32)
                + _HEATMAP_ALPHA * heatmap_rgb.astype(np.float32)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )

        return overlay

    except Exception as exc:
        print(f"[Grad-CAM] Warning: {exc}  — skipping heatmap.")
        # Fallback: return the original image as RGB array
        try:
            return np.array(original_pil.convert("RGB"), dtype=np.uint8)
        except Exception:
            return np.zeros((128, 128, 3), dtype=np.uint8)


def compute_mc_uncertainty(
    model,
    tensor,
    n_samples: int = 30,
    class_keys: Optional[List[str]] = None,
) -> UncertaintyInfo:
    """
    Estimate predictive uncertainty via MC-Dropout.

    Runs the model `n_samples` times with dropout active and reports
    mean entropy + std of the top-class confidence.
    """
    import torch
    import torch.nn.functional as F_nn

    try:
        model.train()  # keeps dropout ON
        probs_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = model(tensor)
                p = F_nn.softmax(logits, dim=1).squeeze().numpy()
                probs_list.append(p)
        model.eval()

        probs_arr = np.stack(probs_list, axis=0)  # (n_samples, n_classes)
        mean_probs = probs_arr.mean(axis=0)  # (n_classes,)
        std_probs = probs_arr.std(axis=0)  # (n_classes,)

        # Shannon entropy of mean distribution
        eps = 1e-8
        entropy = -float(np.sum(mean_probs * np.log(mean_probs + eps)))
        max_entropy = math.log(len(mean_probs))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        top_idx = int(np.argmax(mean_probs))
        std_top_conf = float(std_probs[top_idx])

        if norm_entropy < 0.25:
            label = "Low — model is confident"
        elif norm_entropy < 0.55:
            label = "Moderate — clinician review recommended"
        else:
            label = "High — result should be treated with caution"

        return UncertaintyInfo(
            mean_entropy=round(float(entropy), 4),
            std_confidence=round(std_top_conf, 4),
            mc_samples=n_samples,
            uncertainty_label=label,
        )
    except Exception as exc:
        print(f"[MC-Dropout] Warning: {exc}")
        model.eval()
        return UncertaintyInfo()


# ═════════════════════════════════════════════════════════════════════════════
#  PDF builder
# ═════════════════════════════════════════════════════════════════════════════


class _NumberedCanvas(rl_canvas.Canvas):
    """ReportLab canvas that draws page numbers in the footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()  # type: ignore[attr-defined]

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_footer(self, page_count: int):
        page_num = self._pageNumber  # type: ignore[attr-defined]
        self.saveState()
        self.setFont("Helvetica", 7)
        self.setFillColor(_TEXT_MUTED)
        w, _ = A4
        self.drawCentredString(
            w / 2,
            12 * mm,
            f"Tecnomate Clinical AI  ·  CONFIDENTIAL  ·  Page {page_num} of {page_count}",
        )
        self.restoreState()


def _styles() -> dict:
    """Return a dict of ParagraphStyle objects keyed by name."""
    base = getSampleStyleSheet()
    S = {}

    def ps(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    S["h1"] = ps(
        "h1",
        fontSize=20,
        leading=26,
        textColor=_WHITE,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["h2"] = ps(
        "h2",
        fontSize=13,
        leading=18,
        textColor=_BRAND_DARK,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    S["section_label"] = ps(
        "section_label",
        fontSize=7.5,
        leading=10,
        textColor=_BRAND_BLUE,
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=2,
        # small caps effect via uppercase in text
    )
    S["body"] = ps(
        "body",
        fontSize=9,
        leading=13,
        textColor=_TEXT_BODY,
        fontName="Helvetica",
    )
    S["body_bold"] = ps(
        "body_bold",
        fontSize=9,
        leading=13,
        textColor=_TEXT_BODY,
        fontName="Helvetica-Bold",
    )
    S["muted"] = ps(
        "muted",
        fontSize=8,
        leading=12,
        textColor=_TEXT_MUTED,
        fontName="Helvetica",
    )
    S["caption"] = ps(
        "caption",
        fontSize=7.5,
        leading=11,
        textColor=_TEXT_MUTED,
        fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
    )
    S["disclaimer"] = ps(
        "disclaimer",
        fontSize=7.5,
        leading=11,
        textColor=_TEXT_MUTED,
        fontName="Helvetica",
    )
    S["icd"] = ps(
        "icd",
        fontSize=8.5,
        leading=12,
        textColor=_TEXT_BODY,
        fontName="Courier",
    )
    S["ocr_line"] = ps(
        "ocr_line",
        fontSize=8.5,
        leading=12,
        textColor=_TEXT_BODY,
        fontName="Courier",
        leftIndent=6,
    )
    S["centered"] = ps(
        "centered",
        fontSize=9,
        leading=13,
        textColor=_TEXT_BODY,
        fontName="Helvetica",
        alignment=TA_CENTER,
    )
    S["risk_high"] = ps(
        "risk_high",
        fontSize=14,
        leading=18,
        textColor=_RED,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["risk_medium"] = ps(
        "risk_medium",
        fontSize=14,
        leading=18,
        textColor=_AMBER,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    S["risk_low"] = ps(
        "risk_low",
        fontSize=14,
        leading=18,
        textColor=_GREEN,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    return S


def _hr(width: float = 0) -> HRFlowable:
    return HRFlowable(
        width=width or "100%",
        thickness=0.6,
        color=_BRAND_BORDER,
        spaceAfter=6,
        spaceBefore=6,
    )


def _section_heading(title: str, S: dict) -> List:
    return [
        Paragraph(title.upper(), S["section_label"]),
        _hr(),
    ]


def _kv_table(rows: List[Tuple[str, str]], S: dict, col_widths=None) -> Table:
    """Two-column key/value table."""
    data = [[Paragraph(k, S["body_bold"]), Paragraph(v, S["body"])] for k, v in rows]
    cw = col_widths or [5.5 * cm, 10.5 * cm]
    t = Table(data, colWidths=cw, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_BRAND_LIGHT, _WHITE]),
                ("GRID", (0, 0), (-1, -1), 0.3, _BRAND_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _pil_to_rl_image(pil_img: Image.Image, max_w_cm: float, max_h_cm: float) -> RLImage:
    """Convert a PIL image to a ReportLab Image flowable, respecting max dimensions."""
    buf = io.BytesIO()
    fmt = pil_img.format or "PNG"
    if fmt.upper() not in ("JPEG", "PNG", "BMP", "TIFF", "GIF"):
        fmt = "PNG"
    pil_img.save(buf, format=fmt)
    buf.seek(0)

    w_px, h_px = pil_img.size
    max_w_pt = max_w_cm * cm
    max_h_pt = max_h_cm * cm

    scale = min(max_w_pt / w_px, max_h_pt / h_px, 1.0)
    draw_w = w_px * scale
    draw_h = h_px * scale

    return RLImage(buf, width=draw_w, height=draw_h)


def _np_to_rl_image(arr: np.ndarray, max_w_cm: float, max_h_cm: float) -> RLImage:
    pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return _pil_to_rl_image(pil_img, max_w_cm, max_h_cm)


def _prob_bar_image(
    probs: Dict[str, float],
    mode_keys: List[str],
    mode_labels: Dict[str, str],
    pred_key: str,
    figsize=(6.0, 2.0),
) -> Optional[RLImage]:
    """Render a horizontal bar chart and return as a ReportLab Image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        keys = [k for k in mode_keys if k in probs]
        values = [probs[k] * 100 for k in keys]
        labels = [mode_labels.get(k, k) for k in keys]
        bar_colors = [
            "#e74c3c"
            if k == "glioma"
            else "#e67e22"
            if k in ("meningioma", "pituitary")
            else "#e74c3c"
            if k == "pneumonia"
            else "#27ae60"
            for k in keys
        ]
        # highlight predicted bar
        bar_colors = [
            c if k != pred_key else "#1a6ed8" for k, c in zip(keys, bar_colors)
        ]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(labels, values, color=bar_colors, height=0.55, edgecolor="none")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=7)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("#f8fafc")

        for bar, val in zip(bars, values):
            ax.text(
                min(val + 1.5, 96),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=7,
                color="#1e293b",
            )

        # legend patch for predicted
        patch = mpatches.Patch(color="#1a6ed8", label="AI Prediction")
        ax.legend(handles=[patch], fontsize=7, loc="lower right", framealpha=0.6)

        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_chart = Image.open(buf)

        w_px, h_px = pil_chart.size
        max_w_pt = 14 * cm
        max_h_pt = 6 * cm
        scale = min(max_w_pt / w_px, max_h_pt / h_px, 1.0)
        draw_w = w_px * scale
        draw_h = h_px * scale
        buf.seek(0)
        return RLImage(buf, width=draw_w, height=draw_h)

    except Exception as exc:
        print(f"[PDF] bar chart error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Header band
# ─────────────────────────────────────────────────────────────────────────────


def _build_cover_band(S: dict, req: ReportRequest) -> List:
    """Blue header band with title and report date."""
    ts = req.server_timestamp or datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    # We draw the header as a coloured Table acting as a banner
    header_data = [
        [
            Paragraph("TECNOMATE CLINICAL AI", S["h1"]),
        ]
    ]
    header_table = Table(header_data, colWidths=["100%"])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _BRAND_DARK),
                ("TOPPADDING", (0, 0), (-1, -1), 16),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )

    sub_data = [
        [
            Paragraph(
                "DIAGNOSTIC REPORT  ·  CONFIDENTIAL",
                ParagraphStyle(
                    "subhead",
                    fontSize=9,
                    textColor=colors.HexColor("#94a3b8"),
                    fontName="Helvetica",
                    alignment=TA_CENTER,
                ),
            ),
        ]
    ]
    sub_table = Table(sub_data, colWidths=["100%"])
    sub_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _BRAND_MID),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    meta_data = [
        [
            Paragraph(f"<b>Report Date:</b> {ts}", S["muted"]),
            Paragraph(f"<b>Session ID:</b> {req.session_id[:16]}…", S["muted"]),
            Paragraph(f"<b>Scan Type:</b> {req.scan_type}", S["muted"]),
        ]
    ]
    meta_table = Table(meta_data, colWidths=["33%", "34%", "33%"])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _BRAND_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.3, _BRAND_BORDER),
            ]
        )
    )

    return [header_table, sub_table, meta_table, Spacer(1, 10)]


# ─────────────────────────────────────────────────────────────────────────────
#  Patient section
# ─────────────────────────────────────────────────────────────────────────────


def _build_patient_section(S: dict, req: ReportRequest) -> List:
    p = req.patient
    vdate = p.visit_date or datetime.date.today().strftime("%Y-%m-%d")

    rows = [
        ("Patient Name", p.patient_name),
        ("Patient ID", p.patient_id),
        ("Date of Birth", p.date_of_birth),
        ("Gender", p.gender),
        ("Visit Date", vdate),
        ("Referring Clinician", p.referring_doctor),
        ("Institution", p.institution),
    ]

    out = [*_section_heading("Patient & Visit Information", S)]
    out.append(_kv_table(rows, S))

    if p.clinical_notes.strip():
        out.append(Spacer(1, 6))
        out.append(Paragraph("<b>Clinical Notes:</b>", S["body_bold"]))
        out.append(Paragraph(p.clinical_notes.replace("\n", "<br/>"), S["body"]))

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Images section (scan + heatmap)
# ─────────────────────────────────────────────────────────────────────────────


def _build_images_section(S: dict, req: ReportRequest) -> List:
    out = [*_section_heading("Scan Image & Grad-CAM Explainability", S)]

    has_scan = req.scan_image_path and Path(req.scan_image_path).exists()
    has_heatmap = req.gradcam_image is not None

    if not has_scan and not has_heatmap:
        out.append(Paragraph("No scan image available for this report.", S["muted"]))
        return out

    img_cells = []

    if has_scan:
        try:
            pil_scan = Image.open(req.scan_image_path).convert("RGB")  # type: ignore[arg-type]
            scan_rl = _pil_to_rl_image(pil_scan, 7.0, 7.0)
            img_cells.append([scan_rl, Paragraph("Original Scan", S["caption"])])
        except Exception as exc:
            print(f"[PDF] scan image error: {exc}")
            img_cells.append([Paragraph("(scan unavailable)", S["muted"]), ""])

    if has_heatmap:
        try:
            heatmap_rl = _np_to_rl_image(req.gradcam_image, 7.0, 7.0)  # type: ignore[arg-type]
            img_cells.append(
                [heatmap_rl, Paragraph("Grad-CAM Attention Heatmap", S["caption"])]
            )
        except Exception as exc:
            print(f"[PDF] heatmap error: {exc}")

    if not img_cells:
        return out

    # Lay images side by side if both present, otherwise centred
    if len(img_cells) == 2:
        img_row = [[img_cells[0][0], img_cells[1][0]]]
        cap_row = [[img_cells[0][1], img_cells[1][1]]]
        cws = [8.5 * cm, 8.5 * cm]
        t_img = Table(img_row, colWidths=cws, hAlign="CENTER")
        t_cap = Table(cap_row, colWidths=cws, hAlign="CENTER")
        for t in (t_img, t_cap):
            t.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
        out.extend([t_img, t_cap])
    else:
        # Single image centred
        solo_tbl = Table([[img_cells[0][0]]], hAlign="CENTER")
        solo_cap = Table([[img_cells[0][1]]], hAlign="CENTER")
        for t in (solo_tbl, solo_cap):
            t.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
        out.extend([solo_tbl, solo_cap])

    out.append(Spacer(1, 4))
    out.append(
        Paragraph(
            "The Grad-CAM heatmap highlights image regions most influential to the AI prediction. "
            "Warmer colours (red/yellow) indicate higher activation.",
            S["muted"],
        )
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  AI Prediction section
# ─────────────────────────────────────────────────────────────────────────────

SCAN_MODE_CFG = {
    "Brain MRI": {
        "keys": ["glioma", "meningioma", "notumor", "pituitary"],
        "labels": {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "notumor": "No Tumor",
            "pituitary": "Pituitary Tumor",
        },
    },
    "Chest X-Ray": {
        "keys": ["normal", "pneumonia"],
        "labels": {
            "normal": "Normal (No Pneumonia)",
            "pneumonia": "Pneumonia Detected",
        },
    },
}


def _build_prediction_section(S: dict, req: ReportRequest) -> List:
    pred_key = req.ai_pred_key
    risk = RISK_LEVEL.get(pred_key, "MEDIUM")
    risk_style = S.get(f"risk_{risk.lower()}", S["body"])
    risk_col = RISK_COLOR.get(risk, _AMBER)

    icd_code, icd_desc = ICD10_MAP.get(pred_key, ("N/A", ""))

    out = [*_section_heading("AI Prediction Result", S)]

    # Big result banner
    result_label = SHORT_NAMES.get(pred_key, pred_key)
    banner_data = [[Paragraph(result_label.upper(), risk_style)]]
    banner_tbl = Table(banner_data, colWidths=["100%"])
    # Extract RGB components safely — RISK_COLOR values are HexColor instances
    _rc: Any = risk_col
    _bg = colors.Color(float(_rc.red), float(_rc.green), float(_rc.blue), alpha=0.08)
    banner_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _bg),
                ("LINEABOVE", (0, 0), (-1, 0), 2.5, risk_col),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    out.append(banner_tbl)
    out.append(Spacer(1, 6))

    # Risk badge
    risk_badge_data = [
        [
            Paragraph(
                f"RISK LEVEL: {risk}",
                ParagraphStyle(
                    "risk_badge",
                    fontSize=9,
                    textColor=_WHITE,
                    fontName="Helvetica-Bold",
                    alignment=TA_CENTER,
                ),
            )
        ]
    ]
    risk_badge = Table(risk_badge_data, colWidths=["100%"])
    risk_badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), risk_col),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROUNDEDCORNERS", [4]),
            ]
        )
    )
    out.append(risk_badge)
    out.append(Spacer(1, 8))

    # Key metrics table
    rows = [
        ("AI Predicted Class", SHORT_NAMES.get(pred_key, pred_key)),
        ("Confidence", f"{req.ai_confidence * 100:.2f}%"),
        ("ICD-10 Code", icd_code),
        ("ICD-10 Description", icd_desc),
        ("Scan Type", req.scan_type),
        ("Image File", req.filename),
    ]
    out.append(_kv_table(rows, S))

    # Uncertainty block
    unc = req.uncertainty
    if unc.mc_samples > 0:
        out.append(Spacer(1, 8))
        out.extend(_section_heading("Predictive Uncertainty (MC-Dropout)", S))
        unc_rows = [
            ("MC Samples Used", str(unc.mc_samples)),
            ("Mean Entropy", f"{unc.mean_entropy:.4f}"),
            ("Std. Confidence", f"{unc.std_confidence:.4f}"),
            ("Uncertainty Level", unc.uncertainty_label),
        ]
        out.append(_kv_table(unc_rows, S))

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Probability chart section
# ─────────────────────────────────────────────────────────────────────────────


def _build_prob_section(S: dict, req: ReportRequest) -> List:
    out = [*_section_heading("Class Probabilities", S)]

    mode_cfg = SCAN_MODE_CFG.get(req.scan_type, SCAN_MODE_CFG["Brain MRI"])
    chart = _prob_bar_image(
        req.probabilities,
        mode_cfg["keys"],
        mode_cfg["labels"],
        req.ai_pred_key,
    )
    if chart:
        tbl = Table([[chart]], colWidths=["100%"], hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("GRID", (0, 0), (-1, -1), 0.3, _BRAND_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        out.append(tbl)
    else:
        # Fallback: plain text table
        table_rows = []
        for k in mode_cfg["keys"]:
            prob = req.probabilities.get(k, 0.0) * 100
            lbl = mode_cfg["labels"].get(k, k)
            bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
            table_rows.append((lbl, f"{prob:.2f}%", bar))
        data = [["Class", "Probability", "Bar"]] + [
            [Paragraph(r, S["body"]) for r in row] for row in table_rows
        ]
        t = Table(data, colWidths=[5 * cm, 3 * cm, 8 * cm], hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _BRAND_DARK),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_BRAND_LIGHT, _WHITE]),
                    ("GRID", (0, 0), (-1, -1), 0.3, _BRAND_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        out.append(t)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Clinician review section
# ─────────────────────────────────────────────────────────────────────────────


def _build_clinician_section(S: dict, req: ReportRequest) -> List:
    doc_key = req.doctor_choice_key
    ai_key = req.ai_pred_key
    overridden = doc_key != ai_key

    icd_code, icd_desc = ICD10_MAP.get(doc_key, ("N/A", ""))

    out = [*_section_heading("Clinician Review & Final Diagnosis", S)]

    outcome_text = "OVERRIDDEN BY CLINICIAN" if overridden else "CONFIRMED BY CLINICIAN"
    outcome_col = _AMBER if overridden else _GREEN

    outcome_data = [
        [
            Paragraph(
                outcome_text,
                ParagraphStyle(
                    "out",
                    fontSize=10,
                    textColor=_WHITE,
                    fontName="Helvetica-Bold",
                    alignment=TA_CENTER,
                ),
            )
        ]
    ]
    outcome_tbl = Table(outcome_data, colWidths=["100%"])
    outcome_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), outcome_col),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    out.append(outcome_tbl)
    out.append(Spacer(1, 6))

    rows = [
        ("AI Prediction", SHORT_NAMES.get(ai_key, ai_key)),
        ("Clinician Diagnosis", SHORT_NAMES.get(doc_key, doc_key)),
        ("ICD-10 Code", icd_code),
        ("ICD-10 Description", icd_desc),
        (
            "Overridden",
            "Yes — Clinician disagreed with AI"
            if overridden
            else "No — Clinician agreed with AI",
        ),
        ("Clinician Name", req.clinician_name or "—"),
        ("Clinician ID", req.clinician_id or "—"),
    ]
    out.append(_kv_table(rows, S))

    # Signature block
    out.append(Spacer(1, 14))
    sig_data = [
        [
            Paragraph("Clinician Signature:", S["body_bold"]),
            Paragraph("_" * 38, S["body"]),
            Paragraph("Date:", S["body_bold"]),
            Paragraph("_" * 20, S["body"]),
        ]
    ]
    sig_tbl = Table(sig_data, colWidths=[3.5 * cm, 7.5 * cm, 1.8 * cm, 4 * cm])
    sig_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    out.append(sig_tbl)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  OCR section
# ─────────────────────────────────────────────────────────────────────────────


def _build_ocr_section(S: dict, req: ReportRequest) -> List:
    if not req.ocr_text or not req.ocr_text.strip():
        return []

    out = [*_section_heading("Extracted Text (OCR)", S)]
    out.append(
        Paragraph(
            "The following text was automatically extracted from the scan image:",
            S["muted"],
        )
    )
    out.append(Spacer(1, 4))

    lines = req.ocr_text.strip().split("\n")
    for line in lines:
        if line.strip():
            out.append(Paragraph(line.strip(), S["ocr_line"]))

    if req.ocr_lines:
        out.append(Spacer(1, 6))
        ocr_data = [["Text", "Confidence"]]
        for ln in req.ocr_lines[:30]:  # cap at 30 rows
            txt = str(ln.get("text", ""))[:80]
            conf = f"{float(ln.get('confidence', 0)) * 100:.1f}%"
            ocr_data.append([Paragraph(txt, S["body"]), Paragraph(conf, S["body"])])  # type: ignore[arg-type]
        t = Table(ocr_data, colWidths=[13 * cm, 3 * cm], hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _BRAND_MID),
                    ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_BRAND_LIGHT, _WHITE]),
                    ("GRID", (0, 0), (-1, -1), 0.3, _BRAND_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        out.append(t)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Audit trail section
# ─────────────────────────────────────────────────────────────────────────────


def _build_audit_section(S: dict, req: ReportRequest) -> List:
    ts = req.server_timestamp or datetime.datetime.now().isoformat()

    rows = [
        ("Session ID", req.session_id),
        ("Image File", req.filename),
        ("Report Time", ts),
        ("FL Round", str(req.fl_round) if req.fl_round else "N/A"),
        ("Model Version", req.model_version),
        ("Scan Type", req.scan_type),
        ("Privacy Status", "EXIF / metadata permanently stripped before storage"),
    ]

    out = [*_section_heading("Audit Trail & System Information", S)]
    out.append(_kv_table(rows, S))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Disclaimer / footer section
# ─────────────────────────────────────────────────────────────────────────────


def _build_disclaimer_section(S: dict) -> List:
    out = [Spacer(1, 10), _hr()]

    notices = [
        (
            "MEDICAL DISCLAIMER",
            "This report is generated by an experimental AI assistant and is intended "
            "solely as a decision-support tool for qualified medical professionals. "
            "It MUST NOT be used as the sole basis for clinical diagnosis, treatment, "
            "or any other medical decision. All findings must be verified by a licensed "
            "clinician before any action is taken.",
        ),
        (
            "PRIVACY NOTICE",
            "All EXIF metadata, device identifiers, GPS tags, and hidden patient "
            "identifiers have been permanently removed from the scan image before "
            "any storage or transmission. No patient-identifiable information is "
            "retained in this system beyond what the clinician explicitly entered.",
        ),
        (
            "FEDERATED LEARNING NOTICE",
            "This model was trained using privacy-preserving federated learning. "
            "No raw patient data was shared during the training process. Only model "
            "parameter updates (gradients) were exchanged between participating nodes, "
            "in accordance with data minimisation principles.",
        ),
        (
            "REPORT CONFIDENTIALITY",
            "This document contains confidential clinical information. It is intended "
            "only for the named clinician and institution. Unauthorised disclosure, "
            "copying or distribution is strictly prohibited.",
        ),
    ]

    for title, text in notices:
        out.append(Paragraph(f"<b>{title}</b>", S["disclaimer"]))
        out.append(Paragraph(text, S["disclaimer"]))
        out.append(Spacer(1, 5))

    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════════════


def build_pdf_report(req: ReportRequest) -> bytes:
    """
    Build a complete clinical PDF report and return it as raw bytes.

    Parameters
    ----------
    req : ReportRequest
        All data required for the report.  Optional fields default gracefully.

    Returns
    -------
    bytes
        The fully rendered PDF as a byte-string ready to send to the browser.
    """
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title="Tecnomate Clinical AI — Diagnostic Report",
        author="Tecnomate Health Network",
        subject=f"Diagnostic Report — {req.scan_type}",
        creator="Tecnomate Clinical AI v1.0",
    )

    S = _styles()

    story: List = []

    # ── cover band ────────────────────────────────────────────────────────────
    story.extend(_build_cover_band(S, req))
    story.append(Spacer(1, 10))

    # ── patient info ──────────────────────────────────────────────────────────
    story.extend(_build_patient_section(S, req))
    story.append(Spacer(1, 10))

    # ── scan images + Grad-CAM (try to keep on same page) ────────────────────
    story.append(KeepTogether(_build_images_section(S, req)))
    story.append(Spacer(1, 10))

    # ── AI prediction result ──────────────────────────────────────────────────
    story.append(KeepTogether(_build_prediction_section(S, req)))
    story.append(Spacer(1, 10))

    # ── class probabilities chart ─────────────────────────────────────────────
    story.extend(_build_prob_section(S, req))
    story.append(Spacer(1, 10))

    # ── clinician review ──────────────────────────────────────────────────────
    story.append(KeepTogether(_build_clinician_section(S, req)))
    story.append(Spacer(1, 10))

    # ── OCR text (only if present) ────────────────────────────────────────────
    ocr_items = _build_ocr_section(S, req)
    if ocr_items:
        story.extend(ocr_items)
        story.append(Spacer(1, 10))

    # ── audit trail ───────────────────────────────────────────────────────────
    story.extend(_build_audit_section(S, req))

    # ── disclaimer + footer ───────────────────────────────────────────────────
    story.extend(_build_disclaimer_section(S))

    # ── build ─────────────────────────────────────────────────────────────────
    doc.build(story, canvasmaker=_NumberedCanvas)

    return buf.getvalue()
