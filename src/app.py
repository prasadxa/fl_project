"""
Tecnomate - Clinical Edge Interface (app.py)
=============================================
Streamlit web app for privacy-preserving federated medical image inference
with doctor-in-the-loop feedback and continuous data collection.

Dual-mode: select "Chest X-Ray" or "Brain MRI" in the sidebar to filter
predictions and override options to only the relevant disease classes.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
import uuid
import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# â”€â”€ project imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASS_NAMES, NUM_CLASSES
from model import MedicalCNN
from anonymizer import strip_metadata_and_save
from ocr_reader import extract_text, filter_medical_text, format_for_report, is_ocr_available, ocr_unavailable_reason

# â”€â”€ paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJ_ROOT   = Path(__file__).parent.parent
MODEL_PATH  = PROJ_ROOT / "models" / "global_model.pth"
COLLECT_DIR = PROJ_ROOT / "data" / "new_collected_data"

# â”€â”€ scan-type configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Maps each mode to the model output indices that are valid for that scan type.
SCAN_MODES = {
    "Brain MRI": {
        "indices": [0, 1, 2, 3],
        "labels": {
            0: "Glioma",
            1: "Meningioma",
            2: "No Tumor",
            3: "Pituitary Tumor",
        },
        "class_keys": ["glioma", "meningioma", "notumor", "pituitary"],
        "icon": "ðŸ§ ",
    },
    "Chest X-Ray": {
        "indices": [4, 5],
        "labels": {
            4: "No Pneumonia",
            5: "Pneumonia Detected",
        },
        "class_keys": ["normal", "pneumonia"],
        "icon": "ðŸ«",
    },
}

# Full display names and risk colours for all 6 classes
SHORT_NAMES = {
    "glioma":      "Glioma (Brain Tumor)",
    "meningioma":  "Meningioma (Brain Tumor)",
    "notumor":     "No Tumor Detected",
    "pituitary":   "Pituitary Tumor",
    "normal":      "Normal / Healthy (CXR)",
    "pneumonia":   "Pneumonia Detected (CXR)",
}

RISK_COLOURS = {
    "glioma":      "#e74c3c",
    "meningioma":  "#e67e22",
    "notumor":     "#27ae60",
    "pituitary":   "#e67e22",
    "normal":      "#27ae60",
    "pneumonia":   "#e74c3c",
}

# â”€â”€ page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Tecnomate | Clinical AI",
    page_icon="ðŸ¥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    .tecnomate-header { font-size:2.6rem; font-weight:800; color:#1a73e8; }
    .sub-header       { font-size:1.05rem; color:#555; margin-top:-12px; }
    .result-box       { padding:20px; border-radius:12px; border:2px solid; margin:10px 0; }
    .disclaimer       { background:#fff3cd; border-left:4px solid #ffc107;
                        padding:12px 16px; border-radius:6px; font-size:0.85rem; }
    .privacy-badge    { background:#d4edda; color:#155724; padding:4px 10px;
                        border-radius:20px; font-size:0.8rem; font-weight:600; }
    .mode-badge       { background:#cce5ff; color:#004085; padding:4px 10px;
                        border-radius:20px; font-size:0.85rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# â”€â”€ model loader (cached â€” loads once into 16 GB RAM, never reloaded) â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_resource
def load_model() -> MedicalCNN | None:
    if not MODEL_PATH.exists():
        return None
    mdl = MedicalCNN(num_classes=NUM_CLASSES)
    mdl.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    mdl.eval()
    return mdl


# â”€â”€ image preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_INFER_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def prepare_any_image_for_model(pil_img: Image.Image) -> torch.Tensor:
    """
    Converts any uploaded image to the exact (1,1,128,128) tensor the model
    expects, mirroring the cv2 INTER_AREA pipeline used during training.
    """
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    pil_gray = Image.fromarray(resized, mode="L")
    return _INFER_TRANSFORM(pil_gray).unsqueeze(0)   # (1,1,128,128)


@torch.no_grad()
def run_inference(model: MedicalCNN, tensor: torch.Tensor):
    probs = F.softmax(model(tensor), dim=1).squeeze().numpy()
    pred  = int(np.argmax(probs))
    return pred, probs


def build_report(
    filename: str,
    pred_label: str,
    confirmed_label: str,
    ocr_text: str = "",
) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ocr_block = (
        f"\n--------------------------------------------------------\n"
        f"  EXTRACTED TEXT (OCR)\n"
        f"{ocr_text}\n"
        if ocr_text.strip()
        else ""
    )
    return f"""========================================================
  TECNOMATE CLINICAL AI — DIAGNOSTIC REPORT
========================================================
  Date/Time         : {ts}
  Image File        : {filename}
  AI Prediction     : {pred_label}
  Doctor Confirmed  : {confirmed_label}
  Outcome           : {"CONFIRMED" if pred_label == confirmed_label else "OVERRIDDEN BY CLINICIAN"}
{ocr_block}
--------------------------------------------------------
  PRIVACY NOTICE
  Patient Data Anonymized and Secured.
  All EXIF metadata, device identifiers, and hidden
  tags have been permanently stripped from this image
  before storage.  No patient-identifiable information
  is retained anywhere in this system.
--------------------------------------------------------
  MEDICAL DISCLAIMER
  This report is generated by an AI assistant only.
  It must not be used as the sole basis for clinical
  decisions.  Always rely on qualified medical
  professionals for diagnosis and treatment.
========================================================
"""


# â”€â”€ sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## ðŸ¥ Tecnomate")
    st.markdown("**Federated Clinical AI Platform**")
    st.markdown("---")

    # Scan Type selector â€” this drives all downstream logic
    scan_type = st.radio(
        "**Scan Type**",
        options=list(SCAN_MODES.keys()),
        index=0,
        help="Select the type of scan you are uploading. "
             "The AI will only show predictions relevant to that scan type.",
    )
    mode_cfg = SCAN_MODES[scan_type]
    st.markdown(
        f'<span class="mode-badge">{mode_cfg["icon"]} {scan_type} mode</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Detectable Conditions**")
    for cls_key in mode_cfg["class_keys"]:
        st.markdown(f"- {SHORT_NAMES[cls_key]}")
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown(f"- Architecture: `MedicalCNN`")
    st.markdown(f"- Classes: `{NUM_CLASSES}` (all modalities)")
    st.markdown(f"- Training: `FedAvg FL`")
    st.markdown(f"- Images used: `13,056`")
    st.markdown("---")
    st.markdown('<span class="privacy-badge">HIPAA-Style Privacy Mode ON</span>',
                unsafe_allow_html=True)


# â”€â”€ main header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown('<p class="tecnomate-header">ðŸ¥ Tecnomate</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Privacy-Preserving Federated Medical Image Diagnosis '
    f'â€” {mode_cfg["icon"]} <strong>{scan_type}</strong> mode</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# â”€â”€ load model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model = load_model()
if model is None:
    st.error("Global model not found at `models/global_model.pth`.  "
             "Please complete federated training first.")
    st.stop()

# â”€â”€ image upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader(f"{mode_cfg['icon']} Upload {scan_type}")
    uploaded = st.file_uploader(
        f"Select a {scan_type} image",
        type=["jpg", "jpeg", "png"],
        key=scan_type,   # resets uploader when user switches scan type
    )
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)

# â”€â”€ inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if uploaded:
    tensor         = prepare_any_image_for_model(img_pil)
    raw_pred, probs = run_inference(model, tensor)

    # --- OCR: run on the original full-resolution image, not the 128x128 tensor ---
    _ocr_result = extract_text(img_pil) if is_ocr_available() else None
    _ocr_text   = format_for_report(_ocr_result) if _ocr_result is not None else ""

    # --- restrict prediction to valid indices for this scan type ---
    valid_indices  = mode_cfg["indices"]
    valid_probs    = {i: float(probs[i]) for i in valid_indices}
    pred_idx       = max(valid_probs, key=valid_probs.get)
    pred_conf      = valid_probs[pred_idx] * 100
    pred_class_key = CLASS_NAMES[pred_idx]
    pred_label     = mode_cfg["labels"][pred_idx]

    with col_result:
        st.subheader("AI Diagnostic Result")
        colour   = RISK_COLOURS[pred_class_key]
        top_vals = sorted(valid_probs.values(), reverse=True)
        margin   = (top_vals[0] - top_vals[1]) * 100 if len(top_vals) > 1 else 100.0
        low_conf = pred_conf < 60.0

        st.markdown(
            f'<div class="result-box" style="border-color:{colour};">'
            f'<h3 style="color:{colour}">{pred_label}</h3>'
            f'<p style="font-size:1.1rem">Confidence: <strong>{pred_conf:.1f}%</strong></p>'
            f'<p style="font-size:0.9rem; color:#888">Margin over 2nd class: {margin:.1f}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if low_conf:
            st.warning(
                f"âš ï¸ Low confidence ({pred_conf:.1f}%). "
                "Please ensure the image is the correct scan type and verify manually."
            )

        # probability bar chart â€” only show relevant classes for this scan type
        st.markdown("**Class Probability Distribution**")
        prob_dict = {mode_cfg["labels"][i]: valid_probs[i] for i in valid_indices}
        st.bar_chart(prob_dict)

    # â”€â”€ doctor verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown("---")
    st.subheader("Doctor Verification")
    st.markdown(
        "Review the AI prediction and confirm or override. "
        "Your verified label improves the federated model in the next training round."
    )

    col_confirm, col_override = st.columns(2)

    with col_confirm:
        confirm_btn = st.button("âœ… Confirm Diagnosis", type="primary", use_container_width=True)

    with col_override:
        # Override dropdown shows ONLY the classes valid for the selected scan type
        override_options = mode_cfg["class_keys"]
        override_labels  = [SHORT_NAMES[k] for k in override_options]
        default_override = override_options.index(pred_class_key) if pred_class_key in override_options else 0
        selected_override_label = st.selectbox(
            "Select correct class (if overriding)",
            options=override_labels,
            index=default_override,
        )
        correct_class_key = override_options[override_labels.index(selected_override_label)]
        override_btn = st.button("âœï¸ Override Diagnosis", use_container_width=True)

    # â”€â”€ handle confirmation / override â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    confirmed_class_key = None

    if confirm_btn:
        confirmed_class_key = pred_class_key
        st.success(f"Diagnosis confirmed: **{pred_label}**")

    elif override_btn:
        confirmed_class_key = correct_class_key
        confirmed_label     = SHORT_NAMES[correct_class_key]
        if correct_class_key == pred_class_key:
            st.info(f"Override matches prediction â€” saved as: **{confirmed_label}**")
        else:
            st.warning(
                f"AI predicted **{pred_label}** â€” "
                f"Doctor overrode to **{confirmed_label}**"
            )

    if confirmed_class_key is not None:
        # Generate a unique filename using uuid (no timestamps = no patient metadata leakage)
        unique_name = f"{confirmed_class_key}_{uuid.uuid4().hex}.jpg"
        save_path   = COLLECT_DIR / confirmed_class_key / unique_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # strip_metadata_and_save takes a PIL Image directly
        strip_metadata_and_save(img_pil, save_path)

        count = sum(1 for _ in (COLLECT_DIR / confirmed_class_key).glob("*.jpg"))
        st.markdown(
            f'<div class="disclaimer">'
            f'<strong>Patient Data Anonymized and Secured</strong><br>'
            f'Image saved as <code>{unique_name}</code><br>'
            f'directory: <code>data/new_collected_data/{confirmed_class_key}/</code><br>'
            f'Total images awaiting next FL round in this class: <strong>{count}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Downloadable diagnostic report
        report_txt = build_report(
            uploaded.name,
            pred_label,
            SHORT_NAMES.get(confirmed_class_key, confirmed_class_key),
            ocr_text=_ocr_text,
        )
        st.download_button(
            label="ðŸ“„ Download Diagnostic Report",
            data=report_txt,
            file_name=f"tecnomate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

        # Clear session state so the next upload starts fresh
        if "uploader" in st.session_state:
            del st.session_state["uploader"]

    # â”€â”€ new data queue status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown("---")
    st.subheader("New Data Collection Queue")
    import pandas as pd
    total_new = 0
    rows = []
    for cls_key in CLASS_NAMES:
        cls_dir = COLLECT_DIR / cls_key
        n = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
        total_new += n
        rows.append({"Class": SHORT_NAMES[cls_key], "Pending Images": n})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if total_new > 0:
        st.info(f"**{total_new}** new image(s) queued for the next federated training round.")
    else:
        st.success("No new images pending — model is up to date.")

    # ── OCR text extraction ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Text Extraction (OCR)")
    st.caption(
        "Reads annotations, measurements, labels, and report text "
        "embedded directly in the scan image. Powered by RapidOCR (PP-OCRv3)."
    )

    if not is_ocr_available():
        st.info(
            "**OCR is not available.**  Install the runtime to enable text extraction:\n\n"
            "```\npip install rapidocr-onnxruntime\n```\n\n"
            f"_{ocr_unavailable_reason()}_"
        )
    elif _ocr_result is None:
        st.info("Upload an image above to extract any embedded text.")

    elif _ocr_result.error:
        st.warning(f"OCR error: {_ocr_result.error}")

    elif not _ocr_result.found_text:
        st.success("No text detected in this image.")
        st.caption(f"OCR inference: {_ocr_result.elapsed_ms:.0f} ms")

    else:
        st.caption(_ocr_result.summary() + f"  |  {_ocr_result.elapsed_ms:.0f} ms")

        tab_all, tab_medical = st.tabs(["All Detected Text", "Medical Keywords Only"])

        with tab_all:
            for ln in _ocr_result.lines:
                col_txt, col_conf = st.columns([5, 1])
                with col_txt:
                    st.write(ln.text)
                with col_conf:
                    if ln.confidence >= 0.80:
                        colour = "green"
                    elif ln.confidence >= 0.60:
                        colour = "orange"
                    else:
                        colour = "red"
                    st.markdown(
                        f'<span style="color:{colour};font-size:0.85rem">'
                        f'{ln.confidence_pct}</span>',
                        unsafe_allow_html=True,
                    )

        with tab_medical:
            medical = filter_medical_text(_ocr_result)
            if medical.found_text:
                for ln in medical.lines:
                    st.markdown(f"- **{ln.text}** &nbsp; _{ln.confidence_pct}_")
            else:
                st.info(
                    "No recognised medical keywords found in the detected text.\n\n"
                    "Switch to **All Detected Text** to see everything extracted."
                )
