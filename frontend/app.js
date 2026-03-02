/* ============================================================
   Tecnomate Clinical AI — app.js  v2.1
   SPA logic: drag-drop upload, inference, chart, OCR,
   doctor override, feedback, PDF + TXT report download,
   patient info modal, uncertainty display, ICD-10 badges,
   Grad-CAM toggle, DICOM support, admin stats,
   dark mode, toast notifications, confidence ring,
   history panel, keyboard shortcuts, auto-scroll.
   ============================================================ */

"use strict";

/* ── Constants ──────────────────────────────────────────────── */
const API = {
  health: "/api/health",
  modelInfo: "/api/model-info",
  predict: "/api/predict",
  feedback: "/api/feedback",
  queue: "/api/queue",
  report: "/api/report",
  pdfReport: "/api/pdf-report",
  adminStats: "/api/admin/stats",
  adminFeedback: "/api/admin/feedback",
  exportExcel: "/api/admin/export-excel",
};

const SCAN_MODES = {
  "Brain MRI": {
    keys: ["glioma", "meningioma", "notumor", "pituitary"],
    labels: {
      glioma: "Glioma (Brain Tumor)",
      meningioma: "Meningioma (Brain Tumor)",
      notumor: "No Tumor Detected",
      pituitary: "Pituitary Tumor",
    },
    icon: "🧠",
  },
  "Chest X-Ray": {
    keys: ["normal", "pneumonia"],
    labels: {
      normal: "Normal / Healthy (CXR)",
      pneumonia: "Pneumonia Detected (CXR)",
    },
    icon: "🫁",
  },
};

const RISK_COLOURS = {
  glioma: "#e74c3c",
  meningioma: "#e67e22",
  notumor: "#27ae60",
  pituitary: "#e67e22",
  normal: "#27ae60",
  pneumonia: "#e74c3c",
};

const RISK_LEVEL = {
  glioma: "high",
  meningioma: "medium",
  notumor: "low",
  pituitary: "medium",
  normal: "low",
  pneumonia: "high",
};

const RISK_LABEL = { high: "HIGH RISK", medium: "MODERATE", low: "NORMAL" };

const ICD10 = {
  glioma: { code: "C71.9", desc: "Malignant neoplasm of brain, unspecified" },
  meningioma: {
    code: "D32.9",
    desc: "Benign neoplasm of meninges, unspecified",
  },
  notumor: {
    code: "Z03.89",
    desc: "No pathological finding — observation only",
  },
  pituitary: { code: "D35.2", desc: "Benign neoplasm of pituitary gland" },
  normal: { code: "Z03.89", desc: "CXR within normal limits" },
  pneumonia: { code: "J18.9", desc: "Pneumonia, unspecified organism" },
};

/* ── Upload constraints (mirrors backend) ───────────────────── */
const ALLOWED_MIME_PREFIXES = [
  "image/",
  "application/octet-stream",
  "application/dicom",
];
const ALLOWED_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
  ".tif",
  ".tiff",
  ".gif",
  ".avif",
  ".heic",
  ".heif",
  ".dcm",
]);
const MAX_UPLOAD_BYTES = 30 * 1024 * 1024; // 30 MB

/* ── App State ──────────────────────────────────────────────── */
const state = {
  scanType: "Brain MRI",
  file: null,
  sessionId: null,
  aiPredKey: null,
  aiPredLabel: null,
  aiConfidence: 0,
  _suggestedScanType: null,
  probabilities: null,
  ocrText: "",
  ocrLines: [],
  filename: "",
  doctorChoice: null,
  confirmed: false,
  confirmedSession: false,
  queueCount: 0,
  overrideCount: 0,
  gradcamAvailable: false,
  uncertainty: null,
};

/* ── Chart instance ─────────────────────────────────────────── */
let probChart = null;

/* ── Session history (in-memory) ───────────────────────────── */
let _history = [];

/* ── DOM helpers ────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const show = (id) => {
  const el = $(id);
  if (el) el.hidden = false;
};
const hide = (id) => {
  const el = $(id);
  if (el) el.hidden = true;
};
const setText = (id, txt) => {
  const el = $(id);
  if (el) el.textContent = txt;
};

/* ══════════════════════════════════════════════════════════════
   INITIALISATION
══════════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  // Restore saved theme
  const savedTheme = localStorage.getItem("tecnomate-theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  checkHealth();
  setInterval(checkHealth, 30_000);
  populateDoctorSelect();
  refreshQueue();
  setInterval(refreshQueue, 60_000);

  // Pre-fill today's date in patient modal visit date
  const todayIso = new Date().toISOString().split("T")[0];
  const ptVisit = $("ptVisitDate");
  if (ptVisit && !ptVisit.value) ptVisit.value = todayIso;

  // Global keyboard shortcuts
  document.addEventListener("keydown", handleGlobalKey);

  renderHistoryPanel();
});

/* ── Global keyboard handler ─────────────────────────────────── */
function handleGlobalKey(e) {
  // Ignore when typing in an input / textarea / select
  const tag = document.activeElement?.tagName?.toUpperCase();
  if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;

  // Escape — close modal or clear results
  if (e.key === "Escape") {
    const modal = $("patientModal");
    if (modal && modal.classList.contains("open")) {
      closePatientModal();
      return;
    }
    if (state.file) {
      clearAll();
    }
    return;
  }

  // Enter — run analysis if a file is loaded and no session yet, or re-run
  if (e.key === "Enter" && state.file && !$("predictBtn")?.disabled) {
    e.preventDefault();
    runPredict();
    return;
  }

  // "d" — toggle dark mode
  if (e.key === "d" && !e.ctrlKey && !e.metaKey) {
    toggleTheme();
  }
}

/* ── Health / status check ───────────────────────────────────── */
async function checkHealth() {
  const pill = $("modelStatusPill");
  const dot = $("statusDot");
  const txt = $("statusText");
  const ocrPill = $("ocrStatusPill");
  const ocrDot = $("ocrDot");
  const ocrTxt = $("ocrStatusText");

  pill.className = "status-pill loading";
  txt.textContent = "Connecting\u2026";

  try {
    const res = await fetch(API.health);
    const data = await res.json();

    if (data.model_loaded) {
      pill.className = "status-pill online";
      txt.textContent = "Model Ready";
      document.title = "Tecnomate | Clinical AI v2";
    } else {
      pill.className = "status-pill offline";
      txt.textContent = "Model Missing";
      document.title = "Tecnomate | Model Missing";
    }

    if (data.ocr_available) {
      ocrPill.className = "status-pill ocr-pill ocr-on";
      ocrTxt.textContent = "OCR Active";
    } else {
      ocrPill.className = "status-pill ocr-pill ocr-off";
      ocrTxt.textContent = "OCR Off";
      ocrPill.title = data.ocr_reason || "Install rapidocr-onnxruntime";
    }

    // Sync queue counters from DB
    if (typeof data.feedback_total !== "undefined") {
      setText(
        "queueCount",
        String(data.feedback_total - (data.feedback_overridden || 0)),
      );
      setText("overrideCount", String(data.feedback_overridden || 0));
    }
  } catch {
    pill.className = "status-pill offline";
    txt.textContent = "API Offline";
    ocrPill.className = "status-pill ocr-pill ocr-off";
    document.title = "Tecnomate | API Offline";
  }
}

/* ══════════════════════════════════════════════════════════════
   DARK MODE TOGGLE
══════════════════════════════════════════════════════════════ */
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute("data-theme") || "light";
  const next = current === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  localStorage.setItem("tecnomate-theme", next);
  showToast(
    next === "dark" ? "Dark mode enabled" : "Light mode enabled",
    "info",
    1800,
  );
}

/* ══════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
══════════════════════════════════════════════════════════════ */
let _toastId = 0;

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'info'|'success'|'error'|'warning'} type
 * @param {number} duration  ms before auto-dismiss (0 = sticky)
 */
function showToast(message, type = "info", duration = 3500) {
  const container = $("toastContainer");
  if (!container) return;

  const id = ++_toastId;
  const div = document.createElement("div");
  div.className = `toast toast-${type === "info" ? "" : type}`.trim();
  div.id = `toast-${id}`;

  const icon =
    type === "success"
      ? "✓"
      : type === "error"
        ? "✕"
        : type === "warning"
          ? "⚠"
          : "ℹ";

  div.innerHTML = `
    <span style="font-size:1rem;flex-shrink:0">${icon}</span>
    <span style="flex:1">${escHtml(message)}</span>
    <button class="toast-close" onclick="dismissToast(${id})" aria-label="Dismiss">&times;</button>`;

  container.appendChild(div);

  if (duration > 0) {
    setTimeout(() => dismissToast(id), duration);
  }
}

function dismissToast(id) {
  const el = $(`toast-${id}`);
  if (!el) return;
  el.classList.add("toast-out");
  setTimeout(() => el.remove(), 320);
}

/* ══════════════════════════════════════════════════════════════
   SCAN TYPE SELECTOR
══════════════════════════════════════════════════════════════ */
function setScanType(mode) {
  state.scanType = mode;

  document.querySelectorAll(".scan-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  const badge = $("scanTypeBadge");
  if (badge) badge.textContent = `${SCAN_MODES[mode].icon} ${mode}`;

  if (state.probabilities) {
    renderChart(state.probabilities);
    updateDoctorAIField();
  }
  populateDoctorSelect();
}

/* ══════════════════════════════════════════════════════════════
   DRAG-AND-DROP / FILE SELECTION
══════════════════════════════════════════════════════════════ */
function handleDragOver(event) {
  event.preventDefault();
  event.stopPropagation();
  $("dropZone").classList.add("drag-hover");
}

function handleDragLeave(event) {
  event.preventDefault();
  $("dropZone").classList.remove("drag-hover");
}

function handleDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  $("dropZone").classList.remove("drag-hover");
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) processFile(files[0]);
}

function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (file) processFile(file);
}

function processFile(file) {
  hideError("uploadError");

  // MIME check
  const mimeOk =
    ALLOWED_MIME_PREFIXES.some((p) => file.type.startsWith(p)) ||
    file.type === "";
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const extOk = ALLOWED_EXTENSIONS.has(ext);

  if (!mimeOk && !extOk) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      `Unsupported file type "${file.type || ext}". ` +
        "Accepted: JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, HEIC, DICOM (.dcm).",
    );
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    const mb = (file.size / 1024 / 1024).toFixed(1);
    showError(
      "uploadError",
      "uploadErrorMsg",
      `File is too large (${mb} MB). Maximum allowed size is 30 MB.`,
    );
    return;
  }

  state.file = file;

  // Animate progress bar to 100% while reading
  setProgress(true);

  // Update drop zone appearance
  const zone = $("dropZone");
  const inner = $("dropZoneInner");
  zone.classList.add("has-file");
  inner.innerHTML = `
    <div class="drop-icon" style="opacity:.7">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="18" height="18" rx="3"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
    </div>
    <p class="drop-primary" style="font-size:.95rem">${escHtml(file.name)}</p>
    <p class="drop-secondary">${formatBytes(file.size)} · Click to change</p>`;

  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = $("previewImage");
    if (img) {
      img.src = e.target.result;
      img.alt = file.name;
    }
    show("resultsArea");

    // Update meta
    setText("metaFilename", file.name);
    setText("metaSize", formatBytes(file.size));

    // Reset any previous prediction UI
    hide("predictionResult");
    hide("chartContainer");
    const uncPanel = $("uncertaintyPanel");
    if (uncPanel) uncPanel.style.display = "none";
    const featPanel0 = $("featuresPanel");
    if (featPanel0) featPanel0.style.display = "none";
    const icdBadge = $("icdBadge");
    if (icdBadge) icdBadge.style.display = "none";
    const gcHint = $("gradcamHint");
    if (gcHint) gcHint.classList.add("hidden");
    hide("ocrCard");
    hide("doctorCard");
    hide("feedbackSuccess");
    hide("feedbackError");
    hide("pdfError");

    // Reset confidence ring
    updateConfidenceRing(0, "");

    setProgress(false);
  };
  reader.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 90);
      setProgressValue(pct);
    }
  };

  // DICOM files can't be previewed as images; use a placeholder
  if (ext === ".dcm") {
    const img = $("previewImage");
    if (img) {
      img.src = "";
      img.alt = "DICOM file — no preview available";
      img.style.display = "none";
    }
    show("resultsArea");
    setText("metaFilename", file.name);
    setText("metaSize", formatBytes(file.size));
    hide("predictionResult");
    hide("chartContainer");
    hide("ocrCard");
    hide("doctorCard");
    setProgress(false);
  } else {
    const img = $("previewImage");
    if (img) img.style.display = "";
    reader.readAsDataURL(file);
  }
}

/* ── Progress bar helpers ────────────────────────────────────── */
function setProgress(indeterminate) {
  const wrap = $("uploadProgressWrap");
  const bar = $("uploadProgressBar");
  if (!wrap || !bar) return;
  if (indeterminate) {
    wrap.classList.add("visible");
    bar.style.width = "0%";
    bar.classList.add("indeterminate");
  } else {
    bar.classList.remove("indeterminate");
    bar.style.width = "100%";
    setTimeout(() => {
      wrap.classList.remove("visible");
      bar.style.width = "0%";
    }, 500);
  }
}

function setProgressValue(pct) {
  const bar = $("uploadProgressBar");
  if (bar) {
    bar.classList.remove("indeterminate");
    bar.style.width = `${pct}%`;
  }
}

/* ══════════════════════════════════════════════════════════════
   PREDICT
══════════════════════════════════════════════════════════════ */
async function runPredict() {
  if (!state.file) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      "Please upload a scan image first.",
    );
    return;
  }

  const btn = $("predictBtn");
  const spinner = $("predictionSpinner");
  btn.disabled = true;
  spinner.classList.add("spinning");
  hide("predictionResult");
  hide("chartContainer");
  hideError("uploadError");

  // Show indeterminate progress while running inference
  const wrap = $("uploadProgressWrap");
  const bar = $("uploadProgressBar");
  if (wrap && bar) {
    wrap.classList.add("visible");
    bar.classList.add("indeterminate");
  }

  // Update page title to indicate processing
  document.title = "Tecnomate | Analysing…";

  const useGradcam = $("optGradcam")?.checked || false;
  const useMcDropout = $("optMcDropout")?.checked || false;
  const mcSamples = parseInt($("optMcSamples")?.value || "20", 10);

  try {
    const form = new FormData();
    form.append("image", state.file, state.file.name);
    form.append("scan_type", state.scanType);
    form.append("gradcam", useGradcam ? "true" : "false");
    form.append("mc_dropout", useMcDropout ? "true" : "false");
    form.append("mc_samples", String(Math.max(5, Math.min(50, mcSamples))));

    const res = await fetch(API.predict, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);

    // Persist state
    state.sessionId = data.session_id;
    state.aiPredKey = data.mode_predicted_key;
    state.aiPredLabel = data.mode_predicted_class;
    state.aiConfidence = data.mode_confidence;
    state.probabilities = data.probabilities;
    state.ocrText = data.ocr_text || "";
    state.ocrLines = data.ocr_lines || [];
    state.filename = data.filename || state.file.name;
    state.gradcamAvailable = !!data.gradcam_available;
    state.uncertainty = data.uncertainty || null;
    state.selectedModeMass = data.selected_mode_mass || 0;
    state.otherModeMass = data.other_mode_mass || 0;
    state.scanTypeMismatch = !!(data.scan_type_mismatch);

    // Update preview meta
    const fmt = data.detected_format ? data.detected_format.toUpperCase() : "";
    const dims = Array.isArray(data.image_dimensions)
      ? `${data.image_dimensions[0]}×${data.image_dimensions[1]}px`
      : "";
    const extra = [fmt, dims].filter(Boolean).join("  ·  ");
    const metaEl = $("metaSize");
    if (metaEl && extra) {
      metaEl.textContent = `${formatBytes(state.file.size)}  ·  ${extra}`;
    }

    // Grad-CAM hint
    const gcHint = $("gradcamHint");
    if (gcHint) {
      if (state.gradcamAvailable) {
        gcHint.classList.remove("hidden");
      } else {
        gcHint.classList.add("hidden");
      }
    }

    // Render everything
    renderMismatchWarning(data);
    renderPredictionBanner(data);
    renderIcdBadge(data.mode_predicted_key);
    renderUncertainty(data.uncertainty);
    renderChart(data.probabilities);
    renderFeaturesPanel(data.mode_predicted_key);
    renderOCR(data);

    // Doctor panel
    populateDoctorSelect();
    updateDoctorAIField();
    show("doctorCard");

    // Add to history
    addToHistory({
      label:
        SCAN_MODES[state.scanType].labels[data.mode_predicted_key] ||
        data.mode_predicted_class,
      confidence: data.mode_confidence,
      scanType: state.scanType,
      filename: data.filename || state.file.name,
      timestamp: new Date().toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      mismatch: !!data.scan_type_mismatch,
    });

    // Success toast
    const confPct = (data.mode_confidence * 100).toFixed(1);
    const label =
      SCAN_MODES[state.scanType].labels[data.mode_predicted_key] ||
      data.mode_predicted_class;
    if (data.scan_type_mismatch) {
      showToast(
        "⚠ Scan type mismatch detected — check the warning banner",
        "warning",
      );
    } else {
      showToast(`Analysis complete: ${label} (${confPct}%)`, "success");
    }

    // Enable report download buttons right after analysis
    const dlBtnPost = $("downloadBtn");
    if (dlBtnPost) dlBtnPost.disabled = false;
    const pdfBtnPost = $("downloadPdfBtn");
    if (pdfBtnPost) pdfBtnPost.disabled = false;
    const xlBtnPost = $("downloadExcelBtn");
    if (xlBtnPost) xlBtnPost.disabled = false;

    // Update title
    document.title = `Tecnomate | ${label}`;

    // Auto-scroll to results
    scrollToResults();
  } catch (err) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      `Prediction failed: ${err.message}`,
    );
    showToast(`Analysis failed: ${err.message}`, "error");
    document.title = "Tecnomate | Clinical AI v2";
  } finally {
    btn.disabled = false;
    spinner.classList.remove("spinning");
    // Hide progress bar
    if (wrap && bar) {
      bar.classList.remove("indeterminate");
      bar.style.width = "100%";
      setTimeout(() => {
        wrap.classList.remove("visible");
        bar.style.width = "0%";
      }, 400);
    }
  }
}

/* ── Auto-scroll to results ──────────────────────────────────── */
function scrollToResults() {
  const target = $("resultsArea") || $("predictionResult");
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

/* ── Scan-type mismatch warning ──────────────────────────────── */
function renderMismatchWarning(data) {
  const banner = $("mismatchWarning");
  if (!banner) return;

  if (!data.scan_type_mismatch) {
    banner.classList.remove("visible");
    return;
  }

  // Store the suggested mode so the switch button can use it
  state._suggestedScanType = data.suggested_scan_type;

  const selectedMassPct = ((data.selected_mode_mass || 0) * 100).toFixed(1);
  const otherMassPct = ((data.other_mode_mass || 0) * 100).toFixed(1);
  const currentMode = state.scanType;
  const suggestedMode = data.suggested_scan_type;

  setText(
    "mismatchMainMsg",
    `You selected "${currentMode}" but the model thinks this looks like a ` +
      `${suggestedMode} scan. The result shown below is forced through the ` +
      `"${currentMode}" classes and may be unreliable.`,
  );

  // Suggestion box
  if (data.suggested_class && data.suggested_confidence) {
    const icon = SCAN_MODES[suggestedMode]?.icon || "";
    setText(
      "mismatchSuggestionText",
      `If treated as ${suggestedMode}: ` +
        `${icon} ${data.suggested_class} ` +
        `(${(data.suggested_confidence * 100).toFixed(1)}% confidence)`,
    );
    const sugg = $("mismatchSuggestion");
    if (sugg) sugg.style.display = "flex";
  } else {
    const sugg = $("mismatchSuggestion");
    if (sugg) sugg.style.display = "none";
  }

  // Probability mass bar
  const otherMass = data.other_mode_mass || 0;
  const fill = $("mismatchBarFill");
  if (fill) fill.style.width = `${Math.round(otherMass * 100)}%`;

  setText(
    "mismatchMassLabel",
    `${suggestedMode} mass: ${otherMassPct}%  vs  ${currentMode} mass: ${selectedMassPct}%`,
  );

  setText("mismatchSwitchLabel", suggestedMode);

  banner.classList.add("visible");
}

function switchToSuggestedMode() {
  const suggested = state._suggestedScanType;
  if (!suggested) return;

  setScanType(suggested);

  const banner = $("mismatchWarning");
  if (banner) banner.classList.remove("visible");

  if (state.file) {
    runPredict();
  }
}

/* ── Prediction banner ───────────────────────────────────────── */
function renderPredictionBanner(data) {
  const key = data.mode_predicted_key;
  const label = data.mode_predicted_class;
  const conf = data.mode_confidence;
  const level = RISK_LEVEL[key] || "medium";

  const resultLabel = $("resultLabel");
  const riskBadge = $("resultRiskBadge");

  setText("resultEmoji", SCAN_MODES[state.scanType].icon);
  setText("resultClass", label);
  setText(
    "resultConf",
    `Confidence: ${(conf * 100).toFixed(1)}%  ·  ${state.scanType}`,
  );

  resultLabel.className = `result-label risk-${level}`;
  riskBadge.className = `result-risk-badge risk-${level}`;
  setText("resultRiskBadge", RISK_LABEL[level]);

  // Animate confidence ring
  const color = RISK_COLOURS[key] || "#2563eb";
  updateConfidenceRing(conf, color);

  show("predictionResult");
  show("chartContainer");
}

/* ── Confidence Ring ─────────────────────────────────────────── */
/**
 * Animate the circular confidence gauge.
 * circumference = 2π×40 ≈ 251.2 (radius 40, as set in SVG)
 */
function updateConfidenceRing(confidence, color) {
  const fill = $("confRingFill");
  const text = $("confRingText");
  if (!fill || !text) return;

  const circumference = 251.2;
  const pct = Math.max(0, Math.min(1, confidence));
  const offset = circumference - pct * circumference;

  fill.style.strokeDashoffset = String(offset);
  fill.style.stroke = color || "#2563eb";
  text.textContent = `${(pct * 100).toFixed(1)}%`;
  text.style.fill = color || "#2563eb";
}

/* ── ICD-10 badge ─────────────────────────────────────────────── */
function renderIcdBadge(predKey) {
  const badge = $("icdBadge");
  const codeEl = $("icdCode");
  if (!badge || !codeEl) return;

  const icd = ICD10[predKey];
  if (icd) {
    codeEl.textContent = `${icd.code} — ${icd.desc}`;
    badge.style.display = "inline-flex";
    badge.title = icd.desc;
  } else {
    badge.style.display = "none";
  }
}

/* ── Copy ICD-10 code to clipboard ──────────────────────────── */
function copyIcdCode() {
  const codeEl = $("icdCode");
  if (!codeEl) return;
  const text = codeEl.textContent || "";
  if (!text || text === "—") return;

  navigator.clipboard
    .writeText(text)
    .then(() => {
      showToast("ICD-10 code copied to clipboard", "success", 2000);
      // Brief visual feedback on button
      const btn = $("icdCopyBtn");
      if (btn) {
        const orig = btn.innerHTML;
        btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
        setTimeout(() => {
          btn.innerHTML = orig;
        }, 1200);
      }
    })
    .catch(() => {
      showToast(
        "Could not copy — try selecting and copying manually",
        "warning",
      );
    });
}

/* ── Uncertainty panel ────────────────────────────────────────── */
function renderUncertainty(unc) {
  const panel = $("uncertaintyPanel");
  if (!panel) return;

  if (!unc || !unc.mc_samples) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "block";
  setText("uncSamples", String(unc.mc_samples));
  setText("uncEntropy", unc.mean_entropy.toFixed(4));
  setText("uncStdConf", unc.std_confidence.toFixed(4));

  const uncLabelEl = $("uncLabel");
  if (uncLabelEl) {
    uncLabelEl.textContent = unc.uncertainty_label || "—";
    const lbl = (unc.uncertainty_label || "").toLowerCase();
    uncLabelEl.className = lbl.includes("low")
      ? "unc-label-low"
      : lbl.includes("moderate")
        ? "unc-label-mod"
        : lbl.includes("high")
          ? "unc-label-high"
          : "";
  }
}

/* ══════════════════════════════════════════════════════════════
   CHART.JS PROBABILITY BAR CHART
══════════════════════════════════════════════════════════════ */
function renderChart(probabilities) {
  const mode = SCAN_MODES[state.scanType];
  const keys = mode.keys;
  const labels = keys.map((k) => mode.labels[k] || k);
  const values = keys.map((k) => probabilities[k] ?? 0);
  const colors = keys.map((k) => RISK_COLOURS[k] || "#94a3b8");
  const alphas = colors.map((c) => c + "cc");

  const canvas = $("probChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (probChart) {
    probChart.destroy();
    probChart = null;
  }
  canvas.parentElement.style.height = `${keys.length * 54 + 20}px`;

  // Detect current theme for grid colours
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const gridColor = isDark ? "#30363d" : "#e2e8f0";
  const tickColor = isDark ? "#6e7681" : "#94a3b8";
  const labelColor = isDark ? "#8b949e" : "#475569";

  probChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: alphas,
          borderColor: colors,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: "easeOutQuart" },
      layout: { padding: { right: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: (ctx) => `  ${(ctx.raw * 100).toFixed(2)}%` },
          backgroundColor: isDark ? "#161b22" : "#0f172a",
          titleColor: isDark ? "#c9d1d9" : "#e2e8f0",
          bodyColor: isDark ? "#6e7681" : "#94a3b8",
          borderColor: isDark ? "#30363d" : "#334155",
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
        },
      },
      scales: {
        x: {
          min: 0,
          max: 1,
          grid: { color: gridColor, lineWidth: 1 },
          border: { display: false },
          ticks: {
            callback: (v) => `${(v * 100).toFixed(0)}%`,
            color: tickColor,
            font: { size: 11 },
            maxTicksLimit: 6,
          },
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: { color: labelColor, font: { size: 12, weight: "600" } },
        },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════════
   OCR PANEL
══════════════════════════════════════════════════════════════ */
function renderOCR(data) {
  const ocrBody = $("ocrBody");
  if (!ocrBody) return;

  if (!data.ocr_lines && !data.ocr_text) {
    ocrBody.innerHTML = `
      <div class="ocr-unavailable">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        OCR not installed. Run:
        <span class="ocr-install-hint">pip install rapidocr-onnxruntime</span>
      </div>`;
    show("ocrCard");
    return;
  }

  const lines = data.ocr_lines || [];
  if (lines.length === 0) {
    ocrBody.innerHTML = `<p class="ocr-empty">No text detected in this image.</p>`;
    show("ocrCard");
    setText("ocrElapsed", "");
    return;
  }

  const lineHtml = lines
    .map((ln, i) => {
      const conf = ln.confidence ?? 0;
      const confPct = (conf * 100).toFixed(1);
      const confClass =
        conf >= 0.8
          ? "ocr-conf-high"
          : conf >= 0.5
            ? "ocr-conf-medium"
            : "ocr-conf-low";
      return `
      <div class="ocr-line" style="animation-delay:${i * 40}ms">
        <span class="ocr-line-text">${escHtml(ln.text)}</span>
        <span class="ocr-line-conf ${confClass}">${confPct}%</span>
      </div>`;
    })
    .join("");

  const highConf = lines.filter((l) => (l.confidence ?? 0) >= 0.8).length;
  ocrBody.innerHTML = `
    <div class="ocr-lines">${lineHtml}</div>
    <p class="ocr-summary">
      ${lines.length} text region(s) detected &mdash; ${highConf} high-confidence
    </p>`;
  show("ocrCard");
}

/* ══════════════════════════════════════════════════════════════
   DOCTOR DECISION PANEL
══════════════════════════════════════════════════════════════ */
function populateDoctorSelect() {
  const sel = $("doctorSelect");
  if (!sel) return;
  const mode = SCAN_MODES[state.scanType];
  sel.innerHTML = "";

  mode.keys.forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = mode.labels[key] || key;
    sel.appendChild(opt);
  });

  if (state.aiPredKey && mode.keys.includes(state.aiPredKey)) {
    sel.value = state.aiPredKey;
  }
  state.doctorChoice = sel.value;
  updateOverrideIndicator();
}

function updateDoctorAIField() {
  if (!state.aiPredKey) return;
  const mode = SCAN_MODES[state.scanType];
  const label =
    mode.labels[state.aiPredKey] || state.aiPredLabel || state.aiPredKey;
  setText("doctorAIPred", label);
}

function handleDoctorSelectChange() {
  const sel = $("doctorSelect");
  state.doctorChoice = sel.value;
  state.confirmed = false;
  hide("feedbackSuccess");
  hide("feedbackError");
  const dlBtn = $("downloadBtn");
  if (dlBtn) dlBtn.disabled = true;
  const pdfBtn = $("downloadPdfBtn");
  if (pdfBtn) pdfBtn.disabled = true;
  updateOverrideIndicator();
}

function updateOverrideIndicator() {
  const sel = $("doctorSelect");
  const isOver =
    state.doctorChoice &&
    state.aiPredKey &&
    state.doctorChoice !== state.aiPredKey;

  if (isOver) {
    sel?.classList.add("override-active");
    show("overrideBadge");
  } else {
    sel?.classList.remove("override-active");
    hide("overrideBadge");
  }
}

function hideOverrideBadge() {
  hide("overrideBadge");
  $("doctorSelect")?.classList.remove("override-active");
}

/* ══════════════════════════════════════════════════════════════
   FEEDBACK SUBMISSION
══════════════════════════════════════════════════════════════ */
async function submitFeedback() {
  if (!state.sessionId) {
    showError(
      "feedbackError",
      "feedbackErrorMsg",
      "No prediction session found. Please run AI Analysis first.",
    );
    return;
  }

  const sel = $("doctorSelect");
  const chosenKey = sel.value;
  const confirmBtn = $("confirmBtn");

  hide("feedbackSuccess");
  hide("feedbackError");
  confirmBtn.disabled = true;
  confirmBtn.textContent = "Saving\u2026";

  try {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    form.append("chosen_key", chosenKey);
    form.append("scan_type", state.scanType);
    form.append("ai_predicted_key", state.aiPredKey || "");
    form.append("clinician_name", $("clinicianName")?.value || "");
    form.append("clinician_id", $("clinicianId")?.value || "");
    form.append("notes", $("clinicianNotes")?.value || "");

    const res = await fetch(API.feedback, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);

    state.confirmed = true;
    state.confirmedSession = true;
    state.doctorChoice = chosenKey;

    const overridden = data.overridden;
    const modeLabels = SCAN_MODES[state.scanType].labels;
    const chosenLabel = modeLabels[chosenKey] || chosenKey;

    setText(
      "feedbackSuccessMsg",
      overridden
        ? `Override recorded. Saved as "${chosenLabel}" for next FL training round.`
        : `Confirmed "${chosenLabel}" — image queued for next training round.`,
    );
    show("feedbackSuccess");

    // Enable report downloads
    const dlBtn = $("downloadBtn");
    if (dlBtn) dlBtn.disabled = false;
    const pdfBtn = $("downloadPdfBtn");
    if (pdfBtn) pdfBtn.disabled = false;
    const xlBtn = $("downloadExcelBtn");
    if (xlBtn) xlBtn.disabled = false;

    if (overridden) {
      state.overrideCount += 1;
      showToast(`Override saved: "${chosenLabel}"`, "warning");
    } else {
      state.queueCount += 1;
      showToast(`Diagnosis confirmed: "${chosenLabel}"`, "success");
    }
    updateQueueDisplay();

    confirmBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      Saved`;
    confirmBtn.classList.remove("btn-success");
    confirmBtn.classList.add("btn-outline");

    // Pulse the queue count briefly
    const queueNum = $("queueCount");
    if (queueNum) {
      queueNum.classList.remove("ping-once");
      void queueNum.offsetWidth; // reflow trick to restart animation
      queueNum.classList.add("ping-once");
    }
  } catch (err) {
    showError(
      "feedbackError",
      "feedbackErrorMsg",
      `Could not save: ${err.message}`,
    );
    showToast(`Save failed: ${err.message}`, "error");
    confirmBtn.disabled = false;
    confirmBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      Confirm &amp; Save`;
    confirmBtn.classList.add("btn-success");
    confirmBtn.classList.remove("btn-outline");
  }
}

/* ══════════════════════════════════════════════════════════════
   PDF REPORT DOWNLOAD
══════════════════════════════════════════════════════════════ */
async function downloadPdfReport() {
  if (!state.sessionId || !state.aiPredKey) {
    showError(
      "pdfError",
      "pdfErrorMsg",
      "Please run AI Analysis and confirm diagnosis before downloading the PDF report.",
    );
    show("pdfError");
    return;
  }

  const overlay = $("pdfOverlay");
  if (overlay) overlay.classList.add("open");
  hide("pdfError");

  try {
    const form = new FormData();

    // Core prediction data
    form.append("session_id", state.sessionId);
    form.append("scan_type", state.scanType);
    form.append("ai_pred_key", state.aiPredKey);
    form.append("ai_confidence", String(state.aiConfidence || 0));
    form.append("doctor_choice_key", state.doctorChoice || state.aiPredKey);
    form.append(
      "probabilities_json",
      JSON.stringify(state.probabilities || {}),
    );
    form.append("ocr_text", state.ocrText || "");
    form.append("ocr_lines_json", JSON.stringify(state.ocrLines || []));

    // Patient info from modal
    form.append(
      "patient_name",
      $("ptName")?.value || "Anonymous / De-identified",
    );
    form.append("patient_id", $("ptId")?.value || "N/A");
    form.append("date_of_birth", $("ptDob")?.value || "N/A");
    form.append("gender", $("ptGender")?.value || "N/A");
    form.append("referring_doctor", $("ptRefDoc")?.value || "N/A");
    form.append(
      "institution",
      $("ptInstitution")?.value || "Tecnomate Health Network",
    );
    form.append("visit_date", $("ptVisitDate")?.value || "");
    form.append("clinical_notes", $("ptNotes")?.value || "");

    // Clinician identity
    form.append("clinician_name", $("clinicianName")?.value || "");
    form.append("clinician_id", $("clinicianId")?.value || "");

    // Uncertainty (if available)
    if (state.uncertainty) {
      form.append("mc_entropy", String(state.uncertainty.mean_entropy || 0));
      form.append("mc_std_conf", String(state.uncertainty.std_confidence || 0));
      form.append("mc_samples", String(state.uncertainty.mc_samples || 0));
      form.append("mc_label", state.uncertainty.uncertainty_label || "");
    }

    // Model info
    form.append("fl_round", "0");
    form.append("model_version", "global_model.pth");

    // AI mathematical parameters
    form.append("selected_mode_mass", String(state.selectedModeMass || 0));
    form.append("other_mode_mass", String(state.otherModeMass || 0));
    form.append("scan_type_mismatch", state.scanTypeMismatch ? "true" : "false");

    const res = await fetch(API.pdfReport, { method: "POST", body: form });

    if (!res.ok) {
      let detail = `Server error ${res.status}`;
      try {
        const errData = await res.json();
        detail = errData.detail || detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }

    // Trigger browser download
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const cd = res.headers.get("Content-Disposition") || "";
    const match = cd.match(/filename="?([^"]+)"?/);
    a.download = match ? match[1] : `tecnomate_report_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast("PDF report downloaded successfully", "success");
  } catch (err) {
    showError(
      "pdfError",
      "pdfErrorMsg",
      `PDF generation failed: ${err.message}`,
    );
    show("pdfError");
    showToast(`PDF failed: ${err.message}`, "error");
  } finally {
    if (overlay) overlay.classList.remove("open");
  }
}

/* ══════════════════════════════════════════════════════════════
   PLAIN-TEXT REPORT DOWNLOAD (legacy)
══════════════════════════════════════════════════════════════ */
function downloadReport() {
  const modeLabels = SCAN_MODES[state.scanType].labels;
  const aiLabel = modeLabels[state.aiPredKey] || state.aiPredLabel || "—";
  const docLabel = modeLabels[state.doctorChoice] || state.doctorChoice || "—";
  const outcome =
    state.aiPredKey === state.doctorChoice
      ? "CONFIRMED"
      : "OVERRIDDEN BY CLINICIAN";

  const ts = new Date().toLocaleString("en-GB", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const icd = ICD10[state.doctorChoice] || ICD10[state.aiPredKey] || {};
  const icdLine = icd.code
    ? `  ICD-10 Code       : ${icd.code} — ${icd.desc}`
    : "";

  const uncBlock = buildUncertaintyLines();

  let ocrBlock = "";
  if (state.ocrText && state.ocrText.trim()) {
    ocrBlock = [
      "",
      "--------------------------------------------------------",
      "  EXTRACTED TEXT (OCR)",
      state.ocrText,
    ].join("\n");
  }

  const clinName = $("clinicianName")?.value || "";
  const clinId = $("clinicianId")?.value || "";
  const ptName = $("ptName")?.value || "Anonymous";
  const ptId = $("ptId")?.value || "N/A";
  const ptDob = $("ptDob")?.value || "N/A";
  const ptRef = $("ptRefDoc")?.value || "N/A";
  const instit = $("ptInstitution")?.value || "Tecnomate Health Network";
  const clinNote = $("ptNotes")?.value || "";

  const reportText = [
    "========================================================",
    "  TECNOMATE CLINICAL AI \u2014 DIAGNOSTIC REPORT",
    "========================================================",
    `  Date/Time         : ${ts}`,
    `  Image File        : ${state.filename || "uploaded_scan.jpg"}`,
    `  Scan Type         : ${state.scanType}`,
    "",
    "  PATIENT INFORMATION",
    "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    `  Patient Name      : ${ptName}`,
    `  Patient ID        : ${ptId}`,
    `  Date of Birth     : ${ptDob}`,
    `  Referring Doc     : ${ptRef}`,
    `  Institution       : ${instit}`,
    clinNote ? `  Clinical Notes    : ${clinNote}` : "",
    "",
    "  AI PREDICTION",
    "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    `  AI Prediction     : ${aiLabel}`,
    `  Confidence        : ${((state.aiConfidence || 0) * 100).toFixed(1)}%`,
    `  Risk Level        : ${(RISK_LEVEL[state.aiPredKey] || "").toUpperCase()}`,
    icdLine,
    "",
    "  CLINICIAN REVIEW",
    "  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    `  Doctor Confirmed  : ${docLabel}`,
    `  Outcome           : ${outcome}`,
    clinName ? `  Clinician Name    : ${clinName}` : "",
    clinId ? `  Clinician ID      : ${clinId}` : "",
    "",
    "--------------------------------------------------------",
    "  CLASS PROBABILITIES",
    "--------------------------------------------------------",
    ...buildProbLines(),
    ...uncBlock,
    ocrBlock,
    "",
    "--------------------------------------------------------",
    "  CNN FEATURE ANALYSIS",
    "  What the model extracted from this image",
    "--------------------------------------------------------",
    ...buildFeatureLines(),
    "",
    "--------------------------------------------------------",
    "  PRIVACY NOTICE",
    "  Patient data anonymized. EXIF/metadata stripped.",
    "  No patient-identifiable information retained.",
    "--------------------------------------------------------",
    "  MEDICAL DISCLAIMER",
    "  AI-assisted tool only. Not for sole clinical use.",
    "  Always rely on qualified medical professionals.",
    "========================================================",
    "",
  ]
    .filter((l) => l !== null && l !== undefined)
    .join("\n");

  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `tecnomate_report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast("TXT report downloaded", "success", 2000);
}

/* ══════════════════════════════════════════════════════════════
   EXCEL EXPORT DOWNLOAD
══════════════════════════════════════════════════════════════ */
async function downloadExcelReport() {
  const btn = $("downloadExcelBtn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Exporting…";
  }
  try {
    const res = await fetch(API.exportExcel);
    if (!res.ok) {
      let detail = `Server error ${res.status}`;
      try { const e = await res.json(); detail = e.detail || detail; } catch { /**/ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    const cd   = res.headers.get("Content-Disposition") || "";
    const m    = cd.match(/filename="?([^"]+)"?/);
    a.download = m ? m[1] : `tecnomate_export_${Date.now()}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Excel report downloaded", "success", 2500);
  } catch (err) {
    showToast(`Excel export failed: ${err.message}`, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2.5">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <line x1="3" y1="9" x2="21" y2="9"/>
        <line x1="3" y1="15" x2="21" y2="15"/>
        <line x1="9" y1="3" x2="9" y2="21"/>
        <line x1="15" y1="3" x2="15" y2="21"/>
      </svg> Excel Export`;
    }
  }
}

function buildProbLines() {
  if (!state.probabilities) return ["  No data"];
  const mode = SCAN_MODES[state.scanType];
  return mode.keys.map((k) => {
    const prob = (state.probabilities[k] ?? 0) * 100;
    const label = mode.labels[k] || k;
    const bar = "\u2588".repeat(Math.round(prob / 5)).padEnd(20, "\u2591");
    return `  ${label.padEnd(30)}${bar} ${prob.toFixed(2)}%`;
  });
}

function buildUncertaintyLines() {
  const unc = state.uncertainty;
  if (!unc || !unc.mc_samples) return [];
  return [
    "",
    "--------------------------------------------------------",
    "  PREDICTIVE UNCERTAINTY (MC-DROPOUT)",
    `  MC Samples        : ${unc.mc_samples}`,
    `  Mean Entropy      : ${unc.mean_entropy.toFixed(4)}`,
    `  Std. Confidence   : ${unc.std_confidence.toFixed(4)}`,
    `  Uncertainty Level : ${unc.uncertainty_label}`,
  ];
}

const FEATURE_DETAIL = {
  glioma: {
    primary:         "Focal hyper/hypo-intense mass; irregular ill-defined boundary; heterogeneous signal (necrotic core); surrounding oedema; midline shift.",
    texture:         "High-frequency edge irregularity at mass; abrupt intensity gradients at tumour-parenchyma interface; chaotic internal texture.",
    spatial:         "Supratentorial (frontal/temporal lobe); unilateral mass effect; loss of adjacent gyral pattern.",
    differentiates:  "Ill-defined margin & heterogeneous core vs meningioma; outside sella vs pituitary; any focal mass/oedema vs no-tumour.",
  },
  meningioma: {
    primary:         "Well-circumscribed homogeneous extra-axial mass; smooth dural-based boundary; uniform density; dural tail sign.",
    texture:         "Uniform internal texture; sharp peripheral edges; possible focal hypo-intense calcification spots.",
    spatial:         "Dura/falx/tentorium origin; convexity or parasagittal; compression without parenchymal invasion.",
    differentiates:  "Sharp well-defined boundary vs glioma; extra-sellar location vs pituitary; extra-axial mass vs no-tumour.",
  },
  notumor: {
    primary:         "No focal mass, no abnormal signal, no boundary irregularity; symmetric grey-white distribution; preserved cortical folding.",
    texture:         "Homogeneous intensity throughout both hemispheres; no abrupt local transitions; normal ventricle size/position.",
    spatial:         "Bilateral symmetry; midline centred; no structural displacement; sulci and gyri clearly defined.",
    differentiates:  "Complete absence of mass features (irregular edges, heterogeneous cores, extra-axial masses, sellar enlargement).",
  },
  pituitary: {
    primary:         "Sellar/supra-sellar mass; expanded or eroded sella; variable signal (micro vs macro adenoma); possible optic chiasm compression.",
    texture:         "Focal intensity variation in midline sellar region; subtle gland asymmetry (microadenoma); sellar floor depression (macroadenoma).",
    spatial:         "Strictly midline centred on sella; inferior-to-chiasm extension; cavernous sinus invasion possible.",
    differentiates:  "Strictly midline sellar location vs glioma/meningioma; sellar expansion or asymmetric signal vs no-tumour.",
  },
  pneumonia: {
    primary:         "Pulmonary consolidation (opacification); air bronchogram sign; lobar/segmental/patchy distribution; possible pleural effusion.",
    texture:         "High-intensity lung regions (normally dark aerated); loss of normal lung texture gradient; ill-defined consolidation margins.",
    spatial:         "Unilateral or bilateral; lower-lobe predominance (bacterial); perihilar/diffuse (atypical/viral); heart border obliteration possible.",
    differentiates:  "Focal/diffuse opacification replacing aerated lung vs normal; air bronchogram vs pleural effusion.",
  },
  normal: {
    primary:         "Clear uniformly aerated lung fields; distinct costophrenic angles; sharp cardiac silhouette; visible pulmonary vasculature.",
    texture:         "Low-intensity (dark) parenchyma with fine uniform vascular markings; sharp diaphragm-lung interface; no asymmetric density changes.",
    spatial:         "Bilateral symmetric lung fields; no pleural thickening; no hilar enlargement; normal tracheal midline.",
    differentiates:  "Complete absence of consolidation, opacification, or pleural fluid; all normal landmarks preserved bilaterally.",
  },
};

/* ════════════════════════════════════════════════════════
   FEATURES PANEL — dashboard render
════════════════════════════════════════════════════════ */
function renderFeaturesPanel(predKey) {
  const panel = $("featuresPanel");
  if (!panel) return;

  const feat = FEATURE_DETAIL[predKey];
  if (!feat) {
    panel.style.display = "none";
    return;
  }

  // Pipeline summary line
  const pipelineEl = $("featPipeline");
  if (pipelineEl) {
    pipelineEl.textContent =
      "→ Greyscale → CLAHE contrast enhancement → Resize 128×128 → Normalise " +
      "→ Conv Block×3 (16→32→64 filters) → FC 1024 → Dropout → 6-class output";
  }

  const setText = (id, val) => { const el = $(id); if (el) el.textContent = val || "—"; };
  setText("featPrimary", feat.primary);
  setText("featTexture",  feat.texture);
  setText("featSpatial",  feat.spatial);
  setText("featDiff",     feat.differentiates);

  panel.style.display = "block";
}

function buildFeatureLines() {
  const pred = state.aiPredKey;
  const feat = FEATURE_DETAIL[pred];
  const lines = [
    "  PREPROCESSING PIPELINE:",
    "  Greyscale → CLAHE contrast enhancement → Resize 128x128 → Normalize",
    "",
    "  CNN EXTRACTION BLOCKS:",
    "  Block 1 (Conv 1→16):  Low-level edges, intensity gradients, fine texture.",
    "  Block 2 (Conv 16→32): Mid-level textures, shape outlines, boundary topology.",
    "  Block 3 (Conv 32→64): High-level patterns — mass density, asymmetry, consolidation.",
    "  FC 1024 → Dropout → FC 6: Final 6-class probability scores.",
    "",
  ];
  if (feat) {
    const label = SCAN_MODES[state.scanType]?.labels?.[pred] || pred;
    lines.push(`  FEATURES DETECTED FOR: ${label}`);
    lines.push(`  Primary Features   : ${feat.primary}`);
    lines.push(`  Texture Signals    : ${feat.texture}`);
    lines.push(`  Spatial Context    : ${feat.spatial}`);
    lines.push(`  Differentiates By  : ${feat.differentiates}`);
  }
  return lines;
}

/* ══════════════════════════════════════════════════════════════
   PATIENT INFO MODAL
══════════════════════════════════════════════════════════════ */
function openPatientModal() {
  const modal = $("patientModal");
  if (modal) modal.classList.add("open");
}

function closePatientModal() {
  const modal = $("patientModal");
  if (modal) modal.classList.remove("open");

  // Show hint in sidebar if any field has been filled
  const anyFilled = [
    $("ptName")?.value,
    $("ptId")?.value,
    $("ptDob")?.value,
    $("ptRefDoc")?.value,
    $("ptNotes")?.value,
  ].some((v) => v && v.trim());

  const hint = $("patientFilledHint");
  if (hint) hint.style.display = anyFilled ? "block" : "none";
}

function clearPatientForm() {
  [
    "ptName",
    "ptId",
    "ptDob",
    "ptGender",
    "ptVisitDate",
    "ptRefDoc",
    "ptInstitution",
    "ptNotes",
  ].forEach((id) => {
    const el = $(id);
    if (!el) return;
    if (el.tagName === "SELECT") el.selectedIndex = 0;
    else el.value = "";
  });
  const hint = $("patientFilledHint");
  if (hint) hint.style.display = "none";
  showToast("Patient form cleared", "info", 1800);
}

// Close modal when clicking the overlay backdrop
document.addEventListener("click", (e) => {
  const modal = $("patientModal");
  if (modal && e.target === modal) closePatientModal();
});

/* ══════════════════════════════════════════════════════════════
   HISTORY PANEL
══════════════════════════════════════════════════════════════ */

/**
 * Add an analysis result to the in-memory history list and re-render.
 * @param {{ label: string, confidence: number, scanType: string,
 *           filename: string, timestamp: string, mismatch: boolean }} entry
 */
function addToHistory(entry) {
  _history.unshift(entry); // newest first
  if (_history.length > 20) _history.length = 20; // cap at 20
  renderHistoryPanel();
}

function renderHistoryPanel() {
  const list = $("historyList");
  const empty = $("historyEmpty");
  if (!list) return;

  if (_history.length === 0) {
    if (empty) empty.style.display = "";
    // remove any existing items
    list.querySelectorAll(".history-item").forEach((el) => el.remove());
    return;
  }

  if (empty) empty.style.display = "none";

  // Rebuild
  list.querySelectorAll(".history-item").forEach((el) => el.remove());

  _history.forEach((h, idx) => {
    const confPct = (h.confidence * 100).toFixed(1);
    const icon = SCAN_MODES[h.scanType]?.icon || "";
    const mismatchDot = h.mismatch
      ? '<span style="color:#f59e0b;margin-left:3px;" title="Scan type mismatch">⚠</span>'
      : "";

    const item = document.createElement("div");
    item.className = "history-item";
    item.title = `${h.filename} — ${h.timestamp}`;
    item.innerHTML = `
      <div class="history-item-top">
        <span class="history-item-label">${icon} ${escHtml(h.label)}${mismatchDot}</span>
        <span class="history-item-conf">${confPct}%</span>
      </div>
      <div class="history-item-meta">${escHtml(h.filename)} · ${h.timestamp}</div>`;

    // Clicking a history entry scrolls to results and shows a toast
    item.addEventListener("click", () => {
      showToast(`Session ${idx + 1}: ${h.label} (${confPct}%)`, "info", 2500);
      scrollToResults();
    });

    list.appendChild(item);
  });
}

function clearHistory() {
  _history = [];
  renderHistoryPanel();
  showToast("Analysis history cleared", "info", 1800);
}

/* ══════════════════════════════════════════════════════════════
   CLEAR / RESET
══════════════════════════════════════════════════════════════ */
function clearAll() {
  Object.assign(state, {
    file: null,
    sessionId: null,
    aiPredKey: null,
    aiPredLabel: null,
    aiConfidence: 0,
    probabilities: null,
    ocrText: "",
    ocrLines: [],
    filename: "",
    doctorChoice: null,
    confirmed: false,
    confirmedSession: false,
    gradcamAvailable: false,
    uncertainty: null,
  });

  const zone = $("dropZone");
  if (zone) zone.classList.remove("has-file", "drag-hover");
  const input = $("fileInput");
  if (input) input.value = "";

  const dzInner = $("dropZoneInner");
  if (dzInner) {
    dzInner.innerHTML = `
      <div class="drop-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="3"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      </div>
      <p class="drop-primary">Drag &amp; drop your scan here</p>
      <p class="drop-secondary">or <span class="drop-link">browse files</span>
        &mdash; JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, HEIC, DICOM</p>
      <p class="drop-hint">MRI scans, X-rays, CT images, DICOM (.dcm) &mdash; max 30 MB</p>`;
  }

  hide("resultsArea");
  hide("ocrCard");
  hide("doctorCard");
  hide("predictionResult");
  hide("chartContainer");
  hide("feedbackSuccess");
  hide("feedbackError");
  hide("pdfError");
  hideError("uploadError");

  const uncPanel = $("uncertaintyPanel");
  if (uncPanel) uncPanel.style.display = "none";
  const featPanel1 = $("featuresPanel");
  if (featPanel1) featPanel1.style.display = "none";
  const icdBadge = $("icdBadge");
  if (icdBadge) icdBadge.style.display = "none";
  const gcHint = $("gradcamHint");
  if (gcHint) gcHint.classList.add("hidden");
  const mismatchBanner = $("mismatchWarning");
  if (mismatchBanner) mismatchBanner.classList.remove("visible");
  state._suggestedScanType = null;

  // Reset confidence ring
  updateConfidenceRing(0, "");

  if (probChart) {
    probChart.destroy();
    probChart = null;
  }

  // Reset confirm button
  const confirmBtn = $("confirmBtn");
  if (confirmBtn) {
    confirmBtn.disabled = false;
    confirmBtn.className = "btn btn-success";
    confirmBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      Confirm &amp; Save`;
  }

  const dlBtn = $("downloadBtn");
  if (dlBtn) dlBtn.disabled = true;
  const pdfBtn = $("downloadPdfBtn");
  if (pdfBtn) pdfBtn.disabled = true;
  const xlBtn2 = $("downloadExcelBtn");
  if (xlBtn2) xlBtn2.disabled = true;

  hideOverrideBadge();

  // Clear clinician fields
  ["clinicianName", "clinicianId", "clinicianNotes"].forEach((id) => {
    const el = $(id);
    if (el) el.value = "";
  });

  // Reset progress bar
  const wrap = $("uploadProgressWrap");
  const bar = $("uploadProgressBar");
  if (wrap) wrap.classList.remove("visible");
  if (bar) {
    bar.classList.remove("indeterminate");
    bar.style.width = "0%";
  }

  // Reset title
  document.title = "Tecnomate | Clinical AI v2";
}

/* ══════════════════════════════════════════════════════════════
   SIDEBAR QUEUE DISPLAY
══════════════════════════════════════════════════════════════ */
function updateQueueDisplay() {
  setText("queueCount", String(state.queueCount));
  setText("overrideCount", String(state.overrideCount));
}

async function refreshQueue() {
  try {
    const res = await fetch(API.queue);
    const data = await res.json();
    const confirmed = data.entries
      ? data.entries.filter((e) => !e.overridden).length
      : 0;
    const overridden = data.entries
      ? data.entries.filter((e) => e.overridden).length
      : 0;
    const totalConf = data.count ? data.count - overridden : confirmed;
    setText("queueCount", String(totalConf));
    setText("overrideCount", String(overridden));
    state.queueCount = totalConf;
    state.overrideCount = overridden;
  } catch {
    /* silent */
  }
}

/* ══════════════════════════════════════════════════════════════
   UTILITY FUNCTIONS
══════════════════════════════════════════════════════════════ */
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function showError(containerId, msgId, msg) {
  const el = document.getElementById(containerId);
  const msgEl = document.getElementById(msgId);
  if (el) el.hidden = false;
  if (msgEl) msgEl.textContent = msg;
}

function hideError(containerId) {
  const el = document.getElementById(containerId);
  if (el) el.hidden = true;
}
