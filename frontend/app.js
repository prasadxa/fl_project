/* ============================================================
   Tecnomate Clinical AI — app.js
   SPA logic: drag-drop upload, inference, chart, OCR,
   doctor override, feedback submission, report download.
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

/* ── App State ──────────────────────────────────────────────── */
const state = {
  scanType: "Brain MRI",
  file: null,
  sessionId: null,
  aiPredKey: null,
  aiPredLabel: null,
  probabilities: null,
  ocrText: "",
  ocrLines: [],
  filename: "",
  doctorChoice: null,
  confirmed: false,
  confirmedSession: false,
  queueCount: 0,
  overrideCount: 0,
};

/* ── Chart instance ─────────────────────────────────────────── */
let probChart = null;

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
  checkHealth();
  setInterval(checkHealth, 30_000); // re-check every 30 s
  populateDoctorSelect();
});

/* ── Health / status check ───────────────────────────────────── */
async function checkHealth() {
  const pill = $("modelStatusPill");
  const dot = $("statusDot");
  const txt = $("statusText");
  const ocrPill = $("ocrStatusPill");
  const ocrDot = $("ocrDot");
  const ocrTxt = $("ocrStatusText");

  // show loading state
  pill.className = "status-pill loading";
  txt.textContent = "Connecting…";

  try {
    const res = await fetch(API.health);
    const data = await res.json();

    if (data.model_loaded) {
      pill.className = "status-pill online";
      txt.textContent = "Model Ready";
    } else {
      pill.className = "status-pill offline";
      txt.textContent = "Model Missing";
    }

    // OCR pill
    if (data.ocr_available) {
      ocrPill.className = "status-pill ocr-pill ocr-on";
      ocrTxt.textContent = "OCR Active";
      ocrDot.title = "RapidOCR is available";
    } else {
      ocrPill.className = "status-pill ocr-pill ocr-off";
      ocrTxt.textContent = "OCR Off";
      ocrPill.title = data.ocr_reason || "Install rapidocr-onnxruntime";
    }
  } catch {
    pill.className = "status-pill offline";
    txt.textContent = "API Offline";
    ocrPill.className = "status-pill ocr-pill ocr-off";
  }
}

/* ══════════════════════════════════════════════════════════════
   SCAN TYPE SELECTOR
══════════════════════════════════════════════════════════════ */
function setScanType(mode) {
  state.scanType = mode;

  // Update sidebar buttons
  document.querySelectorAll(".scan-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  // Update the badge on the upload card
  const badge = $("scanTypeBadge");
  if (badge) {
    badge.textContent = `${SCAN_MODES[mode].icon} ${mode}`;
  }

  // If we have results already, re-render the chart for the new mode
  if (state.probabilities) {
    renderChart(state.probabilities);
    updateDoctorAIField();
  }

  // Refresh doctor dropdown
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
  $("dropZone").classList.remove("drag-hover");
  const files = event.dataTransfer?.files;
  if (files && files.length > 0) {
    processFile(files[0]);
  }
}

function handleFileSelect(event) {
  const file = event.target.files?.[0];
  if (file) processFile(file);
}

function processFile(file) {
  hideError("uploadError");

  // Validate type
  const allowed = ["image/jpeg", "image/png", "image/jpg"];
  if (!allowed.includes(file.type)) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      "Unsupported file type. Please upload a JPEG or PNG image.",
    );
    return;
  }

  // Validate size (max 20 MB)
  if (file.size > 20 * 1024 * 1024) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      "File too large. Maximum allowed size is 20 MB.",
    );
    return;
  }

  state.file = file;
  state.filename = file.name;
  state.confirmed = false;
  state.confirmedSession = false;

  // Update drop zone appearance
  const zone = $("dropZone");
  zone.classList.add("has-file");

  // Update drop zone text to show chosen file
  const inner = $("dropZoneInner");
  inner.innerHTML = `
    <div class="drop-icon">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.5">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
    <p class="drop-primary">${escHtml(file.name)}</p>
    <p class="drop-secondary">${formatBytes(file.size)} &mdash; ready to analyse</p>
  `;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    $("previewImage").src = e.target.result;
    setText("metaFilename", file.name);
    setText("metaSize", formatBytes(file.size));
  };
  reader.readAsDataURL(file);

  // Show results area (with predict button)
  show("resultsArea");
  hide("predictionResult");
  hide("chartContainer");
  hide("ocrCard");
  hide("doctorCard");
  hide("feedbackSuccess");
  hide("feedbackError");

  // Reset doctor panel
  $("downloadBtn").disabled = true;
  $("confirmBtn").disabled = false;
  hideOverrideBadge();

  // Reset state
  state.sessionId = null;
  state.aiPredKey = null;
  state.probabilities = null;
  state.ocrLines = [];
  state.ocrText = "";
}

/* ══════════════════════════════════════════════════════════════
   INFERENCE
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

  // Show spinner, disable button
  const btn = $("predictBtn");
  const spinner = $("predictionSpinner");
  btn.disabled = true;
  spinner.classList.add("spinning");
  hide("predictionResult");
  hide("chartContainer");
  hideError("uploadError");

  try {
    const form = new FormData();
    form.append("image", state.file, state.file.name);
    form.append("scan_type", state.scanType);

    const res = await fetch(API.predict, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    // Store results in state
    state.sessionId = data.session_id;
    state.aiPredKey = data.mode_predicted_key;
    state.aiPredLabel = data.mode_predicted_class;
    state.probabilities = data.probabilities;
    state.ocrText = data.ocr_text || "";
    state.ocrLines = data.ocr_lines || [];
    state.filename = data.filename || state.file.name;

    // Render prediction result
    renderPredictionBanner(data);

    // Render probability chart (scan-type filtered)
    renderChart(data.probabilities);

    // Render OCR panel
    renderOCR(data);

    // Show doctor panel
    populateDoctorSelect();
    updateDoctorAIField();
    show("doctorCard");
  } catch (err) {
    showError(
      "uploadError",
      "uploadErrorMsg",
      `Prediction failed: ${err.message}`,
    );
  } finally {
    btn.disabled = false;
    spinner.classList.remove("spinning");
  }
}

/* ── Render prediction banner ──────────────────────────────── */
function renderPredictionBanner(data) {
  const key = data.mode_predicted_key;
  const label = data.mode_predicted_class;
  const conf = data.mode_confidence;
  const level = RISK_LEVEL[key] || "medium";

  const resultLabel = $("resultLabel");
  const riskBadge = $("resultRiskBadge");

  // Emoji
  setText("resultEmoji", SCAN_MODES[state.scanType].icon);

  // Class & confidence
  setText("resultClass", label);
  setText(
    "resultConf",
    `Confidence: ${(conf * 100).toFixed(1)}%  ·  ${state.scanType}`,
  );

  // Risk colouring
  resultLabel.className = `result-label risk-${level}`;
  riskBadge.className = `result-risk-badge risk-${level}`;
  setText("resultRiskBadge", RISK_LABEL[level]);

  show("predictionResult");
  show("chartContainer");
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
  const alphas = keys.map((k) => (RISK_COLOURS[k] || "#94a3b8") + "cc");

  const canvas = $("probChart");
  const ctx = canvas.getContext("2d");

  // Destroy previous instance
  if (probChart) {
    probChart.destroy();
    probChart = null;
  }

  // Set container height BEFORE chart creation to prevent ResizeObserver resize loop
  canvas.parentElement.style.height = `${keys.length * 54 + 20}px`;

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
      animation: { duration: 500, easing: "easeOutQuart" },
      layout: { padding: { right: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `  ${(ctx.raw * 100).toFixed(2)}%`,
          },
          backgroundColor: "#0f172a",
          titleColor: "#e2e8f0",
          bodyColor: "#94a3b8",
          borderColor: "#334155",
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
        },
      },
      scales: {
        x: {
          min: 0,
          max: 1,
          grid: { color: "#e2e8f0", lineWidth: 1 },
          border: { display: false },
          ticks: {
            callback: (v) => `${(v * 100).toFixed(0)}%`,
            color: "#94a3b8",
            font: { size: 11 },
            maxTicksLimit: 6,
          },
        },
        y: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: "#475569",
            font: { size: 12, weight: "600" },
          },
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

  if (!data.ocr_lines && !data.ocr_text) {
    // OCR not available (backend reported unavailable via ocr_text being undefined)
    ocrBody.innerHTML = `
      <div class="ocr-unavailable">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        OCR is not installed. Run:
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
      ${lines.length} text region(s) detected &mdash;
      ${highConf} high-confidence
    </p>`;

  show("ocrCard");
}

/* ══════════════════════════════════════════════════════════════
   DOCTOR DECISION PANEL
══════════════════════════════════════════════════════════════ */

/** Rebuild the dropdown options for the current scan type. */
function populateDoctorSelect() {
  const sel = $("doctorSelect");
  const mode = SCAN_MODES[state.scanType];
  sel.innerHTML = "";

  mode.keys.forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = mode.labels[key] || key;
    sel.appendChild(opt);
  });

  // Pre-select AI prediction if we have one
  if (state.aiPredKey && mode.keys.includes(state.aiPredKey)) {
    sel.value = state.aiPredKey;
  }

  state.doctorChoice = sel.value;
  updateOverrideIndicator();
}

/** Sync the AI prediction display field to current state. */
function updateDoctorAIField() {
  if (!state.aiPredKey) return;
  const mode = SCAN_MODES[state.scanType];
  const label =
    mode.labels[state.aiPredKey] || state.aiPredLabel || state.aiPredKey;
  setText("doctorAIPred", label);
}

/** Called whenever the doctor changes the dropdown. */
function handleDoctorSelectChange() {
  const sel = $("doctorSelect");
  state.doctorChoice = sel.value;
  state.confirmed = false;
  hide("feedbackSuccess");
  hide("feedbackError");
  $("downloadBtn").disabled = true;
  updateOverrideIndicator();
}

/** Show/hide the "AI Override" badge and style the select. */
function updateOverrideIndicator() {
  const sel = $("doctorSelect");
  const badge = $("overrideBadge");
  const isOver =
    state.doctorChoice &&
    state.aiPredKey &&
    state.doctorChoice !== state.aiPredKey;

  if (isOver) {
    sel.classList.add("override-active");
    show("overrideBadge");
  } else {
    sel.classList.remove("override-active");
    hide("overrideBadge");
  }
}

function hideOverrideBadge() {
  hide("overrideBadge");
  const sel = $("doctorSelect");
  if (sel) sel.classList.remove("override-active");
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
  confirmBtn.textContent = "Saving…";

  try {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    form.append("chosen_key", chosenKey);
    form.append("scan_type", state.scanType);
    form.append("ai_predicted_key", state.aiPredKey || "");

    const res = await fetch(API.feedback, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    // Success
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

    // Enable report download
    $("downloadBtn").disabled = false;

    // Update session queue counts
    if (overridden) {
      state.overrideCount += 1;
    } else {
      state.queueCount += 1;
    }
    updateQueueDisplay();

    // Re-label confirm button
    confirmBtn.innerHTML = `
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      Saved`;
    confirmBtn.classList.remove("btn-success");
    confirmBtn.classList.add("btn-outline");
  } catch (err) {
    showError(
      "feedbackError",
      "feedbackErrorMsg",
      `Could not save: ${err.message}`,
    );
    confirmBtn.disabled = false;
    // restore label
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
   REPORT DOWNLOAD
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

  let ocrBlock = "";
  if (state.ocrText && state.ocrText.trim()) {
    ocrBlock = [
      "",
      "--------------------------------------------------------",
      "  EXTRACTED TEXT (OCR)",
      state.ocrText,
    ].join("\n");
  }

  const reportText = [
    "========================================================",
    "  TECNOMATE CLINICAL AI \u2014 DIAGNOSTIC REPORT",
    "========================================================",
    `  Date/Time         : ${ts}`,
    `  Image File        : ${state.filename || "uploaded_scan.jpg"}`,
    `  Scan Type         : ${state.scanType}`,
    `  AI Prediction     : ${aiLabel}`,
    `  Doctor Confirmed  : ${docLabel}`,
    `  Outcome           : ${outcome}`,
    ocrBlock,
    "",
    "--------------------------------------------------------",
    "  CLASS PROBABILITIES",
    "--------------------------------------------------------",
    ...buildProbLines(),
    "",
    "--------------------------------------------------------",
    "  PRIVACY NOTICE",
    "  Patient Data Anonymized and Secured.",
    "  All EXIF metadata, device identifiers, and hidden",
    "  tags have been permanently stripped from this image",
    "  before storage.  No patient-identifiable information",
    "  is retained anywhere in this system.",
    "--------------------------------------------------------",
    "  MEDICAL DISCLAIMER",
    "  This report is generated by an AI assistant only.",
    "  It must not be used as the sole basis for clinical",
    "  decisions.  Always rely on qualified medical",
    "  professionals for diagnosis and treatment.",
    "========================================================",
    "",
  ].join("\n");

  // Trigger browser download
  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `tecnomate_report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function buildProbLines() {
  if (!state.probabilities) return ["  No data"];
  const mode = SCAN_MODES[state.scanType];
  return mode.keys.map((k) => {
    const prob = (state.probabilities[k] ?? 0) * 100;
    const label = mode.labels[k] || k;
    const bar = "█".repeat(Math.round(prob / 5)).padEnd(20, "░");
    return `  ${label.padEnd(30)}${bar} ${prob.toFixed(2)}%`;
  });
}

/* ══════════════════════════════════════════════════════════════
   CLEAR / RESET
══════════════════════════════════════════════════════════════ */
function clearAll() {
  // Reset state
  Object.assign(state, {
    file: null,
    sessionId: null,
    aiPredKey: null,
    aiPredLabel: null,
    probabilities: null,
    ocrText: "",
    ocrLines: [],
    filename: "",
    doctorChoice: null,
    confirmed: false,
    confirmedSession: false,
  });

  // Reset drop zone
  const zone = $("dropZone");
  zone.classList.remove("has-file", "drag-hover");
  $("dropZoneInner").innerHTML = `
    <div class="drop-icon">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="18" height="18" rx="3"/>
        <circle cx="8.5" cy="8.5" r="1.5"/>
        <polyline points="21 15 16 10 5 21"/>
      </svg>
    </div>
    <p class="drop-primary">Drag &amp; drop your scan here</p>
    <p class="drop-secondary">or <span class="drop-link">browse files</span> &mdash; JPEG &amp; PNG supported</p>
    <p class="drop-hint">MRI scans, X-rays, CT images &mdash; any resolution accepted</p>
  `;

  // Reset file input so the same file can be re-selected
  const input = $("fileInput");
  if (input) input.value = "";

  // Hide panels
  hide("resultsArea");
  hide("ocrCard");
  hide("doctorCard");
  hideError("uploadError");

  // Destroy chart
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

  if ($("downloadBtn")) $("downloadBtn").disabled = true;
  hideOverrideBadge();
}

/* ══════════════════════════════════════════════════════════════
   SIDEBAR QUEUE DISPLAY
══════════════════════════════════════════════════════════════ */
function updateQueueDisplay() {
  setText("queueCount", String(state.queueCount));
  setText("overrideCount", String(state.overrideCount));
}

/** Refresh queue stats from the server (called on demand). */
async function refreshQueue() {
  try {
    const res = await fetch(API.queue);
    const data = await res.json();
    const confirmed = data.entries.filter((e) => !e.overridden).length;
    const overridden = data.entries.filter((e) => e.overridden).length;
    setText("queueCount", String(confirmed));
    setText("overrideCount", String(overridden));
    state.queueCount = confirmed;
    state.overrideCount = overridden;
  } catch {
    /* silent */
  }
}

// Refresh queue when sidebar is visible
refreshQueue();
setInterval(refreshQueue, 60_000);

/* ══════════════════════════════════════════════════════════════
   UTILITY FUNCTIONS
══════════════════════

/* ============================================================*/
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
