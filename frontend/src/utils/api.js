const BASE = "/api";

// ── OCR gate error codes ───────────────────────────────────────────────────────
export const OCR_ERROR_CODES = {
  SCAN_REJECTED: "OCR_SCAN_REJECTED", // OCR confirmed non-xray / wrong modality
  XRAY_UNCONFIRMED: "OCR_XRAY_UNCONFIRMED", // inconclusive — no xray markers found
  REQUIRED_UNAVAILABLE: "OCR_REQUIRED_UNAVAILABLE", // OCR engine not installed on server
};

// Scan types that require the OCR + visual gate before inference.
// Both modes are included — a perfume bottle uploaded under "Brain MRI"
// must be rejected just as firmly as one uploaded under "Chest X-Ray".
export const STRICT_OCR_SCAN_TYPES = ["Chest X-Ray", "Brain MRI"];

// ── Scan type canonicalization ─────────────────────────────────────────────────
// Normalizes scan type strings to canonical values expected by the backend.
// Handles case differences, extra spaces, dashes, and common aliases.
export function canonicalizeScanType(scanType) {
  if (!scanType) return "unknown";
  // Normalize: lowercase, remove non-alphanumeric chars
  const key = scanType.toLowerCase().replace(/[^a-z0-9]/g, "");
  // Map aliases to canonical values
  if (["cxr", "chestxray", "chestx ray", "xray"].includes(key)) {
    return "Chest X-Ray";
  }
  if (["mri", "brainmri", "brain mri"].includes(key)) {
    return "Brain MRI";
  }
  // Return trimmed original if no alias match
  return scanType.trim().replace(/\s+/g, " ");
}

async function request(path, opts = {}) {
  // Add saved authorization header if present
  const authHeader = sessionStorage.getItem("adminAuth");
  if (authHeader) {
    opts.headers = { ...opts.headers, "Authorization": authHeader };
  }

  let res = await fetch(`${BASE}${path}`, opts);

  // If unauthorized and we're accessing an admin path, prompt for credentials
  while (res.status === 401 && path.startsWith("/admin")) {
    const user = window.prompt("Admin username:");
    if (user === null) break; // User cancelled
    const pass = window.prompt("Admin password:");
    if (pass === null) break;

    const token = "Basic " + btoa(user + ":" + pass);
    sessionStorage.setItem("adminAuth", token);
    opts.headers = { ...opts.headers, "Authorization": token };
    res = await fetch(`${BASE}${path}`, opts);
  }

  if (!res.ok) {
    let msg = `API ${res.status}: ${res.statusText}`;
    let errorCode = null;
    let errorMeta = {};
    try {
      const body = await res.json();
      if (body.detail) {
        if (typeof body.detail === "object") {
          // Structured error: { message, error_code, scan_type_detected, … }
          msg = body.detail.message || JSON.stringify(body.detail);
          errorCode = body.detail.error_code || null;
          errorMeta = body.detail;
        } else {
          msg = body.detail;
        }
      }
    } catch {}
    const err = new Error(msg);
    if (errorCode) err.errorCode = errorCode;
    if (errorMeta) err.errorMeta = errorMeta;
    throw err;
  }
  return res;
}

/**
 * Run the OCR gate pre-check for a given file + scan type.
 * Call this BEFORE predict() when scanType is in STRICT_OCR_SCAN_TYPES.
 *
 * Returns:
 *   { allowed, is_xray, scan_type_detected, confidence, keywords_found, message, error_code }
 *
 * Throws on network / server error (5xx).
 * Does NOT throw on gate rejection — callers should inspect allowed=false.
 */
export async function ocrCheck(file, scanType = "Chest X-Ray") {
  const form = new FormData();
  form.append("image", file);
  form.append("scan_type", canonicalizeScanType(scanType));
  const res = await fetch(`${BASE}/ocr-check`, { method: "POST", body: form });
  if (!res.ok) {
    let msg = `OCR check failed: ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail)
        msg =
          typeof body.detail === "object"
            ? body.detail.message || JSON.stringify(body.detail)
            : body.detail;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function getHealth() {
  return (await request("/health")).json();
}

export async function getModelInfo() {
  return (await request("/model-info")).json();
}

export async function predict(
  file,
  {
    scanType = "Brain MRI",
    gradcam = false,
    mcDropout = false,
    mcSamples = 20,
  } = {},
) {
  const form = new FormData();
  form.append("image", file);
  form.append("scan_type", canonicalizeScanType(scanType));
  form.append("gradcam", gradcam);
  form.append("mc_dropout", mcDropout);
  form.append("mc_samples", mcSamples);
  return (await request("/predict", { method: "POST", body: form })).json();
}

export async function sendFeedback(data) {
  const form = new FormData();
  Object.entries(data).forEach(([k, v]) => form.append(k, v));
  return (await request("/feedback", { method: "POST", body: form })).json();
}

export async function getQueue(limit = 50) {
  return (await request(`/queue?limit=${limit}`)).json();
}

export async function getAdminStats() {
  return (await request("/admin/stats")).json();
}

export async function getAdminFeedback({ limit = 50, offset = 0 } = {}) {
  return (
    await request(`/admin/feedback?limit=${limit}&offset=${offset}`)
  ).json();
}

export async function getAdminSessions({ limit = 50, offset = 0 } = {}) {
  return (
    await request(`/admin/sessions?limit=${limit}&offset=${offset}`)
  ).json();
}

export async function downloadCSV() {
  const res = await request(`/admin/export-csv`);
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  a.download = `tecnomate_feedback_${ts}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function downloadExcel() {
  const res = await request(`/admin/export-excel`);
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  a.download = `tecnomate_admin_report_${ts}.xlsx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function downloadPdfReport(data, format = "latex") {
  const form = new FormData();
  Object.entries(data).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    if (typeof v === "object" && !(v instanceof File)) {
      form.append(k, JSON.stringify(v));
    } else {
      form.append(k, v);
    }
  });
  form.append("format", format);

  const res = await fetch(`${BASE}/pdf-report`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let msg = `API ${res.status}: ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = body.detail;
    } catch {}
    throw new Error(msg);
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  a.download = `tecnomate_report_${ts}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}
