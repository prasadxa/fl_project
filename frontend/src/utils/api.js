const BASE = "/api";

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res;
}

export async function getHealth() {
  return (await request("/health")).json();
}

export async function getModelInfo() {
  return (await request("/model-info")).json();
}

export async function predict(
  file,
  { gradcam = false, mcDropout = false, mcSamples = 20 } = {},
) {
  const form = new FormData();
  form.append("file", file);
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

export function downloadCSV() {
  window.open(`${BASE}/admin/export-csv`, "_blank");
}

export function downloadExcel() {
  window.open(`${BASE}/admin/export-excel`, "_blank");
}
