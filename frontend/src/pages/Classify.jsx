import { useState, useRef, useCallback } from "react";
import {
  predict,
  sendFeedback,
  downloadPdfReport,
  ocrCheck,
  OCR_ERROR_CODES,
  STRICT_OCR_SCAN_TYPES,
} from "../utils/api";

// ── Browser-level upload validation constants ─────────────────────────────────
const MAX_FILE_BYTES = 20 * 1024 * 1024; // 20 MB

// ── Canvas-based image colour analysis ───────────────────────────────────────
/**
 * Loads a File into an HTMLImageElement and draws it on an off-screen canvas,
 * then samples every pixel to compute grayscale statistics.
 *
 * Returns a promise that resolves to:
 *   { grayRatio, meanSat, highSatRatio, uniqueColors }
 *
 * grayRatio    — fraction of pixels where max(|R-G|,|R-B|,|G-B|) < 18
 * meanSat      — average saturation (0–1) across all pixels
 * highSatRatio — fraction of pixels with saturation > 0.20
 * uniqueColors — number of unique quantised colours (÷16 per channel, 64×64)
 */
function analyseImageColour(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      try {
        // ── Full-size sample (128×128) for saturation stats ────────────────
        const SIZE = 128;
        const canvas = document.createElement("canvas");
        canvas.width = SIZE;
        canvas.height = SIZE;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, SIZE, SIZE);
        const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

        let grayCount = 0;
        let satSum = 0;
        let highSatCount = 0;
        const total = SIZE * SIZE;

        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];

          // Grayscale check
          const maxDiff = Math.max(
            Math.abs(r - g),
            Math.abs(r - b),
            Math.abs(g - b),
          );
          if (maxDiff < 18) grayCount++;

          // HSL saturation
          const rn = r / 255,
            gn = g / 255,
            bn = b / 255;
          const max = Math.max(rn, gn, bn);
          const min = Math.min(rn, gn, bn);
          const l = (max + min) / 2;
          const sat =
            max === min
              ? 0
              : (max - min) / (l > 0.5 ? 2 - max - min : max + min);
          satSum += sat;
          if (sat > 0.2) highSatCount++;
        }

        const grayRatio = grayCount / total;
        const meanSat = satSum / total;
        const highSatRatio = highSatCount / total;

        // ── Unique colour count (64×64) ────────────────────────────────────
        const SMALL = 64;
        const canvas2 = document.createElement("canvas");
        canvas2.width = SMALL;
        canvas2.height = SMALL;
        const ctx2 = canvas2.getContext("2d");
        ctx2.drawImage(img, 0, 0, SMALL, SMALL);
        const { data: d2 } = ctx2.getImageData(0, 0, SMALL, SMALL);
        const colorSet = new Set();
        for (let i = 0; i < d2.length; i += 4) {
          const key =
            (d2[i] >> 4) * 65536 + (d2[i + 1] >> 4) * 256 + (d2[i + 2] >> 4);
          colorSet.add(key);
        }

        resolve({
          grayRatio,
          meanSat,
          highSatRatio,
          uniqueColors: colorSet.size,
        });
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not load image for colour analysis."));
    };
    img.src = url;
  });
}

/**
 * Returns a rejection reason string if the image is clearly non-medical
 * (colourful, too many unique colours), or null if it looks plausibly
 * grayscale enough to be a medical scan.
 *
 * Thresholds mirror the backend _is_likely_medical_scan() function.
 */
async function browserVisualCheck(file, scanType) {
  // Skip check for DICOM — canvas cannot decode raw DICOM bytes
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "dcm") return null;

  // Only enforce strict visual check for Chest X-Ray
  if (!STRICT_OCR_SCAN_TYPES.includes(scanType)) return null;

  let stats;
  try {
    stats = await analyseImageColour(file);
  } catch {
    // If analysis fails, let the server decide
    return null;
  }

  const { grayRatio, meanSat, highSatRatio, uniqueColors } = stats;
  const modality = scanType === "Chest X-Ray" ? "chest X-ray" : "brain MRI";

  if (grayRatio < 0.7) {
    return `This image appears to be a colour photograph, not a ${modality}. Real medical scans are grayscale. Please upload a valid ${modality} image.`;
  }
  if (meanSat > 0.15) {
    return `This image has too much colour saturation to be a ${modality}. Please upload a grayscale medical image.`;
  }
  if (highSatRatio > 0.2) {
    return `Too many colourful pixels detected for a ${modality}. Please upload a valid medical scan.`;
  }
  if (uniqueColors > 900) {
    return `This image contains too many distinct colours to be a ${modality}. Medical scans are typically near-grayscale.`;
  }

  return null; // passes visual check
}

const ALLOWED_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
  "image/tiff",
  "image/gif",
  "image/avif",
  // DICOM has no standard MIME; browsers may report these:
  "application/dicom",
  "application/octet-stream", // generic binary — allow, server validates
]);

const ALLOWED_EXTENSIONS = new Set([
  "jpg",
  "jpeg",
  "png",
  "webp",
  "bmp",
  "tiff",
  "tif",
  "gif",
  "avif",
  "dcm",
]);

function validateFile(f) {
  if (!f) return "No file selected.";

  // Extension check
  const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED_EXTENSIONS.has(ext)) {
    return `Unsupported file type ".${ext}". Please upload a JPEG, PNG, WebP, BMP, TIFF, GIF, AVIF, or DICOM file.`;
  }

  // MIME check — skip for .dcm which browsers often report as octet-stream
  if (ext !== "dcm" && f.type && !ALLOWED_MIME_TYPES.has(f.type)) {
    return `Unsupported file format (${f.type}). Please upload a valid medical image.`;
  }

  // Size check
  if (f.size > MAX_FILE_BYTES) {
    const mb = (f.size / (1024 * 1024)).toFixed(1);
    return `File is too large (${mb} MB). Maximum allowed size is 20 MB.`;
  }

  // Zero-byte guard
  if (f.size === 0) {
    return "The selected file is empty. Please choose a valid image file.";
  }

  return null; // valid (basic checks — colour analysis is async, done in handleFile)
}

const SCAN_TYPES = ["Brain MRI", "Chest X-Ray"];
const CLASS_COLOURS = {
  glioma: "#ef4444",
  meningioma: "#f59e0b",
  notumor: "#10b981",
  pituitary: "#ec4899",
  normal: "#0d9488",
  pneumonia: "#e11d48",
};
const CLASS_LABELS = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
  normal: "Normal (CXR)",
  pneumonia: "Pneumonia",
};

/* ── Heuristic: is this likely NOT a medical image? ── */
function isLikelyNonMedical(result) {
  if (!result || !result.all_probabilities) return false;
  const probs = Object.values(result.all_probabilities);
  const topConf = result.confidence || 0;
  const count = probs.length || 1;
  const evenDist = 1 / count;
  const avgProb = probs.reduce((a, b) => a + b, 0) / count;
  const spreadRatio = avgProb > 0 ? topConf / avgProb : 999;
  // Low confidence AND probabilities are roughly uniform => not a real scan
  if (topConf < 0.55 && spreadRatio < 2.2) return true;
  // Very low confidence alone
  if (topConf < 0.35) return true;
  return false;
}

function ConfidenceRing({ value, color, size = 120 }) {
  const r = 40,
    C = 2 * Math.PI * r;
  const offset = C - value * C;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className="transform -rotate-90"
    >
      <circle
        cx="50"
        cy="50"
        r={r}
        fill="none"
        stroke="#e7e5e4"
        strokeWidth="8"
      />
      <circle
        cx="50"
        cy="50"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="8"
        strokeDasharray={C}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="conf-ring-animate"
        style={{ "--ring-offset": offset }}
      />
      <text
        x="50"
        y="54"
        textAnchor="middle"
        fill={color}
        fontSize="18"
        fontWeight="bold"
        className="transform rotate-90 origin-center"
        dominantBaseline="middle"
      >
        {Math.round(value * 100)}%
      </text>
    </svg>
  );
}

function ProbBar({ label, value, color, max }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-medium text-stone-500 w-24 truncate">
        {label}
      </span>
      <div className="flex-1 h-2.5 bg-stone-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-bold text-stone-600 w-14 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  );
}

/* ── OCR Gate Step Indicator ── */
function OcrGateSteps({ ocrStatus, scanType }) {
  // ocrStatus: 'idle' | 'checking' | 'pass' | 'fail'
  const verifyLabel =
    scanType === "Chest X-Ray" ? "OCR X-Ray Verify" : "Visual Scan Verify";
  const steps = [
    { key: "upload", label: "Image Uploaded" },
    { key: "ocr", label: verifyLabel },
    { key: "infer", label: "AI Inference" },
  ];
  const stepState = (key) => {
    if (key === "upload") return ocrStatus === "idle" ? "active" : "done";
    if (key === "ocr") {
      if (ocrStatus === "idle") return "pending";
      if (ocrStatus === "checking") return "active";
      if (ocrStatus === "pass") return "done";
      if (ocrStatus === "fail") return "fail";
    }
    if (key === "infer") {
      if (ocrStatus === "pass") return "active";
      return "pending";
    }
    return "pending";
  };
  const colours = {
    done: "bg-teal-500 text-white border-teal-500",
    active: "bg-teal-50 text-teal-700 border-teal-400 animate-pulse",
    fail: "bg-red-100 text-red-600 border-red-400",
    pending: "bg-stone-100 text-stone-400 border-stone-200",
  };
  const lineColour = (idx) => {
    const s = stepState(steps[idx].key);
    return s === "done" ? "bg-teal-400" : "bg-stone-200";
  };
  return (
    <div className="flex items-center gap-0 mb-4">
      {steps.map((step, idx) => (
        <div key={step.key} className="flex items-center flex-1 last:flex-none">
          <div className="flex flex-col items-center gap-1">
            <div
              className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-bold transition-all ${colours[stepState(step.key)]}`}
            >
              {stepState(step.key) === "done" ? (
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="2.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m4.5 12.75 6 6 9-13.5"
                  />
                </svg>
              ) : stepState(step.key) === "fail" ? (
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="2.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18 18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                idx + 1
              )}
            </div>
            <span className="text-[10px] text-stone-500 font-medium whitespace-nowrap">
              {step.label}
            </span>
          </div>
          {idx < steps.length - 1 && (
            <div
              className={`flex-1 h-0.5 mb-4 mx-1 transition-all ${lineColour(idx)}`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Strict Gate Badge shown for all scan modes ── */
function StrictGateBadge({ active, scanType }) {
  if (!active) return null;
  const label =
    scanType === "Chest X-Ray"
      ? "Strict X-Ray Gate: ON"
      : "Scan Validation Gate: ON";
  return (
    <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-teal-50 border border-teal-200 text-xs font-semibold text-teal-700">
      <svg
        className="w-3.5 h-3.5"
        fill="none"
        viewBox="0 0 24 24"
        strokeWidth="2"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z"
        />
      </svg>
      {label}
    </div>
  );
}

/* ── OCR Scan Rejection Banner ── */
function OcrRejectionBanner({
  message,
  scanTypeDetected,
  errorCode,
  keywords,
}) {
  const isUnavailable = errorCode === OCR_ERROR_CODES.REQUIRED_UNAVAILABLE;
  const isUnconfirmed = errorCode === OCR_ERROR_CODES.XRAY_UNCONFIRMED;

  const iconPath = isUnavailable
    ? "M9.75 9.75l4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
    : isUnconfirmed
      ? "M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z"
      : "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z";

  // Detect if the rejection is specifically due to a stock-photo watermark
  const WATERMARK_TERMS = [
    "alamy",
    "shutterstock",
    "istock",
    "getty",
    "dreamstime",
    "stock photo",
    "royalty free",
    "123rf",
    "depositphotos",
    "bigstock",
  ];
  const msgLower = (message || "").toLowerCase();
  const isWatermarked =
    scanTypeDetected === "non_medical" &&
    (WATERMARK_TERMS.some((t) => msgLower.includes(t)) ||
      (keywords || []).some((k) =>
        WATERMARK_TERMS.some((t) => k.toLowerCase().includes(t)),
      ));

  const labelMap = {
    non_medical: isWatermarked
      ? "Stock Photo Watermark Detected"
      : "Non-Medical Image Detected",
    ct: "CT Scan — Not Supported",
    xray: "Chest X-Ray — Wrong Mode Selected",
    mri: "MRI Scan — Wrong Mode Selected",
    unknown: "Scan Type Could Not Be Verified",
  };
  const hintMap = {
    non_medical: isWatermarked
      ? "This image has a stock-photo watermark (Alamy, Shutterstock, iStock, etc.). Upload an original, unwatermarked chest X-ray exported directly from a radiology system or PACS (JPEG / PNG / DICOM)."
      : "Upload an original, unedited radiology image (JPEG / PNG / DICOM).",
    ct: "This system only accepts X-ray and MRI images. CT scans are not supported.",
    xray: 'Switch to "Chest X-Ray" scan type, or upload a valid brain MRI image.',
    mri: 'Switch to "Brain MRI" scan type, or upload a chest X-ray instead.',
    unknown: "Please upload a standard chest X-ray from a radiology system.",
  };

  const titleOverride = isUnavailable
    ? "OCR Engine Unavailable"
    : isUnconfirmed
      ? "X-Ray Could Not Be Confirmed"
      : null;

  const title =
    titleOverride || labelMap[scanTypeDetected] || "Scan Validation Failed";
  const hint = isUnavailable
    ? "Contact the system administrator to install rapidocr-onnxruntime on the server."
    : isUnconfirmed
      ? "The ML scan classifier could not confirm this image with sufficient confidence, and no X-ray text markers were found. Upload a clear, original chest X-ray exported from a radiology system or PACS (JPEG / PNG / DICOM)."
      : hintMap[scanTypeDetected] || "";

  const bgColour = isUnavailable
    ? "border-orange-200 bg-orange-50/60"
    : "border-red-200 bg-red-50/60";
  const iconColour = isUnavailable ? "text-orange-500" : "text-red-500";
  const iconBg = isUnavailable ? "bg-orange-100" : "bg-red-100";
  const badgeBg = isUnavailable
    ? "bg-orange-100 border-orange-200 text-orange-600"
    : "bg-red-100 border-red-200 text-red-600";
  const dotColour = isUnavailable ? "bg-orange-500" : "bg-red-500";
  const titleColour = isUnavailable ? "text-orange-700" : "text-red-700";
  const msgColour = isUnavailable ? "text-orange-600/90" : "text-red-600/90";
  const hintColour = isUnavailable ? "text-orange-500/80" : "text-red-500/80";

  return (
    <div className={`rounded-2xl border-2 ${bgColour} p-6 text-center fade-in`}>
      <div
        className={`w-14 h-14 mx-auto mb-3 rounded-full ${iconBg} flex items-center justify-center`}
      >
        <svg
          className={`w-7 h-7 ${iconColour}`}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth="2"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d={iconPath} />
        </svg>
      </div>

      <div
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full ${badgeBg} border text-xs font-semibold mb-3 uppercase tracking-wide`}
      >
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColour} animate-pulse`}
        />
        OCR Scan Validator
      </div>

      <h3 className={`text-lg font-bold ${titleColour} mb-2`}>{title}</h3>
      <p className={`text-sm ${msgColour} max-w-sm mx-auto leading-relaxed`}>
        {message}
      </p>
      {hint && (
        <p className={`mt-3 text-xs ${hintColour} max-w-xs mx-auto`}>{hint}</p>
      )}
      {keywords && keywords.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-1.5 justify-center">
          {keywords.slice(0, 6).map((kw) => (
            <span
              key={kw}
              className="px-2 py-0.5 rounded-full bg-white/70 border border-red-200 text-xs text-red-600 font-mono"
            >
              {kw}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── OCR Checking Spinner ── */
function OcrCheckingPanel({ scanType }) {
  const isCxr = scanType === "Chest X-Ray";
  const heading = isCxr ? "Verifying X-Ray..." : "Verifying Medical Scan...";
  const subtext = isCxr
    ? "Running ML scan classifier + OCR checks to confirm this is a valid chest X-ray before inference."
    : "Running ML scan classifier + visual checks to confirm this is a valid medical scan before inference.";
  return (
    <div className="rounded-2xl border-2 border-teal-200 bg-teal-50/60 p-6 text-center fade-in">
      <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-teal-100 flex items-center justify-center">
        <svg className="animate-spin w-7 h-7 text-teal-500" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
            fill="none"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
          />
        </svg>
      </div>
      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-teal-100 border border-teal-200 text-xs font-semibold text-teal-700 mb-3 uppercase tracking-wide">
        <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
        Scan Validator
      </div>
      <h3 className="text-base font-bold text-teal-700 mb-1">{heading}</h3>
      <p className="text-sm text-teal-600/80 max-w-xs mx-auto">{subtext}</p>
    </div>
  );
}

/* ── OCR Pass Banner ── */
function OcrPassBanner({ keywords, scanType }) {
  const isCxr = scanType === "Chest X-Ray";
  const hasOcrMarkers = keywords && keywords.length > 0;

  const passLabel = isCxr ? "X-Ray Verified ✓" : "Medical Scan Verified ✓";

  // Differentiate message: OCR confirmed vs ML gate confirmed (no text markers)
  const passDetail = isCxr
    ? hasOcrMarkers
      ? "OCR confirmed valid chest X-ray."
      : "Scan classifier confirmed valid chest X-ray (no text markers present — clean scan)."
    : "Scan classifier confirmed valid medical scan.";

  return (
    <div className="rounded-2xl border-2 border-teal-200 bg-teal-50/40 p-4 flex items-center gap-4 fade-in">
      <div className="w-10 h-10 rounded-full bg-teal-100 flex items-center justify-center flex-shrink-0">
        <svg
          className="w-5 h-5 text-teal-600"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth="2.5"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="m4.5 12.75 6 6 9-13.5"
          />
        </svg>
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-bold text-teal-700">{passLabel}</p>
        <p className="text-xs text-teal-600/80 mt-0.5">
          {passDetail}
          {hasOcrMarkers && (
            <span className="ml-1 font-mono">
              ({keywords.slice(0, 4).join(", ")})
            </span>
          )}
        </p>
      </div>
    </div>
  );
}

function NonMedicalWarning({ message }) {
  return (
    <div className="warning-shimmer rounded-2xl border-2 border-red-200 p-6 text-center fade-in">
      <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-red-50 flex items-center justify-center">
        <svg
          className="w-7 h-7 text-red-500"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth="2"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
          />
        </svg>
      </div>
      <h3 className="text-lg font-bold text-red-700 mb-1">
        Medical Scan Not Detected
      </h3>
      <p className="text-sm text-red-600/80 max-w-sm mx-auto">
        {message ||
          "This image does not appear to be a valid X-ray or MRI scan. Classification results may be unreliable. Please upload a proper medical image."}
      </p>
      <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-50 border border-red-200 text-xs font-medium text-red-600">
        <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
        Low confidence — results not trustworthy
      </div>
    </div>
  );
}

export default function Classify() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scanType, setScanType] = useState("Brain MRI");
  const [gradcam, setGradcam] = useState(false);
  const [mcDropout, setMcDropout] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [visualCheckPending, setVisualCheckPending] = useState(false);
  // ocrRejected: { message, scanTypeDetected, errorCode, keywords }
  const [ocrRejected, setOcrRejected] = useState(null);
  // ocrStatus: 'idle' | 'checking' | 'pass' | 'fail'
  const [ocrStatus, setOcrStatus] = useState("idle");
  const [ocrPassKeywords, setOcrPassKeywords] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [fbSent, setFbSent] = useState(false);
  const inputRef = useRef();

  const [downloadingReport, setDownloadingReport] = useState(false);
  const [reportFormat, setReportFormat] = useState("reportlab");

  const isStrictMode = STRICT_OCR_SCAN_TYPES.includes(scanType);

  const handleFile = useCallback(
    (f) => {
      if (!f) return;

      // ── Step 1: Basic synchronous checks (type, size) ─────────────────────
      const validationError = validateFile(f);
      if (validationError) {
        setError(validationError);
        setFile(null);
        setPreview(null);
        setResult(null);
        setOcrRejected(null);
        setOcrStatus("idle");
        setOcrPassKeywords([]);
        setFbSent(false);
        if (inputRef.current) inputRef.current.value = "";
        return;
      }

      // Show preview immediately so user sees what they selected
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
      setOcrRejected(null);
      setOcrStatus("idle");
      setOcrPassKeywords([]);
      setFbSent(false);

      // ── Step 2: Async canvas-based colour analysis ─────────────────────────
      // Run in background right after file selection — before the user clicks
      // "Verify & Classify" — so we can show an error immediately.
      setVisualCheckPending(true);
      browserVisualCheck(f, scanType)
        .then((colourError) => {
          setVisualCheckPending(false);
          if (colourError) {
            setError(colourError);
            setFile(null);
            setPreview(null);
            if (inputRef.current) inputRef.current.value = "";
          }
        })
        .catch(() => {
          // Analysis failed — let server handle it, don't block the user
          setVisualCheckPending(false);
        });
    },
    [scanType],
  );

  // Reset OCR gate state when scan type switches, and re-run visual check
  // on the already-selected file since different modes have different rules.
  const handleScanTypeChange = (t) => {
    setScanType(t);
    setResult(null);
    setError(null);
    setOcrRejected(null);
    setOcrStatus("idle");
    setOcrPassKeywords([]);
    setFbSent(false);

    // Re-validate the current file under the new scan type
    if (file) {
      setVisualCheckPending(true);
      browserVisualCheck(file, t)
        .then((colourError) => {
          setVisualCheckPending(false);
          if (colourError) {
            setError(colourError);
            setFile(null);
            setPreview(null);
            if (inputRef.current) inputRef.current.value = "";
          }
        })
        .catch(() => {
          setVisualCheckPending(false);
        });
    }
  };

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragActive(false);
      handleFile(e.dataTransfer.files[0]);
    },
    [handleFile],
  );

  const classify = async () => {
    if (!file) return;

    // Guard: if canvas visual check is still running, wait for it first
    if (visualCheckPending) {
      setError(
        "Image analysis in progress, please wait a moment and try again.",
      );
      return;
    }

    // Re-run visual check synchronously as a final gate before any network call
    const colourError = await browserVisualCheck(file, scanType);
    if (colourError) {
      setError(colourError);
      setFile(null);
      setPreview(null);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    setLoading(true);
    setError(null);
    setOcrRejected(null);
    setResult(null);

    // ── Step 1: OCR pre-check for strict scan types ────────────────────────
    if (isStrictMode) {
      setOcrStatus("checking");
      try {
        const checkResult = await ocrCheck(file, scanType);
        if (!checkResult.allowed) {
          // Strict mode: reject all non-allowed results including unconfirmed X-rays.
          // Random images without X-ray markers should not pass through.
          setOcrStatus("fail");
          setOcrRejected({
            message: checkResult.message,
            scanTypeDetected: checkResult.scan_type_detected || "unknown",
            errorCode: checkResult.error_code,
            keywords: checkResult.keywords_found || [],
          });
          setLoading(false);
          return;
        } else {
          setOcrStatus("pass");
          setOcrPassKeywords(checkResult.keywords_found || []);
        }
      } catch (err) {
        // Network/server error during pre-check — surface as a plain error
        setOcrStatus("idle");
        setError(`OCR pre-check failed: ${err.message}`);
        setLoading(false);
        return;
      }
    }

    // ── Step 2: Inference ──────────────────────────────────────────────────
    try {
      const res = await predict(file, { scanType, gradcam, mcDropout });
      // Fail-closed guard on client side:
      // If backend returns a mismatch response instead of rejecting (e.g. stale
      // backend process), block obviously wrong-mode uploads in Brain MRI mode.
      if (
        scanType === "Brain MRI" &&
        res?.scan_type_mismatch === true &&
        Number(res?.other_mode_mass ?? 0) >= 0.95
      ) {
        setOcrStatus("fail");
        setOcrRejected({
          message:
            "This upload appears to be a Chest X-Ray (or non-MRI image), but 'Brain MRI' mode is selected. Please switch to 'Chest X-Ray' mode or upload a valid brain MRI image.",
          scanTypeDetected: "xray",
          errorCode: OCR_ERROR_CODES.SCAN_REJECTED,
          keywords: [],
        });
        setResult(null);
        setLoading(false);
        return;
      }
      setResult(res);
    } catch (err) {
      // Catch any gate rejections that slip through at the API level too
      if (
        err.errorCode === OCR_ERROR_CODES.SCAN_REJECTED ||
        err.errorCode === OCR_ERROR_CODES.XRAY_UNCONFIRMED ||
        err.errorCode === OCR_ERROR_CODES.REQUIRED_UNAVAILABLE
      ) {
        setOcrStatus("fail");
        setOcrRejected({
          message: err.message,
          scanTypeDetected: err.errorMeta?.scan_type_detected || "unknown",
          errorCode: err.errorCode,
          keywords: err.errorMeta?.keywords_found || [],
        });
      } else {
        setError(err.message);
      }
    }
    setLoading(false);
  };

  const submitFeedback = async (agreed) => {
    if (!result) return;
    const chosenKey = agreed
      ? result.predicted_key || result.predicted_class
      : overrideClass;
    if (!chosenKey) return;
    try {
      await sendFeedback({
        session_id: result.session_id,
        chosen_key: chosenKey,
        ai_predicted_key: result.predicted_key || result.predicted_class,
        scan_type: scanType,
      });
      setFbSent(true);
    } catch {}
  };
  const [overrideClass, setOverrideClass] = useState("");

  const handleDownloadReport = async () => {
    if (!result) return;
    setDownloadingReport(true);
    try {
      await downloadPdfReport(
        {
          session_id: result.session_id,
          scan_type: scanType,
          ai_pred_key: result.predicted_key || result.predicted_class,
          ai_confidence: result.confidence || 0,
          doctor_choice_key:
            overrideClass || result.predicted_key || result.predicted_class,
          probabilities_json: JSON.stringify(
            result.all_probabilities || result.probabilities || {},
          ),
          mc_entropy: result.uncertainty?.mean_entropy || 0,
          mc_std_conf: result.uncertainty?.std_confidence || 0,
          mc_samples: result.uncertainty?.mc_samples || 0,
          mc_label: result.uncertainty?.uncertainty_label || "",
          selected_mode_mass: result.selected_mode_mass || 0,
          other_mode_mass: result.other_mode_mass || 0,
          scan_type_mismatch: result.scan_type_mismatch || false,
        },
        reportFormat,
      );
    } catch (err) {
      alert("Failed to download report: " + err.message);
    }
    setDownloadingReport(false);
  };

  const topProbs = result
    ? Object.entries(
        result.all_probabilities || result.probabilities || {},
      ).sort((a, b) => b[1] - a[1])
    : [];
  const maxProb = topProbs.length > 0 ? topProbs[0][1] : 1;
  const nonMedical = result
    ? result.likely_medical_scan === false || isLikelyNonMedical(result)
    : false;

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-extrabold text-stone-800 mb-6 fade-in">
        Classify Medical Image
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left: Upload ── */}
        <div className="space-y-4 fade-in">
          {/* Drop Zone */}
          <label
            htmlFor="file-upload"
            className={`relative block border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer glass focus-within:ring-2 focus-within:ring-teal-500 focus-within:ring-offset-2 ${
              dragActive
                ? "border-teal-500 bg-teal-50/40 drag-pulse"
                : "border-stone-300 hover:border-teal-400"
            } ${preview ? "bg-white/70" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
          >
            <input
              id="file-upload"
              ref={inputRef}
              type="file"
              accept="image/*,.dcm"
              className="sr-only"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-xl shadow-lg"
              />
            ) : (
              <div className="py-8">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-teal-50 flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-teal-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth="1.5"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
                    />
                  </svg>
                </div>
                <p className="text-sm font-medium text-stone-600">
                  Drop a medical image here or click to browse
                </p>
                <p className="text-xs text-stone-400 mt-1">
                  JPEG, PNG, WebP, BMP, TIFF, DICOM
                </p>
              </div>
            )}
          </label>

          {/* Options Panel */}
          <div className="glass rounded-2xl p-5 space-y-4">
            {/* Scan Type */}
            <div>
              <label className="text-xs font-semibold text-stone-500 uppercase tracking-wide">
                Scan Type
              </label>
              <div className="flex gap-2 mt-2">
                {SCAN_TYPES.map((t) => (
                  <button
                    key={t}
                    onClick={() => handleScanTypeChange(t)}
                    className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      scanType === t
                        ? "bg-teal-600 text-white shadow-lg shadow-teal-600/20"
                        : "bg-stone-100 text-stone-600 hover:bg-stone-200"
                    }`}
                  >
                    {t === "Brain MRI" ? "🧠" : "🫁"} {t}
                  </button>
                ))}
              </div>
              {isStrictMode && (
                <p className="mt-2 text-xs text-teal-600/80">
                  🔒 Strict OCR verification is required for Chest X-Ray
                  uploads.
                </p>
              )}
            </div>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={gradcam}
                  onChange={(e) => setGradcam(e.target.checked)}
                  className="w-4 h-4 rounded border-stone-300 text-teal-600 focus:ring-teal-500"
                />
                <span className="text-sm text-stone-600">Grad-CAM</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={mcDropout}
                  onChange={(e) => setMcDropout(e.target.checked)}
                  className="w-4 h-4 rounded border-stone-300 text-teal-600 focus:ring-teal-500"
                />
                <span className="text-sm text-stone-600">MC Dropout</span>
              </label>
            </div>
            <button
              onClick={classify}
              disabled={!file || loading || visualCheckPending}
              className="w-full py-3 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-teal-600/20 transition-all active:scale-[0.98]"
            >
              {visualCheckPending ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                  Analysing Image...
                </span>
              ) : loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                  {ocrStatus === "checking"
                    ? "Verifying X-Ray..."
                    : "Classifying..."}
                </span>
              ) : isStrictMode ? (
                "🔒 Verify & Classify"
              ) : (
                "Classify Image"
              )}
            </button>
          </div>
          {/* Strict Gate Badge */}
          <div className="flex items-center justify-between">
            <StrictGateBadge active={isStrictMode} scanType={scanType} />
            {isStrictMode && ocrStatus !== "idle" && (
              <OcrGateSteps ocrStatus={ocrStatus} scanType={scanType} />
            )}
          </div>

          {/* OCR checking in progress */}
          {ocrStatus === "checking" && <OcrCheckingPanel scanType={scanType} />}

          {/* OCR pass confirmation */}
          {ocrStatus === "pass" && !loading && (
            <OcrPassBanner keywords={ocrPassKeywords} scanType={scanType} />
          )}

          {/* OCR rejection */}
          {ocrRejected && (
            <OcrRejectionBanner
              message={ocrRejected.message}
              scanTypeDetected={ocrRejected.scanTypeDetected}
              errorCode={ocrRejected.errorCode}
              keywords={ocrRejected.keywords}
            />
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-600">
              {error}
            </div>
          )}
        </div>

        {/* ── Right: Results ── */}
        <div className="space-y-4 fade-in">
          {/* Gate blocked — show full-panel rejection on right side too */}
          {ocrRejected && !result && (
            <div className="glass rounded-2xl p-12 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-50 flex items-center justify-center">
                <svg
                  className="w-8 h-8 text-red-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M18.364 18.364A9 9 0 0 0 5.636 5.636m12.728 12.728A9 9 0 0 1 5.636 5.636m12.728 12.728L5.636 5.636"
                  />
                </svg>
              </div>
              <p className="text-sm font-semibold text-red-600 mb-1">
                Upload Blocked
              </p>
              <p className="text-xs text-stone-400 max-w-xs mx-auto">
                {isStrictMode
                  ? "Only verified chest X-ray images are accepted in this mode. Correct the issue on the left and try again."
                  : "This image was rejected by the scan validator. Please upload a valid medical scan."}
              </p>
            </div>
          )}

          {!result && !loading && !ocrRejected && (
            <div className="glass rounded-2xl p-12 text-center">
              <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-stone-100 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-stone-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                  />
                </svg>
              </div>
              <p className="text-sm text-stone-400">
                Results will appear here after classification
              </p>
            </div>
          )}

          {loading && ocrStatus !== "checking" && (
            <div className="glass rounded-2xl p-12 text-center">
              <svg
                className="animate-spin h-10 w-10 mx-auto text-teal-500"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
              <p className="text-sm text-stone-500 mt-3">
                Running inference...
              </p>
            </div>
          )}

          {result && (
            <>
              {/* Non-Medical Image Warning */}
              {nonMedical && (
                <NonMedicalWarning message={result.medical_scan_warning} />
              )}

              {/* Primary Result */}
              <div
                className={`glass rounded-2xl p-6 ${nonMedical ? "opacity-60" : ""}`}
              >
                <div className="flex items-center gap-6">
                  <ConfidenceRing
                    value={result.confidence || 0}
                    color={
                      nonMedical
                        ? "#a8a29e"
                        : CLASS_COLOURS[result.predicted_class] || "#0d9488"
                    }
                  />
                  <div>
                    <p className="text-xs text-stone-400 uppercase font-medium mb-1">
                      Prediction
                    </p>
                    <h2 className="text-xl font-bold text-stone-800">
                      {result.short_name ||
                        CLASS_LABELS[result.predicted_class] ||
                        result.predicted_class}
                    </h2>
                    <span
                      className="inline-block mt-2 px-3 py-1 rounded-full text-xs font-medium"
                      style={{
                        backgroundColor:
                          (nonMedical
                            ? "#a8a29e"
                            : CLASS_COLOURS[result.predicted_class] ||
                              "#0d9488") + "15",
                        color: nonMedical
                          ? "#78716c"
                          : CLASS_COLOURS[result.predicted_class] || "#0d9488",
                      }}
                    >
                      {result.risk_level || result.scan_type}
                    </span>
                  </div>
                </div>
                {result.scan_type_mismatch && (
                  <div className="mt-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-700">
                    ⚠️ {result.mismatch_detail}
                  </div>
                )}
              </div>

              {/* Probability Bars */}
              <div
                className={`glass rounded-2xl p-6 ${nonMedical ? "opacity-60" : ""}`}
              >
                <h3 className="text-sm font-bold text-stone-600 mb-4">
                  All Probabilities
                </h3>
                <div className="space-y-3">
                  {topProbs.map(([cls, prob]) => (
                    <ProbBar
                      key={cls}
                      label={CLASS_LABELS[cls] || cls}
                      value={prob}
                      color={
                        nonMedical ? "#d6d3d1" : CLASS_COLOURS[cls] || "#0d9488"
                      }
                      max={maxProb}
                    />
                  ))}
                </div>
              </div>

              {/* MC Dropout Uncertainty */}
              {result.uncertainty && (
                <div className="glass rounded-2xl p-6">
                  <h3 className="text-sm font-bold text-stone-600 mb-3">
                    Uncertainty Analysis
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-xs text-stone-400">Mean Entropy</p>
                      <p className="font-bold text-stone-600">
                        {result.uncertainty.mean_entropy?.toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-stone-400">Std Confidence</p>
                      <p className="font-bold text-stone-600">
                        {result.uncertainty.std_confidence?.toFixed(4)}
                      </p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-xs text-stone-400">Label</p>
                      <span
                        className={`inline-block mt-1 px-3 py-1 rounded-full text-xs font-medium ${
                          result.uncertainty.uncertainty_label === "Low"
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                            : result.uncertainty.uncertainty_label === "Medium"
                              ? "bg-amber-50 text-amber-700 border border-amber-200"
                              : "bg-red-50 text-red-700 border border-red-200"
                        }`}
                      >
                        {result.uncertainty.uncertainty_label} Uncertainty
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Grad-CAM */}
              {result.gradcam_path && (
                <div className="glass rounded-2xl p-6">
                  <h3 className="text-sm font-bold text-stone-600 mb-3">
                    Grad-CAM Visualization
                  </h3>
                  <p className="text-xs text-stone-400 mb-3">
                    Highlights regions the model focused on for its prediction.
                  </p>
                  <div className="bg-stone-50 rounded-xl p-3 text-xs text-stone-500 text-center border border-stone-100">
                    Grad-CAM generated — available in PDF report
                  </div>
                </div>
              )}

              {/* Feedback */}
              {!nonMedical && (
                <div className="glass rounded-2xl p-6">
                  <h3 className="text-sm font-bold text-stone-600 mb-3">
                    Clinician Feedback
                  </h3>
                  {fbSent ? (
                    <div className="text-sm text-emerald-600 font-medium flex items-center gap-2">
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth="2"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                        />
                      </svg>
                      Feedback submitted. Thank you!
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex gap-3">
                        <button
                          onClick={() => submitFeedback(true)}
                          className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition"
                        >
                          Agree
                        </button>
                        <button
                          onClick={() =>
                            setOverrideClass(
                              overrideClass
                                ? ""
                                : topProbs[1]?.[0] || topProbs[0]?.[0] || "",
                            )
                          }
                          className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition"
                        >
                          Override
                        </button>
                      </div>
                      {overrideClass && (
                        <div className="flex gap-2">
                          <select
                            value={overrideClass}
                            onChange={(e) => setOverrideClass(e.target.value)}
                            className="flex-1 px-3 py-2 rounded-xl text-sm bg-white border border-stone-200 text-stone-700"
                          >
                            {topProbs.map(([cls]) => (
                              <option key={cls} value={cls}>
                                {CLASS_LABELS[cls] || cls}
                              </option>
                            ))}
                          </select>
                          <button
                            onClick={() => submitFeedback(false)}
                            className="px-4 py-2 rounded-xl text-sm font-medium bg-teal-600 text-white hover:bg-teal-700 transition"
                          >
                            Submit
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
              {/* Download Report */}
              <div className="glass rounded-2xl p-6">
                <h3 className="text-sm font-bold text-stone-600 mb-3">
                  Diagnostic Report
                </h3>
                <p className="text-xs text-stone-400 mb-4">
                  Export a professional clinical diagnostic report.
                </p>
                <div className="flex flex-col gap-3">
                  <div className="flex bg-stone-100 rounded-xl p-1">
                    <button
                      onClick={() => setReportFormat("latex")}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition ${reportFormat === "latex" ? "bg-white text-teal-700 shadow-sm" : "text-stone-500 hover:text-stone-700"}`}
                    >
                      LaTeX PDF
                    </button>
                    <button
                      onClick={() => setReportFormat("reportlab")}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition ${reportFormat === "reportlab" ? "bg-white text-teal-700 shadow-sm" : "text-stone-500 hover:text-stone-700"}`}
                    >
                      Standard PDF
                    </button>
                  </div>
                  <button
                    onClick={handleDownloadReport}
                    disabled={downloadingReport}
                    className="w-full py-2.5 rounded-xl font-bold text-sm text-teal-700 bg-teal-50 border border-teal-200 hover:bg-teal-100 disabled:opacity-50 transition flex items-center justify-center gap-2"
                  >
                    {downloadingReport ? (
                      <>
                        <svg
                          className="animate-spin h-4 w-4"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                          />
                        </svg>
                        Generating...
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth="2"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"
                          />
                        </svg>
                        Download Report (
                        {reportFormat === "latex" ? "LaTeX" : "Standard"})
                      </>
                    )}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
