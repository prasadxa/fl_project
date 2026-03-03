import { useState, useRef, useCallback } from 'react'
import { predict, sendFeedback } from '../utils/api'

const SCAN_TYPES = ['Brain MRI', 'Chest X-Ray']
const CLASS_COLOURS = {
  glioma: '#ef4444', meningioma: '#f59e0b', notumor: '#10b981',
  pituitary: '#8b5cf6', normal: '#06b6d4', pneumonia: '#e11d48',
}
const CLASS_LABELS = {
  glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'No Tumor',
  pituitary: 'Pituitary', normal: 'Normal (CXR)', pneumonia: 'Pneumonia',
}

function ConfidenceRing({ value, color, size = 120 }) {
  const r = 40, C = 2 * Math.PI * r
  const offset = C - (value * C)
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" className="transform -rotate-90">
      <circle cx="50" cy="50" r={r} fill="none" stroke="currentColor" strokeWidth="8"
        className="text-slate-200 dark:text-slate-700" />
      <circle cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="8"
        strokeDasharray={C} strokeDashoffset={offset} strokeLinecap="round"
        className="conf-ring-animate" style={{ '--ring-offset': offset }} />
      <text x="50" y="54" textAnchor="middle" fill={color} fontSize="18" fontWeight="bold"
        className="transform rotate-90 origin-center" dominantBaseline="middle">
        {Math.round(value * 100)}%
      </text>
    </svg>
  )
}

function ProbBar({ label, value, color, max }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-medium text-slate-500 dark:text-slate-400 w-24 truncate">{label}</span>
      <div className="flex-1 h-2.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-bold text-slate-700 dark:text-slate-300 w-14 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  )
}

export default function Classify() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [scanType, setScanType] = useState('Brain MRI')
  const [gradcam, setGradcam] = useState(false)
  const [mcDropout, setMcDropout] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [fbSent, setFbSent] = useState(false)
  const inputRef = useRef()

  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
    setFbSent(false)
  }, [])

  const onDrop = useCallback(e => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const classify = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await predict(file, { gradcam, mcDropout })
      setResult(res)
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  const submitFeedback = async (correct) => {
    if (!result) return
    try {
      await sendFeedback({
        session_id: result.session_id,
        doctor_agrees: correct,
        corrected_label: correct ? result.predicted_class : '',
      })
      setFbSent(true)
    } catch {}
  }

  const topProbs = result ? Object.entries(result.all_probabilities || {}).sort((a, b) => b[1] - a[1]) : []
  const maxProb = topProbs.length > 0 ? topProbs[0][1] : 1

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white mb-6">
        🔬 Classify Medical Image
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Upload */}
        <div className="space-y-4">
          {/* Drop Zone */}
          <div
            className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
              dragActive
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950 drag-pulse'
                : 'border-slate-300 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-600'
            } ${preview ? 'bg-slate-50 dark:bg-slate-900' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragActive(true) }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="image/*,.dcm" className="hidden"
              onChange={e => handleFile(e.target.files[0])} />
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-xl shadow-lg" />
            ) : (
              <div className="py-8">
                <div className="text-5xl mb-3 opacity-60">🏥</div>
                <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Drop a medical image here or click to browse
                </p>
                <p className="text-xs text-slate-400 mt-1">JPEG, PNG, WebP, BMP, TIFF, DICOM</p>
              </div>
            )}
          </div>

          {/* Options */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 space-y-4">
            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase">Scan Type</label>
              <div className="flex gap-2 mt-2">
                {SCAN_TYPES.map(t => (
                  <button key={t} onClick={() => setScanType(t)}
                    className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      scanType === t
                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}>
                    {t === 'Brain MRI' ? '🧠' : '🫁'} {t}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={gradcam} onChange={e => setGradcam(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                <span className="text-sm text-slate-700 dark:text-slate-300">Grad-CAM</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={mcDropout} onChange={e => setMcDropout(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                <span className="text-sm text-slate-700 dark:text-slate-300">MC Dropout</span>
              </label>
            </div>
            <button onClick={classify} disabled={!file || loading}
              className="w-full py-3 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-blue-500/25 transition-all active:scale-[0.98]">
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>
                  Classifying…
                </span>
              ) : '🔬 Classify Image'}
            </button>
          </div>
          {error && (
            <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-xl p-4 text-sm text-red-600 dark:text-red-400">{error}</div>
          )}
        </div>

        {/* Right: Results */}
        <div className="space-y-4">
          {!result && !loading && (
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-12 text-center">
              <div className="text-5xl mb-3 opacity-30">📋</div>
              <p className="text-sm text-slate-400">Results will appear here after classification</p>
            </div>
          )}
          {loading && (
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-12 text-center">
              <svg className="animate-spin h-10 w-10 mx-auto text-blue-500" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
              </svg>
              <p className="text-sm text-slate-500 mt-3">Running inference...</p>
            </div>
          )}
          {result && (
            <>
              {/* Primary Result */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
                <div className="flex items-center gap-6">
                  <ConfidenceRing value={result.confidence || 0} color={CLASS_COLOURS[result.predicted_class] || '#3b82f6'} />
                  <div>
                    <p className="text-xs text-slate-400 uppercase font-medium mb-1">Prediction</p>
                    <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                      {result.short_name || CLASS_LABELS[result.predicted_class] || result.predicted_class}
                    </h2>
                    <span className="inline-block mt-2 px-3 py-1 rounded-full text-xs font-medium" style={{
                      backgroundColor: (CLASS_COLOURS[result.predicted_class] || '#3b82f6') + '15',
                      color: CLASS_COLOURS[result.predicted_class] || '#3b82f6'
                    }}>
                      {result.risk_level || result.scan_type}
                    </span>
                  </div>
                </div>
                {result.scan_type_mismatch && (
                  <div className="mt-4 p-3 rounded-xl bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 text-xs text-amber-700 dark:text-amber-300">
                    ⚠️ {result.mismatch_detail}
                  </div>
                )}
              </div>

              {/* Probability Bars */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-4">All Probabilities</h3>
                <div className="space-y-3">
                  {topProbs.map(([cls, prob]) => (
                    <ProbBar key={cls} label={CLASS_LABELS[cls] || cls} value={prob}
                      color={CLASS_COLOURS[cls] || '#6366f1'} max={maxProb} />
                  ))}
                </div>
              </div>

              {/* MC Dropout */}
              {result.uncertainty && (
                <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
                  <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">Uncertainty Analysis</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-xs text-slate-400">Mean Entropy</p>
                      <p className="font-bold text-slate-700 dark:text-slate-300">{result.uncertainty.mean_entropy?.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Std Confidence</p>
                      <p className="font-bold text-slate-700 dark:text-slate-300">{result.uncertainty.std_confidence?.toFixed(4)}</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-xs text-slate-400">Label</p>
                      <span className={`inline-block mt-1 px-3 py-1 rounded-full text-xs font-medium ${
                        result.uncertainty.uncertainty_label === 'Low' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                        : result.uncertainty.uncertainty_label === 'Medium' ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300'
                        : 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                      }`}>
                        {result.uncertainty.uncertainty_label} Uncertainty
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Grad-CAM */}
              {result.gradcam_path && (
                <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
                  <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">Grad-CAM Visualization</h3>
                  <p className="text-xs text-slate-400 mb-3">Highlights regions the model focused on for its prediction.</p>
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-xl p-3 text-xs text-slate-500 text-center">
                    Grad-CAM generated — available in PDF report
                  </div>
                </div>
              )}

              {/* Feedback */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
                <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">Clinician Feedback</h3>
                {fbSent ? (
                  <div className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">✅ Feedback submitted. Thank you!</div>
                ) : (
                  <div className="flex gap-3">
                    <button onClick={() => submitFeedback(true)}
                      className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 dark:hover:bg-emerald-900 transition">
                      ✅ Agree
                    </button>
                    <button onClick={() => submitFeedback(false)}
                      className="flex-1 py-2.5 rounded-xl text-sm font-medium bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800 hover:bg-red-100 dark:hover:bg-red-900 transition">
                      ❌ Override
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
