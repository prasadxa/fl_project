import { useState, useEffect } from 'react'
import { getHealth, getModelInfo, getAdminStats } from '../utils/api'

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 hover:shadow-lg hover:shadow-blue-500/5 transition-all hover:-translate-y-0.5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{label}</p>
          <p className={`text-3xl font-bold mt-1 ${color || 'text-slate-900 dark:text-white'}`}>{value}</p>
          {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </div>
  )
}

function ClassCard({ name, shortName, colour }) {
  return (
    <div className="flex items-center gap-3 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 hover:border-blue-300 dark:hover:border-blue-700 transition">
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: colour }} />
      <div>
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">{shortName}</p>
        <p className="text-xs text-slate-400">{name}</p>
      </div>
    </div>
  )
}

export default function Dashboard({ health }) {
  const [info, setInfo] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    getModelInfo().then(setInfo).catch(() => {})
    getAdminStats().then(setStats).catch(() => {})
  }, [])

  const modelParams = '11.2M'

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero */}
      <div className="mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">
          Clinical AI Dashboard
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-2 max-w-xl">
          Privacy-preserving federated learning for medical image classification.
          Brain tumor MRI & chest X-ray analysis with 95.10% accuracy.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <StatCard icon="🎯" label="Model Accuracy" value="95.1%" sub="Weighted F1: 95.11%" color="text-emerald-600" />
        <StatCard icon="🧠" label="Parameters" value={modelParams} sub="ResNet-18 backbone" />
        <StatCard icon="📊" label="Classes" value={info?.num_classes || 6} sub="Brain tumor + Chest X-ray" color="text-blue-600" />
        <StatCard icon="🔒" label="Status" value={health?.model_loaded ? 'Online' : 'Offline'} sub={health?.model_loaded ? 'Model loaded & ready' : 'Model not loaded'} color={health?.model_loaded ? 'text-emerald-600' : 'text-red-500'} />
      </div>

      {/* Class Registry */}
      <div className="mb-10">
        <h2 className="text-lg font-bold text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-gradient-to-b from-blue-500 to-indigo-500 rounded-full" />
          Classification Classes
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {info && Object.entries(info.short_names).map(([key, name]) => (
            <ClassCard key={key} name={key} shortName={name} colour={info.risk_colours[key] || '#6366f1'} />
          ))}
          {!info && Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-16 rounded-xl bg-slate-100 dark:bg-slate-800 animate-pulse" />
          ))}
        </div>
      </div>

      {/* Scan Modes */}
      {info && (
        <div className="mb-10">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-purple-500 to-pink-500 rounded-full" />
            Scan Modes
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(info.scan_modes).map(([mode, cfg]) => (
              <div key={mode} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
                <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                  <span className="text-xl">{cfg.icon}</span>{mode}
                </h3>
                <div className="mt-3 flex flex-wrap gap-2">
                  {cfg.class_keys.map(k => (
                    <span key={k} className="px-3 py-1 rounded-full text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                      {info.short_names[k] || k}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System Info */}
      {health && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-amber-500 to-orange-500 rounded-full" />
            System Info
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-slate-400 text-xs uppercase font-medium">OCR</p>
              <p className={`font-semibold ${health.ocr_available ? 'text-emerald-600' : 'text-slate-400'}`}>
                {health.ocr_available ? 'Available' : 'Unavailable'}
              </p>
            </div>
            <div>
              <p className="text-slate-400 text-xs uppercase font-medium">Max Upload</p>
              <p className="font-semibold text-slate-700 dark:text-slate-300">{health.max_upload_mb} MB</p>
            </div>
            <div>
              <p className="text-slate-400 text-xs uppercase font-medium">Feedback Total</p>
              <p className="font-semibold text-slate-700 dark:text-slate-300">{health.feedback_total}</p>
            </div>
            <div>
              <p className="text-slate-400 text-xs uppercase font-medium">Rate Limit</p>
              <p className="font-semibold text-slate-700 dark:text-slate-300">{health.security?.rate_limit_general}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
