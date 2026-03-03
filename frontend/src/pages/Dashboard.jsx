import { useState, useEffect } from 'react'
import { getHealth, getModelInfo, getAdminStats } from '../utils/api'

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div className="glass rounded-2xl p-6 card-lift fade-in">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-stone-500">{label}</p>
          <p className={`text-3xl font-bold mt-1 ${color || 'text-stone-800'}`}>{value}</p>
          {sub && <p className="text-xs text-stone-400 mt-1">{sub}</p>}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </div>
  )
}

function ClassCard({ name, shortName, colour }) {
  return (
    <div className="flex items-center gap-3 glass rounded-xl p-4 card-lift fade-in hover:scale-[1.01] transition-transform">
      <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: colour, boxShadow: `0 0 8px ${colour}30` }} />
      <div>
        <p className="text-sm font-semibold text-stone-700">{shortName}</p>
        <p className="text-xs text-stone-400">{name}</p>
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
      <div className="mb-8 fade-in">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-stone-800">
          Clinical AI Dashboard
        </h1>
        <p className="text-stone-500 mt-2 max-w-xl">
          Privacy-preserving federated learning for medical image classification.
          Brain tumor MRI & chest X-ray analysis with 95.10% accuracy.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <StatCard icon="🎯" label="Model Accuracy" value="95.1%" sub="Weighted F1: 95.11%" color="text-emerald-600" />
        <StatCard icon="🧠" label="Parameters" value={modelParams} sub="ResNet-18 backbone" />
        <StatCard icon="📊" label="Classes" value={info?.num_classes || 6} sub="Brain tumor + Chest X-ray" color="text-teal-600" />
        <StatCard icon="🔒" label="Status" value={health?.model_loaded ? 'Online' : 'Offline'} sub={health?.model_loaded ? 'Model loaded & ready' : 'Model not loaded'} color={health?.model_loaded ? 'text-emerald-600' : 'text-red-500'} />
      </div>

      {/* Class Registry */}
      <div className="mb-10">
        <h2 className="text-lg font-bold text-stone-700 mb-4 flex items-center gap-2">
          <span className="w-1 h-6 bg-gradient-to-b from-teal-500 to-emerald-500 rounded-full" />
          Classification Classes
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {info && Object.entries(info.short_names).map(([key, name]) => (
            <ClassCard key={key} name={key} shortName={name} colour={info.risk_colours[key] || '#0d9488'} />
          ))}
          {!info && Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-16 rounded-xl bg-stone-100 animate-pulse" />
          ))}
        </div>
      </div>

      {/* Scan Modes */}
      {info && (
        <div className="mb-10">
          <h2 className="text-lg font-bold text-stone-700 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-rose-400 to-amber-400 rounded-full" />
            Scan Modes
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.entries(info.scan_modes).map(([mode, cfg]) => (
              <div key={mode} className="glass rounded-2xl p-5 card-lift fade-in">
                <h3 className="text-base font-bold text-stone-700 flex items-center gap-2">
                  <span className="text-xl">{cfg.icon}</span>{mode}
                </h3>
                <div className="mt-3 flex flex-wrap gap-2">
                  {cfg.class_keys.map(k => (
                    <span key={k} className="px-3 py-1 rounded-full text-xs font-medium bg-teal-50 text-teal-700 border border-teal-100">
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
        <div className="glass rounded-2xl p-6 fade-in">
          <h2 className="text-lg font-bold text-stone-700 mb-4 flex items-center gap-2">
            <span className="w-1 h-6 bg-gradient-to-b from-amber-500 to-orange-500 rounded-full" />
            System Info
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-stone-400 text-xs uppercase font-medium">OCR</p>
              <p className={`font-semibold ${health.ocr_available ? 'text-emerald-600' : 'text-stone-400'}`}>
                {health.ocr_available ? 'Available' : 'Unavailable'}
              </p>
            </div>
            <div>
              <p className="text-stone-400 text-xs uppercase font-medium">Max Upload</p>
              <p className="font-semibold text-stone-600">{health.max_upload_mb} MB</p>
            </div>
            <div>
              <p className="text-stone-400 text-xs uppercase font-medium">Feedback Total</p>
              <p className="font-semibold text-stone-600">{health.feedback_total}</p>
            </div>
            <div>
              <p className="text-stone-400 text-xs uppercase font-medium">Rate Limit</p>
              <p className="font-semibold text-stone-600">{health.security?.rate_limit_general}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
