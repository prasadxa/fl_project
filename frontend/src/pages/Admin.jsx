import { useState, useEffect } from 'react'
import { getAdminStats, getAdminFeedback, downloadCSV, downloadExcel } from '../utils/api'

export default function Admin() {
  const [stats, setStats] = useState(null)
  const [feedback, setFeedback] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const LIMIT = 20

  useEffect(() => {
    setLoading(true)
    Promise.all([
      getAdminStats().catch(() => null),
      getAdminFeedback({ limit: LIMIT, offset: page * LIMIT }).catch(() => ({ rows: [] })),
    ]).then(([s, f]) => {
      setStats(s)
      setFeedback(Array.isArray(f?.rows) ? f.rows : Array.isArray(f) ? f : [])
    }).finally(() => setLoading(false))
  }, [page])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white">⚙️ Admin Panel</h1>
        <div className="flex gap-2">
          <button onClick={downloadCSV}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-emerald-50 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 hover:bg-emerald-100 transition">
            📥 CSV
          </button>
          <button onClick={downloadExcel}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 transition">
            📊 Excel
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Sessions', value: stats.total_sessions ?? stats.total_predictions ?? 0, icon: '📊', color: 'text-blue-600' },
            { label: 'Feedback Count', value: stats.total_feedback ?? stats.feedback_count ?? 0, icon: '💬', color: 'text-purple-600' },
            { label: 'Overridden', value: stats.total_overridden ?? stats.overridden_count ?? 0, icon: '❌', color: 'text-red-500' },
            { label: 'Override Rate', value: stats.override_rate_pct != null ? `${stats.override_rate_pct.toFixed(1)}%` : stats.agreement_rate != null ? `${((1 - stats.agreement_rate) * 100).toFixed(1)}%` : '—', icon: '📈', color: 'text-amber-600' },
          ].map(s => (
            <div key={s.label} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5">
              <p className="text-xs font-medium text-slate-400 uppercase">{s.label}</p>
              <p className={`text-2xl font-bold mt-1 ${s.color}`}>{s.value}</p>
              <span className="text-lg">{s.icon}</span>
            </div>
          ))}
        </div>
      )}

      {/* Class Distribution */}
      {stats?.class_distribution && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 mb-8">
          <h2 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-4">Class Distribution</h2>
          <div className="space-y-2">
            {Object.entries(stats.class_distribution).map(([cls, cnt]) => {
              const total = Object.values(stats.class_distribution).reduce((a, b) => a + b, 0)
              const pct = total > 0 ? (cnt / total * 100) : 0
              return (
                <div key={cls} className="flex items-center gap-3">
                  <span className="text-xs font-medium text-slate-500 w-24 truncate">{cls}</span>
                  <div className="flex-1 h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-slate-400 w-16 text-right">{cnt} ({pct.toFixed(0)}%)</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Feedback Table */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        <h2 className="text-sm font-bold text-slate-700 dark:text-slate-300 px-6 py-4 border-b border-slate-200 dark:border-slate-800">
          Feedback Log
        </h2>
        {loading ? (
          <div className="p-12 text-center">
            <svg className="animate-spin h-8 w-8 mx-auto text-blue-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
          </div>
        ) : feedback.length === 0 ? (
          <div className="p-12 text-center text-slate-400 text-sm">No feedback entries yet</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-800/50">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Time</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Predicted</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Doctor</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Agreed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {feedback.map((fb, i) => (
                  <tr key={fb.id || i} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="px-5 py-3 text-slate-500 whitespace-nowrap">
                      {fb.timestamp ? new Date(fb.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3 font-medium text-slate-700 dark:text-slate-300">{fb.predicted_label || fb.predicted_class || '—'}</td>
                    <td className="px-5 py-3 text-slate-600 dark:text-slate-400">{fb.corrected_label || fb.confirmed_label || '—'}</td>
                    <td className="px-5 py-3">
                      {fb.doctor_agrees != null ? (
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          fb.doctor_agrees ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
                          : 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
                        }`}>
                          {fb.doctor_agrees ? 'Yes' : 'No'}
                        </span>
                      ) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex items-center justify-between px-5 py-3 bg-slate-50 dark:bg-slate-800/30 border-t border-slate-200 dark:border-slate-800">
          <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
            ← Prev
          </button>
          <span className="text-xs text-slate-400">Page {page + 1}</span>
          <button onClick={() => setPage(p => p + 1)} disabled={feedback.length < LIMIT}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
