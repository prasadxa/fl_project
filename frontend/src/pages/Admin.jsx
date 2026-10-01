import { useState, useEffect } from 'react'
import { getAdminStats, getAdminFeedback, downloadCSV, downloadExcel, setAdminAuth } from '../utils/api'

export default function Admin() {
  const [stats, setStats] = useState(null)
  const [feedback, setFeedback] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [needsAuth, setNeedsAuth] = useState(false)
  const [user, setUser] = useState('')
  const [pass, setPass] = useState('')
  const [retryCount, setRetryCount] = useState(0)
  const LIMIT = 20

  useEffect(() => {
    setLoading(true)
    Promise.all([
      getAdminStats().catch(err => err),
      getAdminFeedback({ limit: LIMIT, offset: page * LIMIT }).catch(err => err),
    ]).then(([s, f]) => {
      if (s?.errorCode === 'UNAUTHORIZED' || f?.errorCode === 'UNAUTHORIZED') {
        setNeedsAuth(true)
        setLoading(false)
        return
      }
      setStats(s?.errorCode ? null : s)
      const rows = f?.errorCode ? [] : f?.rows
      f = f?.errorCode ? [] : f
      setFeedback(Array.isArray(rows) ? rows : Array.isArray(f) ? f : [])
    }).finally(() => setLoading(false))
  }, [page, retryCount])

  if (needsAuth) {
    return (
      <div className="max-w-md mx-auto mt-12 p-6 glass rounded-2xl">
        <h2 className="text-xl font-bold mb-4 text-stone-800">Admin Authentication Required</h2>
        <form onSubmit={(e) => {
          e.preventDefault()
          setAdminAuth(user, pass)
          setNeedsAuth(false)
          setRetryCount(c => c + 1)
        }} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-stone-600 mb-1">Username</label>
            <input type="text" value={user} onChange={e => setUser(e.target.value)} required
              className="w-full px-3 py-2 border border-stone-200 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-teal-500" />
          </div>
          <div>
            <label className="block text-sm font-medium text-stone-600 mb-1">Password</label>
            <input type="password" value={pass} onChange={e => setPass(e.target.value)} required
              className="w-full px-3 py-2 border border-stone-200 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-teal-500" />
          </div>
          <button type="submit" className="w-full py-2 bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-xl transition">
            Login
          </button>
        </form>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6 fade-in">
        <h1 className="text-2xl font-extrabold text-stone-800">Admin Panel</h1>
        <div className="flex gap-2">
          <button onClick={downloadCSV}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition">
            CSV Export
          </button>
          <button onClick={downloadExcel}
            className="px-4 py-2 rounded-xl text-xs font-medium bg-teal-50 text-teal-700 border border-teal-200 hover:bg-teal-100 transition">
            Excel Export
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Total Sessions', value: stats.total_sessions ?? stats.total_predictions ?? 0, icon: '📊', color: 'text-teal-600' },
            { label: 'Feedback Count', value: stats.total_feedback ?? stats.feedback_count ?? 0, icon: '💬', color: 'text-emerald-600' },
            { label: 'Overridden', value: stats.total_overridden ?? stats.overridden_count ?? 0, icon: '❌', color: 'text-red-500' },
            { label: 'Override Rate', value: stats.override_rate_pct != null ? `${stats.override_rate_pct.toFixed(1)}%` : stats.agreement_rate != null ? `${((1 - stats.agreement_rate) * 100).toFixed(1)}%` : '—', icon: '📈', color: 'text-amber-600' },
          ].map(s => (
            <div key={s.label} className="glass rounded-2xl p-5 card-lift fade-in">
              <p className="text-xs font-medium text-stone-400 uppercase">{s.label}</p>
              <p className={`text-2xl font-bold mt-1 ${s.color}`}>{s.value}</p>
              <span className="text-lg">{s.icon}</span>
            </div>
          ))}
        </div>
      )}

      {/* Class Distribution */}
      {stats?.feedback_by_class && (
        <div className="glass rounded-2xl p-6 mb-8 fade-in">
          <h2 className="text-sm font-bold text-stone-600 mb-4">Class Distribution</h2>
          <div className="space-y-2">
            {Object.entries(stats.feedback_by_class).map(([cls, cnt]) => {
              const total = Object.values(stats.feedback_by_class).reduce((a, b) => a + b, 0)
              const pct = total > 0 ? (cnt / total * 100) : 0
              return (
                <div key={cls} className="flex items-center gap-3">
                  <span className="text-xs font-medium text-stone-500 w-24 truncate">{cls}</span>
                  <div className="flex-1 h-2 bg-stone-100 rounded-full overflow-hidden">
                    <div className="h-full bg-teal-500 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="text-xs text-stone-400 w-16 text-right">{cnt} ({pct.toFixed(0)}%)</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Feedback Table */}
      <div className="glass rounded-2xl overflow-hidden fade-in">
        <h2 className="text-sm font-bold text-stone-600 px-6 py-4"
          style={{ borderBottom: '1px solid rgba(0,0,0,0.04)' }}>
          Feedback Log
        </h2>
        {loading ? (
          <div className="p-12 text-center">
            <svg className="animate-spin h-8 w-8 mx-auto text-teal-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
          </div>
        ) : feedback.length === 0 ? (
          <div className="p-12 text-center text-stone-400 text-sm">No feedback entries yet</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-stone-50/60">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Time</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Predicted</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Doctor</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Agreed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100">
                {feedback.map((fb, i) => (
                  <tr key={fb.id || i} className="hover:bg-teal-50/30 transition-colors">
                    <td className="px-5 py-3 text-stone-500 whitespace-nowrap">
                      {fb.timestamp ? new Date(fb.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3 font-medium text-stone-700">{fb.ai_predicted_key || '—'}</td>
                    <td className="px-5 py-3 text-stone-600">{fb.chosen_label || fb.chosen_key || '—'}</td>
                    <td className="px-5 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        !fb.overridden
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : 'bg-red-50 text-red-700 border border-red-200'
                      }`}>
                        {fb.overridden ? 'No' : 'Yes'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between px-5 py-3 bg-stone-50/40"
          style={{ borderTop: '1px solid rgba(0,0,0,0.04)' }}>
          <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white border border-stone-200 text-stone-600 disabled:opacity-40 hover:bg-stone-50 transition">
            Prev
          </button>
          <span className="text-xs text-stone-400">Page {page + 1}</span>
          <button onClick={() => setPage(p => p + 1)} disabled={feedback.length < LIMIT}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white border border-stone-200 text-stone-600 disabled:opacity-40 hover:bg-stone-50 transition">
            Next
          </button>
        </div>
      </div>
    </div>
  )
}
