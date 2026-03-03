import { useState, useEffect } from 'react'
import { getAdminSessions } from '../utils/api'

export default function History() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const LIMIT = 20

  useEffect(() => {
    setLoading(true)
    getAdminSessions({ limit: LIMIT, offset: page * LIMIT })
      .then(data => setSessions(data.sessions || data || []))
      .catch(() => setSessions([]))
      .finally(() => setLoading(false))
  }, [page])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white mb-6 flex items-center gap-2">
        📋 Prediction History
      </h1>

      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden">
        {loading ? (
          <div className="p-12 text-center">
            <svg className="animate-spin h-8 w-8 mx-auto text-blue-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
          </div>
        ) : sessions.length === 0 ? (
          <div className="p-12 text-center text-slate-400">
            <div className="text-4xl mb-2 opacity-40">📭</div>
            <p className="text-sm">No prediction sessions found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 dark:bg-slate-800/50">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Time</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Prediction</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Confidence</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Scan Type</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase">Session ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {sessions.map((s, i) => (
                  <tr key={s.session_id || i} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                    <td className="px-5 py-3 text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {s.timestamp ? new Date(s.timestamp).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3">
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {s.predicted_class || s.short_name || '—'}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      {s.confidence != null ? (
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${(s.confidence * 100)}%` }} />
                          </div>
                          <span className="text-xs font-medium text-slate-600 dark:text-slate-400">{(s.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ) : '—'}
                    </td>
                    <td className="px-5 py-3 text-slate-500 dark:text-slate-400">{s.scan_type || '—'}</td>
                    <td className="px-5 py-3 text-slate-400 font-mono text-xs">{(s.session_id || '—').slice(0, 8)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between px-5 py-3 bg-slate-50 dark:bg-slate-800/30 border-t border-slate-200 dark:border-slate-800">
          <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
            ← Prev
          </button>
          <span className="text-xs text-slate-400">Page {page + 1}</span>
          <button onClick={() => setPage(p => p + 1)} disabled={sessions.length < LIMIT}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-700 transition">
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
