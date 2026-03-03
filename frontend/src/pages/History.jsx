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
      <h1 className="text-2xl font-extrabold text-stone-800 mb-6 fade-in">
        Prediction History
      </h1>

      <div className="glass rounded-2xl overflow-hidden fade-in">
        {loading ? (
          <div className="p-12 text-center">
            <svg className="animate-spin h-8 w-8 mx-auto text-teal-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
          </div>
        ) : sessions.length === 0 ? (
          <div className="p-12 text-center text-stone-400">
            <div className="text-4xl mb-2 opacity-40">📭</div>
            <p className="text-sm">No prediction sessions found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-stone-50/60">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Time</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Prediction</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Confidence</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Scan Type</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-stone-500 uppercase">Session ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100">
                {sessions.map((s, i) => (
                  <tr key={s.session_id || i} className="hover:bg-teal-50/30 transition-colors">
                    <td className="px-5 py-3 text-stone-500 whitespace-nowrap">
                      {s.created_at ? new Date(s.created_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3">
                      <span className="font-semibold text-stone-700">
                        {s.ai_pred_key || '—'}
                      </span>
                    </td>
                    <td className="px-5 py-3">
                      {s.ai_confidence != null ? (
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-stone-100 rounded-full overflow-hidden">
                            <div className="h-full bg-teal-500 rounded-full" style={{ width: `${(s.ai_confidence * 100)}%` }} />
                          </div>
                          <span className="text-xs font-medium text-stone-500">{(s.ai_confidence * 100).toFixed(1)}%</span>
                        </div>
                      ) : '—'}
                    </td>
                    <td className="px-5 py-3 text-stone-500">{s.scan_type || '—'}</td>
                    <td className="px-5 py-3 text-stone-400 font-mono text-xs">{(s.session_id || '—').slice(0, 8)}...</td>
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
          <button onClick={() => setPage(p => p + 1)} disabled={sessions.length < LIMIT}
            className="px-4 py-1.5 rounded-lg text-sm font-medium bg-white border border-stone-200 text-stone-600 disabled:opacity-40 hover:bg-stone-50 transition">
            Next
          </button>
        </div>
      </div>
    </div>
  )
}
