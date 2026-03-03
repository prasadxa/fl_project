import { Routes, Route, NavLink } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard'
import Classify from './pages/Classify'
import History from './pages/History'
import Admin from './pages/Admin'
import { getHealth } from './utils/api'

const NAV = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/classify', label: 'Classify', icon: '🔬' },
  { to: '/history', label: 'History', icon: '📋' },
  { to: '/admin', label: 'Admin', icon: '⚙️' },
]

export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('theme') === 'dark')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  useEffect(() => { getHealth().then(setHealth).catch(() => {}) }, [])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                <span className="text-white text-lg">🧠</span>
              </div>
              <span className="text-lg font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Tecnomate AI
              </span>
            </div>
            <div className="hidden sm:flex items-center gap-1">
              {NAV.map(n => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.to === '/'}
                  className={({ isActive }) =>
                    `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-blue-50 dark:bg-blue-950 text-blue-600 dark:text-blue-400'
                        : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
                    }`
                  }
                >
                  <span className="mr-1.5">{n.icon}</span>{n.label}
                </NavLink>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <div className={`w-2.5 h-2.5 rounded-full ${health?.model_loaded ? 'bg-emerald-500 shadow-lg shadow-emerald-500/30' : 'bg-red-500 animate-pulse'}`} title={health?.model_loaded ? 'Model loaded' : 'Model offline'} />
              <button
                onClick={() => setDark(d => !d)}
                className="p-2 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              >
                {dark ? '☀️' : '🌙'}
              </button>
            </div>
          </div>
        </div>
        {/* Mobile nav */}
        <div className="sm:hidden flex border-t border-slate-200 dark:border-slate-800">
          {NAV.map(n => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === '/'}
              className={({ isActive }) =>
                `flex-1 text-center py-2.5 text-xs font-medium transition-all ${
                  isActive ? 'text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950' : 'text-slate-500'
                }`
              }
            >
              <div className="text-base">{n.icon}</div>
              {n.label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* Page Content */}
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Dashboard health={health} />} />
          <Route path="/classify" element={<Classify />} />
          <Route path="/history" element={<History />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="py-4 text-center text-xs text-slate-400 dark:text-slate-600 border-t border-slate-200 dark:border-slate-800">
        Tecnomate Clinical AI v2.0 — Privacy-preserving Federated Learning
      </footer>
    </div>
  )
}
