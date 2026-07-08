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
  const [health, setHealth] = useState(null)

  useEffect(() => { getHealth().then(setHealth).catch(() => {}) }, [])

  return (
    <div className="min-h-screen flex flex-col">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:z-[100] px-4 py-2 bg-teal-600 text-white rounded-br-xl font-medium shadow-lg"
      >
        Skip to main content
      </a>

      {/* ── Navbar ── */}
      <nav className="sticky top-0 z-50 glass rounded-none"
        style={{ borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-600 to-emerald-600 flex items-center justify-center shadow-lg shadow-teal-600/20">
                <span className="text-white text-lg">🧠</span>
              </div>
              <span className="text-lg font-bold bg-gradient-to-r from-teal-700 to-emerald-600 bg-clip-text text-transparent">
                Tecnomate AI
              </span>
            </div>

            {/* Desktop nav */}
            <div className="hidden sm:flex items-center gap-1">
              {NAV.map(n => (
                <NavLink
                  key={n.to}
                  to={n.to}
                  end={n.to === '/'}
                  className={({ isActive }) =>
                    `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-teal-50 text-teal-700 shadow-sm'
                        : 'text-stone-500 hover:bg-white/60 hover:text-stone-700'
                    }`
                  }
                >
                  <span className="mr-1.5">{n.icon}</span>{n.label}
                </NavLink>
              ))}
            </div>

            {/* Status indicator */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-stone-400">
                {health?.model_loaded ? 'Online' : 'Offline'}
              </span>
              <div className={`w-2.5 h-2.5 rounded-full ${
                health?.model_loaded
                  ? 'bg-emerald-500 shadow-lg shadow-emerald-500/30'
                  : 'bg-red-400 animate-pulse'
              }`} />
            </div>
          </div>
        </div>

        {/* Mobile nav */}
        <div className="sm:hidden flex" style={{ borderTop: '1px solid rgba(0,0,0,0.04)' }}>
          {NAV.map(n => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === '/'}
              className={({ isActive }) =>
                `flex-1 text-center py-2.5 text-xs font-medium transition-all ${
                  isActive ? 'text-teal-700 bg-teal-50' : 'text-stone-400'
                }`
              }
            >
              <div className="text-base">{n.icon}</div>
              {n.label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* ── Page Content ── */}
      <main id="main-content" tabIndex={-1} className="flex-1 outline-none">
        <Routes>
          <Route path="/" element={<Dashboard health={health} />} />
          <Route path="/classify" element={<Classify />} />
          <Route path="/history" element={<History />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </main>

      {/* ── Footer ── */}
      <footer className="py-4 text-center text-xs text-stone-400"
        style={{ borderTop: '1px solid rgba(0,0,0,0.04)' }}>
        Tecnomate Clinical AI v2.0 — Privacy-preserving Federated Learning
      </footer>
    </div>
  )
}
