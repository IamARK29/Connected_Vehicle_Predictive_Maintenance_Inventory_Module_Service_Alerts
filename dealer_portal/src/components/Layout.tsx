import { useEffect } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { logout } from '../api/client'
import { useJob, type ToastMsg } from '../context/JobContext'

function GlobalToast({ t, onClose }: { t: ToastMsg; onClose: () => void }) {
  useEffect(() => {
    const id = setTimeout(onClose, 6000)
    return () => clearTimeout(id)
  }, [t.id, onClose])

  const bg   = t.type === 'success' ? 'bg-green-600' : t.type === 'error' ? 'bg-red-600' : 'bg-blue-600'
  const icon = t.type === 'success' ? '✓' : t.type === 'error' ? '✗' : 'i'

  return (
    <div className={`fixed top-6 right-6 z-50 ${bg} text-white px-5 py-3 rounded-xl shadow-2xl max-w-md flex items-start gap-3`}>
      <span className="font-bold leading-none mt-0.5 w-4 text-center">{icon}</span>
      <p className="flex-1 text-sm font-medium">{t.message}</p>
      <button onClick={onClose} className="text-white/70 hover:text-white text-lg leading-none">&times;</button>
    </div>
  )
}

function GlobalProgressBar() {
  const { job, stopJob } = useJob()
  const isRunning = job.phase === 'generating' || job.phase === 'training'
  if (!isRunning) return null

  return (
    <div className="fixed bottom-6 right-6 z-40 bg-white border border-gray-200 shadow-xl rounded-xl p-4 w-80">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-700">
          {job.phase === 'generating' ? 'Generating data...' : 'Training models...'}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-blue-600 tabular-nums">{job.pct}%</span>
          <button
            onClick={stopJob}
            className="text-xs text-red-500 hover:text-red-700 font-medium px-1.5 py-0.5 rounded border border-red-200 hover:border-red-400 transition-colors"
          >
            Stop
          </button>
        </div>
      </div>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${job.phase === 'training' ? 'bg-green-600' : 'bg-blue-600'}`}
          style={{ width: `${job.pct}%` }}
        />
      </div>
      <p className="text-xs text-gray-400 mt-1.5 truncate">{job.message}</p>
    </div>
  )
}

const DEALER_NAV = [
  { to: '/',              label: 'Dashboard',     icon: '📊', exact: true },
  { to: '/alerts',        label: 'Alerts',        icon: '🔔' },
  { to: '/service-bay',   label: 'Service Bay',   icon: '🔧' },
  { to: '/inventory',     label: 'Inventory',     icon: '📦' },
  { to: '/driver-scores', label: 'Driver Scores', icon: '🏆' },
  { to: '/workflows',     label: 'Workflows',     icon: '🤖' },
]

const OEM_NAV = [
  { to: '/oem/fleet',     label: 'Fleet Intelligence', icon: '🌐' },
  { to: '/oem/models',    label: 'Model Health',       icon: '🧠' },
  { to: '/oem/eda',       label: 'EDA Explorer',       icon: '🔬' },
  { to: '/oem/whatif',    label: 'What-If Simulator',  icon: '⚡' },
  { to: '/oem/retrain',   label: 'Retrain Control',    icon: '🔄' },
  { to: '/upload',        label: 'Data Upload',        icon: '📤' },
]

const ADMIN_NAV = [
  { to: '/admin/users', label: 'User Management', icon: '👥' },
]

function NavItem({ to, label, icon, exact }: { to: string; label: string; icon: string; exact?: boolean }) {
  return (
    <NavLink
      to={to}
      end={exact}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-slate-400 hover:text-white hover:bg-slate-800'
        }`
      }
    >
      <span className="text-base leading-none w-5 text-center">{icon}</span>
      {label}
    </NavLink>
  )
}

export default function Layout() {
  const navigate = useNavigate()
  const { toast, dismissToast } = useJob()
  const role = localStorage.getItem('ap_role') ?? 'DEALER'
  const dealerCode = localStorage.getItem('ap_dealer_code') ?? 'DL001'
  const username = localStorage.getItem('ap_user') ?? 'user'
  const isOem = role === 'OEM' || role === 'ADMIN'

  const handleLogout = () => {
    logout()
    localStorage.removeItem('ap_dealer_code')
    navigate('/login')
  }

  const portalLabel = role === 'OEM' ? 'OEM Portal' : role === 'ADMIN' ? 'Admin Portal' : `Dealer · ${dealerCode}`

  return (
    <div className="flex h-screen bg-slate-50">
      {toast && <GlobalToast t={toast} onClose={dismissToast} />}
      <GlobalProgressBar />
      <aside className="w-60 bg-slate-900 flex flex-col shrink-0">
        <div className="px-5 py-5 border-b border-slate-700">
          <h1 className="text-white text-lg font-bold tracking-tight">AutoPredict</h1>
          <p className="text-slate-400 text-xs mt-0.5">{portalLabel}</p>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {/* Dealer section — always visible */}
          {isOem && (
            <p className="px-3 pt-1 pb-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Dealer View
            </p>
          )}
          {DEALER_NAV.map(item => (
            <NavItem key={item.to} {...item} />
          ))}

          {/* OEM section — only for OEM / ADMIN */}
          {isOem && (
            <>
              <div className="my-3 border-t border-slate-700" />
              <p className="px-3 pb-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                OEM Intelligence
              </p>
              {OEM_NAV.map(item => (
                <NavItem key={item.to} {...item} />
              ))}
            </>
          )}

          {/* Admin section — only for ADMIN */}
          {role === 'ADMIN' && (
            <>
              <div className="my-3 border-t border-slate-700" />
              <p className="px-3 pb-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Admin
              </p>
              {ADMIN_NAV.map(item => (
                <NavItem key={item.to} {...item} />
              ))}
            </>
          )}
        </nav>

        <div className="px-3 py-4 border-t border-slate-700">
          <div className="flex items-center gap-2 px-3 py-2 mb-1">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-white text-xs font-bold ${
              isOem ? 'bg-purple-600' : 'bg-blue-600'
            }`}>
              {username[0]?.toUpperCase()}
            </div>
            <div className="min-w-0">
              <p className="text-white text-xs font-medium truncate">{username}</p>
              <p className="text-slate-500 text-xs capitalize">{role.toLowerCase()}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2 w-full rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <span className="w-5 text-center">🚪</span>
            Sign out
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
