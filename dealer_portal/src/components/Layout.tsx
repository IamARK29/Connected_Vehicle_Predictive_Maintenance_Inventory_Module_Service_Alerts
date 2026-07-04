import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { logout } from '../api/client'

const NAV = [
  { to: '/',             label: 'Dashboard',    emoji: '📊', exact: true },
  { to: '/alerts',       label: 'Alerts',       emoji: '🔔' },
  { to: '/service-bay',  label: 'Service Bay',  emoji: '🔧' },
  { to: '/inventory',    label: 'Inventory',    emoji: '📦' },
  { to: '/driver-scores',label: 'Driver Scores',emoji: '🏆' },
  { to: '/workflows',    label: 'Workflows',    emoji: '🤖' },
  { to: '/upload',       label: 'Data Upload',  emoji: '📤' },
]

export default function Layout() {
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  const dealerCode = localStorage.getItem('ap_dealer_code') ?? import.meta.env.VITE_DEALER_CODE ?? 'DL001'

  return (
    <div className="flex h-screen bg-slate-50">
      <aside className="w-60 bg-slate-900 flex flex-col shrink-0">
        <div className="px-5 py-5 border-b border-slate-700">
          <h1 className="text-white text-lg font-bold tracking-tight">AutoPredict</h1>
          <p className="text-slate-400 text-xs mt-0.5">Dealer Portal · {dealerCode}</p>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          {NAV.map(({ to, label, emoji, exact }) => (
            <NavLink
              key={to}
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
              <span className="text-base leading-none">{emoji}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-3 py-4 border-t border-slate-700">
          <div className="flex items-center gap-2 px-3 py-2 mb-1">
            <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold">
              {(localStorage.getItem('ap_role') ?? 'D')[0]}
            </div>
            <div className="min-w-0">
              <p className="text-white text-xs font-medium truncate">{localStorage.getItem('ap_user') ?? 'dealer'}</p>
              <p className="text-slate-500 text-xs capitalize">{(localStorage.getItem('ap_role') ?? 'DEALER').toLowerCase()}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2 w-full rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <span>🚪</span>
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
