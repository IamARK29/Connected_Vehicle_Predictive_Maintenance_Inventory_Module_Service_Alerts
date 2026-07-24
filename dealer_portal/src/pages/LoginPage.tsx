import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { login } from '../api/client'

/* ── CSS keyframe animations injected once ─────────────────────────────────── */
const STYLES = `
  @keyframes ap-float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-12px); }
  }
  @keyframes ap-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  @keyframes ap-pulse-head {
    0%, 100% { opacity: 1; filter: brightness(1.2); }
    50%       { opacity: 0.55; filter: brightness(0.8); }
  }
  @keyframes ap-pulse-tail {
    0%, 100% { opacity: 0.95; }
    50%       { opacity: 0.35; }
  }
  @keyframes ap-speedline {
    0%   { opacity: 0;   transform: translateX(18px) scaleX(0.4); }
    35%  { opacity: 0.8; transform: translateX(0px)  scaleX(1); }
    65%  { opacity: 0.6; transform: translateX(-8px) scaleX(0.8); }
    100% { opacity: 0;   transform: translateX(-28px) scaleX(0.3); }
  }
  @keyframes ap-glow-ring {
    0%, 100% { opacity: 0.2; r: 5; }
    50%       { opacity: 0.7; r: 7; }
  }
  @keyframes ap-fade-in {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
`

/* ── Spoke helper ─────────────────────────────────────────────────────────── */
function Spokes({ cx, cy, inner = 10, outer = 25 }: { cx: number; cy: number; inner?: number; outer?: number }) {
  return (
    <>
      {[0, 72, 144, 216, 288].map(deg => {
        const r = (deg * Math.PI) / 180
        return (
          <line key={deg}
            x1={cx + inner * Math.cos(r)} y1={cy + inner * Math.sin(r)}
            x2={cx + outer * Math.cos(r)} y2={cy + outer * Math.sin(r)}
            stroke="#9ca3af" strokeWidth="3" strokeLinecap="round"
          />
        )
      })}
    </>
  )
}

/* ── Animated Car SVG ─────────────────────────────────────────────────────── */
function AnimatedCar() {
  const WF = { cx: 138, cy: 192 }   // front wheel centre
  const WR = { cx: 390, cy: 192 }   // rear wheel centre
  const WR_SIZE = 40                 // tyre radius

  return (
    <div className="relative w-full select-none"
      style={{ animation: 'ap-float 4.5s ease-in-out infinite' }}>

      {/* Speed lines – rendered behind the car */}
      <div className="absolute left-0 top-[42%] -translate-y-1/2 flex flex-col gap-2 pointer-events-none z-0">
        {[
          { w: 56, delay: '0s',    opacity: 0.65 },
          { w: 80, delay: '0.25s', opacity: 0.42 },
          { w: 40, delay: '0.5s',  opacity: 0.55 },
          { w: 64, delay: '0.7s',  opacity: 0.30 },
        ].map((l, i) => (
          <div key={i}
            className="h-px rounded-full"
            style={{
              width: l.w,
              opacity: l.opacity,
              background: 'linear-gradient(to right, #60a5fa, transparent)',
              animation: `ap-speedline 1.6s ease-in-out infinite`,
              animationDelay: l.delay,
            }}
          />
        ))}
      </div>

      <svg viewBox="0 0 540 230" xmlns="http://www.w3.org/2000/svg"
        className="relative z-10 w-full drop-shadow-2xl">
        <defs>
          <linearGradient id="ap-body" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#1d4ed8" />
            <stop offset="55%"  stopColor="#1e3a8a" />
            <stop offset="100%" stopColor="#0c1a3a" />
          </linearGradient>
          <linearGradient id="ap-glass" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"   stopColor="#bfdbfe" stopOpacity="0.88" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.50" />
          </linearGradient>
          <radialGradient id="ap-tyre" cx="42%" cy="35%" r="65%">
            <stop offset="0%"   stopColor="#374151" />
            <stop offset="70%"  stopColor="#1f2937" />
            <stop offset="100%" stopColor="#0d1117" />
          </radialGradient>
          <radialGradient id="ap-hub" cx="50%" cy="42%" r="58%">
            <stop offset="0%"   stopColor="#d1d5db" />
            <stop offset="100%" stopColor="#6b7280" />
          </radialGradient>
          <radialGradient id="ap-shadow" cx="50%" cy="50%" r="50%">
            <stop offset="0%"   stopColor="#000" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#000" stopOpacity="0" />
          </radialGradient>
          <filter id="ap-glow-h" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="ap-glow-t" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <linearGradient id="ap-shine" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#fff" stopOpacity="0.18" />
            <stop offset="60%"  stopColor="#fff" stopOpacity="0.04" />
            <stop offset="100%" stopColor="#fff" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Ground shadow */}
        <ellipse cx="264" cy="224" rx="232" ry="7" fill="url(#ap-shadow)" />

        {/* ── BODY ──────────────────────────────────────────────────────── */}
        {/*
          Compound path (fill-rule=evenodd):
          outer silhouette clockwise, then two wheel-arch arcs (CCW sweep=0)
          to punch the arch openings into the body.
        */}
        <path
          fillRule="evenodd"
          fill="url(#ap-body)"
          stroke="#2563eb"
          strokeWidth="0.6"
          d={`
            M 52,216
            Q 38,216 36,202
            L 36,170
            Q 36,152 52,142
            L 112,112
            L 150,64
            Q 158,52 172,49
            L 368,46
            Q 387,46 398,60
            L 426,108
            Q 433,119 433,132
            L 433,170
            Q 433,196 448,204
            L 456,216
            Q 441,198 ${WR.cx + WR_SIZE - 2},180
            A ${WR_SIZE},${WR_SIZE} 0 0 0 ${WR.cx - WR_SIZE + 2},180
            L ${WF.cx + WR_SIZE - 2},180
            A ${WR_SIZE},${WR_SIZE} 0 0 0 ${WF.cx - WR_SIZE + 2},180
            Q 88,196 68,206
            L 52,216
            Z
          `}
        />

        {/* Body shine overlay */}
        <path
          d="M 112,112 L 150,64 Q 158,52 172,49 L 368,46 Q 387,46 398,60 L 422,104"
          fill="url(#ap-shine)"
        />

        {/* Side crease highlight */}
        <path d="M 68,158 Q 252,150 433,154"
          stroke="#3b82f6" strokeWidth="1.2" fill="none" opacity="0.45" />
        <path d="M 68,158 Q 252,150 433,154"
          stroke="#93c5fd" strokeWidth="0.5" fill="none" opacity="0.35" />

        {/* Lower body panel line */}
        <path d="M 68,172 L 434,172"
          stroke="#1e3a8a" strokeWidth="1" fill="none" opacity="0.6" />

        {/* ── WINDOWS ────────────────────────────────────────────────────── */}
        {/* Windshield */}
        <path d="M 118,110 L 152,66 Q 160,54 174,51 L 213,50 L 213,110 Z"
          fill="url(#ap-glass)" stroke="#93c5fd" strokeWidth="0.8" />
        {/* Front side */}
        <path d="M 217,50 L 299,48 L 299,110 L 217,110 Z"
          fill="url(#ap-glass)" stroke="#93c5fd" strokeWidth="0.8" opacity="0.92" />
        {/* Rear side */}
        <path d="M 303,48 L 364,46 L 367,68 L 367,110 L 303,110 Z"
          fill="url(#ap-glass)" stroke="#93c5fd" strokeWidth="0.8" opacity="0.85" />
        {/* Rear quarter glass */}
        <path d="M 371,68 Q 396,68 399,76 L 420,106 L 371,110 Z"
          fill="url(#ap-glass)" stroke="#93c5fd" strokeWidth="0.8" opacity="0.72" />

        {/* Window pillars */}
        <rect x="213" y="49" width="4" height="63" rx="1.5" fill="#0a1628" />
        <rect x="299" y="47" width="4" height="65" rx="1.5" fill="#0a1628" />
        <rect x="365" y="44" width="4" height="68" rx="1.5" fill="#0a1628" />

        {/* Roof rail */}
        <path d="M 172,49 L 365,46" stroke="#60a5fa" strokeWidth="1.6" opacity="0.65" />

        {/* Glass highlight reflections */}
        <path d="M 126,80 L 140,62 L 158,60 L 148,78 Z" fill="#fff" opacity="0.10" />
        <path d="M 230,62 L 265,59 L 267,74 L 232,77 Z" fill="#fff" opacity="0.07" />

        {/* ── FRONT HEADLIGHT ────────────────────────────────────────────── */}
        <g filter="url(#ap-glow-h)"
          style={{ animation: 'ap-pulse-head 3s ease-in-out infinite' }}>
          <path d="M 36,148 Q 32,144 33,138 L 52,130 L 60,145 L 42,152 Z"
            fill="#fef3c7" />
          <path d="M 35,148 Q 31,144 33,139 L 50,132 L 57,145 Z"
            fill="#fde68a" />
        </g>
        {/* DRL strip */}
        <line x1="38" y1="160" x2="63" y2="152"
          stroke="#fbbf24" strokeWidth="2.8" strokeLinecap="round"
          filter="url(#ap-glow-h)"
          style={{ animation: 'ap-pulse-head 3s ease-in-out infinite' }} />

        {/* ── REAR TAILLIGHT ─────────────────────────────────────────────── */}
        <g filter="url(#ap-glow-t)"
          style={{ animation: 'ap-pulse-tail 3s ease-in-out infinite 0.6s' }}>
          <path d="M 433,122 L 436,108 Q 434,103 431,103 L 424,108 L 427,138 Z"
            fill="#dc2626" />
        </g>
        <line x1="426" y1="150" x2="434" y2="146"
          stroke="#ef4444" strokeWidth="2.8" strokeLinecap="round"
          filter="url(#ap-glow-t)" />

        {/* ── DOOR DETAILS ───────────────────────────────────────────────── */}
        <line x1="217" y1="110" x2="217" y2="180" stroke="#1d4ed8" strokeWidth="1" opacity="0.38" />
        <line x1="303" y1="110" x2="303" y2="180" stroke="#1d4ed8" strokeWidth="1" opacity="0.38" />
        <rect x="232" y="147" width="26" height="5" rx="2.5" fill="#1e293b" stroke="#334155" strokeWidth="0.5" />
        <rect x="318" y="147" width="26" height="5" rx="2.5" fill="#1e293b" stroke="#334155" strokeWidth="0.5" />

        {/* Rearview mirror */}
        <path d="M 154,88 L 163,88 L 161,100 L 150,100 Z" fill="#0a1628" stroke="#1e3a8a" strokeWidth="0.8" />

        {/* Rear wing/spoiler hint */}
        <rect x="368" y="43" width="20" height="4" rx="2" fill="#1e40af" stroke="#3b82f6" strokeWidth="0.5" />

        {/* Front bumper detail */}
        <path d="M 36,185 Q 40,192 50,194 L 95,194" stroke="#1d4ed8" strokeWidth="1" fill="none" opacity="0.5" />
        {/* Rear bumper detail */}
        <path d="M 456,190 Q 452,194 445,196 L 430,196" stroke="#1d4ed8" strokeWidth="1" fill="none" opacity="0.5" />

        {/* ── FRONT WHEEL ────────────────────────────────────────────────── */}
        <g style={{ transformBox: 'fill-box', transformOrigin: 'center', animation: 'ap-spin 2.8s linear infinite' } as React.CSSProperties}>
          <circle cx={WF.cx} cy={WF.cy} r={WR_SIZE} fill="#0d1117" stroke="#374151" strokeWidth="1.5" />
          <circle cx={WF.cx} cy={WF.cy} r={28} fill="url(#ap-tyre)" />
          <Spokes cx={WF.cx} cy={WF.cy} />
          {/* Brake caliper hint */}
          <path d={`M ${WF.cx - 4},${WF.cy - 28} L ${WF.cx + 4},${WF.cy - 28} L ${WF.cx + 4},${WF.cy - 22} L ${WF.cx - 4},${WF.cy - 22} Z`}
            fill="#dc2626" opacity="0.7" />
          <circle cx={WF.cx} cy={WF.cy} r={9} fill="#1f2937" stroke="#4b5563" strokeWidth="1" />
          <circle cx={WF.cx} cy={WF.cy} r={4.5} fill="url(#ap-hub)" />
        </g>
        {/* Wheel arch liner shadow */}
        <path d={`M ${WF.cx - WR_SIZE + 2},180 A ${WR_SIZE},${WR_SIZE} 0 0 1 ${WF.cx + WR_SIZE - 2},180`}
          stroke="#060d1f" strokeWidth="2.5" fill="none" />

        {/* ── REAR WHEEL ─────────────────────────────────────────────────── */}
        <g style={{ transformBox: 'fill-box', transformOrigin: 'center', animation: 'ap-spin 2.8s linear infinite' } as React.CSSProperties}>
          <circle cx={WR.cx} cy={WR.cy} r={WR_SIZE} fill="#0d1117" stroke="#374151" strokeWidth="1.5" />
          <circle cx={WR.cx} cy={WR.cy} r={28} fill="url(#ap-tyre)" />
          <Spokes cx={WR.cx} cy={WR.cy} />
          <path d={`M ${WR.cx - 4},${WR.cy - 28} L ${WR.cx + 4},${WR.cy - 28} L ${WR.cx + 4},${WR.cy - 22} L ${WR.cx - 4},${WR.cy - 22} Z`}
            fill="#dc2626" opacity="0.7" />
          <circle cx={WR.cx} cy={WR.cy} r={9} fill="#1f2937" stroke="#4b5563" strokeWidth="1" />
          <circle cx={WR.cx} cy={WR.cy} r={4.5} fill="url(#ap-hub)" />
        </g>
        <path d={`M ${WR.cx - WR_SIZE + 2},180 A ${WR_SIZE},${WR_SIZE} 0 0 1 ${WR.cx + WR_SIZE - 2},180`}
          stroke="#060d1f" strokeWidth="2.5" fill="none" />
      </svg>
    </div>
  )
}

/* ── Login form ───────────────────────────────────────────────────────────── */
function CredentialHint({ role, user, password, color }: { role: string; user: string; password: string; color: string }) {
  return (
    <div className={`flex items-center justify-between px-3 py-2 rounded-lg border ${color} text-xs`}>
      <span className="font-semibold text-gray-700">{role}</span>
      <span className="font-mono text-gray-500">{user} / {password}</span>
    </div>
  )
}

/* ── Page ─────────────────────────────────────────────────────────────────── */
export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await login(username, password)
      localStorage.setItem('ap_user',        username)
      localStorage.setItem('ap_role',        data.role ?? 'DEALER')
      localStorage.setItem('ap_dealer_code', data.dealer_code ?? 'ALL')
      navigate('/')
    } catch {
      setError('Invalid username or password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <style>{STYLES}</style>
      <div className="min-h-screen flex overflow-hidden bg-slate-950">

        {/* ── LEFT PANEL ─────────────────────────────────────────────────── */}
        <div className="hidden lg:flex lg:w-[58%] flex-col relative overflow-hidden"
          style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 35%, #0c1a3a 60%, #0f172a 100%)' }}>

          {/* Background grid lines (subtle) */}
          <div className="absolute inset-0 opacity-[0.04]"
            style={{ backgroundImage: 'linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px)', backgroundSize: '48px 48px' }} />

          {/* Glow orbs */}
          <div className="absolute top-1/3 left-1/4 w-80 h-80 rounded-full bg-blue-600/10 blur-3xl pointer-events-none" />
          <div className="absolute bottom-1/4 right-1/3 w-64 h-64 rounded-full bg-indigo-500/8 blur-3xl pointer-events-none" />

          {/* Top branding */}
          <div className="relative z-10 px-12 pt-10 flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-900/50">
              <svg viewBox="0 0 24 24" className="w-6 h-6 fill-white">
                <path d="M12 2C8 2 4 4 3 8L2 12h3l1-3h12l1 3h3l-1-4C20 4 16 2 12 2zm-4 4h8l1 2H7l1-2zM4 14v6h3v-2h10v2h3v-6H4zm2 2h12v2H6v-2z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-white text-xl font-bold tracking-tight">AutoPredict</h1>
              <p className="text-blue-400 text-xs font-medium">Connected Vehicle · Predictive Maintenance</p>
            </div>
          </div>

          {/* Car animation + headline */}
          <div className="relative z-10 flex-1 flex flex-col items-center justify-center px-8 -mt-6">
            <div className="w-full max-w-xl">
              <AnimatedCar />
            </div>

            <div className="mt-6 text-center" style={{ animation: 'ap-fade-in 0.8s ease-out 0.3s both' }}>
              <h2 className="text-3xl font-bold text-white leading-tight">
                Predict before it breaks.
              </h2>
              <p className="text-blue-300 text-sm mt-2 max-w-sm mx-auto leading-relaxed">
                AI-driven maintenance intelligence for your entire connected vehicle fleet.
              </p>
            </div>
          </div>

          {/* Stats row */}
          <div className="relative z-10 px-10 pb-10 grid grid-cols-3 gap-4"
            style={{ animation: 'ap-fade-in 0.8s ease-out 0.6s both' }}>
            {[
              { val: '12',    label: 'ML Models',     icon: '🧠' },
              { val: '8',     label: 'Dealer Sites',  icon: '🏭' },
              { val: '<2ms',  label: 'Inference',     icon: '⚡' },
            ].map(s => (
              <div key={s.label} className="bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-center backdrop-blur-sm">
                <p className="text-lg">{s.icon}</p>
                <p className="text-white text-lg font-bold tabular-nums">{s.val}</p>
                <p className="text-blue-400 text-xs font-medium">{s.label}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── RIGHT PANEL ────────────────────────────────────────────────── */}
        <div className="flex-1 flex items-center justify-center px-6 py-12 bg-white">
          <div className="w-full max-w-sm" style={{ animation: 'ap-fade-in 0.6s ease-out both' }}>

            {/* Mobile-only logo */}
            <div className="lg:hidden flex items-center gap-3 mb-8">
              <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-5 h-5 fill-white">
                  <path d="M12 2C8 2 4 4 3 8L2 12h3l1-3h12l1 3h3l-1-4C20 4 16 2 12 2zm-4 4h8l1 2H7l1-2zM4 14v6h3v-2h10v2h3v-6H4zm2 2h12v2H6v-2z"/>
                </svg>
              </div>
              <div>
                <p className="font-bold text-gray-900">AutoPredict</p>
                <p className="text-xs text-gray-500">Predictive Maintenance</p>
              </div>
            </div>

            {/* Heading */}
            <div className="mb-8">
              <h2 className="text-2xl font-bold text-gray-900">Welcome back</h2>
              <p className="text-gray-500 text-sm mt-1">Sign in to your portal</p>
            </div>

            {/* Error */}
            {error && (
              <div className="mb-5 flex items-start gap-2.5 p-3.5 bg-red-50 border border-red-200 rounded-xl">
                <span className="text-red-500 text-base leading-none mt-0.5">⚠</span>
                <p className="text-sm text-red-700 font-medium">{error}</p>
              </div>
            )}

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1.5">Username</label>
                <input
                  data-testid="login-username"
                  type="text"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  className="w-full border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow placeholder-gray-400"
                  placeholder="dealer / oem / admin"
                  required
                  autoFocus
                />
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1.5">Password</label>
                <input
                  data-testid="login-password"
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow placeholder-gray-400"
                  placeholder="••••••••"
                  required
                />
              </div>
              <button
                data-testid="login-submit"
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 text-white rounded-xl py-3 text-sm font-semibold hover:bg-blue-700 active:scale-[0.98] disabled:opacity-50 transition-all shadow-md shadow-blue-600/20 mt-2"
              >
                {loading
                  ? <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                      </svg>
                      Signing in…
                    </span>
                  : 'Sign In →'
                }
              </button>
            </form>

            {/* Demo credentials */}
            <div className="mt-8">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Demo credentials</p>
              <div className="space-y-2">
                <CredentialHint role="Dealer"    user="dealer"  password="dealer123" color="border-green-200 bg-green-50/50" />
                <CredentialHint role="OEM"       user="oem"     password="oem123"    color="border-purple-200 bg-purple-50/50" />
                <CredentialHint role="Admin"     user="admin"   password="admin123"  color="border-blue-200 bg-blue-50/50" />
              </div>
            </div>

            <p className="mt-8 text-center text-xs text-gray-400">
              AutoPredict v2.0 · Connected Vehicle Platform ©{new Date().getFullYear()}
            </p>
          </div>
        </div>
      </div>
    </>
  )
}
