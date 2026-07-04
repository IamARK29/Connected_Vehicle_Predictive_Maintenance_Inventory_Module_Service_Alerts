interface Props {
  value: number        // 0–100
  size?: number        // px (default 120)
  label?: string
  showValue?: boolean
}

const RADIUS = 40
const CIRC   = 2 * Math.PI * RADIUS   // ≈ 251.33

function scoreColor(v: number) {
  if (v >= 80) return '#16a34a'
  if (v >= 60) return '#d97706'
  if (v >= 40) return '#ea580c'
  return '#dc2626'
}

export function HealthGauge({ value, size = 120, label, showValue = true }: Props) {
  const clipped = Math.max(0, Math.min(100, value))
  const offset  = CIRC * (1 - clipped / 100)
  const color   = scoreColor(clipped)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        style={{ transform: 'rotate(-90deg)' }}
        aria-label={`Health score: ${Math.round(clipped)}%`}
      >
        {/* Track */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="10"
        />
        {/* Fill */}
        <circle
          cx="50" cy="50" r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.7s ease, stroke 0.3s' }}
        />
      </svg>
      {showValue && (
        <div className="-mt-1 text-center" style={{ marginTop: `-${size * 0.56}px`, lineHeight: 1 }}>
          <span className="text-2xl font-bold" style={{ color }}>{Math.round(clipped)}</span>
          <span className="text-xs text-gray-400">%</span>
        </div>
      )}
      {label && <p className="text-xs text-gray-500 font-medium mt-1">{label}</p>}
    </div>
  )
}

export default HealthGauge
