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
  const fontSize = size < 80 ? 14 : size < 100 ? 18 : 22

  return (
    <div className="flex flex-col items-center gap-1">
      {/* Gauge — value text absolutely centered inside the SVG */}
      <div className="relative flex-shrink-0" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox="0 0 100 100"
          style={{ transform: 'rotate(-90deg)' }}
          aria-label={`Health score: ${Math.round(clipped)}%`}
        >
          <circle cx="50" cy="50" r={RADIUS} fill="none" stroke="#e5e7eb" strokeWidth="10" />
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
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span style={{ fontSize, fontWeight: 700, color, lineHeight: 1 }}>{Math.round(clipped)}</span>
            <span style={{ fontSize: fontSize * 0.5, color: '#9ca3af', lineHeight: 1, alignSelf: 'flex-end', marginBottom: 2 }}>%</span>
          </div>
        )}
      </div>
      {label && <p className="text-xs text-gray-500 font-medium text-center">{label}</p>}
    </div>
  )
}

export default HealthGauge
