interface Props {
  value: number      // 0–100
  size?: number      // px, default 120
  label?: string
  showValue?: boolean
}

// ── colour by score ───────────────────────────────────────────────────────────
function scoreColor(v: number) {
  if (v >= 80) return '#16a34a'
  if (v >= 60) return '#d97706'
  if (v >= 40) return '#ea580c'
  return '#dc2626'
}

// ── arc maths ─────────────────────────────────────────────────────────────────
const VB   = 120          // viewBox square side
const CX   = 60           // circle centre x
const CY   = 60           // circle centre y
const R    = 44           // arc radius
const SW   = 10           // stroke width
const START_DEG = 135     // 7-o'clock (bottom-left)
const SPAN_DEG  = 270     // opens at the bottom

const toRad = (d: number) => (d * Math.PI) / 180
const pt    = (deg: number) => ({
  x: CX + R * Math.cos(toRad(deg)),
  y: CY + R * Math.sin(toRad(deg)),
})

// ── component ─────────────────────────────────────────────────────────────────
export function HealthGauge({ value, size = 120, label, showValue = true }: Props) {
  const pct    = Math.max(0, Math.min(100, value))
  const color  = scoreColor(pct)

  // Background track: full 270° arc
  const s   = pt(START_DEG)
  const te  = pt(START_DEG + SPAN_DEG)
  const trackPath = `M ${s.x.toFixed(2)},${s.y.toFixed(2)} A ${R},${R} 0 1 1 ${te.x.toFixed(2)},${te.y.toFixed(2)}`

  // Progress arc
  const progressDeg = SPAN_DEG * pct / 100
  const pe          = pt(START_DEG + progressDeg)
  const largeArc    = progressDeg > 180 ? 1 : 0
  const progressPath = progressDeg > 0.1
    ? `M ${s.x.toFixed(2)},${s.y.toFixed(2)} A ${R},${R} 0 ${largeArc} 1 ${pe.x.toFixed(2)},${pe.y.toFixed(2)}`
    : null

  // Responsive font sizes (in SVG units, not px)
  const numSvg = size < 80 ? 20 : size < 110 ? 24 : 28
  const pctSvg = numSvg * 0.44

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${VB} ${VB}`}
        aria-label={`Health score ${Math.round(pct)}%`}
      >
        {/* Background track */}
        <path
          d={trackPath}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={SW}
          strokeLinecap="round"
        />

        {/* Coloured progress arc */}
        {progressPath && (
          <path
            d={progressPath}
            fill="none"
            stroke={color}
            strokeWidth={SW}
            strokeLinecap="round"
            style={{ transition: 'all 0.75s cubic-bezier(.4,0,.2,1)' }}
          />
        )}

        {/* Score number + % — fully inside SVG, no DOM overlap */}
        {showValue && (
          <text
            x={CX}
            y={CY + numSvg * 0.18}   // optical centre (slightly below true centre)
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fontFamily: 'system-ui,-apple-system,sans-serif' }}
          >
            <tspan
              fontSize={numSvg}
              fontWeight="700"
              fill={color}
            >
              {Math.round(pct)}
            </tspan>
            <tspan
              fontSize={pctSvg}
              fill="#9ca3af"
              dy={-numSvg * 0.32}     // superscript lift
              dx="1"
            >
              %
            </tspan>
          </text>
        )}
      </svg>

      {label && (
        <p className="text-xs text-gray-500 font-medium text-center leading-tight px-1">
          {label}
        </p>
      )}
    </div>
  )
}

export default HealthGauge
