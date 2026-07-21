import { useEVHealth } from '../api/hooks'
import { HealthGauge } from './HealthGauge'

// ── Types ─────────────────────────────────────────────────────────────────────

interface FeatureValue {
  label: string
  value: number | null
  unit: string
}

interface AlertEntry {
  feature: string
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  value: number
  threshold: number
  comparison: 'gt' | 'lt'
}

interface ComponentHealth {
  health_score: number | null
  status: string
  features: Record<string, FeatureValue>
  alerts: AlertEntry[]
}

interface RangeData {
  predicted_range_km: number
  range_p10_km: number
  range_p90_km: number
  range_anxiety_flag: boolean
  anxiety_reason: string | null
  energy_available_kwh: number
  effective_kwh_per_100km: number
  temp_efficiency_factor: number
  driver_efficiency_factor: number
  current_soc_pct: number
  outside_temp_c: number
  ac_is_on: boolean
}

interface EVHealthData {
  vin: string
  model_name: string
  fuel_type: string
  rated_range_km: number
  battery_capacity_kwh: number
  overall_ev_health_score: number
  overall_status: string
  computed_at: string
  range: RangeData
  components: {
    charging: ComponentHealth
    motor: ComponentHealth
    dcdc: ComponentHealth
  }
}

// ── Severity helpers ──────────────────────────────────────────────────────────

const SEV_STYLES: Record<string, { pill: string; dot: string; border: string; bg: string }> = {
  CRITICAL: { pill: 'bg-red-100 text-red-700 border-red-200',       dot: 'bg-red-500',    border: 'border-red-300',    bg: 'bg-red-50'    },
  HIGH:     { pill: 'bg-orange-100 text-orange-700 border-orange-200', dot: 'bg-orange-500', border: 'border-orange-300', bg: 'bg-orange-50' },
  MEDIUM:   { pill: 'bg-yellow-100 text-yellow-700 border-yellow-200', dot: 'bg-yellow-500', border: 'border-yellow-300', bg: 'bg-yellow-50' },
  LOW:      { pill: 'bg-blue-100 text-blue-700 border-blue-200',     dot: 'bg-blue-400',   border: 'border-blue-200',   bg: 'bg-blue-50'   },
}

const STATUS_COLOR: Record<string, string> = {
  'Good':     'text-green-600',
  'Fair':     'text-amber-600',
  'Poor':     'text-orange-600',
  'Critical': 'text-red-600',
  'No Data':  'text-gray-400',
}

const STATUS_BG: Record<string, string> = {
  'Good':     'bg-green-50 border-green-200 text-green-700',
  'Fair':     'bg-amber-50 border-amber-200 text-amber-700',
  'Poor':     'bg-orange-50 border-orange-200 text-orange-700',
  'Critical': 'bg-red-50 border-red-200 text-red-700',
  'No Data':  'bg-gray-50 border-gray-200 text-gray-500',
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ScoreBadge({ score, status }: { score: number | null; status: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border ${STATUS_BG[status] ?? STATUS_BG['No Data']}`}>
      {score !== null && <span className="tabular-nums">{score}</span>}
      <span>{status}</span>
    </span>
  )
}

function AlertList({ alerts }: { alerts: AlertEntry[] }) {
  if (alerts.length === 0) return null
  return (
    <div className="mt-3 space-y-1.5">
      {alerts.map((a, i) => {
        const s = SEV_STYLES[a.severity] ?? SEV_STYLES.LOW
        const dir = a.comparison === 'gt' ? 'above' : 'below'
        return (
          <div key={i} className={`flex items-start gap-2 px-3 py-2 rounded-md border text-xs ${s.border} ${s.bg}`}>
            <span className={`mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${s.dot}`} />
            <div>
              <span className={`font-semibold uppercase tracking-wide text-[10px] mr-1.5 ${s.pill.split(' ')[1]}`}>{a.severity}</span>
              <span className="text-gray-700">
                {a.feature.replace(/_/g, ' ')} is {a.comparison === 'gt' ? `${a.value} > ${a.threshold}` : `${a.value} < ${a.threshold}`}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FeatureTable({ features }: { features: Record<string, FeatureValue> }) {
  const rows = Object.entries(features).filter(([, f]) => f.value !== null && f.value !== undefined)
  if (rows.length === 0)
    return <p className="text-xs text-gray-400 py-4 text-center">No telemetry data available for this component</p>
  return (
    <div className="divide-y divide-gray-100">
      {rows.map(([key, f]) => (
        <div key={key} className="flex items-center justify-between py-1.5">
          <span className="text-xs text-gray-500 leading-tight pr-3">{f.label}</span>
          <span className="text-xs font-semibold tabular-nums text-gray-900 whitespace-nowrap">
            {f.value}{f.unit ? <span className="text-gray-400 font-normal ml-0.5">{f.unit}</span> : null}
          </span>
        </div>
      ))}
    </div>
  )
}

function ComponentCard({
  title, subtitle, icon, component,
}: {
  title: string
  subtitle?: string
  icon: React.ReactNode
  component: ComponentHealth
}) {
  return (
    <div className="card flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2 min-w-0">
          <span className="text-blue-500 flex-shrink-0 mt-0.5">{icon}</span>
          <div className="min-w-0">
            <h3 className="font-semibold text-gray-900 text-sm leading-snug">{title}</h3>
            {subtitle && <p className="text-[11px] text-gray-400 mt-0.5 leading-snug">{subtitle}</p>}
          </div>
        </div>
        <div className="flex-shrink-0">
          <ScoreBadge score={component.health_score} status={component.status} />
        </div>
      </div>

      {/* Gauge + features side-by-side when there's a score */}
      {component.health_score !== null ? (
        <div className="flex gap-4 items-start">
          <div className="flex-shrink-0">
            <HealthGauge value={component.health_score} size={88} />
          </div>
          <div className="flex-1 min-w-0">
            <FeatureTable features={component.features} />
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-6 text-gray-400 gap-2">
          <svg className="w-8 h-8 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-xs font-medium text-gray-400">No telemetry data</p>
          <p className="text-xs text-gray-300">Requires live TBox signals</p>
        </div>
      )}

      <AlertList alerts={component.alerts} />
    </div>
  )
}

function RangeCard({ range, ratedRangeKm }: { range: RangeData; ratedRangeKm: number }) {
  const { predicted_range_km: pred, range_p10_km: p10, range_p90_km: p90 } = range
  const bandKm = Math.max(p90 - p10, 1)
  const predPct = Math.min(100, Math.max(0, ((pred - p10) / bandKm) * 100))
  const isLowRange = range.range_anxiety_flag
  const ratedPct = ratedRangeKm > 0 ? Math.round(pred / ratedRangeKm * 100) : null

  return (
    <div className="card">
      {/* Title row */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <h3 className="font-semibold text-gray-900 text-sm">Estimated Range</h3>
        </div>
        {isLowRange ? (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border bg-amber-50 border-amber-200 text-amber-700">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Charge Recommended
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border bg-green-50 border-green-200 text-green-700">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            Range Adequate
          </span>
        )}
      </div>
      <p className="text-[11px] text-gray-400 mb-4">
        Projected distance on current charge — based on SoC, ambient temperature, and this vehicle's usage pattern.
        {isLowRange && ' Low range detected: advise customer to charge before next long trip.'}
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Range display */}
        <div className="space-y-4">
          {/* Big predicted range */}
          <div>
            <p className="text-4xl font-bold tabular-nums text-gray-900">
              {Math.round(pred)}
              <span className="text-base font-normal text-gray-400 ml-1">km</span>
            </p>
            <div className="flex items-center gap-2 mt-0.5">
              <p className="text-xs text-gray-400">at {Math.round(range.current_soc_pct)}% SoC</p>
              {ratedPct !== null && (
                <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                  ratedPct >= 90 ? 'bg-green-50 text-green-700' :
                  ratedPct >= 75 ? 'bg-amber-50 text-amber-700' :
                  'bg-red-50 text-red-700'
                }`}>
                  {ratedPct}% of rated {Math.round(ratedRangeKm)} km
                </span>
              )}
            </div>
          </div>

          {/* P10–P90 band */}
          <div>
            <div className="flex justify-between text-[10px] text-gray-400 mb-1.5 font-mono">
              <span>Low estimate · {Math.round(p10)} km</span>
              <span>High estimate · {Math.round(p90)} km</span>
            </div>
            <div className="relative h-2 rounded-full bg-gray-100 overflow-visible">
              <div
                className={`absolute inset-y-0 rounded-full ${isLowRange ? 'bg-amber-200' : 'bg-blue-100'}`}
                style={{ left: '0%', right: '0%' }}
              />
              <div
                className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full border-2 border-white shadow-sm ${isLowRange ? 'bg-amber-500' : 'bg-blue-500'}`}
                style={{ left: `${predPct}%` }}
              />
            </div>
            <p className="text-[10px] text-gray-400 mt-1.5">
              Confidence band — actual range will fall within this window for 80% of similar trips.
            </p>
          </div>

          {/* Energy */}
          <div className="flex items-center gap-4 pt-1">
            <div>
              <p className="text-xs text-gray-400">Usable energy</p>
              <p className="text-sm font-semibold tabular-nums text-gray-900">{range.energy_available_kwh} <span className="text-gray-400 font-normal text-xs">kWh</span></p>
            </div>
            <div>
              <p className="text-xs text-gray-400">Consumption rate</p>
              <p className="text-sm font-semibold tabular-nums text-gray-900">{range.effective_kwh_per_100km} <span className="text-gray-400 font-normal text-xs">kWh/100km</span></p>
            </div>
          </div>
        </div>

        {/* Right: efficiency factors */}
        <div className="space-y-2 pt-1">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400 mb-2">Range Reduction Factors</p>

          {[
            {
              label: 'Ambient Temperature',
              value: range.temp_efficiency_factor,
              sub: `${range.outside_temp_c}°C — battery loses efficiency outside 20–30°C`,
              color: range.temp_efficiency_factor >= 0.95 ? 'bg-green-500' : range.temp_efficiency_factor >= 0.85 ? 'bg-amber-500' : 'bg-red-500',
            },
            {
              label: 'Driver Behaviour',
              value: range.driver_efficiency_factor,
              sub: range.driver_efficiency_factor >= 1.0
                ? 'Efficient — better than fleet average'
                : 'Above-average energy use vs. fleet',
              color: range.driver_efficiency_factor >= 1.0 ? 'bg-green-500' : range.driver_efficiency_factor >= 0.85 ? 'bg-amber-500' : 'bg-red-500',
            },
            {
              label: 'Air Conditioning',
              value: range.ac_is_on ? 0.9 : 1.0,
              sub: range.ac_is_on ? 'AC on — approx. 10% range reduction' : 'AC off — no reduction',
              color: range.ac_is_on ? 'bg-amber-500' : 'bg-green-500',
            },
          ].map(f => (
            <div key={f.label}>
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-xs text-gray-600">{f.label}</span>
                <span className="text-xs font-semibold tabular-nums text-gray-900">{(f.value * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${f.color}`}
                  style={{ width: `${Math.min(100, f.value * 100)}%` }}
                />
              </div>
              <p className="text-[10px] text-gray-400 mt-0.5">{f.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function EVHealthPanel({ vin }: { vin: string }) {
  const { data, isLoading, error } = useEVHealth(vin)
  const ev = data as EVHealthData | undefined

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="card animate-pulse">
            <div className="h-4 w-40 bg-gray-100 rounded mb-3" />
            <div className="h-24 bg-gray-50 rounded" />
          </div>
        ))}
      </div>
    )
  }

  if (error || !ev) {
    return (
      <div className="card text-center py-16">
        <svg className="w-12 h-12 text-gray-300 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-gray-500 font-medium">EV health data unavailable</p>
        <p className="text-gray-400 text-xs mt-1">
          {(error as any)?.response?.data?.detail ?? 'No EV telemetry found for this vehicle'}
        </p>
      </div>
    )
  }

  const { overall_ev_health_score: overall, overall_status, components, range, computed_at } = ev
  const totalAlerts = [...components.charging.alerts, ...components.motor.alerts, ...components.dcdc.alerts]
  const criticalCount = totalAlerts.filter(a => a.severity === 'CRITICAL').length
  const highCount     = totalAlerts.filter(a => a.severity === 'HIGH').length

  return (
    <div className="space-y-5">

      {/* ── Summary header ───────────────────────────────────────────────── */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-6">
          {/* Overall gauge */}
          <div className="flex items-center gap-4">
            <HealthGauge value={overall} size={76} />
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">EV Powertrain Health</p>
              <p className={`text-lg font-bold mt-0.5 ${STATUS_COLOR[overall_status] ?? 'text-gray-900'}`}>
                {overall_status}
              </p>
              <p className="text-[10px] text-gray-400 mt-0.5">
                Weighted: motor 40% · charging 35% · DC-DC 25%
              </p>
            </div>
          </div>

          <div className="w-px h-10 bg-gray-200 hidden sm:block" />

          {/* Component scores */}
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Charging',      c: components.charging },
              { label: 'Motor',         c: components.motor    },
              { label: 'DC-DC',         c: components.dcdc     },
            ].map(({ label, c }) => (
              <div key={label} className="text-center">
                <p className="text-[10px] text-gray-400 uppercase tracking-wider font-medium mb-1">{label}</p>
                <ScoreBadge score={c.health_score} status={c.status} />
              </div>
            ))}
          </div>

          <div className="w-px h-10 bg-gray-200 hidden sm:block" />

          {/* Alert summary */}
          <div className="flex items-center gap-3">
            {criticalCount > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-700 border border-red-200">
                {criticalCount} CRITICAL
              </span>
            )}
            {highCount > 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-orange-100 text-orange-700 border border-orange-200">
                {highCount} HIGH
              </span>
            )}
            {totalAlerts.length === 0 && (
              <span className="text-xs text-green-600 font-medium">No threshold violations</span>
            )}
          </div>

          <p className="ml-auto text-[10px] text-gray-300 hidden lg:block whitespace-nowrap">
            Computed {new Date(computed_at).toLocaleString()}
          </p>
        </div>
      </div>

      {/* ── Range card ───────────────────────────────────────────────────── */}
      <RangeCard range={range} ratedRangeKm={ev.rated_range_km} />

      {/* ── Component cards grid ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ComponentCard
          title="Charging Health"
          subtitle="Battery's ability to accept and hold charge; degrades with age and DC fast-charge usage"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          component={components.charging}
        />
        <ComponentCard
          title="Motor & Inverter"
          subtitle="Traction motor and power electronics health; thermal stress and torque delivery efficiency"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          }
          component={components.motor}
        />
        <ComponentCard
          title="DC-DC Converter"
          subtitle="High-voltage to 12V converter; keeps auxiliary systems powered when driving"
          icon={
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
            </svg>
          }
          component={components.dcdc}
        />
      </div>
    </div>
  )
}
