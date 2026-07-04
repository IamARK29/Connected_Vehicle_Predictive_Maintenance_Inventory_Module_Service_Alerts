import { useState, useEffect, useRef, useCallback } from 'react'
import { postOemWhatIf } from '../api/client'
import { HealthGauge } from '../components/HealthGauge'
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, Tooltip,
} from 'recharts'

const DRIVER_PROFILES = [
  'eco_driver', 'highway_cruiser', 'urban_commuter', 'elderly_cautious',
  'hill_region', 'aggressive', 'taxi_fleet', 'delivery_driver',
]
const FUEL_TYPES  = ['ICE', 'EV', 'PHEV', 'CNG']
const MODEL_NAMES = ['MG Hector', 'MG ZS EV', 'MG Astor', 'MG Comet EV', 'MG Gloster', 'MG Windsor']

const SEVERITY_STYLES: Record<string, string> = {
  HIGH:   'bg-red-50 border-red-300 text-red-800',
  MEDIUM: 'bg-amber-50 border-amber-300 text-amber-800',
  LOW:    'bg-green-50 border-green-300 text-green-800',
}

const COMPONENT_LABELS: Record<string, string> = {
  brake_wear:     'Brake Wear',
  engine_oil:     'Engine Oil',
  tyre_wear:      'Tyre Wear',
  battery_12v:    '12V Battery',
  hv_battery_soh: 'HV Battery SoH',
}

// Which input parameters drive each component — shown as tooltip
const COMPONENT_DRIVERS: Record<string, string[]> = {
  brake_wear:     ['Harsh Braking / Trip', 'Km since Last Brake Service', 'Overspeed Fraction'],
  engine_oil:     ['Odometer', 'Idle Time Fraction', 'Overspeed Fraction'],
  tyre_wear:      ['Odometer', 'Overspeed Fraction', 'Harsh Braking / Trip'],
  battery_12v:    ['Battery Age (months)', 'Short Trip Fraction'],
  hv_battery_soh: ['Charge Cycles', 'Fast Charge Fraction', 'Avg SoC'],
}

function RangeInput({ label, name, value, min, max, step, unit, hint, format, onChange }: {
  label: string; name: string; value: number; min: number; max: number;
  step: number; unit: string; hint?: string; format?: (v: number) => string;
  onChange: (n: string, v: number) => void
}) {
  const displayVal = format ? format(value) : `${value}${unit ? ' ' + unit : ''}`
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <div>
          <label className="text-xs font-medium text-gray-700">{label}</label>
          {hint && <p className="text-xs text-gray-400 leading-tight">{hint}</p>}
        </div>
        <span className="text-xs font-mono text-blue-700 font-bold">{displayVal}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, parseFloat(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded appearance-none accent-blue-600 cursor-pointer"
      />
      <div className="flex justify-between text-xs text-gray-300 mt-0.5">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  )
}

function SelectInput({ label, name, value, options, onChange }: {
  label: string; name: string; value: string; options: string[];
  onChange: (n: string, v: string) => void
}) {
  return (
    <div>
      <label className="text-xs font-medium text-gray-700 block mb-1">{label}</label>
      <select
        value={value}
        onChange={e => onChange(name, e.target.value)}
        className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
      >
        {options.map(o => <option key={o} value={o}>{o.replace(/_/g, ' ')}</option>)}
      </select>
    </div>
  )
}

function SourceBadge({ source }: { source: string }) {
  if (source === 'survival_model' || source === 'trained_model') {
    return (
      <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-medium">
        ML model
      </span>
    )
  }
  return (
    <span className="text-xs px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200 font-medium">
      physics formula
    </span>
  )
}

function ComponentResult({ name, result }: { name: string; result: any }) {
  const label         = COMPONENT_LABELS[name] ?? name.replace(/_/g, ' ')
  const drivers       = COMPONENT_DRIVERS[name] ?? []
  const severityClass = SEVERITY_STYLES[result.severity] ?? SEVERITY_STYLES.LOW
  const isModel       = result.source === 'trained_model' || result.source === 'survival_model'

  return (
    <div className={`rounded-xl border p-4 ${severityClass}`}>
      <div className="flex items-start justify-between mb-2 gap-2">
        <p className="text-sm font-semibold">{label}</p>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            result.severity === 'HIGH' ? 'bg-red-100 text-red-800' :
            result.severity === 'MEDIUM' ? 'bg-amber-100 text-amber-800' :
            'bg-green-100 text-green-800'
          }`}>{result.severity}</span>
          <SourceBadge source={result.source} />
        </div>
      </div>

      <div className="mt-1">
        <div className="flex justify-between text-xs mb-1">
          <span>Health</span>
          <span className="font-bold">{result.health_pct}%</span>
        </div>
        <div className="h-2 bg-white/50 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${result.health_pct}%`,
              backgroundColor: result.health_pct >= 60 ? '#22c55e' : result.health_pct >= 30 ? '#f97316' : '#ef4444',
            }}
          />
        </div>
      </div>

      {result.confidence != null && (
        <div className="mt-1.5">
          <div className="flex justify-between text-xs mb-0.5">
            <span className="font-medium">Confidence</span>
            <span className="font-bold">{Math.round(result.confidence * 100)}%</span>
          </div>
          <div className="h-1.5 bg-white/50 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.round(result.confidence * 100)}%`,
                backgroundColor: isModel ? '#3b82f6' : '#a855f7',
              }}
            />
          </div>
        </div>
      )}

      <div className="mt-2 space-y-0.5">
        {result.rul_days != null && (
          <p className="text-xs font-medium">RUL: {result.rul_days} days{result.rul_km ? ` / ${result.rul_km.toLocaleString()} km` : ''}</p>
        )}
        {result.rul_days_p10 != null && (
          <p className="text-xs opacity-60">Pessimistic: {Math.round(result.rul_days_p10)} d · Optimistic: {Math.round(result.rul_days_p90)} d</p>
        )}
        {result.predicted_date && (
          <p className="text-xs opacity-70">
            Est: {new Date(result.predicted_date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}
          </p>
        )}
        {result.survival_30d != null && (
          <p className="text-xs opacity-60">Survival 30d: {(result.survival_30d * 100).toFixed(0)}% · 90d: {(result.survival_90d * 100).toFixed(0)}%</p>
        )}
      </div>

      {/* What drives this component */}
      {drivers.length > 0 && (
        <div className="mt-2 pt-2 border-t border-current/10">
          <p className="text-xs opacity-50 mb-1">Driven by</p>
          <div className="flex flex-wrap gap-1">
            {drivers.map(d => (
              <span key={d} className="text-xs px-1.5 py-0.5 rounded bg-white/40 opacity-80">{d}</span>
            ))}
          </div>
        </div>
      )}

      {/* Key drivers for battery */}
      {result.key_drivers && (
        <div className="mt-2 pt-2 border-t border-current/10 text-xs space-y-0.5">
          {Object.entries(result.key_drivers).map(([k, v]: [string, any]) => (
            <div key={k} className="flex justify-between opacity-70">
              <span>{k.replace(/_/g, ' ')}</span>
              <span className="font-mono">{typeof v === 'number' ? v.toFixed(2) : v}</span>
            </div>
          ))}
        </div>
      )}

      {/* Heuristic note */}
      {result.heuristic_note && (
        <p className="text-xs mt-2 pt-1 border-t border-current/10 opacity-50 italic">{result.heuristic_note}</p>
      )}
    </div>
  )
}

const DEFAULT_PARAMS = {
  odometer_km: 45000,
  driver_profile: 'urban_commuter',
  fuel_type: 'ICE',
  model_name: 'MG Hector',
  days_owned: 730,
  harsh_braking_per_trip: 2.5,
  idle_fraction: 0.15,
  avg_max_speed_kph: 85,
  overspeed_fraction: 0.05,
  km_since_last_brake_service: 15000,
  km_since_last_oil_change: 3000,
  km_since_last_tyre_service: 20000,
  battery_age_months: 24,
  short_trip_fraction: 0.20,
  fast_charge_fraction: 0.3,
  battery_soc_avg: 0.6,
  charge_cycle_count: 150,
}

export default function OemWhatIf() {
  const [params, setParams]   = useState<Record<string, any>>(DEFAULT_PARAMS)
  const [result, setResult]   = useState<any>(null)
  const [isPending, setIsPending] = useState(false)
  const [simError, setSimError]   = useState<string | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // PHEV is a plug-in hybrid — it has both ICE (needs engine oil) AND HV battery
  const isPureEV    = params.fuel_type === 'EV'
  const hasHVBattery = params.fuel_type === 'EV' || params.fuel_type === 'PHEV'

  const setField = (name: string, value: number | string) =>
    setParams(p => ({ ...p, [name]: value }))

  const buildPayload = useCallback((p: Record<string, any>) => {
    const payload: any = { ...p }
    if (payload.fuel_type !== 'EV' && payload.fuel_type !== 'PHEV') {
      delete payload.fast_charge_fraction
      delete payload.battery_soc_avg
      delete payload.charge_cycle_count
    }
    return payload
  }, [])

  // Debounce: fire POST 500ms after last param change
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setIsPending(true)
      setSimError(null)
      try {
        const res = await postOemWhatIf(buildPayload(params))
        setResult(res)
      } catch (e: any) {
        const detail = e?.response?.data?.detail ?? e?.message ?? 'Simulation failed'
        setSimError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      } finally {
        setIsPending(false)
      }
    }, 500)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [params, buildPayload])

  // Filter fuel-specific predictions: pure EV has no ICE engine (skip engine oil);
  // HV battery only for EV/PHEV; guarded here in case user switches fuel type mid-session.
  const visiblePredictions = result
    ? Object.entries(result.predictions).filter(([key]) => {
        if (key === 'hv_battery_soh' && !hasHVBattery) return false
        if (key === 'engine_oil'     &&  isPureEV)      return false
        return true
      })
    : []

  const radarData = visiblePredictions.map(([key, val]: [string, any]) => ({
    component: COMPONENT_LABELS[key] ?? key,
    health: val.health_pct,
  }))

  const modelCount = visiblePredictions.filter(([, v]: [string, any]) =>
    v.source === 'trained_model' || v.source === 'survival_model'
  ).length
  const totalCount = visiblePredictions.length

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">What-If Simulator</h1>
        <p className="text-gray-400 text-sm mt-0.5">
          Adjust parameters and see real-time predictions. ML model results are labelled — heuristic estimates are clearly flagged.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Input panel */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 p-5 space-y-4 overflow-y-auto max-h-[80vh]">
          <h3 className="font-semibold text-gray-900 border-b border-gray-100 pb-3">Vehicle Parameters</h3>

          <SelectInput label="Driver Profile" name="driver_profile" value={params.driver_profile} options={DRIVER_PROFILES} onChange={setField} />
          <SelectInput label="Fuel Type"      name="fuel_type"      value={params.fuel_type}      options={FUEL_TYPES}        onChange={setField} />
          <SelectInput label="Model"          name="model_name"     value={params.model_name}     options={MODEL_NAMES}       onChange={setField} />

          <RangeInput label="Odometer" name="odometer_km" value={params.odometer_km} min={0} max={200000} step={1000} unit="km" onChange={setField} />
          <RangeInput label="Vehicle Age" name="days_owned" value={params.days_owned} min={30} max={3650} step={30} unit="days" onChange={setField} />

          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2 border-t border-gray-100">
            Brake &amp; Driving
          </p>
          <RangeInput label="Harsh Braking" name="harsh_braking_per_trip" value={params.harsh_braking_per_trip} min={0} max={20} step={0.5} unit="/trip" onChange={setField} hint="Drives: brake wear" />
          <RangeInput label="Km since Last Brake Service" name="km_since_last_brake_service" value={params.km_since_last_brake_service} min={0} max={45000} step={500} unit="km" onChange={setField} hint="Drives: brake wear (limit 45,000 km)" />
          <RangeInput label="Overspeed (>80 kph)" name="overspeed_fraction" value={params.overspeed_fraction} min={0} max={0.8} step={0.05} unit="" format={v => `${Math.round(v * 100)}%`} onChange={setField} hint="Drives: tyre wear, brake wear" />
          <RangeInput label="Peak Speed" name="avg_max_speed_kph" value={params.avg_max_speed_kph} min={40} max={180} step={5} unit="kph" onChange={setField} />

          {!isPureEV && (
            <>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2 border-t border-gray-100">
                Engine Oil
              </p>
              <RangeInput label="Km since Last Oil Change" name="km_since_last_oil_change" value={params.km_since_last_oil_change} min={0} max={7500} step={250} unit="km" onChange={setField} hint="Drives: engine oil (change interval 7,500 km)" />
              <RangeInput label="Idle Time" name="idle_fraction" value={params.idle_fraction} min={0} max={0.8} step={0.05} unit="" format={v => `${Math.round(v * 100)}%`} onChange={setField} hint="High idle degrades oil faster" />
            </>
          )}

          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2 border-t border-gray-100">
            Tyre Wear
          </p>
          <RangeInput label="Km since Last Tyre Service" name="km_since_last_tyre_service" value={params.km_since_last_tyre_service} min={0} max={55000} step={1000} unit="km" onChange={setField} hint="Drives: tyre wear (replace at ~55,000 km)" />

          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2 border-t border-gray-100">
            12V Battery Parameters
          </p>
          <p className="text-xs text-gray-400 -mt-2">These directly determine 12V battery health</p>
          <RangeInput label="Battery Age" name="battery_age_months" value={params.battery_age_months} min={0} max={84} step={1} unit="months" onChange={setField} hint="~1.2% health lost per month" />
          <RangeInput label="Short Trips (<5 km)" name="short_trip_fraction" value={params.short_trip_fraction} min={0} max={0.9} step={0.05} unit="" format={v => `${Math.round(v * 100)}%`} onChange={setField} hint="Short trips prevent full recharge" />

          {hasHVBattery && (
            <>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider pt-2 border-t border-gray-100">
                {isPureEV ? 'EV Parameters' : 'PHEV / EV Parameters'}
              </p>
              <RangeInput label="Fast Charge Fraction" name="fast_charge_fraction" value={params.fast_charge_fraction ?? 0.3} min={0} max={1} step={0.05} unit="" format={v => `${Math.round(v * 100)}%`} onChange={setField} hint="Drives: HV battery degradation" />
              <RangeInput label="Avg State of Charge" name="battery_soc_avg" value={params.battery_soc_avg ?? 0.6} min={0.1} max={1} step={0.05} unit="" format={v => `${Math.round(v * 100)}%`} onChange={setField} />
              <RangeInput label="Charge Cycles" name="charge_cycle_count" value={params.charge_cycle_count ?? 150} min={0} max={2000} step={10} unit="" onChange={setField} />
            </>
          )}

          <div className="flex gap-2 pt-2 border-t border-gray-100 items-center">
            <div className="flex-1 text-xs text-gray-400">
              {isPending ? (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 border border-blue-500 border-t-transparent rounded-full animate-spin" />
                  Updating…
                </span>
              ) : (
                'Auto-updates on every change'
              )}
            </div>
            <button
              onClick={() => { setParams(DEFAULT_PARAMS) }}
              className="px-3 py-2 border border-gray-200 text-gray-500 rounded-lg text-sm hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-3 space-y-4">
          {isPending && (
            <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
              <div className="animate-spin w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-3" />
              <p className="text-gray-400 text-sm">Running simulation…</p>
            </div>
          )}

          {simError && !isPending && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
              <p className="font-semibold mb-1">Simulation error</p>
              <p className="text-xs opacity-80">{simError}</p>
            </div>
          )}

          {result && (
            <>
              {/* Overall scores */}
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-900">Simulation Results</h3>
                  {/* Model vs heuristic banner */}
                  <div className={`text-xs px-3 py-1.5 rounded-full font-medium border ${
                    modelCount === totalCount
                      ? 'bg-blue-50 text-blue-700 border-blue-200'
                      : modelCount > 0
                      ? 'bg-amber-50 text-amber-700 border-amber-200'
                      : 'bg-gray-100 text-gray-500 border-gray-200'
                  }`}>
                    {modelCount}/{totalCount} from trained models
                    {modelCount < totalCount && ` · ${totalCount - modelCount} formula estimate${totalCount - modelCount > 1 ? 's' : ''}`}
                  </div>
                </div>
                <div className="flex items-center gap-8">
                  <div className="flex flex-col items-center gap-2">
                    <HealthGauge value={result.overall_health} size={110} label="Overall Health" />
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <HealthGauge value={result.drive_score} size={110} label="Driver Score" />
                  </div>
                  {radarData.length > 0 && (
                    <div className="flex-1">
                      <ResponsiveContainer width="100%" height={160}>
                        <RadarChart data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="component" tick={{ fontSize: 9 }} />
                          <PolarRadiusAxis domain={[0, 100]} tick={false} />
                          <Radar name="Health" dataKey="health" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                          <Tooltip formatter={(v: any) => `${v}%`} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>

              {/* Component predictions */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {visiblePredictions.map(([key, val]: [string, any]) => (
                  <ComponentResult key={key} name={key} result={val} />
                ))}
              </div>

              {/* Scenario presets */}
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                <p className="text-sm font-medium text-blue-800 mb-2">Quick Scenarios</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    { label: 'High Mileage',
                      params: { fuel_type: 'ICE', model_name: 'MG Hector', odometer_km: 150000, days_owned: 2000, battery_age_months: 36 } },
                    { label: 'Aggressive',
                      params: { fuel_type: 'ICE', model_name: 'MG Hector', driver_profile: 'aggressive', harsh_braking_per_trip: 12, overspeed_fraction: 0.45, idle_fraction: 0.2 } },
                    { label: 'Old 12V Battery',
                      params: { fuel_type: 'ICE', model_name: 'MG Astor', battery_age_months: 60, short_trip_fraction: 0.6 } },
                    { label: 'Eco Driver',
                      params: { fuel_type: 'ICE', model_name: 'MG Hector', driver_profile: 'eco_driver', harsh_braking_per_trip: 0.5, idle_fraction: 0.05, overspeed_fraction: 0, battery_age_months: 12, short_trip_fraction: 0.05 } },
                    { label: 'Taxi Fleet',
                      params: { fuel_type: 'ICE', model_name: 'MG Hector', driver_profile: 'taxi_fleet', odometer_km: 180000, days_owned: 1800, battery_age_months: 48, short_trip_fraction: 0.1 } },
                    { label: 'EV Heavy Fast Charge',
                      params: { fuel_type: 'EV', model_name: 'MG ZS EV', driver_profile: 'urban_commuter', odometer_km: 80000, days_owned: 1200, fast_charge_fraction: 0.7, battery_soc_avg: 0.4, charge_cycle_count: 600, battery_age_months: 36, short_trip_fraction: 0.1 } },
                  ].map(s => (
                    <button
                      key={s.label}
                      onClick={() => {
                        setParams({ ...DEFAULT_PARAMS, ...s.params })
                      }}
                      className="text-xs px-3 py-1 bg-white border border-blue-200 text-blue-700 rounded-full hover:bg-blue-100 transition-colors"
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
