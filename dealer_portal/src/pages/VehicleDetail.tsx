import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useVehicle, useVehicleAlerts, useVehiclePredictions, useVehicleServiceHistory, useVehicleTrips } from '../api/hooks'
import { HealthGauge } from '../components/HealthGauge'
import { AlertBadge } from '../components/AlertBadge'
import { TelemetryChart } from '../components/TelemetryChart'
import { PredictionCard } from '../components/PredictionCard'
import { EVHealthPanel } from '../components/EVHealthPanel'
import type { Alert, MLPrediction, ServiceRecord } from '../types'

const EV_FUEL_TYPES = new Set(['EV', 'PHEV', 'BEV'])

type Tab = 'Health Overview' | 'Live Telemetry' | 'Predictions' | 'Service History' | 'Driver Score' | 'EV Systems'
const BASE_TABS: Tab[] = ['Health Overview', 'Live Telemetry', 'Predictions', 'Service History', 'Driver Score']

function Row({ label, value, unit }: { label: string; value: unknown; unit?: string }) {
  const display = value == null || value === '' ? '—' : `${value}${unit ?? ''}`
  return (
    <div className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-medium text-gray-900 tabular-nums">{display}</span>
    </div>
  )
}

function scoreLabel(s: number) {
  if (s >= 80) return { text: 'Good',     cls: 'text-green-600 bg-green-50  border-green-200' }
  if (s >= 60) return { text: 'Fair',     cls: 'text-amber-600 bg-amber-50  border-amber-200' }
  if (s >= 40) return { text: 'Poor',     cls: 'text-orange-600 bg-orange-50 border-orange-200' }
  return              { text: 'Critical', cls: 'text-red-600   bg-red-50    border-red-200'   }
}

function severityConfig(sev: string) {
  switch (sev?.toLowerCase()) {
    case 'critical': return { dot: 'bg-red-500',    border: 'border-red-200',    bg: 'bg-red-50'   }
    case 'high':     return { dot: 'bg-orange-500', border: 'border-orange-200', bg: 'bg-orange-50'}
    case 'medium':   return { dot: 'bg-yellow-500', border: 'border-yellow-200', bg: 'bg-yellow-50'}
    default:         return { dot: 'bg-blue-500',   border: 'border-blue-200',   bg: 'bg-blue-50'  }
  }
}

function HealthOverview({ vin }: { vin: string }) {
  const { data: vehicle, isLoading: vehicleLoading }  = useVehicle(vin)
  const { data: alertsRaw } = useVehicleAlerts(vin)
  const alerts = (Array.isArray(alertsRaw) ? alertsRaw : (alertsRaw as any)?.alerts ?? []) as Alert[]
  const score = vehicle?.health_score != null ? Number(vehicle.health_score) : null
  const v = vehicle ?? {}
  const label = score != null ? scoreLabel(score) : null

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Health gauge */}
      <div className="card flex flex-col items-center justify-center gap-3 py-8">
        {vehicleLoading || score === null ? (
          <>
            <div className="w-40 h-40 rounded-full border-[10px] border-gray-100 animate-pulse" />
            <div className="h-4 w-28 bg-gray-100 rounded animate-pulse" />
          </>
        ) : (
          <>
            <HealthGauge value={score} size={160} />
            <p className="text-sm font-semibold text-gray-700 -mt-1">Overall Health</p>
            <span className={`text-xs font-semibold px-3 py-1 rounded-full border ${label!.cls}`}>
              {label!.text} condition
            </span>
          </>
        )}
      </div>

      {/* Vehicle details */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Vehicle Info
        </h3>
        <Row label="Model"          value={v.model_name} />
        <Row label="Fuel Type"      value={v.fuel_type} />
        <Row label="Year"           value={v.manufacture_year} />
        <Row label="Odometer"       value={v.current_odometer_km ?? v.odometer_km} unit=" km" />
        <Row label="Color"          value={v.color} />
        <Row label="Dealer"         value={[v.dealer_code, v.dealer_city ? `(${v.dealer_city})` : ''].filter(Boolean).join(' ') || undefined} />
        <Row label="Driver Profile" value={v.driver_profile?.replace(/_/g, ' ')} />
        <Row label="Last Seen"      value={v.last_seen ? new Date(v.last_seen).toLocaleString() : undefined} />
      </div>

      {/* Active alerts */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <svg className="w-4 h-4 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
          Active Alerts
          {alerts.length > 0 && (
            <span className="ml-auto text-xs font-bold text-white bg-red-500 rounded-full px-2 py-0.5">{alerts.length}</span>
          )}
        </h3>
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-gray-400 gap-2">
            <svg className="w-10 h-10 text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm font-medium text-gray-500">No active alerts</p>
            <p className="text-xs text-gray-400">Vehicle operating normally</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
            {alerts.map((a, i) => {
              const sc = severityConfig(a.severity)
              return (
                <div key={i} className={`p-3 rounded-lg border ${sc.border} ${sc.bg}`}>
                  <div className="flex items-start gap-2 mb-1">
                    <span className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${sc.dot}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-1">
                        <p className="text-xs font-semibold text-gray-800 leading-snug">{a.title}</p>
                        <AlertBadge severity={a.severity} />
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{a.message_customer}</p>
                      {(a.estimated_cost_min || a.estimated_cost_max) && (
                        <p className="text-xs text-gray-400 mt-1 font-mono">
                          ₹{a.estimated_cost_min?.toLocaleString('en-IN')} – ₹{a.estimated_cost_max?.toLocaleString('en-IN')}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

function Predictions({ vin }: { vin: string }) {
  const { data: predsRaw, isLoading } = useVehiclePredictions(vin)
  const predictions = ((predsRaw as any)?.predictions ?? predsRaw ?? {}) as Record<string, MLPrediction>

  if (isLoading) return <div className="text-gray-400 text-sm">Loading predictions…</div>

  const entries = Object.entries(predictions)
  if (entries.length === 0) {
    return <div className="text-gray-400 text-sm py-8 text-center">No ML predictions available. Train models first.</div>
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {entries.map(([model, pred]) => (
        <PredictionCard key={model} modelName={model} prediction={pred} />
      ))}
    </div>
  )
}

function ServiceHistory({ vin }: { vin: string }) {
  const { data: historyRaw = [], isLoading } = useVehicleServiceHistory(vin)
  const history = historyRaw as ServiceRecord[]

  if (isLoading) return <div className="text-gray-400 text-sm">Loading service history…</div>

  return (
    <div className="card p-0 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            {['Date', 'Job Type', 'Description', 'Cost (INR)', 'Dealer'].map(h => (
              <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {history.length === 0 && (
            <tr><td colSpan={5} className="px-4 py-8 text-center text-gray-400">No service records found</td></tr>
          )}
          {history.map((r: any, i) => (
            <tr key={i} className="hover:bg-gray-50">
              <td className="px-4 py-2.5 text-gray-700 whitespace-nowrap">{r.service_date ?? r.ServiceDate ?? '—'}</td>
              <td className="px-4 py-2.5 font-medium">{r.job_type ?? r.JobType ?? '—'}</td>
              <td className="px-4 py-2.5 text-gray-500 max-w-xs truncate">{r.description ?? r.Description ?? '—'}</td>
              <td className="px-4 py-2.5 font-mono">₹{Number(r.cost ?? r.Cost ?? 0).toLocaleString('en-IN')}</td>
              <td className="px-4 py-2.5 text-gray-500">{r.dealer_code ?? r.DealerCode ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DriverScore({ vin }: { vin: string }) {
  const { data: predsRaw } = useVehiclePredictions(vin)
  const preds = ((predsRaw as any)?.predictions ?? predsRaw ?? {}) as Record<string, MLPrediction>
  const drv   = preds.driver_score
  const raw   = (drv?.raw ?? {}) as Record<string, any>
  const score = Number(raw.composite_drive_score ?? drv?.value ?? 75)

  type Metric = { key: string; label: string; unit: string; badAbove: number; format: (v: number) => string }
  const METRICS: Metric[] = [
    { key: 'harsh_braking_per_trip',  label: 'Harsh Braking',        unit: 'events/trip', badAbove: 5,   format: v => v.toFixed(1) },
    { key: 'harsh_accel_per_trip',    label: 'Harsh Acceleration',   unit: 'events/trip', badAbove: 3,   format: v => v.toFixed(1) },
    { key: 'overspeed_fraction',      label: 'Overspeed (>80 kph)',  unit: '%',           badAbove: 0.1, format: v => (v * 100).toFixed(1) },
    { key: 'idle_fraction',           label: 'Idle Time',            unit: '%',           badAbove: 0.3, format: v => (v * 100).toFixed(0) },
    { key: 'avg_max_speed_kph',       label: 'Avg Peak Speed',       unit: 'kph',         badAbove: 100, format: v => v.toFixed(0) },
    { key: 'avg_speed_kph',           label: 'Average Speed',        unit: 'kph',         badAbove: 999, format: v => v.toFixed(0) },
    { key: 'avg_trip_distance_km',    label: 'Avg Trip Distance',    unit: 'km',          badAbove: 999, format: v => v.toFixed(1) },
    { key: 'fuel_efficiency_l100km',  label: 'Fuel Consumption',     unit: 'L/100km',     badAbove: 12,  format: v => v.toFixed(1) },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="card flex flex-col items-center justify-center gap-4 py-10">
        <HealthGauge value={score} size={160} label="Composite Driver Score" />
        <div className={`text-sm font-semibold px-4 py-1.5 rounded-full ${
          score >= 70 ? 'bg-green-100 text-green-800' :
          score >= 50 ? 'bg-yellow-100 text-yellow-800' :
          'bg-red-100 text-red-800'
        }`}>
          {score >= 70 ? 'Low Risk' : score >= 50 ? 'Medium Risk' : 'High Risk'}
        </div>
        {raw.driver_profile && (
          <p className="text-xs text-gray-500 font-medium mt-1">{raw.driver_profile}</p>
        )}
        {raw.total_trips_analysed != null && (
          <p className="text-xs text-gray-400">Based on {raw.total_trips_analysed} trips</p>
        )}
        {raw.score_min != null && raw.score_max != null && (
          <p className="text-xs text-gray-400">Range: {raw.score_min} - {raw.score_max} (std: {raw.score_std})</p>
        )}
      </div>

      <div className="card space-y-4">
        <h3 className="font-semibold text-gray-900">Behaviour Breakdown</h3>
        <p className="text-xs text-gray-500">Computed from actual trip telemetry data</p>
        {METRICS.map(m => {
          const val = raw[m.key]
          if (val == null) return null
          const numVal = Number(val)
          const isBad = numVal > m.badAbove
          return (
            <div key={m.key} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
              <span className="text-sm text-gray-600">{m.label}</span>
              <div className="flex items-center gap-2">
                <span className={`text-sm font-bold tabular-nums ${isBad ? 'text-red-600' : 'text-green-600'}`}>
                  {m.format(numVal)}
                </span>
                <span className="text-xs text-gray-400">{m.unit}</span>
                <span className={`w-2 h-2 rounded-full ${isBad ? 'bg-red-500' : 'bg-green-500'}`} />
              </div>
            </div>
          )
        })}
        {Object.keys(raw).length === 0 && (
          <p className="text-sm text-gray-400">No trip data available for this vehicle.</p>
        )}
      </div>
    </div>
  )
}

export default function VehicleDetail() {
  const { vin } = useParams<{ vin: string }>()
  const [activeTab, setActiveTab] = useState<Tab>('Health Overview')
  const { data: vehicle } = useVehicle(vin!)

  if (!vin) return <div className="p-6 text-gray-400">Invalid VIN</div>

  const isEV = EV_FUEL_TYPES.has((vehicle?.fuel_type ?? '').toUpperCase())
  const tabs: Tab[] = isEV ? [...BASE_TABS, 'EV Systems'] : BASE_TABS

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Link to="/" className="text-xs text-blue-600 hover:underline">← Fleet</Link>
          </div>
          <h1 className="text-2xl font-bold text-gray-900">
            {vehicle?.model_name ?? 'Vehicle'}
            {vehicle?.license_plate ? <span className="text-gray-400 font-normal"> · {vehicle.license_plate}</span> : null}
          </h1>
          <div className="flex items-center gap-2 mt-0.5">
            <p className="text-gray-400 font-mono text-xs">{vin}</p>
            {isEV && (
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 uppercase tracking-wide">
                {vehicle?.fuel_type}
              </span>
            )}
          </div>
        </div>
        {vehicle?.health_score != null && (
          <div className="card px-4 py-3 flex items-center gap-3">
            <HealthGauge value={Number(vehicle.health_score)} size={76} />
            <div className="flex flex-col gap-0.5">
              <p className="text-xs text-gray-400 font-medium uppercase tracking-wide whitespace-nowrap">Health</p>
              <p className={`text-sm font-bold ${scoreLabel(Number(vehicle.health_score)).cls.split(' ')[0]}`}>
                {scoreLabel(Number(vehicle.health_score)).text}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap flex items-center gap-1.5 ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'EV Systems' && (
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              )}
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div>
        {activeTab === 'Health Overview' && <HealthOverview vin={vin} />}
        {activeTab === 'Live Telemetry' && (
          <div className="card">
            <h3 className="font-semibold text-gray-900 mb-4">Live Telemetry: {vin}</h3>
            <TelemetryChart vin={vin} />
          </div>
        )}
        {activeTab === 'Predictions' && <Predictions vin={vin} />}
        {activeTab === 'Service History' && <ServiceHistory vin={vin} />}
        {activeTab === 'Driver Score' && <DriverScore vin={vin} />}
        {activeTab === 'EV Systems' && <EVHealthPanel vin={vin} />}
      </div>
    </div>
  )
}
