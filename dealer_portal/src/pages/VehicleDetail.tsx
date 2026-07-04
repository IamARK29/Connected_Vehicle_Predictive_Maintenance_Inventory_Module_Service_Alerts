import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useVehicle, useVehicleAlerts, useVehiclePredictions, useVehicleServiceHistory, useVehicleTrips } from '../api/hooks'
import { HealthGauge } from '../components/HealthGauge'
import { AlertBadge } from '../components/AlertBadge'
import { TelemetryChart } from '../components/TelemetryChart'
import { PredictionCard } from '../components/PredictionCard'
import type { Alert, MLPrediction, ServiceRecord } from '../types'

const TABS = ['Health Overview', 'Live Telemetry', 'Predictions', 'Service History', 'Driver Score'] as const
type Tab = typeof TABS[number]

function Row({ label, value, unit }: { label: string; value: unknown; unit?: string }) {
  const display = value == null || value === '' ? '—' : `${value}${unit ?? ''}`
  return (
    <div className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-medium text-gray-900 tabular-nums">{display}</span>
    </div>
  )
}

function HealthOverview({ vin }: { vin: string }) {
  const { data: vehicle, isLoading: vehicleLoading }  = useVehicle(vin)
  const { data: alertsRaw } = useVehicleAlerts(vin)
  const alerts = (Array.isArray(alertsRaw) ? alertsRaw : (alertsRaw as any)?.alerts ?? []) as Alert[]
  const score = vehicle?.health_score != null ? Number(vehicle.health_score) : null
  const v = vehicle ?? {}

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Health gauge */}
      <div className="card flex flex-col items-center justify-center gap-4 py-6">
        {vehicleLoading || score === null ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-[140px] h-[140px] rounded-full border-[10px] border-gray-100 animate-pulse" />
            <div className="h-4 w-24 bg-gray-100 rounded animate-pulse" />
          </div>
        ) : (
          <>
            <HealthGauge value={score} size={140} />
            <div className="text-center">
              <p className="text-sm font-semibold text-gray-700">Overall Health</p>
              <p className="text-xs text-gray-400 mt-0.5">
                {score >= 80 ? 'Vehicle in good condition' : score >= 60 ? 'Maintenance recommended' : 'Immediate attention required'}
              </p>
            </div>
          </>
        )}
      </div>

      {/* Vehicle details */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">Vehicle Info</h3>
        <Row label="Model"              value={v.model_name} />
        <Row label="Fuel Type"          value={v.fuel_type} />
        <Row label="Year"               value={v.manufacture_year} />
        <Row label="Odometer"           value={v.current_odometer_km ?? v.odometer_km} unit=" km" />
        <Row label="Color"              value={v.color} />
        <Row label="Dealer"             value={[v.dealer_code, v.dealer_city ? `(${v.dealer_city})` : ''].filter(Boolean).join(' ') || undefined} />
        <Row label="Driver Profile"     value={v.driver_profile?.replace(/_/g, ' ')} />
        <Row label="Last Seen"          value={v.last_seen ? new Date(v.last_seen).toLocaleString() : undefined} />
      </div>

      {/* Active alerts */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">
          Active Alerts
          {alerts.length > 0 && (
            <span className="ml-2 text-xs text-red-600 font-bold">({alerts.length})</span>
          )}
        </h3>
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-6 text-gray-400">
            <span className="text-3xl mb-2">✅</span>
            <p className="text-sm">No active alerts</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {alerts.map((a, i) => (
              <div key={i} className="p-3 rounded-lg border border-gray-200 bg-gray-50">
                <div className="flex items-start justify-between gap-2 mb-1">
                  <p className="text-xs font-semibold text-gray-800">{a.title}</p>
                  <AlertBadge severity={a.severity} />
                </div>
                <p className="text-xs text-gray-500 line-clamp-2">{a.message_customer}</p>
                {(a.estimated_cost_min || a.estimated_cost_max) && (
                  <p className="text-xs text-gray-400 mt-1">
                    Est. ₹{a.estimated_cost_min?.toLocaleString('en-IN')} – ₹{a.estimated_cost_max?.toLocaleString('en-IN')}
                  </p>
                )}
              </div>
            ))}
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
          <p className="text-gray-400 font-mono text-xs mt-0.5">{vin}</p>
        </div>
        {vehicle?.health_score != null && (
          <div className="card p-3">
            <HealthGauge value={Number(vehicle.health_score)} size={64} label="Health" />
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1">
          {TABS.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
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
      </div>
    </div>
  )
}
