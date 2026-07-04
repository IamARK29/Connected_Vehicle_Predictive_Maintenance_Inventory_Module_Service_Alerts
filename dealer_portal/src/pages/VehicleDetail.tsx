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
  const { data: vehicle }  = useVehicle(vin)
  const { data: alertsRaw = [] } = useVehicleAlerts(vin)
  const alerts = alertsRaw as Alert[]
  const score = Number(vehicle?.health_score ?? 80)
  const v = vehicle ?? {}

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Health gauge */}
      <div className="card flex flex-col items-center justify-center gap-4">
        <HealthGauge value={score} size={140} />
        <div className="text-center">
          <p className="text-sm font-semibold text-gray-700">Overall Health</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {score >= 80 ? 'Vehicle in good condition' : score >= 60 ? 'Maintenance recommended' : 'Immediate attention required'}
          </p>
        </div>
      </div>

      {/* Vehicle details */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-3">Vehicle Info</h3>
        <Row label="Model"              value={v.model_name ?? v.ModelSalesCodeDescription} />
        <Row label="Fuel Type"          value={v.fuel_type} />
        <Row label="Manufacture Date"   value={v.manufacture_date} />
        <Row label="Odometer"           value={v.current_odometer_km} unit=" km" />
        <Row label="Dealer"             value={v.dealer_code} />
        <Row label="Owner"              value={v.owner_name} />
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
  const predictions = (predsRaw ?? {}) as Record<string, MLPrediction>

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
  const preds = (predsRaw ?? {}) as Record<string, MLPrediction>
  const drv   = preds.driver_score
  const raw   = (drv?.raw ?? {}) as Record<string, number>
  const score = Number(raw.composite_drive_score ?? raw.driver_score ?? 75)

  const METRICS: [string, string][] = [
    ['harsh_accel_rate_30d',    'Harsh Acceleration'],
    ['harsh_brake_rate_30d',    'Harsh Braking'],
    ['overspeed_80_fraction_30d','Overspeed (>80 km/h)'],
    ['idle_fraction_30d',        'Idle Fraction'],
    ['fuel_efficiency_score',    'Fuel Efficiency'],
    ['night_driving_fraction_30d','Night Driving'],
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
      </div>

      <div className="card space-y-4">
        <h3 className="font-semibold text-gray-900">Behaviour Breakdown</h3>
        {METRICS.map(([key, label]) => {
          const val = raw[key]
          if (val == null) return null
          const pct = Math.min(100, Math.round(val * 100))
          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600">{label}</span>
                <span className="font-medium text-gray-800">{pct}%</span>
              </div>
              <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${pct < 15 ? 'bg-green-500' : pct < 40 ? 'bg-yellow-400' : 'bg-red-500'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
        {Object.keys(raw).length === 0 && (
          <p className="text-sm text-gray-400">Driver score predictions not available. Train the driver model first.</p>
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
            {vehicle?.model_name ?? 'Vehicle'} — {vehicle?.license_plate ?? vin}
          </h1>
          <p className="text-gray-400 font-mono text-xs mt-0.5">{vin}</p>
        </div>
        {vehicle?.health_score !== undefined && (
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
            <h3 className="font-semibold text-gray-900 mb-4">Live Telemetry — {vin}</h3>
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
