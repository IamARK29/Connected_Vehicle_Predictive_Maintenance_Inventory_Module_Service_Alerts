import { AlertBadge } from './AlertBadge'
import type { MLPrediction } from '../types'

interface Props {
  modelName: string
  prediction: MLPrediction
}

const MODEL_LABELS: Record<string, string> = {
  brake_wear:      'Brake Wear',
  engine_oil:      'Engine Oil',
  hv_battery_soh:  'HV Battery SoH',
  battery_12v:     '12V Battery',
  tyre_wear:       'Tyre Wear',
  fuel_anomaly:    'Fuel Anomaly',
  driver_score:    'Driver Behaviour',
}

const MODEL_ICONS: Record<string, string> = {
  brake_wear:      '🛑',
  engine_oil:      '🛢️',
  hv_battery_soh:  '⚡',
  battery_12v:     '🔋',
  tyre_wear:       '🔄',
  fuel_anomaly:    '⛽',
  driver_score:    '🏎️',
}

function sevBg(sev: string) {
  const s = sev?.toLowerCase()
  if (s === 'critical') return 'border-red-300 bg-red-50'
  if (s === 'warning')  return 'border-orange-200 bg-orange-50'
  return 'border-gray-200 bg-white'
}

export function PredictionCard({ modelName, prediction }: Props) {
  const label = MODEL_LABELS[modelName] ?? modelName.replace(/_/g, ' ')
  const icon  = MODEL_ICONS[modelName] ?? '📊'
  const conf  = Math.max(0, Math.min(1, prediction.confidence ?? 0))
  const raw   = (prediction.raw ?? {}) as Record<string, any>
  const sev   = prediction.severity?.toLowerCase() ?? 'ok'
  const value = prediction.value ?? raw.remaining_life_pct ?? raw.health_score ?? raw.composite_drive_score

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${sevBg(sev)}`}>
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xl">{icon}</span>
          <div>
            <p className="text-sm font-bold text-gray-900">{raw.component ?? label}</p>
            {prediction.message && (
              <p className="text-xs text-gray-600 mt-0.5 leading-snug">{prediction.message}</p>
            )}
          </div>
        </div>
        <AlertBadge severity={prediction.severity?.toUpperCase() ?? 'OK'} />
      </div>

      {/* Health gauge + RUL */}
      <div className="flex items-center gap-4">
        {value != null && (
          <div className="text-center">
            <p className={`text-2xl font-bold ${
              Number(value) >= 70 ? 'text-green-600' : Number(value) >= 45 ? 'text-yellow-600' : 'text-red-600'
            }`}>{Math.round(Number(value))}%</p>
            <p className="text-xs text-gray-400">Health</p>
          </div>
        )}
        <div className="flex-1 space-y-1.5">
          {raw.rul_days_median != null && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">RUL (days)</span>
              <span className={`font-bold ${Number(raw.rul_days_median) < 30 ? 'text-red-600' : Number(raw.rul_days_median) < 90 ? 'text-orange-500' : 'text-green-600'}`}>
                {raw.rul_days_median}
              </span>
            </div>
          )}
          {raw.rul_km_estimate != null && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">RUL (km)</span>
              <span className="font-medium text-gray-800">{Number(raw.rul_km_estimate).toLocaleString()}</span>
            </div>
          )}
          {prediction.predicted_date && (
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Service by</span>
              <span className="font-medium text-gray-800">{prediction.predicted_date}</span>
            </div>
          )}
        </div>
      </div>

      {/* Confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Confidence</span>
          <span className="font-medium">{Math.round(conf * 100)}%</span>
        </div>
        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${
            conf >= 0.7 ? 'bg-blue-500' : conf >= 0.4 ? 'bg-blue-400' : 'bg-blue-300'
          }`} style={{ width: `${Math.round(conf * 100)}%` }} />
        </div>
      </div>

      {/* Key details */}
      <div className="border-t border-gray-200 pt-2 space-y-1">
        {Object.entries(raw)
          .filter(([k]) => !['component', 'rul_days_median', 'rul_km_estimate'].includes(k))
          .slice(0, 4)
          .map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs">
              <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
              <span className="text-gray-800 font-medium">
                {typeof v === 'number' ? (v < 1 && v > 0 ? v.toFixed(3) : v.toFixed(1)) : String(v ?? '-')}
              </span>
            </div>
          ))}
      </div>
    </div>
  )
}

export default PredictionCard
