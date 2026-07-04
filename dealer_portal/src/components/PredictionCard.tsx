import { AlertBadge } from './AlertBadge'
import type { MLPrediction } from '../types'

interface Props {
  modelName: string
  prediction: MLPrediction
}

function confidenceColor(c: number) {
  if (c >= 0.8) return 'bg-red-500'
  if (c >= 0.6) return 'bg-orange-400'
  if (c >= 0.4) return 'bg-yellow-400'
  return 'bg-blue-400'
}

const MODEL_LABELS: Record<string, string> = {
  brake_wear:    '🛑 Brake Wear',
  oil_change:    '🛢 Oil Change',
  hv_battery:    '⚡ HV Battery SoH',
  battery_12v:   '🔋 12V Battery',
  tyre_wear:     '🔄 Tyre Wear',
  fuel_anomaly:  '⛽ Fuel Anomaly',
  driver_score:  '🏎 Driver Score',
}

export function PredictionCard({ modelName, prediction }: Props) {
  const label   = MODEL_LABELS[modelName] ?? modelName.replace(/_/g, ' ')
  const conf    = Math.max(0, Math.min(1, prediction.confidence ?? 0))
  const confPct = Math.round(conf * 100)

  const raw = prediction.raw ?? {}

  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-gray-900">{label}</p>
          {prediction.predicted_date && (
            <p className="text-xs text-gray-500 mt-0.5">
              Predicted: {prediction.predicted_date}
              {prediction.days_until !== undefined && (
                <span className={`ml-1 font-medium ${prediction.days_until <= 7 ? 'text-red-600' : prediction.days_until <= 30 ? 'text-orange-500' : 'text-gray-500'}`}>
                  ({prediction.days_until}d)
                </span>
              )}
            </p>
          )}
        </div>
        <AlertBadge severity={prediction.severity?.toUpperCase() ?? 'LOW'} />
      </div>

      {/* Confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Confidence</span>
          <span className="font-medium">{confPct}%</span>
        </div>
        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${confidenceColor(conf)}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Key raw metrics */}
      {Object.entries(raw).slice(0, 3).map(([k, v]) => (
        <div key={k} className="flex justify-between text-xs">
          <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
          <span className="text-gray-800 font-mono font-medium">
            {typeof v === 'number' ? v.toFixed(2) : String(v)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default PredictionCard
