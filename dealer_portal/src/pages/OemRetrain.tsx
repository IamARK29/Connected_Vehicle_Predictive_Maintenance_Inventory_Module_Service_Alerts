import { useState } from 'react'
import { useOemRetrainHistory, useTriggerOemRetrain, useOemRetrainStatus, useStopOemRetrain } from '../api/hooks'

const AVAILABLE_MODELS = [
  { id: 'brake_wear',        label: 'Brake Wear',            algo: 'XGBoost Classifier + Regressor' },
  { id: 'engine_oil',        label: 'Engine Oil',            algo: 'XGBoost Classifier + Regressor' },
  { id: 'tyre_wear',         label: 'Tyre Wear',             algo: 'LightGBM' },
  { id: 'battery_12v',       label: '12V Battery',           algo: 'XGBoost + Logistic Regression' },
  { id: 'hv_battery_soh',    label: 'HV Battery SoH',       algo: 'XGBoost + Ridge  (EV/PHEV only)' },
  { id: 'fuel_anomaly',      label: 'Fuel Anomaly',          algo: 'IsolationForest' },
  { id: 'driver_score',      label: 'Driver Score',          algo: 'XGBoost Regressor' },
  { id: 'inventory_demand',  label: 'Inventory Demand',      algo: 'LightGBM Regressor (30d forecast)' },
]

const EV_PHYSICS_ENGINES = [
  { id: 'ev_motor_health',         label: 'EV Motor & Inverter',      algo: 'Physics + Heuristic Rule Engine' },
  { id: 'ev_dcdc_health',          label: 'DC-DC Converter',           algo: 'Physics + Heuristic Rule Engine' },
  { id: 'ev_charging_degradation', label: 'EV Charging Degradation',   algo: 'Coulomb Counting + Statistical Rules' },
]

const STATUS_STYLES: Record<string, string> = {
  completed:  'bg-green-100 text-green-800',
  queued:     'bg-yellow-100 text-yellow-800',
  running:    'bg-blue-100 text-blue-800',
  failed:     'bg-red-100 text-red-800',
}

function RunCard({ run }: { run: any }) {
  const badge = STATUS_STYLES[run.status] ?? 'bg-gray-100 text-gray-600'
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="flex items-start justify-between mb-3">
        <div>
          <p className="text-sm font-mono text-gray-500">{run.job_id}</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {new Date(run.started_at).toLocaleString('en-IN')}
            {run.duration_minutes ? ` · ${run.duration_minutes} min` : ''}
          </p>
        </div>
        <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ${badge}`}>
          {run.status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3 text-sm">
        <div>
          <p className="text-xs text-gray-400">Triggered by</p>
          <p className="font-medium text-gray-800 capitalize">{run.triggered_by ?? 'oem'} ({run.trigger_type ?? 'manual'})</p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Training samples</p>
          <p className="font-medium text-gray-800">{run.training_samples?.toLocaleString() ?? '—'}</p>
        </div>
      </div>

      <div className="mb-3">
        <p className="text-xs text-gray-400 mb-1.5">Models retrained</p>
        <div className="flex flex-wrap gap-1.5">
          {(run.models_retrained ?? run.models ?? []).map((m: string) => (
            <span key={m} className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
              {m.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>

      {run.champion_promoted?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-400 mb-1.5">Promoted to champion</p>
          <div className="flex flex-wrap gap-1.5">
            {run.champion_promoted.map((m: string) => (
              <span key={m} className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded-full font-medium border border-green-200">
                ✓ {m.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {run.notes && (
        <p className="text-xs text-gray-500 border-t border-gray-100 pt-2 mt-2">{run.notes}</p>
      )}
    </div>
  )
}

export default function OemRetrain() {
  const { data: historyData, isLoading } = useOemRetrainHistory()
  const { mutate: trigger, data: triggerResult, isPending } = useTriggerOemRetrain()
  const { mutate: stopJob, isPending: isStopping } = useStopOemRetrain()
  const [selectedModels, setSelectedModels] = useState<string[]>(AVAILABLE_MODELS.map(m => m.id))
  const [notes, setNotes] = useState('')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  const { data: jobStatus } = useOemRetrainStatus(activeJobId)
  const history: any[] = historyData?.history ?? []

  const toggleModel = (id: string) =>
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    )

  const handleTrigger = (models: string[], note = notes) => {
    trigger(
      { models, notes: note },
      {
        onSuccess: (data) => {
          setActiveJobId(data.job_id)
          if (models.length === selectedModels.length) setNotes('')
        },
      }
    )
  }

  const handleStop = () => {
    if (!activeJobId) return
    stopJob(activeJobId, {
      onSuccess: () => setActiveJobId(null),
    })
  }

  const isJobActive = activeJobId && (!jobStatus || ['queued', 'running'].includes(jobStatus.status))

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Retrain Control</h1>
        <p className="text-gray-400 text-sm mt-0.5">
          Trigger model retraining, review champion promotions, and track training history
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trigger panel */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
          <h3 className="font-semibold text-gray-900 border-b border-gray-100 pb-3">Trigger Retraining</h3>

          <div>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Models</p>
            <div className="space-y-1.5">
              {AVAILABLE_MODELS.map(m => (
                <div key={m.id} className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-50">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m.id)}
                    onChange={() => toggleModel(m.id)}
                    className="accent-blue-600 w-4 h-4 flex-shrink-0"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-800 leading-tight">{m.label}</p>
                    <p className="text-xs text-gray-400 truncate">{m.algo}</p>
                  </div>
                  {/* Individual train button */}
                  <button
                    onClick={() => handleTrigger([m.id], `Individual retrain: ${m.label}`)}
                    disabled={isPending || !!isJobActive}
                    title={`Train ${m.label} only`}
                    className="px-2 py-1 text-xs font-medium text-blue-600 border border-blue-200 rounded-md hover:bg-blue-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
                  >
                    Train
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* EV physics engines — read-only, no retraining needed */}
          <div className="border-t border-gray-100 pt-3">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
              EV Physics Engines
              <span className="ml-2 normal-case font-normal text-blue-500 bg-blue-50 px-1.5 py-0.5 rounded text-[10px]">
                No training required
              </span>
            </p>
            <div className="space-y-1.5">
              {EV_PHYSICS_ENGINES.map(m => (
                <div key={m.id} className="flex items-center gap-2 p-2 rounded-lg bg-blue-50/50">
                  <div className="w-4 h-4 flex-shrink-0 flex items-center justify-center">
                    <svg className="w-3.5 h-3.5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-700 leading-tight">{m.label}</p>
                    <p className="text-xs text-gray-400 truncate">{m.algo}</p>
                  </div>
                  <span className="text-[10px] text-blue-500 border border-blue-200 px-1.5 py-0.5 rounded bg-white flex-shrink-0">
                    Active
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-gray-400 mt-2 leading-snug">
              These engines run continuously without retraining. Thresholds are derived from MG Motor EV specifications and Coulomb-counting physics.
            </p>
          </div>

          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider block mb-1.5">
              Notes (optional)
            </label>
            <textarea
              value={notes}
              onChange={e => setNotes(e.target.value)}
              rows={2}
              placeholder="Reason for retraining…"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
            />
          </div>

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-700">
            <p className="font-semibold mb-0.5">Before retraining</p>
            <p>Training {selectedModels.length} model(s) takes ~{selectedModels.length * 8}–{selectedModels.length * 12} min.</p>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => handleTrigger(selectedModels)}
              disabled={isPending || selectedModels.length === 0 || !!isJobActive}
              className="flex-1 bg-blue-600 text-white rounded-lg py-2.5 text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {isPending ? 'Queuing…' : `Retrain ${selectedModels.length} model(s)`}
            </button>
            {isJobActive && (
              <button
                onClick={handleStop}
                disabled={isStopping}
                className="px-3 py-2.5 rounded-lg text-sm font-medium border border-red-300 text-red-600 hover:bg-red-50 disabled:opacity-50 transition-colors"
                title="Cancel active training job"
              >
                {isStopping ? '…' : 'Stop'}
              </button>
            )}
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setSelectedModels(AVAILABLE_MODELS.map(m => m.id))}
              className="flex-1 border border-gray-200 text-gray-500 rounded-lg py-1.5 text-xs hover:bg-gray-50 transition-colors"
            >
              Select All
            </button>
            <button
              onClick={() => setSelectedModels([])}
              className="flex-1 border border-gray-200 text-gray-500 rounded-lg py-1.5 text-xs hover:bg-gray-50 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Right side */}
        <div className="lg:col-span-2 space-y-4">
          {/* Active job status */}
          {activeJobId && (
            <div className={`bg-white rounded-xl border p-5 ${isJobActive ? 'border-blue-200' : 'border-gray-200'}`}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-gray-900">Active Job</h3>
                <div className="flex gap-2 items-center">
                  {isJobActive && (
                    <button
                      onClick={handleStop}
                      disabled={isStopping}
                      className="text-xs px-3 py-1 rounded-lg border border-red-300 text-red-600 hover:bg-red-50 disabled:opacity-50 transition-colors font-medium"
                    >
                      {isStopping ? 'Stopping…' : 'Stop Training'}
                    </button>
                  )}
                  <button
                    onClick={() => setActiveJobId(null)}
                    className="text-xs text-gray-400 hover:text-gray-600"
                  >
                    dismiss
                  </button>
                </div>
              </div>
              {triggerResult && (
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    {isJobActive
                      ? <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full" />
                      : <span className="text-green-500">✓</span>
                    }
                    <p className="text-sm text-gray-700">
                      Job <span className="font-mono text-blue-600">{activeJobId}</span> is{' '}
                      <span className={`font-semibold ${
                        jobStatus?.status === 'cancelled' ? 'text-red-600' :
                        jobStatus?.status === 'completed' ? 'text-green-600' : 'text-blue-600'
                      }`}>{jobStatus?.status ?? 'queued'}</span>
                    </p>
                  </div>
                  <div className="text-xs text-gray-500 space-y-0.5 pl-7">
                    <p>Models: {(triggerResult.models ?? []).join(', ')}</p>
                    <p>Started: {new Date(triggerResult.started_at).toLocaleString()}</p>
                    <p className="text-amber-600">
                      Actual training runs locally via: <span className="font-mono">py models/train_all.py</span>
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Metrics overview */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <p className="text-2xl font-bold text-gray-900">{historyData?.total_runs ?? '—'}</p>
              <p className="text-xs text-gray-400 mt-0.5">Total Runs</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <p className="text-2xl font-bold text-green-600">
                {history.filter((r: any) => r.status === 'completed').length}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">Completed</p>
            </div>
            <div className="bg-white rounded-xl border border-gray-200 p-4 text-center">
              <p className="text-2xl font-bold text-blue-600">
                {history.reduce((acc: number, r: any) => acc + (r.champion_promoted?.length ?? 0), 0)}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">Champions Promoted</p>
            </div>
          </div>

          {/* History */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Training History</h3>
            {isLoading && (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full" />
              </div>
            )}
            {!isLoading && (
              <div className="space-y-4">
                {history.length === 0 && (
                  <div className="bg-white rounded-xl border border-gray-200 p-8 text-center">
                    <p className="text-3xl mb-2">📋</p>
                    <p className="text-gray-500 text-sm">No training runs yet</p>
                  </div>
                )}
                {history.map((run: any) => (
                  <RunCard key={run.job_id} run={run} />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
