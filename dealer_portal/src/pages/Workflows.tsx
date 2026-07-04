import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useWorkflows, useAdvanceWorkflow, useTriggerWorkflow } from '../api/hooks'
import type { Workflow } from '../types'

const STAGES = [
  'detection', 'customer_alert', 'appointment_booking', 'parts_pre_order',
  'pre_service_reminder', 'workshop_receipt', 'progress_updates',
  'costing_approval', 'delivery_notification', 'post_service_followup',
]

function StageProgress({ current, history }: { current: string; history: string[] }) {
  const currentIdx = STAGES.indexOf(current)
  return (
    <div className="flex gap-0.5">
      {STAGES.map((s, i) => (
        <div
          key={s}
          title={s.replace(/_/g, ' ')}
          className={`h-2 flex-1 rounded-full ${
            i < currentIdx || history.includes(s) ? 'bg-green-500'
            : i === currentIdx                    ? 'bg-blue-500 animate-pulse'
            : 'bg-gray-200'
          }`}
        />
      ))}
    </div>
  )
}

export default function Workflows() {
  const [showCompleted, setShowCompleted] = useState(false)
  const [triggerVin, setTriggerVin]       = useState('')

  const { data: workflowsRaw = [], isLoading } = useWorkflows(showCompleted)
  const advance = useAdvanceWorkflow()
  const trigger = useTriggerWorkflow()

  const workflows = workflowsRaw as Workflow[]

  const handleTrigger = async () => {
    if (!triggerVin.trim()) return
    await trigger.mutateAsync(triggerVin.trim())
    setTriggerVin('')
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Agent Workflows</h1>
          <p className="text-gray-500 text-sm mt-1">10-stage service workflow tracker</p>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input
            type="checkbox"
            checked={showCompleted}
            onChange={e => setShowCompleted(e.target.checked)}
            className="rounded"
          />
          Show completed
        </label>
      </div>

      {/* Trigger new workflow */}
      <div className="card">
        <h2 className="font-semibold text-gray-900 mb-3">Trigger Workflow for VIN</h2>
        <p className="text-xs text-gray-500 mb-3">Runs rule + ML alert engines and starts a new service workflow for the highest-severity alert found.</p>
        <div className="flex gap-2">
          <input
            type="text"
            value={triggerVin}
            onChange={e => setTriggerVin(e.target.value)}
            placeholder="Enter VIN…"
            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
            onKeyDown={e => e.key === 'Enter' && handleTrigger()}
          />
          <button
            onClick={handleTrigger}
            disabled={trigger.isPending || !triggerVin.trim()}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {trigger.isPending ? 'Triggering…' : '▶ Trigger'}
          </button>
        </div>
        {trigger.data && (
          <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200 text-xs text-green-800">
            Workflow started: <span className="font-mono">{(trigger.data as any).workflow_id}</span> —{' '}
            alert: {(trigger.data as any).alert_type}
          </div>
        )}
      </div>

      {/* Stage legend */}
      <div className="flex flex-wrap gap-2 text-xs text-gray-500">
        {[{ c: 'bg-green-500', l: 'Completed' }, { c: 'bg-blue-500', l: 'Active' }, { c: 'bg-gray-200', l: 'Pending' }].map(({ c, l }) => (
          <span key={l} className="flex items-center gap-1.5">
            <span className={`w-3 h-2 rounded-full ${c}`} /> {l}
          </span>
        ))}
      </div>

      {/* Workflows table */}
      {isLoading ? (
        <div className="text-gray-400 text-sm">Loading workflows…</div>
      ) : workflows.length === 0 ? (
        <div className="card text-center py-16">
          <span className="text-5xl">🤖</span>
          <p className="text-gray-500 mt-4 font-medium">No active workflows</p>
          <p className="text-gray-400 text-sm mt-1">Trigger a workflow above or wait for an alert to arrive</p>
        </div>
      ) : (
        <div className="space-y-3">
          {workflows.map(wf => (
            <div key={wf.workflow_id} className={`card p-4 ${wf.escalated ? 'border-red-300 bg-red-50/30' : ''}`}>
              <div className="flex items-start justify-between gap-4 mb-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Link to={`/vehicles/${wf.vin}`} className="font-mono text-sm text-blue-600 hover:underline font-bold">{wf.vin}</Link>
                    {wf.escalated && (
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-red-100 text-red-800">ESCALATED</span>
                    )}
                    {wf.completed_at && (
                      <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">COMPLETED</span>
                    )}
                    {wf.alert_type && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-800 font-medium">
                        {wf.alert_type?.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1 font-mono">{wf.workflow_id}</p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-xs font-semibold text-blue-700 capitalize">
                    {wf.current_stage?.replace(/_/g, ' ')}
                  </p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(wf.updated_at).toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Stage progress bar */}
              <StageProgress current={wf.current_stage} history={wf.stage_history ?? []} />

              {/* Actions */}
              {!wf.completed_at && (
                <div className="mt-3 flex items-center justify-between">
                  <p className="text-xs text-gray-400">
                    Stage {STAGES.indexOf(wf.current_stage) + 1} of {STAGES.length}
                    {wf.appointment_id && (
                      <span className="ml-2">· Appt: <span className="font-mono">{wf.appointment_id.slice(0, 8)}…</span></span>
                    )}
                  </p>
                  <button
                    onClick={() => advance.mutate(wf.vin)}
                    disabled={advance.isPending}
                    className="text-xs text-blue-600 hover:text-blue-800 font-semibold border border-blue-200 px-2.5 py-1 rounded-lg hover:bg-blue-50 disabled:opacity-50"
                  >
                    Advance Stage →
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
