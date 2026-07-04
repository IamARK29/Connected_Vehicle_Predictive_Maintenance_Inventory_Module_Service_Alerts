import { useState, useEffect, useCallback, useRef } from 'react'
import { UploadPanel } from '../components/UploadPanel'
import { uploadTelemetryFile, uploadTripsFile, uploadServiceHistoryFile, generateSynthetic, trainModels, getSyntheticStatus } from '../api/client'

const TABS = ['Upload File', 'Connect Live Feed', 'Generate Synthetic'] as const
type Tab = typeof TABS[number]

// ── Toast ────────────────────────────────────────────────────────────────────

type ToastType = 'success' | 'error' | 'info'
type ToastMsg  = { message: string; type: ToastType; key: number }

function Toast({ t, onClose }: { t: ToastMsg; onClose: () => void }) {
  useEffect(() => {
    const id = setTimeout(onClose, 6000)
    return () => clearTimeout(id)
  }, [t.key, onClose])

  const bg = t.type === 'success' ? 'bg-green-600' : t.type === 'error' ? 'bg-red-600' : 'bg-blue-600'
  const icon = t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : 'ℹ️'

  return (
    <div className={`fixed top-6 right-6 z-50 ${bg} text-white px-5 py-3 rounded-xl shadow-2xl max-w-md animate-slide-in flex items-start gap-3`}>
      <span className="text-lg leading-none mt-0.5">{icon}</span>
      <p className="flex-1 text-sm font-medium">{t.message}</p>
      <button onClick={onClose} className="text-white/70 hover:text-white text-lg leading-none">&times;</button>
    </div>
  )
}

// ── Job poller (lives in parent, survives tab switches) ──────────────────────

type Phase = 'idle' | 'generating' | 'generated' | 'training' | 'done'

interface JobState {
  phase: Phase; pct: number; message: string; jobId: string | null
}

function useJobPoller() {
  const [job, setJob] = useState<JobState>({ phase: 'idle', pct: 0, message: '', jobId: null })
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }, [])

  const startPolling = useCallback((jobId: string, phase: Phase, onDone: (r: any) => void) => {
    stopPolling()
    setJob(j => ({ ...j, jobId, phase, pct: 0 }))
    intervalRef.current = setInterval(async () => {
      try {
        const d = await getSyntheticStatus(jobId)
        setJob(j => ({ ...j, pct: d.pct ?? 0, message: d.message ?? '' }))
        if (d.pct >= 100) {
          if (intervalRef.current) clearInterval(intervalRef.current)
          intervalRef.current = null
          onDone(d.result ?? {})
        }
      } catch { /* keep polling */ }
    }, 2000)
  }, [stopPolling])

  useEffect(() => stopPolling, [stopPolling])
  return { job, setJob, startPolling }
}

// ── Tab 1: File Upload ───────────────────────────────────────────────────────

function FileUploadTab() {
  const [mode, setMode] = useState<'telemetry' | 'trips' | 'service'>('telemetry')
  const uploaders = { telemetry: uploadTelemetryFile, trips: uploadTripsFile, service: uploadServiceHistoryFile }
  const templates: Record<string, string> = {
    telemetry: '/api/upload/templates/telemetry.csv',
    trips: '/api/upload/templates/trips.csv',
    service: '/api/upload/templates/service.csv',
  }

  return (
    <div className="space-y-5">
      <div className="flex gap-2">
        {(['telemetry', 'trips', 'service'] as const).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors border capitalize ${
              mode === m ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
            }`}>
            {m === 'service' ? 'Service History' : m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>
      <UploadPanel label={`Drop your ${mode} CSV here`} onUpload={uploaders[mode]} templateUrl={templates[mode]} />
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <p className="text-xs font-semibold text-blue-800 mb-2">Expected columns for {mode}:</p>
        <p className="text-xs text-blue-700 font-mono leading-relaxed">
          {mode === 'telemetry' && 'StartTime-TimeStamp, VIN, VehSpeed, VehRPM, VehBatt, VehOdo, VehCoolantTemp, ...'}
          {mode === 'trips'     && 'vin, startTime, endTime, odometer, averageSpeed, driveScore, harshBreakingNum, ...'}
          {mode === 'service'   && 'VIN, CreatedOn, DealerCode, DescriptionOne, OrderQuantity, NetValue, Mileage, ...'}
        </p>
      </div>
    </div>
  )
}

// ── Tab 2: Live Feed ─────────────────────────────────────────────────────────

function LiveFeedTab({ toast }: { toast: (m: string, t: ToastType) => void }) {
  const [form, setForm] = useState({ protocol: 'mqtt', host: 'localhost', port: '1883', topic: 'autopredict/telemetry/#', username: '', password: '' })
  const [saved, setSaved] = useState(false)
  const save = (e: React.FormEvent) => {
    e.preventDefault()
    localStorage.setItem('ap_live_feed_config', JSON.stringify(form))
    setSaved(true); toast('Live feed configuration saved', 'success')
    setTimeout(() => setSaved(false), 3000)
  }
  return (
    <form onSubmit={save} className="space-y-4 max-w-lg">
      <p className="text-sm text-gray-600">Configure the MQTT / HTTP TBox endpoint for real-time telemetry ingestion.</p>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">Protocol</label>
        <select value={form.protocol} onChange={e => setForm(f => ({ ...f, protocol: e.target.value }))}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="mqtt">MQTT (TCP)</option><option value="mqtts">MQTT over TLS</option>
          <option value="http">HTTP webhook</option><option value="https">HTTPS webhook</option>
        </select>
      </div>
      {[{ label:'Broker Host',key:'host',ph:'mqtt.example.com' },{ label:'Port',key:'port',ph:'1883' },
        { label:'Topic Pattern',key:'topic',ph:'fleet/telemetry/#' },{ label:'Username',key:'username',ph:'(optional)' },
        { label:'Password',key:'password',ph:'(optional)' }].map(({ label, key, ph }) => (
        <div key={key}>
          <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
          <input type={key==='password'?'password':'text'} value={(form as any)[key]} placeholder={ph}
            onChange={e => setForm(f => ({...f,[key]:e.target.value}))}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
      ))}
      <button type="submit" className="bg-blue-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-blue-700">
        {saved ? 'Saved!' : 'Save Configuration'}
      </button>
    </form>
  )
}

// ── Tab 3: Generate Synthetic ────────────────────────────────────────────────

function SyntheticTab({ job, onGenerate }: { job: JobState; onGenerate: (f: any) => void }) {
  const [form, setForm] = useState({ num_vehicles: 10, num_days: 90, failure_rate: 0.05 })
  const isRunning = job.phase === 'generating' || job.phase === 'training'

  return (
    <form onSubmit={e => { e.preventDefault(); onGenerate(form) }} className="space-y-4 max-w-lg">
      <p className="text-sm text-gray-600">
        Generate a synthetic fleet dataset with telemetry, trips, service history, DTCs, and OTA events.
      </p>

      {[{ label:'Number of Vehicles', key:'num_vehicles', min:1, max:100, step:1, unit:'vehicles' },
        { label:'Days of Data', key:'num_days', min:7, max:365, step:1, unit:'days' }].map(({ label, key, min, max, step, unit }) => (
        <div key={key}>
          <label className="block text-xs font-medium text-gray-600 mb-1">
            {label}: <span className="font-bold text-gray-900">{(form as any)[key]} {unit}</span>
          </label>
          <input type="range" min={min} max={max} step={step} value={(form as any)[key]}
            onChange={e => setForm(f => ({...f,[key]:Number(e.target.value)}))}
            disabled={isRunning} className="w-full accent-blue-600" />
          <div className="flex justify-between text-xs text-gray-400 mt-0.5"><span>{min}</span><span>{max}</span></div>
        </div>
      ))}

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">
          Failure Rate: <span className="font-bold text-gray-900">{(form.failure_rate*100).toFixed(0)}%</span>
        </label>
        <input type="range" min={0} max={0.3} step={0.01} value={form.failure_rate}
          onChange={e => setForm(f => ({...f, failure_rate:Number(e.target.value)}))}
          disabled={isRunning} className="w-full accent-blue-600" />
        <div className="flex justify-between text-xs text-gray-400 mt-0.5"><span>0%</span><span>30%</span></div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-sm">
        <p className="font-medium text-gray-800 mb-1">Will generate:</p>
        <ul className="text-gray-600 space-y-0.5 text-xs">
          <li>- {form.num_vehicles} vehicle profiles with driver archetypes</li>
          <li>- ~{(form.num_vehicles * form.num_days * 3).toLocaleString()} telemetry sessions ({form.num_days}d)</li>
          <li>- ~{Math.round(form.num_vehicles * form.num_days / 2).toLocaleString()} trip records</li>
          <li>- ~{(form.num_vehicles * 3).toLocaleString()} service history + DTC + OTA records</li>
        </ul>
      </div>

      <button type="submit" disabled={isRunning}
        className="w-full bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors">
        {job.phase === 'generating' ? 'Generating...' : 'Generate Data'}
      </button>
    </form>
  )
}

// ── Global Train Models Bar ──────────────────────────────────────────────────

function TrainModelsBar({ job, onTrain }: { job: JobState; onTrain: () => void }) {
  const isRunning = job.phase === 'generating' || job.phase === 'training'

  return (
    <div className={`rounded-xl border p-4 ${
      job.phase === 'done'
        ? 'bg-green-50 border-green-200'
        : isRunning
          ? 'bg-blue-50 border-blue-200'
          : 'bg-white border-gray-200'
    }`}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-gray-900">Train ML Models</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            {job.phase === 'done'
              ? 'Models trained and ready. Predictions are live on the Dashboard.'
              : job.phase === 'training'
                ? job.message
                : 'Train prediction models on whatever data is available (uploaded CSV, live feed, or synthetic).'}
          </p>
        </div>
        <button onClick={onTrain} disabled={isRunning}
          className={`px-5 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
            job.phase === 'done'
              ? 'bg-green-600 text-white hover:bg-green-700'
              : job.phase === 'training'
                ? 'bg-gray-300 text-gray-500 cursor-wait'
                : 'bg-green-600 text-white hover:bg-green-700'
          } disabled:opacity-50`}>
          {job.phase === 'training' ? 'Training...' : job.phase === 'done' ? 'Re-train Models' : 'Train Models'}
        </button>
      </div>

      {/* Progress */}
      {(isRunning || (job.phase === 'done' && job.message)) && (
        <div className="mt-3 space-y-1.5">
          <div className="flex justify-between text-xs text-gray-600">
            <span className="truncate flex-1">{job.message}</span>
            <span className="ml-2 tabular-nums font-medium">{job.pct}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${
              job.phase === 'training' ? 'bg-green-500' : job.phase === 'generating' ? 'bg-blue-500' : 'bg-green-500'
            }`} style={{ width: `${job.pct}%` }} />
          </div>
        </div>
      )}

      {job.phase === 'done' && (
        <p className="mt-2 text-xs text-green-700 font-medium">
          &#9989; Models trained successfully. Go to Dashboard to see fleet health and predictions.
        </p>
      )}
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function Upload() {
  const [activeTab, setActiveTab] = useState<Tab>('Upload File')
  const [toast, setToast] = useState<ToastMsg | null>(null)
  const { job, setJob, startPolling } = useJobPoller()

  const showToast = useCallback((message: string, type: ToastType) => {
    setToast({ message, type, key: Date.now() })
  }, [])

  const handleGenerate = useCallback(async (form: any) => {
    setJob(j => ({ ...j, phase: 'generating', pct: 0, message: 'Starting data generation...' }))
    try {
      const res = await generateSynthetic(form)
      startPolling(res.job_id, 'generating', (result) => {
        if (result.status === 'error') {
          setJob(j => ({ ...j, phase: 'idle', message: '' }))
          showToast(`Generation failed: ${result.error}`, 'error')
        } else {
          setJob(j => ({ ...j, phase: 'idle', message: '' }))
          showToast(`Data generated: ${form.num_vehicles} VINs, ${form.num_days} days of telemetry`, 'success')
        }
      })
    } catch (err: unknown) {
      setJob(j => ({ ...j, phase: 'idle', message: '' }))
      showToast(err instanceof Error ? err.message : 'Generation failed', 'error')
    }
  }, [setJob, startPolling, showToast])

  const handleTrain = useCallback(async () => {
    setJob(j => ({ ...j, phase: 'training', pct: 0, message: 'Starting model training...' }))
    try {
      const res = await trainModels()
      startPolling(res.job_id, 'training', (result) => {
        if (result.status === 'error') {
          setJob(j => ({ ...j, phase: 'idle' }))
          showToast(`Training failed: ${result.error}`, 'error')
        } else {
          setJob(j => ({ ...j, phase: 'done' }))
          showToast(`Training complete: ${result.trained ?? ''}/${result.total ?? ''} models trained`, 'success')
        }
      })
    } catch (err: unknown) {
      setJob(j => ({ ...j, phase: 'idle', message: '' }))
      showToast(err instanceof Error ? err.message : 'Training failed', 'error')
    }
  }, [setJob, startPolling, showToast])

  const isRunning = job.phase === 'generating' || job.phase === 'training'

  return (
    <div className="p-6 space-y-6">
      {toast && <Toast t={toast} onClose={() => setToast(null)} />}

      <div>
        <h1 className="text-2xl font-bold text-gray-900">Data Upload & Training</h1>
        <p className="text-gray-500 text-sm mt-1">Load data via any method below, then train ML models</p>
      </div>

      {/* Global Train Models bar — always visible */}
      <TrainModelsBar job={job} onTrain={handleTrain} />

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1">
          {TABS.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`px-5 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeTab === tab ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}>
              {tab}
              {tab === 'Generate Synthetic' && job.phase === 'generating' && (
                <span className="ml-2 inline-block w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        {activeTab === 'Upload File'        && <FileUploadTab />}
        {activeTab === 'Connect Live Feed'  && <LiveFeedTab toast={showToast} />}
        {activeTab === 'Generate Synthetic' && <SyntheticTab job={job} onGenerate={handleGenerate} />}
      </div>

      {/* Floating progress when navigated away from this page */}
      {isRunning && (
        <div className="fixed bottom-6 right-6 z-40 bg-white border border-gray-200 shadow-xl rounded-xl p-4 w-80">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-gray-700">
              {job.phase === 'generating' ? 'Generating data...' : 'Training models...'}
            </span>
            <span className="text-xs font-bold text-blue-600 tabular-nums">{job.pct}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${
              job.phase === 'training' ? 'bg-green-600' : 'bg-blue-600'
            }`} style={{ width: `${job.pct}%` }} />
          </div>
          <p className="text-xs text-gray-400 mt-1.5 truncate">{job.message}</p>
        </div>
      )}
    </div>
  )
}
