import { useState } from 'react'
import { UploadPanel } from '../components/UploadPanel'
import { uploadTelemetryFile, uploadTripsFile, uploadServiceHistoryFile, generateSynthetic, trainModels } from '../api/client'
import { useJob } from '../context/JobContext'

const TABS = ['Upload File', 'Connect Live Feed', 'Generate Synthetic'] as const
type Tab = typeof TABS[number]

// ── Tab 1: File Upload ───────────────────────────────────────────────────────

const UPLOAD_MODES = [
  {
    id: 'telemetry' as const,
    label: 'Telemetry',
    template: '/api/upload/templates/telemetry.csv',
    columns: 'StartTime-TimeStamp, VIN, vehSpeed, vehRPM, vehBatt, vehOdo, vehCoolantTemp, vehBMSPackSOC, vehBMSPackVol, ...',
    desc: 'Raw TBox sensor data — one row per second per session',
  },
  {
    id: 'trips' as const,
    label: 'Trips',
    template: '/api/upload/templates/trips.csv',
    columns: 'vin, startTime, endTime, odometer, averageSpeed, driveScore, harshBreakingNum, accelerationNum, ...',
    desc: 'Aggregated trip records — one row per trip per vehicle',
  },
  {
    id: 'service' as const,
    label: 'Service History',
    template: '/api/upload/templates/service.csv',
    columns: 'VIN, CreatedOn, ServiceType, DealerCode, DealerName, DescriptionOne, OrderQuantity, NetValue, Mileage, ...',
    desc: 'DMS service orders — used for RUL model training',
  },
]

function FileUploadTab() {
  const [mode, setMode] = useState<'telemetry' | 'trips' | 'service'>('telemetry')
  const uploaders = { telemetry: uploadTelemetryFile, trips: uploadTripsFile, service: uploadServiceHistoryFile }
  const current = UPLOAD_MODES.find(m => m.id === mode)!

  return (
    <div className="space-y-5">
      {/* Mode selector */}
      <div className="flex gap-2 flex-wrap">
        {UPLOAD_MODES.map(m => (
          <button key={m.id} onClick={() => setMode(m.id)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors border ${
              mode === m.id ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
            }`}>
            {m.label}
          </button>
        ))}
      </div>

      {/* Template downloads — all three always visible */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
        <p className="text-xs font-semibold text-slate-700 mb-2">CSV Templates — download before uploading:</p>
        <div className="flex flex-wrap gap-3">
          {UPLOAD_MODES.map(m => (
            <a key={m.id} href={m.template} download
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                mode === m.id
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-blue-600 border-blue-200 hover:bg-blue-50'
              }`}>
              ⬇ {m.label} template
            </a>
          ))}
        </div>
      </div>

      <UploadPanel label={`Drop your ${current.label} CSV here`} onUpload={uploaders[mode]} />

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <div className="flex items-start justify-between mb-1">
          <p className="text-xs font-semibold text-blue-800">Expected columns — {current.label}</p>
          <p className="text-xs text-blue-500 italic">{current.desc}</p>
        </div>
        <p className="text-xs text-blue-700 font-mono leading-relaxed mt-1">{current.columns}</p>
      </div>
    </div>
  )
}

// ── Tab 2: Live Feed ─────────────────────────────────────────────────────────

function LiveFeedTab({ onSaved }: { onSaved: () => void }) {
  const [form, setForm] = useState({ protocol: 'mqtt', host: 'localhost', port: '1883', topic: 'autopredict/telemetry/#', username: '', password: '' })
  const [saved, setSaved] = useState(false)
  const save = (e: React.FormEvent) => {
    e.preventDefault()
    localStorage.setItem('ap_live_feed_config', JSON.stringify(form))
    setSaved(true); onSaved()
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

function SyntheticTab({ isRunning, phase, onGenerate, onStop }: {
  isRunning: boolean; phase: string
  onGenerate: (f: any) => void; onStop: () => void
}) {
  const [form, setForm] = useState({ num_vehicles: 10, num_days: 90, failure_rate: 0.05, sessions_per_day: 8 })

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
          Sessions / Vehicle / Day: <span className="font-bold text-gray-900">{form.sessions_per_day}</span>
          <span className="ml-2 text-gray-400 font-normal">(higher = richer ML training data)</span>
        </label>
        <input type="range" min={2} max={24} step={1} value={form.sessions_per_day}
          onChange={e => setForm(f => ({...f, sessions_per_day: Number(e.target.value)}))}
          disabled={isRunning} className="w-full accent-blue-600" />
        <div className="flex justify-between text-xs text-gray-400 mt-0.5"><span>2 (sparse)</span><span>24 (hourly)</span></div>
      </div>

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
          <li>fleet.csv — {form.num_vehicles} vehicle profiles with driver archetypes</li>
          <li>telemetry_*.csv — ~{(form.num_vehicles * form.num_days * form.sessions_per_day * 30).toLocaleString()} rows across {form.num_days} days</li>
          <li>trips.csv — ~{Math.round(form.num_vehicles * form.num_days / 2).toLocaleString()} trip records</li>
          <li>service_history.csv — ~{(form.num_vehicles * 3).toLocaleString()} service events</li>
          <li>dtc_events.csv — diagnostic trouble code events linked to failures</li>
          <li>ota_events.csv — firmware OTA update events (TBOX / VCU / BMS)</li>
          <li>parts_inventory.csv — 13 SKUs with stock levels calibrated to fleet size</li>
        </ul>
      </div>

      <div className="flex gap-3">
        <button type="submit" disabled={isRunning}
          className="flex-1 bg-blue-600 text-white py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors">
          {phase === 'generating' ? 'Generating...' : 'Generate Data'}
        </button>
        {isRunning && (
          <button type="button" onClick={onStop}
            className="px-4 py-2.5 rounded-lg text-sm font-medium border border-red-300 text-red-600 hover:bg-red-50 transition-colors">
            Stop
          </button>
        )}
      </div>
    </form>
  )
}

// ── Global Train Models Bar ──────────────────────────────────────────────────

function TrainModelsBar({ phase, pct, message, onTrain, onStop }: {
  phase: string; pct: number; message: string; onTrain: () => void; onStop: () => void
}) {
  const isRunning  = phase === 'generating' || phase === 'training'
  const isTraining = phase === 'training'

  return (
    <div className={`rounded-xl border p-4 ${
      phase === 'done' ? 'bg-green-50 border-green-200' :
      isRunning        ? 'bg-blue-50 border-blue-200'   :
                         'bg-white border-gray-200'
    }`}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-gray-900">Train ML Models</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            {phase === 'done'
              ? 'Models trained and ready. Predictions are live on the Dashboard.'
              : isTraining
                ? message
                : 'Train prediction models on whatever data is available (uploaded CSV, live feed, or synthetic).'}
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {isTraining && (
            <button onClick={onStop}
              className="px-3 py-1.5 rounded-lg text-sm font-medium border border-red-300 text-red-600 hover:bg-red-50 transition-colors whitespace-nowrap">
              Stop
            </button>
          )}
          <button onClick={onTrain} disabled={isRunning}
            className={`px-5 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors disabled:opacity-50 ${
              phase === 'done' ? 'bg-green-600 text-white hover:bg-green-700' :
              isTraining       ? 'bg-gray-300 text-gray-500 cursor-wait' :
                                 'bg-green-600 text-white hover:bg-green-700'
            }`}>
            {isTraining ? 'Training...' : phase === 'done' ? 'Re-train Models' : 'Train Models'}
          </button>
        </div>
      </div>

      {(isRunning || (phase === 'done' && message)) && (
        <div className="mt-3 space-y-1.5">
          <div className="flex justify-between text-xs text-gray-600">
            <span className="truncate flex-1">{message}</span>
            <span className="ml-2 tabular-nums font-medium">{pct}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${
              isTraining ? 'bg-green-500' : 'bg-blue-500'
            }`} style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      {phase === 'done' && (
        <p className="mt-2 text-xs text-green-700 font-medium">
          Models trained successfully. Go to Dashboard to see fleet health and predictions.
        </p>
      )}
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function Upload() {
  const [activeTab, setActiveTab] = useState<Tab>('Upload File')
  const { job, setJob, startPolling, stopJob, showToast } = useJob()

  const handleGenerate = async (form: any) => {
    setJob(j => ({ ...j, phase: 'generating', pct: 0, message: 'Starting data generation...' }))
    try {
      const res = await generateSynthetic(form)
      startPolling(res.job_id, 'generating', (result) => {
        if (result.status === 'error') {
          setJob(j => ({ ...j, phase: 'idle', message: '' }))
          showToast(`Generation failed: ${result.error}`, 'error')
        } else {
          setJob(j => ({ ...j, phase: 'idle', message: '' }))
          showToast(`Data generated: ${form.num_vehicles} VINs, ${form.num_days} days — telemetry, trips & service history`, 'success')
        }
      })
    } catch (err: unknown) {
      setJob(j => ({ ...j, phase: 'idle', message: '' }))
      showToast(err instanceof Error ? err.message : 'Generation failed', 'error')
    }
  }

  const handleTrain = async () => {
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
  }

  const isRunning = job.phase === 'generating' || job.phase === 'training'

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Data Upload &amp; Training</h1>
        <p className="text-gray-500 text-sm mt-1">
          Load data via any method below, then train ML models. Progress continues even if you navigate away.
        </p>
      </div>

      <TrainModelsBar
        phase={job.phase} pct={job.pct} message={job.message}
        onTrain={handleTrain} onStop={stopJob}
      />

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
        {activeTab === 'Connect Live Feed'  && <LiveFeedTab onSaved={() => showToast('Live feed configuration saved', 'success')} />}
        {activeTab === 'Generate Synthetic' && (
          <SyntheticTab
            isRunning={isRunning} phase={job.phase}
            onGenerate={handleGenerate} onStop={stopJob}
          />
        )}
      </div>

      {isRunning && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl px-4 py-3 flex items-center gap-3 text-sm text-blue-700">
          <span className="inline-block w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0" />
          <span className="font-medium">
            {job.phase === 'generating' ? 'Data generation' : 'Model training'} is running in the background —
            you can navigate away and it will keep going.
          </span>
        </div>
      )}
    </div>
  )
}
