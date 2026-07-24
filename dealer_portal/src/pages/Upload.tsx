import { useState } from 'react'
import { UploadPanel } from '../components/UploadPanel'
import { uploadTelemetryFile, uploadTripsFile, uploadServiceHistoryFile, generateSynthetic, trainModels } from '../api/client'
import { useJob } from '../context/JobContext'

const TABS = ['Upload File', 'Connect Live Feed', 'Generate Synthetic', 'Data Specifications'] as const
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

// ── Tab 4: Data Specifications ───────────────────────────────────────────────

type SpecColumn = {
  name: string; type: string; unit: string; required: boolean; example: string; desc: string
}
type SpecSection = {
  id: string; label: string; purpose: string; note: string; columns: SpecColumn[]
}

const SPEC_SECTIONS: SpecSection[] = [
  {
    id: 'telemetry',
    label: 'Telemetry CSV',
    purpose: 'Raw TBox sensor data — one row per second per ignition session. Powers RUL models, fault detection, and driver scoring.',
    note: 'VIN is the only field rejected by the ingestor if missing. The five additional required fields are functionally mandatory: without timestamp the time-series is meaningless; without speed/odometer/power-mode the ML feature pipeline cannot compute brake stress, km-since-service, or session boundaries. BMS fields apply to EV/PHEV only.',
    columns: [
      { name: 'VIN',                     type: 'string',    unit: '—',       required: true,  example: 'MA3EUDDS5P0123456', desc: '17-character Vehicle Identification Number — rows missing this are rejected by the ingestor' },
      { name: 'StartTime-TimeStamp',      type: 'timestamp', unit: 'ISO 8601',required: true,  example: '2024-03-15T09:23:11', desc: 'Timestamp per row — falls back to ingest time if absent, making every time-series feature meaningless' },
      { name: 'VehSysPwrMod',            type: 'integer',   unit: '0–3',     required: true,  example: '2', desc: 'Power mode: 0=Off, 1=Acc, 2=Running, 3=Charging — required for session boundary detection and idle_hours features' },
      { name: 'VehSpeed',                 type: 'float',     unit: 'km/h',    required: true,  example: '68.5', desc: 'Vehicle ground speed — drives brake_stress_cumulative, harsh_brake_rate, high_speed_stop_count, and km calculations' },
      { name: 'VehOdo',                   type: 'float',     unit: 'km',      required: true,  example: '12453.7', desc: 'Odometer reading — required for km_since_last_[part]_service features used by every RUL model' },
      { name: 'VehRPM',                   type: 'float',     unit: 'RPM',     required: false, example: '2200', desc: 'Engine/motor RPM — powers high_rpm_stress_index and rpm_to_speed_ratio_anomaly features' },
      { name: 'VehBatt',                  type: 'float',     unit: 'V',       required: false, example: '12.6', desc: '12V auxiliary battery voltage — required for battery_12v health model (resting_voltage, cranking_voltage_dip features)' },
      { name: 'VehAccelPos',              type: 'float',     unit: '%',       required: false, example: '43.2', desc: 'Accelerator pedal position (0–100) — powers wot_event_count and accel_smoothness features' },
      { name: 'VehBrakePos',              type: 'float',     unit: '%',       required: false, example: '0.0', desc: 'Brake pedal position (0–100) — powers brake_pedal_travel_proxy and brake stress features' },
      { name: 'VehGearPos',               type: 'integer',   unit: '0–8',     required: false, example: '4', desc: 'Gear position: 0=Neutral, 1–8=gear — powers gear_efficiency_score' },
      { name: 'VehSteeringAngle',         type: 'float',     unit: 'degrees', required: false, example: '-12.5', desc: 'Steering wheel angle; negative = left — powers cornering_score' },
      { name: 'FuelTankLevel',            type: 'float',     unit: '%',       required: false, example: '72.0', desc: 'Fuel tank fill level — ICE/PHEV only; powers fuel_level alerts' },
      { name: 'BMSPackSOC',               type: 'float',     unit: '%',       required: false, example: '85.3', desc: 'HV battery state of charge — EV/PHEV; required for battery_hv model and range predictor' },
      { name: 'BMSPackVol',               type: 'float',     unit: 'V',       required: false, example: '370.2', desc: 'HV battery pack voltage — EV/PHEV; powers cell_voltage_spread and SOH estimation' },
      { name: 'BMSPackCrnt',              type: 'float',     unit: 'A',       required: false, example: '-12.5', desc: 'Pack current; negative = charging — EV/PHEV; powers charge_c_rate and regen features' },
      { name: 'BMSCellMaxVol',            type: 'float',     unit: 'V',       required: false, example: '4.12', desc: 'Highest individual cell voltage — EV/PHEV; cell_voltage_spread = max minus min' },
      { name: 'BMSCellMinVol',            type: 'float',     unit: 'V',       required: false, example: '4.08', desc: 'Lowest individual cell voltage — EV/PHEV; see BMSCellMaxVol' },
      { name: 'BMSCellMaxTemp',           type: 'float',     unit: '°C',      required: false, example: '32.1', desc: 'Hottest cell temperature — EV/PHEV; powers cell_temp_delta and thermal runaway detection' },
      { name: 'BMSCellMinTemp',           type: 'float',     unit: '°C',      required: false, example: '28.4', desc: 'Coolest cell temperature — EV/PHEV; see BMSCellMaxTemp' },
      { name: 'frontLeftTyrePressure',    type: 'float',     unit: 'bar',     required: false, example: '2.4', desc: 'Front-left tyre pressure — powers tyre_wear model and TPMS alerts' },
      { name: 'frontRrightTyrePressure',  type: 'float',     unit: 'bar',     required: false, example: '2.4', desc: 'Front-right tyre pressure — note: double-r spelling is per TBox Big Data Spec' },
      { name: 'rearLeftTyrePressure',     type: 'float',     unit: 'bar',     required: false, example: '2.2', desc: 'Rear-left tyre pressure' },
      { name: 'rearRightTyrePressure',    type: 'float',     unit: 'bar',     required: false, example: '2.2', desc: 'Rear-right tyre pressure' },
      { name: 'GNSSLat',                  type: 'float',     unit: 'degrees', required: false, example: '12.9716', desc: 'GPS latitude (decimal degrees) — enables trip mapping' },
      { name: 'GNSSLong',                 type: 'float',     unit: 'degrees', required: false, example: '77.5946', desc: 'GPS longitude (decimal degrees) — enables trip mapping' },
      { name: 'GNSSAlt',                  type: 'float',     unit: 'm',       required: false, example: '920', desc: 'GPS altitude above sea level' },
      { name: 'GNSSHead',                 type: 'float',     unit: 'degrees', required: false, example: '182.0', desc: 'GPS heading (0=North, 90=East)' },
      { name: 'GNSSSats',                 type: 'integer',   unit: 'count',   required: false, example: '9', desc: 'Number of GPS satellites locked' },
    ],
  },
  {
    id: 'trips',
    label: 'Trips CSV',
    purpose: 'Aggregated trip summaries — one row per trip per vehicle. Used for driver scoring, fuel efficiency analysis, and maintenance interval calculation.',
    note: 'vin, timestamps, and odometer are required. Either odometer alone or a startOdometer/endOdometer pair is accepted for distance. Fuel fields are needed for ICE/PHEV fuel efficiency scoring.',
    columns: [
      { name: 'vin',              type: 'string',    unit: '—',        required: true,  example: 'MA3EUDDS5P0123456', desc: '17-character Vehicle Identification Number — required to link the trip to a vehicle' },
      { name: 'startTime',        type: 'timestamp', unit: 'ISO 8601', required: true,  example: '2024-03-15T09:23:11', desc: 'Trip start timestamp — required for temporal analysis and maintenance scheduling' },
      { name: 'endTime',          type: 'timestamp', unit: 'ISO 8601', required: true,  example: '2024-03-15T10:05:43', desc: 'Trip end timestamp — required to calculate trip duration' },
      { name: 'odometer',         type: 'float',     unit: 'km',       required: true,  example: '12453.7', desc: 'Odometer at trip end — required for km_since_last_service maintenance calculations; use this OR startOdometer+endOdometer' },
      { name: 'startOdometer',    type: 'float',     unit: 'km',       required: false, example: '12400.0', desc: 'Odometer at trip start — alternative to odometer; used with endOdometer to compute trip distance' },
      { name: 'endOdometer',      type: 'float',     unit: 'km',       required: false, example: '12453.7', desc: 'Odometer at trip end — alternative to odometer; see startOdometer' },
      { name: 'fuelEfficiency',   type: 'float',     unit: 'L/100km',  required: false, example: '9.2', desc: 'Fuel efficiency for this trip — ICE/PHEV; directly used for fuel_efficiency_l100km driver score metric' },
      { name: 'vehFuelConsumed',  type: 'float',     unit: 'L',        required: false, example: '3.82', desc: 'Total fuel consumed in this trip — ICE/PHEV; used to derive fuelEfficiency if not provided' },
      { name: 'driveScore',       type: 'float',     unit: '0–100',    required: false, example: '78.5', desc: 'Pre-computed driver safety score — used directly if raw telemetry is not available' },
      { name: 'averageSpeed',     type: 'float',     unit: 'km/h',     required: false, example: '42.3', desc: 'Average speed for the trip — powers speed_compliance_score' },
      { name: 'maxSpeed',         type: 'float',     unit: 'km/h',     required: false, example: '98.0', desc: 'Maximum speed recorded in trip' },
      { name: 'harshBreakingNum', type: 'integer',   unit: 'count',    required: false, example: '2', desc: 'Number of harsh braking events — powers harsh_brake_rate and brake_stress features' },
      { name: 'accelerationNum',  type: 'integer',   unit: 'count',    required: false, example: '1', desc: 'Number of sudden acceleration events — powers accel_smoothness_score' },
      { name: 'overSpeedNum',     type: 'integer',   unit: 'count',    required: false, example: '0', desc: 'Total over-speed events across the trip' },
      { name: 'overSpeed80',      type: 'integer',   unit: 'count',    required: false, example: '3', desc: 'Events above 80 km/h in a speed-limited zone' },
      { name: 'overSpeed120',     type: 'integer',   unit: 'count',    required: false, example: '0', desc: 'Events above 120 km/h' },
      { name: 'suddenTurnNum',    type: 'integer',   unit: 'count',    required: false, example: '0', desc: 'Number of sudden turn events — powers cornering_score' },
      { name: 'tripId',           type: 'string',    unit: '—',        required: false, example: 'TRIP-00123', desc: 'Unique trip ID — auto-generated if blank' },
      { name: 'startPoint_lat',   type: 'float',     unit: 'degrees',  required: false, example: '12.9716', desc: 'Trip start GPS latitude — enables route mapping' },
      { name: 'startPoint_long',  type: 'float',     unit: 'degrees',  required: false, example: '77.5946', desc: 'Trip start GPS longitude' },
      { name: 'endPoint_lat',     type: 'float',     unit: 'degrees',  required: false, example: '13.0827', desc: 'Trip end GPS latitude' },
      { name: 'endPoint_long',    type: 'float',     unit: 'degrees',  required: false, example: '80.2707', desc: 'Trip end GPS longitude' },
    ],
  },
  {
    id: 'service',
    label: 'Service History CSV',
    purpose: 'DMS service orders — one row per line item per visit. Used to train part failure prediction models and calculate remaining useful life (RUL).',
    note: 'VIN, CreatedOn, DescriptionOne, and Mileage are all required. Mileage is the odometer at service time — without it the system can only do time-based RUL and cannot compute km_since_last_[part]_service features used by every maintenance model.',
    columns: [
      { name: 'VIN',                  type: 'string', unit: '—',          required: true,  example: 'MA3EUDDS5P0123456', desc: '17-character Vehicle Identification Number — required to link the service event to a vehicle' },
      { name: 'CreatedOn',            type: 'date',   unit: 'YYYY-MM-DD', required: true,  example: '2024-03-15', desc: 'Date the service order was created — required for temporal RUL labelling and days_since_last_service features' },
      { name: 'DescriptionOne',       type: 'string', unit: '—',          required: true,  example: 'Engine Air Filter', desc: 'Part or service description — required for keyword matching to ML model labels (brake, oil, battery, tyre)' },
      { name: 'Mileage',              type: 'float',  unit: 'km',         required: true,  example: '25000', desc: 'Vehicle odometer at time of service — required for km_since_oil_change, km_since_last_brake_service, and all mileage-based RUL features' },
      { name: 'DealerCode',           type: 'string', unit: '—',          required: false, example: 'DL001', desc: 'Dealer identifier — links the service event to dealer-level analytics and demand forecasting' },
      { name: 'DealerName',           type: 'string', unit: '—',          required: false, example: 'MG Motors Delhi', desc: 'Dealer display name' },
      { name: 'DealerCity',           type: 'string', unit: '—',          required: false, example: 'Delhi', desc: 'City where dealer is located' },
      { name: 'Region',               type: 'string', unit: '—',          required: false, example: 'North', desc: 'Geographic region: North, South, East, or West' },
      { name: 'ServiceType',          type: 'string', unit: '—',          required: false, example: 'MAINTENANCE', desc: 'MAINTENANCE, REPAIR, WARRANTY, or RECALL' },
      { name: 'OrderItem',            type: 'string', unit: '—',          required: false, example: 'P-EAF-001', desc: 'Part or labour code from DMS — used as secondary part identifier alongside DescriptionOne' },
      { name: 'MaterialGroup',        type: 'string', unit: '—',          required: false, example: 'FILTERS', desc: 'Part category group from DMS — enables category-level demand analysis' },
      { name: 'OrderQuantity',        type: 'float',  unit: 'units',      required: false, example: '1', desc: 'Quantity of parts ordered or used — powers demand forecasting accuracy' },
      { name: 'UnitPrice',            type: 'float',  unit: 'INR',        required: false, example: '850.00', desc: 'Price per unit' },
      { name: 'NetValue',             type: 'float',  unit: 'INR',        required: false, example: '850.00', desc: 'Net order value before tax' },
      { name: 'Tax',                  type: 'float',  unit: 'INR',        required: false, example: '153.00', desc: 'Tax amount (GST)' },
      { name: 'TotalValue',           type: 'float',  unit: 'INR',        required: false, example: '1003.00', desc: 'Net value + tax' },
      { name: 'GrossValue',           type: 'float',  unit: 'INR',        required: false, example: '1003.00', desc: 'Total gross value including all contributions' },
      { name: 'Status',               type: 'string', unit: '—',          required: false, example: 'CLOSED', desc: 'Order status: OPEN, CLOSED, or CANCELLED' },
      { name: 'ModelSalesCode',       type: 'string', unit: '—',          required: false, example: 'HEC-CVT', desc: 'MG model and variant sales code' },
      { name: 'IssueType',            type: 'string', unit: '—',          required: false, example: 'WARRANTY', desc: 'WARRANTY, PAY, INSURANCE, or RECALL' },
      { name: 'WarrantyContribution', type: 'float',  unit: 'INR',        required: false, example: '850.00', desc: 'Amount covered under warranty' },
      { name: 'InsuranceContribution',type: 'float',  unit: 'INR',        required: false, example: '0.00', desc: 'Amount covered by insurance' },
      { name: 'DiscountContribution', type: 'float',  unit: 'INR',        required: false, example: '0.00', desc: 'Discount applied' },
      { name: 'LicensePlateNumber',   type: 'string', unit: '—',          required: false, example: 'MH01AB1234', desc: 'Vehicle registration plate number' },
      { name: 'Zone',                 type: 'string', unit: '—',          required: false, example: 'WEST', desc: 'Sales/service zone from DMS' },
      { name: 'CompanyCode',          type: 'string', unit: '—',          required: false, example: 'MG01', desc: 'Company code from DMS' },
    ],
  },
]

function DataSpecsTab() {
  const [section, setSection] = useState<string>('telemetry')
  const current = SPEC_SECTIONS.find(s => s.id === section)!

  return (
    <div className="space-y-5">
      <div>
        <p className="text-sm text-gray-600">
          Reference for all CSV files this platform accepts. Download a template from the <strong>Upload File</strong> tab to see column headers pre-filled.
        </p>
      </div>

      {/* Sub-tab selector */}
      <div className="flex gap-2 flex-wrap">
        {SPEC_SECTIONS.map(s => (
          <button key={s.id} onClick={() => setSection(s.id)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors border ${
              section === s.id ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
            }`}>
            {s.label}
          </button>
        ))}
      </div>

      {/* Purpose + note */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-1">
        <p className="text-sm font-medium text-blue-900">{current.purpose}</p>
        <p className="text-xs text-blue-700">{current.note}</p>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> Required
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-gray-300" /> Optional
        </span>
      </div>

      {/* Scrollable table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-gray-50 text-left">
              <th className="px-3 py-2.5 font-semibold text-gray-700 border-b border-gray-200 whitespace-nowrap">Column Name</th>
              <th className="px-3 py-2.5 font-semibold text-gray-700 border-b border-gray-200 whitespace-nowrap">Type</th>
              <th className="px-3 py-2.5 font-semibold text-gray-700 border-b border-gray-200 whitespace-nowrap">Unit</th>
              <th className="px-3 py-2.5 font-semibold text-gray-700 border-b border-gray-200 whitespace-nowrap">Example</th>
              <th className="px-3 py-2.5 font-semibold text-gray-700 border-b border-gray-200">Description</th>
            </tr>
          </thead>
          <tbody>
            {current.columns.map((col, i) => (
              <tr key={col.name} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}>
                <td className="px-3 py-2 font-mono font-medium whitespace-nowrap border-b border-gray-100">
                  <span className={`inline-flex items-center gap-1.5 ${col.required ? 'text-gray-900' : 'text-gray-500'}`}>
                    <span className={`flex-shrink-0 w-1.5 h-1.5 rounded-full ${col.required ? 'bg-red-500' : 'bg-gray-300'}`} />
                    {col.name}
                  </span>
                </td>
                <td className="px-3 py-2 text-gray-500 whitespace-nowrap border-b border-gray-100">{col.type}</td>
                <td className="px-3 py-2 text-gray-500 whitespace-nowrap border-b border-gray-100 font-mono">{col.unit}</td>
                <td className="px-3 py-2 text-gray-600 whitespace-nowrap border-b border-gray-100 font-mono">{col.example}</td>
                <td className="px-3 py-2 text-gray-600 border-b border-gray-100">{col.desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-400">
        Extra columns in your CSV are ignored. Column names are case-sensitive. Null/blank values are accepted for optional fields.
      </p>
    </div>
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
        {activeTab === 'Upload File'           && <FileUploadTab />}
        {activeTab === 'Connect Live Feed'     && <LiveFeedTab onSaved={() => showToast('Live feed configuration saved', 'success')} />}
        {activeTab === 'Data Specifications'   && <DataSpecsTab />}
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
