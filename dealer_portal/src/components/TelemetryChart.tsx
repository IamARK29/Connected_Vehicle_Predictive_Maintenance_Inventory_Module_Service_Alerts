import { useEffect, useRef, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

interface DataPoint {
  ts: string
  speed?: number
  engineTemp?: number
  soc?: number
  voltage12v?: number
}

interface Props {
  vin: string
  maxPoints?: number
}

const wsBase = () =>
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`

export function TelemetryChart({ vin, maxPoints = 60 }: Props) {
  const [data, setData]           = useState<DataPoint[]>([])
  const [status, setStatus]       = useState<'connecting' | 'live' | 'no_data' | 'error'>('connecting')
  const [lastAlert, setLastAlert] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const url = `${wsBase()}/ws/live/${vin}`
    const ws  = new WebSocket(url)
    wsRef.current = ws

    ws.onopen  = () => setStatus('connecting')
    ws.onerror = () => setStatus('error')
    ws.onclose = () => setStatus('error')

    ws.onmessage = evt => {
      try {
        const msg = JSON.parse(evt.data as string)
        if (msg.type === 'connected') { setStatus('connecting'); return }
        if (msg.type === 'no_data')   { setStatus('no_data');   return }
        if (msg.type === 'error')     { setStatus('error');     return }
        if (msg.type === 'telemetry') {
          setStatus('live')
          const d = msg.data ?? {}
          const point: DataPoint = {
            ts:        new Date(msg.ts).toLocaleTimeString(),
            speed:     d.vehSpeed      ?? d.speed           ?? undefined,
            engineTemp:d.vehEngineTemp ?? d.engine_temp     ?? undefined,
            soc:       d.vehHvSoc      ?? d.soc_current     ?? undefined,
            voltage12v:d.vehBattVolt   ?? d.battery_voltage  ?? undefined,
          }
          setData(prev => {
            const next = [...prev, point]
            return next.length > maxPoints ? next.slice(-maxPoints) : next
          })
          if (msg.alerts?.length) setLastAlert(msg.alerts[0]?.title)
        }
      } catch { /* ignore parse errors */ }
    }

    return () => ws.close()
  }, [vin, maxPoints])

  const statusBadge: Record<typeof status, { label: string; cls: string }> = {
    connecting: { label: '⏳ Connecting…', cls: 'text-yellow-600' },
    live:       { label: '🟢 Live',        cls: 'text-green-600' },
    no_data:    { label: '⚪ No data',      cls: 'text-gray-400' },
    error:      { label: '🔴 Offline',     cls: 'text-red-500' },
  }
  const badge = statusBadge[status]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className={`text-xs font-semibold ${badge.cls}`}>{badge.label}</span>
        {lastAlert && (
          <span className="text-xs text-orange-600 bg-orange-50 border border-orange-200 px-2 py-0.5 rounded-full">
            ⚠ {lastAlert}
          </span>
        )}
      </div>

      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
          {status === 'error' ? 'WebSocket unavailable — start the API server' : 'Waiting for telemetry data…'}
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="ts" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip
              contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
              labelStyle={{ fontWeight: 600 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="speed"      name="Speed (km/h)"  stroke="#3b82f6" dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="engineTemp" name="Eng Temp (°C)" stroke="#ef4444" dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="soc"        name="SoC (%)"       stroke="#22c55e" dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="voltage12v" name="12V (V)"       stroke="#a855f7" dot={false} strokeWidth={1.5} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

export default TelemetryChart
