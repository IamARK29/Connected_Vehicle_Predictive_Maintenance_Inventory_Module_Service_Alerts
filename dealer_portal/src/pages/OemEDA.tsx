import { useState } from 'react'
import { useOemEda } from '../api/hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell,
} from 'recharts'

const FEATURE_GROUPS = [
  { value: 'fleet',             label: 'Fleet Master',         desc: 'Vehicle inventory: odometer, fuel type, driver profile, model' },
  { value: 'trips',             label: 'Trip Behaviour',       desc: 'Drive scores, harsh events, speed distribution, correlations' },
  { value: 'failures',          label: 'Failure Events',       desc: 'Failure types, odometer at failure, timeline analysis' },
  { value: 'telemetry_summary', label: 'Telemetry Summary',    desc: 'Engine temp, battery voltage, speed — from sampled VINs' },
]

const PALETTE = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6']

function HistogramChart({ data, name }: { data: any; name: string }) {
  const chartData = (data.bins ?? []).map((bin: string, i: number) => ({
    bin,
    count: data.counts?.[i] ?? 0,
  }))

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-sm font-semibold text-gray-900">{name.replace(/_/g, ' ')}</h4>
          {data.mean != null && (
            <p className="text-xs text-gray-400 mt-0.5">
              Mean: {data.mean} · Median: {data.median} · Std: {data.std}
            </p>
          )}
        </div>
        <span className="text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-600 font-medium">histogram</span>
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={chartData} barSize={28}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
          <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="count" radius={[3, 3, 0, 0]}>
            {chartData.map((_: any, i: number) => (
              <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function CategoricalChart({ data, name }: { data: any; name: string }) {
  const chartData = (data.labels ?? []).map((label: string, i: number) => ({
    label: label.length > 14 ? label.slice(0, 13) + '…' : label,
    count: data.counts?.[i] ?? 0,
  }))

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-3">
        <h4 className="text-sm font-semibold text-gray-900">{name.replace(/_/g, ' ')}</h4>
        <span className="text-xs px-2 py-0.5 rounded bg-purple-50 text-purple-600 font-medium">categorical</span>
      </div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={chartData} barSize={24} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="label" width={100} tick={{ fontSize: 9 }} />
          <Tooltip />
          <Bar dataKey="count" radius={[0, 3, 3, 0]}>
            {chartData.map((_: any, i: number) => (
              <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function BoxChart({ data, name }: { data: any; name: string }) {
  const displayName = data.label ?? name.replace(/_/g, ' ')
  const whisker = [
    { metric: 'Min',    value: data.min    },
    { metric: 'P5',     value: data.p5     },
    { metric: 'P25',    value: data.p25    },
    { metric: 'Median', value: data.median },
    { metric: 'Mean',   value: data.mean   },
    { metric: 'P75',    value: data.p75    },
    { metric: 'P95',    value: data.p95    },
    { metric: 'Max',    value: data.max    },
  ]
  const range = Math.max(data.max - data.min, 1e-9)
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h4 className="text-sm font-semibold text-gray-900">{displayName}</h4>
          <p className="text-xs text-gray-400 mt-0.5">
            Mean: {data.mean} · Std: {data.std}
            {data.n != null && ` · n=${data.n.toLocaleString()}`}
          </p>
        </div>
        <span className="text-xs px-2 py-0.5 rounded bg-teal-50 text-teal-600 font-medium">box stats</span>
      </div>
      <div className="space-y-1.5 mt-2">
        {whisker.map(w => (
          <div key={w.metric} className="flex items-center gap-2">
            <span className="text-xs text-gray-400 w-14">{w.metric}</span>
            <div className="flex-1 h-1.5 bg-gray-100 rounded-full">
              <div
                className="h-full bg-teal-400 rounded-full"
                style={{ width: `${Math.min(100, ((w.value - data.min) / range) * 100)}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-700 w-20 text-right">{w.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function CorrelationHeatmap({ data }: { data: any }) {
  const features: string[] = data.features ?? []
  const matrix: number[][] = data.matrix ?? []

  const getColor = (v: number) => {
    if (v >= 0.7) return 'bg-blue-600 text-white'
    if (v >= 0.4) return 'bg-blue-300 text-blue-900'
    if (v >= 0.1) return 'bg-blue-100 text-blue-700'
    if (v >= -0.1) return 'bg-gray-50 text-gray-500'
    if (v >= -0.4) return 'bg-red-100 text-red-700'
    if (v >= -0.7) return 'bg-red-300 text-red-900'
    return 'bg-red-600 text-white'
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 lg:col-span-2">
      <h4 className="text-sm font-semibold text-gray-900 mb-3">Feature Correlation Matrix</h4>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="w-24" />
              {features.map(f => (
                <th key={f} className="p-1 text-gray-500 font-medium writing-mode-vertical max-w-[60px]" style={{ writingMode: 'vertical-rl', textOrientation: 'mixed' }}>
                  <span className="block max-w-[60px] overflow-hidden text-ellipsis">{f}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((row, ri) => (
              <tr key={row}>
                <td className="pr-2 text-gray-500 font-medium text-right">{row}</td>
                {(matrix[ri] ?? []).map((val, ci) => (
                  <td key={ci} className={`w-12 h-12 text-center font-bold rounded ${getColor(val)}`}>
                    {val.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex items-center gap-4 text-xs text-gray-400 flex-wrap">
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-600 rounded inline-block" /> Strong positive (≥0.7)</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-blue-200 rounded inline-block" /> Moderate positive</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-200 rounded inline-block" /> Negative</span>
      </div>
    </div>
  )
}

export default function OemEDA() {
  const [group, setGroup] = useState('fleet')
  const { data, isLoading } = useOemEda(group)
  const distributions = data?.distributions ?? {}

  const selectedGroup = FEATURE_GROUPS.find(g => g.value === group)

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">EDA Explorer</h1>
        <p className="text-gray-400 text-sm mt-0.5">Exploratory data analysis — feature distributions, correlations, and failure patterns</p>
      </div>

      {/* Group selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {FEATURE_GROUPS.map(g => (
          <button
            key={g.value}
            onClick={() => setGroup(g.value)}
            className={`text-left p-3 rounded-xl border transition-all ${
              group === g.value
                ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <p className={`text-sm font-medium ${group === g.value ? 'text-blue-700' : 'text-gray-900'}`}>{g.label}</p>
            <p className="text-xs text-gray-400 mt-0.5 leading-relaxed">{g.desc}</p>
          </button>
        ))}
      </div>

      {/* Stats bar */}
      {data && (
        <div className="bg-gray-50 rounded-xl border border-gray-200 px-5 py-3 flex items-center gap-6 flex-wrap text-sm">
          <span className="text-gray-500">
            Feature group: <span className="font-semibold text-gray-800">{selectedGroup?.label}</span>
          </span>
          {data.row_count && (
            <span className="text-gray-500">Rows: <span className="font-semibold text-gray-800">{data.row_count.toLocaleString()}</span></span>
          )}
          {data.total_failure_events && (
            <span className="text-gray-500">Failures: <span className="font-semibold text-red-600">{data.total_failure_events}</span></span>
          )}
          {data.files_sampled && (
            <span className="text-gray-500">Files sampled: <span className="font-semibold text-gray-800">{data.files_sampled}</span></span>
          )}
          <span className="text-gray-400 text-xs ml-auto">{new Date(data.generated_at).toLocaleString()}</span>
        </div>
      )}

      {isLoading && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-3" />
            <p className="text-gray-400 text-sm">Loading {selectedGroup?.label}…</p>
          </div>
        </div>
      )}

      {!isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(distributions).map(([key, dist]: [string, any]) => {
            if (key === '_correlation') {
              return <CorrelationHeatmap key={key} data={dist} />
            }
            if (dist.type === 'histogram') {
              return <HistogramChart key={key} data={dist} name={key} />
            }
            if (dist.type === 'categorical') {
              return <CategoricalChart key={key} data={dist} name={key} />
            }
            if (dist.type === 'box') {
              return <BoxChart key={key} data={dist} name={key} />
            }
            if (dist.type === 'timeline') {
              const chartData = (dist.labels ?? []).map((l: string, i: number) => ({
                month: l, count: dist.counts?.[i] ?? 0,
              }))
              return (
                <div key={key} className="bg-white rounded-xl border border-gray-200 p-4 md:col-span-2">
                  <h4 className="text-sm font-semibold text-gray-900 mb-3">Failure Timeline</h4>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={chartData} barSize={24}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" />
                      <XAxis dataKey="month" tick={{ fontSize: 9 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#ef4444" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )
            }
            return null
          })}

          {Object.keys(distributions).length === 0 && !isLoading && (
            <div className="col-span-3 text-center py-16 text-gray-400">
              <p className="text-4xl mb-3">📊</p>
              <p className="font-medium">No data available</p>
              <p className="text-sm mt-1">Generate synthetic data or upload real data first.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
