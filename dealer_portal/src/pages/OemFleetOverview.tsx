import { useState } from 'react'
import { useOemFleetOverview } from '../api/hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts'

const GROUP_OPTIONS = [
  { value: 'dealer_code', label: 'By Dealer' },
  { value: 'model_name',  label: 'By Model' },
  { value: 'fuel_type',   label: 'By Fuel Type' },
  { value: 'region',      label: 'By Region' },
]

const HEALTH_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e']
const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

function KpiCard({ label, value, sub, color = 'blue' }: { label: string; value: string | number; sub?: string; color?: string }) {
  const colors: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    orange: 'bg-orange-50 text-orange-700 border-orange-200',
  }
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <p className="text-xs font-medium opacity-70">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
      {sub && <p className="text-xs opacity-60 mt-0.5">{sub}</p>}
    </div>
  )
}

function HealthBar({ value, label }: { value: number; label: string }) {
  const color = value >= 75 ? '#22c55e' : value >= 60 ? '#eab308' : value >= 40 ? '#f97316' : '#ef4444'
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-500 w-24 truncate">{label}</span>
      <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-bold tabular-nums w-10 text-right" style={{ color }}>{value}%</span>
    </div>
  )
}

export default function OemFleetOverview() {
  const [groupBy, setGroupBy] = useState('dealer_code')
  const { data, isLoading } = useOemFleetOverview(groupBy)

  const groups: any[] = data?.groups ?? []
  const totals = data?.totals ?? {}

  const pieData = groups.map((g: any) => ({ name: g.label, value: g.vehicle_count }))

  const alertChartData = groups.map((g: any) => ({
    name: g.label,
    Healthy: g.alerts_healthy,
    Warning: g.alerts_medium + g.alerts_high,
    Critical: g.alerts_critical,
  }))

  const radarData = groups.slice(0, 6).map((g: any) => ({
    group: g.label,
    Health: g.avg_health_score,
    DriverScore: g.avg_driver_score,
    'EV Mix': g.ev_count > 0 ? Math.round((g.ev_count / g.vehicle_count) * 100) : 0,
    'Alert-Free': Math.round((g.alerts_healthy / Math.max(g.vehicle_count, 1)) * 100),
  }))

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-gray-400 text-sm">Loading fleet intelligence…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Fleet Intelligence</h1>
          <p className="text-gray-400 text-sm mt-0.5">Cross-dealer fleet health, distribution, and risk analysis</p>
        </div>
        <div className="flex gap-2">
          {GROUP_OPTIONS.map(opt => (
            <button
              key={opt.value}
              data-testid={`group-by-${opt.value}`}
              onClick={() => setGroupBy(opt.value)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                groupBy === opt.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <KpiCard label="Total Vehicles" value={totals.total_vehicles ?? '—'} color="blue" />
        <KpiCard label="Avg Health" value={`${totals.avg_health_score ?? '—'}%`} color="green" />
        <KpiCard label="EV / PHEV" value={totals.ev_vehicles ?? 0} sub="electric vehicles" color="purple" />
        <KpiCard label="Critical Alerts" value={totals.critical_alerts ?? 0} color="red" />
        <KpiCard label="High Alerts" value={totals.high_alerts ?? 0} color="orange" />
        <KpiCard label={`${GROUP_OPTIONS.find(o => o.value === groupBy)?.label ?? 'Groups'}`} value={totals.group_count ?? 0} color="blue" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Health bar list */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-4">Health Score Ranking</h3>
          <div className="space-y-3">
            {[...groups].sort((a, b) => b.avg_health_score - a.avg_health_score).map((g: any) => (
              <HealthBar key={g.key} value={g.avg_health_score} label={g.label} />
            ))}
          </div>
        </div>

        {/* Alert stacked bar */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-4">Alert Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={alertChartData} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Healthy" stackId="a" fill="#22c55e" />
              <Bar dataKey="Warning" stackId="a" fill="#f97316" />
              <Bar dataKey="Critical" stackId="a" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie — vehicle distribution */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-4">Vehicle Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} aria-label={`${entry.name}: ${entry.value} vehicles`} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar chart */}
      {radarData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-1">Multi-Dimension Comparison</h3>
          <p className="text-xs text-gray-400 mb-4">Health, Driver Score, EV Mix and Alert-Free rate across groups</p>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="group" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
              <Radar name="Health" dataKey="Health" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
              <Radar name="Driver Score" dataKey="DriverScore" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.15} />
              <Radar name="Alert-Free %" dataKey="Alert-Free" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Detail table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100">
          <h3 className="font-semibold text-gray-900">Group Detail</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                {['Group', 'Vehicles', 'EV/PHEV', 'Avg Health', 'Avg Driver Score', 'Critical', 'High', 'Medium', 'Healthy'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {groups.map((g: any) => (
                <tr key={g.key} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-gray-900">{g.label}</td>
                  <td className="px-4 py-3 text-gray-600 tabular-nums">{g.vehicle_count}</td>
                  <td className="px-4 py-3 text-gray-600 tabular-nums">{g.ev_count}</td>
                  <td className="px-4 py-3">
                    <span className={`font-bold tabular-nums ${
                      g.avg_health_score >= 75 ? 'text-green-600' :
                      g.avg_health_score >= 60 ? 'text-yellow-600' : 'text-red-600'
                    }`}>{g.avg_health_score}%</span>
                  </td>
                  <td className="px-4 py-3 tabular-nums text-gray-600">{g.avg_driver_score}</td>
                  <td className="px-4 py-3 text-red-600 font-medium tabular-nums">{g.alerts_critical}</td>
                  <td className="px-4 py-3 text-orange-500 font-medium tabular-nums">{g.alerts_high}</td>
                  <td className="px-4 py-3 text-yellow-600 font-medium tabular-nums">{g.alerts_medium}</td>
                  <td className="px-4 py-3 text-green-600 font-medium tabular-nums">{g.alerts_healthy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
