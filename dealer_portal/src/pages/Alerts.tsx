import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useFleetAlerts } from '../api/hooks'
import { AlertBadge } from '../components/AlertBadge'
import { SortTh, useSortable } from '../components/SortHeader'
import type { Severity, Alert } from '../types'

const SEVERITIES: Array<Severity | 'ALL'> = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
const HOURS_OPTIONS = [24, 48, 168, 336, 720]
const SEV_ORDER: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 }

type AlertSortKey = 'severity' | 'vin' | 'confidence_score' | 'triggered_at' | 'estimated_cost_min'

export default function Alerts() {
  const [severity, setSeverity] = useState<Severity | 'ALL'>('ALL')
  const [hours, setHours]       = useState(168)
  const { sortKey, sortDir, onSort } = useSortable<AlertSortKey>('triggered_at', 'desc')

  const { data: raw, isLoading, refetch } = useFleetAlerts(
    hours,
    severity === 'ALL' ? undefined : severity,
  )

  const alertsData: Alert[] = Array.isArray(raw?.alerts) ? raw.alerts : []

  const sortedAlerts = [...alertsData].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1
    if (sortKey === 'severity') return (SEV_ORDER[a.severity] - SEV_ORDER[b.severity]) * dir
    const av: any = (a as any)[sortKey] ?? ''
    const bv: any = (b as any)[sortKey] ?? ''
    if (av < bv) return -dir
    if (av > bv) return  dir
    return 0
  })

  const severityCounts = SEVERITIES.slice(1).reduce((acc, s) => {
    acc[s] = alertsData.filter(a => a.severity === s).length
    return acc
  }, {} as Record<string, number>)

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Fleet Alerts</h1>
          <p className="text-gray-500 text-sm mt-1">{raw?.count ?? 0} total alerts in last {hours}h</p>
        </div>
        <button
          data-testid="alerts-refresh"
          onClick={() => refetch()}
          className="text-sm text-blue-600 hover:text-blue-800 font-medium border border-blue-200 px-3 py-1.5 rounded-lg"
        >
          ↻ Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Severity filter */}
        <div className="flex gap-1.5 flex-wrap">
          {SEVERITIES.map(s => (
            <button
              key={s}
              data-testid={`filter-${s.toLowerCase()}`}
              onClick={() => setSeverity(s)}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors border ${
                severity === s
                  ? s === 'CRITICAL' ? 'bg-red-600 text-white border-red-600'
                  : s === 'HIGH'     ? 'bg-orange-500 text-white border-orange-500'
                  : s === 'MEDIUM'   ? 'bg-yellow-500 text-white border-yellow-500'
                  : s === 'LOW'      ? 'bg-blue-500 text-white border-blue-500'
                  : 'bg-gray-800 text-white border-gray-800'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
              }`}
            >
              {s} {s !== 'ALL' && severityCounts[s] !== undefined ? `(${severityCounts[s]})` : ''}
            </button>
          ))}
        </div>

        {/* Hours filter */}
        <select
          data-testid="alerts-hours"
          aria-label="Time range"
          value={hours}
          onChange={e => setHours(Number(e.target.value))}
          className="border border-gray-300 rounded-lg px-2.5 py-1.5 text-xs text-gray-600"
        >
          {HOURS_OPTIONS.map(h => (
            <option key={h} value={h}>{h < 48 ? `${h}h` : `${h / 24}d`}</option>
          ))}
        </select>
      </div>

      {/* Alert table */}
      {isLoading ? (
        <div className="text-gray-400 text-sm">Loading alerts…</div>
      ) : alertsData.length === 0 ? (
        <div className="card text-center py-16">
          <span className="text-5xl">✅</span>
          <p className="text-gray-500 mt-4 font-medium">No {severity !== 'ALL' ? severity : ''} alerts in this period</p>
        </div>
      ) : (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <SortTh label="Severity"   col="severity"           cur={sortKey} dir={sortDir} onSort={onSort} />
                <SortTh label="VIN"        col="vin"                cur={sortKey} dir={sortDir} onSort={onSort} />
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Alert</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Recommended Action</th>
                <SortTh label="Cost (INR)" col="estimated_cost_min" cur={sortKey} dir={sortDir} onSort={onSort} />
                <SortTh label="Confidence" col="confidence_score"   cur={sortKey} dir={sortDir} onSort={onSort} />
                <SortTh label="Time"       col="triggered_at"       cur={sortKey} dir={sortDir} onSort={onSort} />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sortedAlerts.map((a, i) => (
                <tr
                  key={i}
                  className={`hover:bg-gray-50 transition-colors ${
                    a.severity === 'CRITICAL' ? 'bg-red-50/40' :
                    a.severity === 'HIGH'     ? 'bg-orange-50/40' : ''
                  }`}
                >
                  <td className="px-4 py-3">
                    <AlertBadge severity={a.severity} />
                  </td>
                  <td className="px-4 py-3">
                    <Link to={`/vehicles/${a.vin}`} className="font-mono text-xs text-blue-600 hover:underline">
                      {a.vin}
                    </Link>
                  </td>
                  <td className="px-4 py-3">
                    <p className="font-medium text-gray-900 text-xs">{a.title}</p>
                    <p className="text-gray-500 text-xs mt-0.5 max-w-xs truncate">{a.message_customer}</p>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-600 max-w-xs truncate">
                    {a.recommended_action}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-700 tabular-nums whitespace-nowrap">
                    ₹{a.estimated_cost_min?.toLocaleString('en-IN')} – {a.estimated_cost_max?.toLocaleString('en-IN')}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      <div className="w-12 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${(a.confidence_score ?? 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">{Math.round((a.confidence_score ?? 0) * 100)}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-400 whitespace-nowrap">
                    {new Date(a.triggered_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
