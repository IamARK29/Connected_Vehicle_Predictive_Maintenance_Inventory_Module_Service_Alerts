import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useDriverScores } from '../api/hooks'
import { SortTh, useSortable } from '../components/SortHeader'
import type { DriverScore } from '../types'

type DSSortKey = 'score' | 'vin' | 'risk_category' | 'percentile'

const RISK_CONFIG: Record<string, { bg: string; text: string; label: string }> = {
  low:    { bg: 'bg-green-100',  text: 'text-green-800',  label: 'Low Risk' },
  medium: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Medium Risk' },
  high:   { bg: 'bg-red-100',    text: 'text-red-800',    label: 'High Risk' },
}

function ScoreBar({ score }: { score: number }) {
  const color = score >= 70 ? 'bg-green-500' : score >= 50 ? 'bg-yellow-400' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className={`text-sm font-bold tabular-nums ${score >= 70 ? 'text-green-600' : score >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
        {score.toFixed(1)}
      </span>
    </div>
  )
}

const RISK_ORDER: Record<string, number> = { high: 0, medium: 1, low: 2 }

export default function DriverScores() {
  const { data: scoresRaw = [], isLoading } = useDriverScores()
  const scores = scoresRaw as DriverScore[]
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const { sortKey, sortDir, onSort } = useSortable<DSSortKey>('score')

  const riskFiltered = filter === 'all' ? scores : scores.filter(s => s.risk_category === filter)

  const filtered = useMemo(() => [...riskFiltered].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1
    if (sortKey === 'risk_category') {
      return ((RISK_ORDER[a.risk_category ?? 'medium'] ?? 1) - (RISK_ORDER[b.risk_category ?? 'medium'] ?? 1)) * dir
    }
    const av: any = (a as any)[sortKey] ?? ''
    const bv: any = (b as any)[sortKey] ?? ''
    if (av < bv) return -dir
    if (av > bv) return  dir
    return 0
  }), [riskFiltered, sortKey, sortDir])

  const counts = {
    high:   scores.filter(s => s.risk_category === 'high').length,
    medium: scores.filter(s => s.risk_category === 'medium').length,
    low:    scores.filter(s => s.risk_category === 'low').length,
  }
  const avg = scores.length ? (scores.reduce((a, s) => a + s.score, 0) / scores.length) : 0

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Driver Scores</h1>
        <p className="text-gray-500 text-sm mt-1">
          Ranked by composite drive score from trip data (harsh braking, acceleration, speed compliance, idle time).
          Lower score = higher risk.
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Fleet Avg Score', value: avg.toFixed(1), emoji: '📊', bg: 'bg-blue-50' },
          { label: 'High Risk',       value: counts.high,    emoji: '🔴', bg: 'bg-red-50' },
          { label: 'Medium Risk',     value: counts.medium,  emoji: '🟡', bg: 'bg-yellow-50' },
          { label: 'Low Risk',        value: counts.low,     emoji: '🟢', bg: 'bg-green-50' },
        ].map(c => (
          <div key={c.label} className={`card flex items-start gap-3 ${c.bg}`}>
            <span className="text-2xl">{c.emoji}</span>
            <div>
              <p className="text-xs text-gray-500 font-medium">{c.label}</p>
              <p className="text-2xl font-bold text-gray-900 tabular-nums">{c.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {(['all', 'high', 'medium', 'low'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold capitalize transition-colors border ${
              filter === f
                ? 'bg-blue-600 text-white border-blue-600'
                : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
            }`}
          >
            {f === 'all' ? 'All Drivers' : `${f} Risk`}
            {f !== 'all' && ` (${counts[f]})`}
          </button>
        ))}
      </div>

      {/* Leaderboard table */}
      <div className="card p-0 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Rank</th>
              <SortTh label="Vehicle"        col="vin"           cur={sortKey} dir={sortDir} onSort={onSort} />
              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Driver Profile</th>
              <SortTh label="Score"          col="score"         cur={sortKey} dir={sortDir} onSort={onSort} />
              <SortTh label="Risk"           col="risk_category" cur={sortKey} dir={sortDir} onSort={onSort} />
              <SortTh label="Percentile"     col="percentile"    cur={sortKey} dir={sortDir} onSort={onSort} />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {isLoading && (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-400">Loading driver scores…</td></tr>
            )}
            {!isLoading && filtered.length === 0 && (
              <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-400">No matching drivers for this filter.</td></tr>
            )}
            {filtered.map((d, i) => {
              const risk   = (d.risk_category ?? 'medium').toLowerCase()
              const config = RISK_CONFIG[risk] ?? RISK_CONFIG.medium
              const medal  = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : null
              return (
                <tr key={d.vin} className={`hover:bg-gray-50 ${d.risk_category === 'high' ? 'bg-red-50/30' : ''}`}>
                  <td className="px-4 py-3 font-bold tabular-nums text-gray-500">
                    {medal ? <span>{medal} {d.rank}</span> : d.rank}
                  </td>
                  <td className="px-4 py-3">
                    <Link to={`/vehicles/${d.vin}`} className="font-mono text-xs text-blue-600 hover:underline">{d.vin}</Link>
                  </td>
                  <td className="px-4 py-3 text-gray-700 capitalize">{d.driver_name ?? d.license_plate ?? '—'}</td>
                  <td className="px-4 py-3"><ScoreBar score={d.score} /></td>
                  <td className="px-4 py-3">
                    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${config.bg} ${config.text}`}>
                      {config.label}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-500 tabular-nums text-xs">
                    {d.percentile !== undefined ? `${d.percentile.toFixed(0)}th pct` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
