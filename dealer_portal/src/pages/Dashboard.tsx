import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { getFleetHealth, getMaintenanceCalendar, getInventory, getVehicles } from '../api/client'
import type { FleetHealth, InventoryItem, VehicleRow, MaintenanceEvent } from '../types'

type SortKey = 'vin' | 'model_name' | 'fuel_type' | 'health_score' | 'active_alert_count'
type SortDir = 'asc' | 'desc'

function isOnline(v: any): boolean {
  const ls = v.last_seen
  if (!ls) return false
  return (Date.now() - new Date(ls).getTime()) < 2 * 60 * 60 * 1000  // within 2h
}

function SortHeader({ label, sortKey, current, dir, onSort }: {
  label: string; sortKey: SortKey; current: SortKey; dir: SortDir; onSort: (k: SortKey) => void
}) {
  const active = current === sortKey
  return (
    <th
      className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap cursor-pointer select-none hover:text-gray-800 hover:bg-gray-100 transition-colors"
      onClick={() => onSort(sortKey)}
    >
      <span className="flex items-center gap-1">
        {label}
        <span className={`text-gray-300 ${active ? 'text-blue-500' : ''}`}>
          {active ? (dir === 'asc' ? '↑' : '↓') : '↕'}
        </span>
      </span>
    </th>
  )
}

const DEALER_CODE = localStorage.getItem('ap_dealer_code') ?? 'DL001'

function KpiCard({
  label, value, sub, emoji, bg, to,
}: {
  label: string; value: string | number; sub?: string; emoji: string; bg: string; to?: string
}) {
  const navigate = useNavigate()
  return (
    <div
      className={`card flex items-start gap-4 ${to ? 'cursor-pointer hover:shadow-md transition-shadow' : ''}`}
      onClick={to ? () => navigate(to) : undefined}
    >
      <div className={`rounded-xl p-3 text-2xl leading-none flex-shrink-0 ${bg}`}>{emoji}</div>
      <div className="min-w-0">
        <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">{label}</p>
        <p className="text-2xl font-bold text-gray-900 mt-0.5 tabular-nums">{value}</p>
        {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [vinSearch, setVinSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('health_score')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  function handleSort(key: SortKey) {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('asc') }
  }

  const { data: health } = useQuery<FleetHealth>({
    queryKey: ['fleet', 'health'],
    queryFn: getFleetHealth,
    refetchInterval: 60_000,
  })

  const { data: calRaw } = useQuery({
    queryKey: ['fleet', 'maintenance', 7],
    queryFn: () => getMaintenanceCalendar(7),
  })
  const calendar: MaintenanceEvent[] = Array.isArray(calRaw) ? calRaw : []

  const { data: invRaw = [] } = useQuery({
    queryKey: ['dealer', DEALER_CODE, 'inventory'],
    queryFn: () => getInventory(DEALER_CODE),
  })
  const inventory = invRaw as InventoryItem[]

  const { data: vehiclesRaw = [] } = useQuery({
    queryKey: ['vehicles', { limit: 100 }],
    queryFn: () => getVehicles({ limit: 100 }),
    refetchInterval: 120_000,
  })
  const vehicles = vehiclesRaw as VehicleRow[]

  const reorderCount = inventory.filter(i => i.reorder_qty > 0).length
  const avgScore     = health?.fleet_avg_health_score ?? 0

  const filteredVehicles = useMemo(() => {
    const q = vinSearch.toLowerCase()
    const filtered = vehicles.filter(v =>
      !q || v.vin?.toLowerCase().includes(q) || v.model_name?.toLowerCase().includes(q)
        || (v as any).driver_profile?.toLowerCase().includes(q)
    )
    return [...filtered].sort((a, b) => {
      let av: any = (a as any)[sortKey] ?? ''
      let bv: any = (b as any)[sortKey] ?? ''
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av
      }
      av = String(av).toLowerCase(); bv = String(bv).toLowerCase()
      return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
    })
  }, [vehicles, vinSearch, sortKey, sortDir])

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fleet Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          Real-time health and predictive maintenance: {DEALER_CODE}
        </p>
      </div>

      {/* KPI grid — each card links to its relevant page */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard label="Total Vehicles" value={health?.total_vehicles ?? '—'}
          sub={`${health?.online_now ?? 0} online now`} emoji="🚗" bg="bg-blue-50" />
        <KpiCard label="Critical Alerts" value={health?.active_alerts_critical ?? 0}
          sub={`${health?.active_alerts_high ?? 0} high · ${health?.active_alerts_medium ?? 0} medium`}
          emoji="🔴" bg="bg-red-50" to="/alerts" />
        <KpiCard label="Vehicles in Service" value={health?.vehicles_due_service ?? 0}
          sub="HIGH / CRITICAL alert pending" emoji="🔧" bg="bg-orange-50" to="/service-bay" />
        <KpiCard label="Fleet Health Score" value={`${Math.round(avgScore)}%`}
          sub={avgScore >= 80 ? 'Good' : avgScore >= 60 ? 'Fair' : 'Needs attention'}
          emoji="❤️" bg={avgScore >= 80 ? 'bg-green-50' : avgScore >= 60 ? 'bg-yellow-50' : 'bg-red-50'} />
        <KpiCard label="Upcoming Service (7d)" value={calendar.length}
          sub="Predicted maintenance events" emoji="📅" bg="bg-purple-50" to="/service-bay" />
        <KpiCard label="Parts to Reorder" value={reorderCount}
          sub={reorderCount > 0 ? 'Below safety stock' : 'All levels OK'}
          emoji={reorderCount > 0 ? '📦' : '✅'}
          bg={reorderCount > 0 ? 'bg-amber-50' : 'bg-green-50'} to="/inventory" />
        <KpiCard label="EV / PHEV Count"
          value={vehicles.filter(v => v.fuel_type === 'EV').length}
          sub={`${vehicles.filter(v => v.fuel_type === 'PHEV').length} PHEV · ${vehicles.filter(v => !['EV','PHEV'].includes(v.fuel_type)).length} ICE`}
          emoji="⚡" bg="bg-teal-50" />
        <KpiCard label="High Alerts" value={health?.active_alerts_high ?? 0}
          sub="Need attention within 1h" emoji="🟠" bg="bg-orange-50" to="/alerts" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Vehicle table */}
        <div className="xl:col-span-2 card p-0 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-200 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">Fleet Vehicles</h2>
              <span className="text-xs text-gray-400">{vehicles.length} total</span>
            </div>
            <form onSubmit={e => { e.preventDefault(); const q = vinSearch.trim(); if (q) navigate(`/vehicles/${q}`) }} className="flex gap-2">
              <input type="text" value={vinSearch} onChange={e => setVinSearch(e.target.value)}
                placeholder="Search by VIN, model, or driver profile..."
                className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
              <button type="submit" className="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-blue-700">Go</button>
            </form>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <SortHeader label="VIN"    sortKey="vin"              current={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Model"  sortKey="model_name"       current={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Fuel"   sortKey="fuel_type"        current={sortKey} dir={sortDir} onSort={handleSort} />
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Driver</th>
                  <SortHeader label="Health" sortKey="health_score"     current={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortHeader label="Alerts" sortKey="active_alert_count" current={sortKey} dir={sortDir} onSort={handleSort} />
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredVehicles.slice(0, 30).map(v => {
                  const score  = Number(v.health_score ?? 0)
                  const online = isOnline(v)
                  return (
                    <tr key={v.vin} className="hover:bg-blue-50 transition-colors cursor-pointer" onClick={() => navigate(`/vehicles/${v.vin}`)}>
                      <td className="px-4 py-2.5 font-mono text-xs text-gray-600">{v.vin?.slice(-8) ?? '—'}</td>
                      <td className="px-4 py-2.5 font-medium text-gray-700 whitespace-nowrap">{v.model_name ?? '—'}</td>
                      <td className="px-4 py-2.5">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          v.fuel_type === 'EV' ? 'bg-green-50 text-green-700' :
                          v.fuel_type === 'PHEV' ? 'bg-teal-50 text-teal-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>{v.fuel_type}</span>
                      </td>
                      <td className="px-4 py-2.5 text-xs text-gray-500 capitalize">{(v as any).driver_profile?.replace(/_/g, ' ') ?? '—'}</td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full ${score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-400' : 'bg-red-500'}`}
                              style={{ width: `${score}%` }} />
                          </div>
                          <span className={`text-xs font-bold tabular-nums ${score >= 80 ? 'text-green-600' : score >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                            {score > 0 ? `${Math.round(score)}%` : '—'}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        {v.active_alert_count > 0
                          ? <span className="text-xs font-bold text-red-600">{v.active_alert_count} alerts</span>
                          : <span className="text-xs text-green-600">OK</span>}
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${
                          online ? 'bg-green-50 text-green-700' : 'bg-gray-100 text-gray-400'
                        }`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-green-500' : 'bg-gray-300'}`} />
                          {online ? 'Online' : 'Offline'}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">
                        <Link to={`/vehicles/${v.vin}`} className="text-xs text-blue-600 hover:underline font-medium"
                          onClick={e => e.stopPropagation()}>Detail</Link>
                      </td>
                    </tr>
                  )
                })}
                {vehicles.length === 0 && (
                  <tr><td colSpan={8} className="px-4 py-10 text-center text-gray-400 text-sm">Loading fleet data...</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Upcoming service — clickable items */}
        <div className="card p-0 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
            <h2 className="font-semibold text-gray-900">Upcoming Service (7d)</h2>
            <Link to="/service-bay" className="text-xs text-blue-600 hover:underline">View all</Link>
          </div>
          <div className="divide-y divide-gray-100 max-h-96 overflow-y-auto">
            {calendar.length === 0 && (
              <p className="px-5 py-8 text-center text-gray-400 text-sm">No events in next 7 days</p>
            )}
            {calendar.map((ev, i) => (
              <Link key={i} to={`/vehicles/${ev.vin}`} className="block px-5 py-3 hover:bg-blue-50 transition-colors">
                <div className="flex items-center justify-between gap-2">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                    ev.severity === 'CRITICAL' ? 'bg-red-100 text-red-700' :
                    ev.severity === 'HIGH'     ? 'bg-orange-100 text-orange-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>{ev.severity}</span>
                  <span className="text-xs text-gray-400">{ev.days_until}d</span>
                </div>
                <p className="text-xs font-medium text-gray-800 mt-1">{ev.alert_type?.replace(/_/g, ' ')}</p>
                <div className="flex justify-between mt-0.5">
                  <p className="text-xs text-gray-500">{ev.model_name}</p>
                  <p className="text-xs font-mono text-gray-400">{ev.vin?.slice(-8)}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
