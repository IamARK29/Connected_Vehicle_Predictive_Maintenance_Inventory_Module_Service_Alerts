import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { getFleetHealth, getMaintenanceCalendar, getInventory, getVehicles } from '../api/client'
import type { FleetHealth, InventoryItem, VehicleRow, MaintenanceEvent } from '../types'

const DEALER_CODE = localStorage.getItem('ap_dealer_code') ?? 'DL001'

function KpiCard({
  label, value, sub, emoji, bg,
}: {
  label: string; value: string | number; sub?: string; emoji: string; bg: string
}) {
  return (
    <div className="card flex items-start gap-4">
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

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fleet Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          Real-time health and predictive maintenance — {DEALER_CODE}
        </p>
      </div>

      {/* KPI grid */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          label="Total Vehicles"
          value={health?.total_vehicles ?? '—'}
          sub={`${health?.online_now ?? 0} online now`}
          emoji="🚗"
          bg="bg-blue-50"
        />
        <KpiCard
          label="Critical Alerts"
          value={health?.active_alerts_critical ?? 0}
          sub={`${health?.active_alerts_high ?? 0} high · ${health?.active_alerts_medium ?? 0} medium`}
          emoji="🔴"
          bg="bg-red-50"
        />
        <KpiCard
          label="Vehicles in Service"
          value={health?.vehicles_due_service ?? 0}
          sub="HIGH / CRITICAL alert pending"
          emoji="🔧"
          bg="bg-orange-50"
        />
        <KpiCard
          label="Fleet Health Score"
          value={`${Math.round(avgScore)}%`}
          sub={avgScore >= 80 ? 'Good ✓' : avgScore >= 60 ? 'Fair ⚠' : 'Needs attention ✗'}
          emoji="❤️"
          bg={avgScore >= 80 ? 'bg-green-50' : avgScore >= 60 ? 'bg-yellow-50' : 'bg-red-50'}
        />
        <KpiCard
          label="Upcoming Service (7d)"
          value={calendar.length}
          sub="Predicted maintenance events"
          emoji="📅"
          bg="bg-purple-50"
        />
        <KpiCard
          label="Parts to Reorder"
          value={reorderCount}
          sub={reorderCount > 0 ? 'Below safety stock' : 'All levels OK'}
          emoji={reorderCount > 0 ? '📦' : '✅'}
          bg={reorderCount > 0 ? 'bg-amber-50' : 'bg-green-50'}
        />
        <KpiCard
          label="EV / PHEV Count"
          value={vehicles.filter(v => v.fuel_type === 'EV').length}
          sub={`${vehicles.filter(v => v.fuel_type === 'PHEV').length} PHEV · ${vehicles.filter(v => !['EV','PHEV'].includes(v.fuel_type)).length} ICE`}
          emoji="⚡"
          bg="bg-teal-50"
        />
        <KpiCard
          label="High Alerts"
          value={health?.active_alerts_high ?? 0}
          sub="Need attention within 1h"
          emoji="🟠"
          bg="bg-orange-50"
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Vehicle table */}
        <div className="xl:col-span-2 card p-0 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-200 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">Fleet Vehicles</h2>
              <span className="text-xs text-gray-400">{vehicles.length} total</span>
            </div>
            <form
              onSubmit={e => {
                e.preventDefault()
                const q = vinSearch.trim()
                if (q) navigate(`/vehicles/${q}`)
              }}
              className="flex gap-2"
            >
              <input
                type="text"
                value={vinSearch}
                onChange={e => setVinSearch(e.target.value)}
                placeholder="Search by VIN (e.g. MZ7X...)"
                className="flex-1 border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm font-medium hover:bg-blue-700"
              >
                Go
              </button>
            </form>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  {['Plate', 'Model', 'Fuel', 'Health', 'Alerts', ''].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {vehicles
                  .filter(v => !vinSearch || v.vin?.toLowerCase().includes(vinSearch.toLowerCase()) || v.license_plate?.toLowerCase().includes(vinSearch.toLowerCase()) || v.model_name?.toLowerCase().includes(vinSearch.toLowerCase()))
                  .slice(0, 20).map(v => {
                  const score = Number(v.health_score ?? 80)
                  return (
                    <tr key={v.vin} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-2.5 font-medium text-gray-700">{v.license_plate ?? '—'}</td>
                      <td className="px-4 py-2.5 text-gray-600 whitespace-nowrap">{v.model_name ?? '—'}</td>
                      <td className="px-4 py-2.5">
                        <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 font-medium">{v.fuel_type}</span>
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${score >= 80 ? 'bg-green-500' : score >= 60 ? 'bg-yellow-400' : 'bg-red-500'}`}
                              style={{ width: `${score}%` }}
                            />
                          </div>
                          <span className={`text-xs font-bold ${score >= 80 ? 'text-green-600' : score >= 60 ? 'text-yellow-600' : 'text-red-600'}`}>
                            {Math.round(score)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        {v.active_alert_count > 0
                          ? <span className="text-xs font-bold text-red-600">{v.active_alert_count} alerts</span>
                          : <span className="text-xs text-gray-400">—</span>}
                      </td>
                      <td className="px-4 py-2.5">
                        <Link to={`/vehicles/${v.vin}`} className="text-xs text-blue-600 hover:underline font-medium">View →</Link>
                      </td>
                    </tr>
                  )
                })}
                {vehicles.length === 0 && (
                  <tr><td colSpan={6} className="px-4 py-10 text-center text-gray-400 text-sm">Loading fleet data…</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Upcoming service */}
        <div className="card p-0 overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-200">
            <h2 className="font-semibold text-gray-900">Upcoming Service (7d)</h2>
          </div>
          <div className="divide-y divide-gray-100 max-h-96 overflow-y-auto">
            {calendar.length === 0 && (
              <p className="px-5 py-8 text-center text-gray-400 text-sm">No events in next 7 days</p>
            )}
            {calendar.map((ev, i) => (
              <div key={i} className="px-5 py-3 hover:bg-gray-50">
                <div className="flex items-center justify-between gap-2">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                    ev.severity === 'CRITICAL' ? 'bg-red-100 text-red-700' :
                    ev.severity === 'HIGH'     ? 'bg-orange-100 text-orange-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>{ev.severity}</span>
                  <span className="text-xs text-gray-400">{ev.days_until}d</span>
                </div>
                <p className="text-xs font-medium text-gray-800 mt-1">{ev.alert_type?.replace(/_/g, ' ')}</p>
                <p className="text-xs text-gray-500 font-mono">{ev.license_plate}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
