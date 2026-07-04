import { useInventory, useDemandForecast } from '../api/hooks'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { InventoryItem, DemandForecast } from '../types'

const DEALER_CODE = localStorage.getItem('ap_dealer_code') ?? 'DL001'

export default function Inventory() {
  const { data: invRaw = [], isLoading: invLoading }  = useInventory(DEALER_CODE)
  const { data: fcastRaw = [], isLoading: fcLoading } = useDemandForecast(DEALER_CODE)

  const inventory = invRaw as InventoryItem[]
  const forecast  = fcastRaw as DemandForecast[]

  const reorderItems  = inventory.filter(i => i.reorder_qty > 0)
  const outOfStock    = inventory.filter(i => !i.in_stock)

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Parts Inventory</h1>
        <p className="text-gray-500 text-sm mt-1">Stock levels and 30/90-day demand forecast — {DEALER_CODE}</p>
      </div>

      {/* Summary badges */}
      <div className="flex gap-3 flex-wrap">
        <span className="px-3 py-1.5 rounded-full text-sm font-medium bg-blue-50 text-blue-800 border border-blue-200">
          {inventory.length} parts tracked
        </span>
        {outOfStock.length > 0 && (
          <span className="px-3 py-1.5 rounded-full text-sm font-medium bg-red-50 text-red-800 border border-red-200">
            {outOfStock.length} out of stock
          </span>
        )}
        {reorderItems.length > 0 && (
          <span className="px-3 py-1.5 rounded-full text-sm font-medium bg-amber-50 text-amber-800 border border-amber-200">
            {reorderItems.length} need reorder
          </span>
        )}
      </div>

      {/* Inventory table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-200">
          <h2 className="font-semibold text-gray-900">Stock Levels</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                {['Part Code', 'Description', 'In Stock', 'Qty', 'Reorder Qty', 'Status'].map(h => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {invLoading && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-gray-400">Loading inventory…</td></tr>
              )}
              {inventory.map(item => {
                const isLow       = item.reorder_qty > 0
                const isOutStock  = !item.in_stock
                return (
                  <tr
                    key={item.part_code}
                    className={`hover:bg-gray-50 ${isOutStock ? 'bg-red-50' : isLow ? 'bg-amber-50' : ''}`}
                  >
                    <td className="px-4 py-2.5 font-mono text-xs text-gray-700">{item.part_code}</td>
                    <td className="px-4 py-2.5 text-gray-700">{item.description}</td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs font-bold ${item.in_stock ? 'text-green-600' : 'text-red-600'}`}>
                        {item.in_stock ? '✓ Yes' : '✗ No'}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-bold tabular-nums">
                      <span className={item.qty === 0 ? 'text-red-600' : item.qty <= 2 ? 'text-amber-600' : 'text-gray-900'}>
                        {item.qty}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-gray-600">{item.reorder_qty}</td>
                    <td className="px-4 py-2.5">
                      {isOutStock ? (
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-800">OUT OF STOCK</span>
                      ) : isLow ? (
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800">REORDER</span>
                      ) : (
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-800">OK</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Demand forecast chart */}
      {(forecast.length > 0 || fcLoading) && (
        <div className="card">
          <h2 className="font-semibold text-gray-900 mb-4">Demand Forecast (top parts)</h2>
          {fcLoading ? (
            <div className="h-64 flex items-center justify-center text-gray-400 text-sm">Calculating forecast…</div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={forecast.slice(0, 10)} margin={{ top: 4, right: 8, bottom: 40, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="part_code" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                  formatter={(v, name) => [`${v} units`, name]}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="demand_30d" name="30-day demand" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="demand_90d" name="90-day demand" fill="#93c5fd" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  )
}
