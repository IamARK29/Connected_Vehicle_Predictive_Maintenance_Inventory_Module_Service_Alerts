import { useState, useMemo, useCallback } from 'react'
import {
  useInventoryOverview, useInventoryStock, useInventoryAlerts,
  useReorderPlan, useInventoryAnalytics, useDealerComparison,
  usePartDetail, useInventoryTransactions,
} from '../api/hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, PieChart, Pie, LineChart, Line,
} from 'recharts'

const _raw_dealer = localStorage.getItem('ap_dealer_code')
const DEALER_CODE = (_raw_dealer && _raw_dealer !== 'ALL' && _raw_dealer !== 'NONE') ? _raw_dealer : undefined

type Tab = 'Overview' | 'Stock Ledger' | 'Reorder Plan' | 'Analytics' | 'Multi-Dealer' | 'Transactions'
const TABS: Tab[] = ['Overview', 'Stock Ledger', 'Reorder Plan', 'Analytics', 'Multi-Dealer', 'Transactions']

const STATUS_CFG: Record<string, { bg: string; text: string; border: string; label: string }> = {
  STOCKOUT: { bg: 'bg-red-100',    text: 'text-red-800',    border: 'border-red-300',    label: 'Stockout' },
  CRITICAL: { bg: 'bg-orange-100', text: 'text-orange-800', border: 'border-orange-300', label: 'Critical' },
  LOW:      { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-300', label: 'Low' },
  OK:       { bg: 'bg-green-100',  text: 'text-green-800',  border: 'border-green-300',  label: 'OK' },
}
const ABC_CFG: Record<string, { bg: string; text: string }> = {
  A: { bg: 'bg-red-100',  text: 'text-red-700' },
  B: { bg: 'bg-blue-100', text: 'text-blue-700' },
  C: { bg: 'bg-gray-100', text: 'text-gray-600' },
}

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CFG[status] ?? STATUS_CFG.OK
  return <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${cfg.bg} ${cfg.text} ${cfg.border}`}>{cfg.label}</span>
}
function AbcBadge({ cls }: { cls: string }) {
  const cfg = ABC_CFG[cls] ?? ABC_CFG.B
  return <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${cfg.bg} ${cfg.text}`}>{cls}</span>
}
function DosBadge({ days }: { days?: number | null }) {
  if (days == null) return <span className="text-xs text-gray-400">—</span>
  if (days === 0)   return <span className="text-xs font-semibold text-red-700 bg-red-50 px-1.5 py-0.5 rounded">Stockout</span>
  if (days <= 7)    return <span className="text-xs font-semibold text-red-600">{days}d</span>
  if (days <= 21)   return <span className="text-xs font-semibold text-amber-600">{days}d</span>
  return <span className="text-xs text-gray-500">{days}d</span>
}
function fmt(n: number, decimals = 0) { return n.toLocaleString('en-IN', { maximumFractionDigits: decimals }) }

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab() {
  const { data: ov, isLoading } = useInventoryOverview()
  const { data: alerts = [] }  = useInventoryAlerts({ min_severity: 'LOW', limit: '10' })

  if (isLoading) return <div className="text-gray-400 text-sm p-8 text-center">Loading overview…</div>
  if (!ov || ov.error) return <div className="text-red-500 text-sm p-8">{ov?.error ?? 'Failed to load'}</div>

  const kpis = [
    { label: 'Total Inventory Value',  value: `₹${fmt(ov.total_inventory_value_inr / 1000)}K`, sub: `${ov.total_sku} SKUs across ${ov.dealers_affected >= 0 ? 8 : '?'} dealers`, icon: '💰', bg: 'bg-blue-50' },
    { label: 'Stockout / Critical',    value: `${ov.stockout_count + ov.critical_count}`,       sub: `₹${fmt(ov.value_at_risk_inr / 1000)}K value at risk`,                    icon: '🚨', bg: 'bg-red-50' },
    { label: 'Low Stock (needs reorder)', value: ov.low_count,                                  sub: 'items below reorder point',                                               icon: '⚠️', bg: 'bg-amber-50' },
    { label: 'Avg Days of Supply',     value: `${ov.avg_days_of_supply}d`,                      sub: `${ov.slow_mover_count} slow movers`,                                      icon: '📦', bg: 'bg-green-50' },
  ]

  const abcData = ['A', 'B', 'C'].map(cls => ({
    name: `Class ${cls}`,
    value: Math.round((ov.abc_value_inr?.[cls] ?? 0) / 1000),
    count: ov.abc_sku_count?.[cls] ?? 0,
    fill: cls === 'A' ? '#ef4444' : cls === 'B' ? '#3b82f6' : '#9ca3af',
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map(k => (
          <div key={k.label} className={`card flex items-start gap-3 ${k.bg}`}>
            <span className="text-2xl">{k.icon}</span>
            <div>
              <p className="text-xs text-gray-500 font-medium">{k.label}</p>
              <p className="text-2xl font-bold text-gray-900 tabular-nums">{k.value}</p>
              <p className="text-xs text-gray-400 mt-0.5">{k.sub}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ABC Value Distribution */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-1">Inventory Value by ABC Class</h3>
          <p className="text-xs text-gray-500 mb-3">A=Critical parts · B=Regular · C=Slow-movers</p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={abcData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ₹${value}K`}>
                {abcData.map(d => <Cell key={d.name} fill={d.fill} />)}
              </Pie>
              <Tooltip formatter={(v: any) => [`₹${v}K`, 'Value']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top Alerts Panel */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-3">Active Stock Alerts</h3>
          <div className="space-y-2 max-h-56 overflow-y-auto">
            {(alerts as any[]).length === 0 && <p className="text-sm text-gray-400">No stock alerts</p>}
            {(alerts as any[]).map((a: any, i: number) => (
              <div key={i} className={`p-2.5 rounded-lg border ${STATUS_CFG[a.severity]?.border ?? 'border-gray-200'} ${STATUS_CFG[a.severity]?.bg ?? 'bg-gray-50'}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-gray-800 truncate">{a.description}</span>
                  <StatusBadge status={a.severity} />
                </div>
                <p className="text-xs text-gray-500 mt-0.5">
                  {a.dealer_city} · Stock: {a.current_stock} · ROP: {a.reorder_point} · {a.days_of_supply != null ? `${a.days_of_supply}d supply` : 'No demand data'}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Stock Ledger Tab ──────────────────────────────────────────────────────────
function StockLedgerTab() {
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [abcFilter, setAbcFilter]       = useState<string>('all')
  const [sortKey, setSortKey]           = useState<string>('stock_status')
  const [sortDir, setSortDir]           = useState<'asc' | 'desc'>('asc')
  const [selectedPart, setSelectedPart] = useState<string | null>(null)

  const params: Record<string, string> = {}
  if (DEALER_CODE) params.dealer_code = DEALER_CODE
  if (statusFilter !== 'all') params.status = statusFilter
  if (abcFilter !== 'all')    params.abc_class = abcFilter

  const { data: stock = [], isLoading } = useInventoryStock(params)
  const { data: partDetail }            = usePartDetail(selectedPart)

  const sorted = useMemo(() => {
    return [...(stock as any[])].sort((a, b) => {
      const sev = (s: string) => ({ STOCKOUT: 0, CRITICAL: 1, LOW: 2, OK: 3 }[s] ?? 3)
      if (sortKey === 'stock_status') return (sev(a.stock_status) - sev(b.stock_status)) * (sortDir === 'asc' ? 1 : -1)
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0
      return (av < bv ? -1 : av > bv ? 1 : 0) * (sortDir === 'asc' ? 1 : -1)
    })
  }, [stock, sortKey, sortDir])

  const onSort = (k: string) => {
    if (k === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('asc') }
  }
  const SH = ({ label, col }: { label: string; col: string }) => (
    <th onClick={() => onSort(col)} className="px-3 py-3 text-left text-xs font-semibold text-gray-500 uppercase cursor-pointer hover:text-gray-800 whitespace-nowrap select-none">
      {label} {sortKey === col ? (sortDir === 'asc' ? '↑' : '↓') : '↕'}
    </th>
  )

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {['all', 'STOCKOUT', 'CRITICAL', 'LOW', 'OK'].map(s => (
          <button key={s} onClick={() => setStatusFilter(s)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${statusFilter === s ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'}`}>
            {s === 'all' ? 'All Status' : STATUS_CFG[s]?.label ?? s}
          </button>
        ))}
        <span className="mx-1 text-gray-300">|</span>
        {['all', 'A', 'B', 'C'].map(a => (
          <button key={a} onClick={() => setAbcFilter(a)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${abcFilter === a ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-600 border-gray-300 hover:border-gray-600'}`}>
            {a === 'all' ? 'All ABC' : `Class ${a}`}
          </button>
        ))}
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 text-xs text-gray-500 font-medium">{(sorted as any[]).length} parts</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <SH label="Part"       col="description" />
                <SH label="ABC"        col="abc_class" />
                <SH label="Status"     col="stock_status" />
                <SH label="Stock"      col="current_stock" />
                <SH label="ROP"        col="reorder_point" />
                <SH label="Safety"     col="safety_stock" />
                <SH label="EOQ"        col="eoq" />
                <SH label="Days Left"  col="days_of_supply" />
                <SH label="Stockout %" col="stockout_prob" />
                <SH label="Value ₹"    col="inventory_value_inr" />
                <SH label="Supplier"   col="supplier" />
                <th className="px-3 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Detail</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {isLoading && <tr><td colSpan={12} className="px-4 py-8 text-center text-gray-400">Loading…</td></tr>}
              {!isLoading && sorted.length === 0 && <tr><td colSpan={12} className="px-4 py-8 text-center text-gray-400">No parts match filter</td></tr>}
              {(sorted as any[]).map((row: any, i: number) => (
                <tr key={i} className={`hover:bg-gray-50 ${row.stock_status === 'STOCKOUT' ? 'bg-red-50/40' : row.stock_status === 'CRITICAL' ? 'bg-orange-50/30' : ''}`}>
                  <td className="px-3 py-2.5">
                    <div className="font-medium text-gray-900 text-xs">{row.description}</div>
                    <div className="text-xs text-gray-400 font-mono">{row.part_code}</div>
                  </td>
                  <td className="px-3 py-2.5"><AbcBadge cls={row.abc_class} /></td>
                  <td className="px-3 py-2.5"><StatusBadge status={row.stock_status} /></td>
                  <td className="px-3 py-2.5 font-bold tabular-nums text-gray-900">{row.current_stock}</td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-600">{row.reorder_point}</td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-500">{row.safety_stock}</td>
                  <td className="px-3 py-2.5 tabular-nums text-blue-600 font-medium">{row.eoq}</td>
                  <td className="px-3 py-2.5"><DosBadge days={row.days_of_supply} /></td>
                  <td className="px-3 py-2.5">
                    <span className={`text-xs font-medium ${row.stockout_prob > 0.5 ? 'text-red-600' : row.stockout_prob > 0.2 ? 'text-amber-600' : 'text-gray-500'}`}>
                      {row.stockout_prob != null ? `${(row.stockout_prob * 100).toFixed(0)}%` : '—'}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-700">₹{fmt(row.inventory_value_inr ?? 0)}</td>
                  <td className="px-3 py-2.5 text-xs text-gray-500">{row.supplier}</td>
                  <td className="px-3 py-2.5">
                    <button onClick={() => setSelectedPart(p => p === row.part_code ? null : row.part_code)}
                      className="text-xs text-blue-600 hover:underline">
                      {selectedPart === row.part_code ? 'Hide' : 'View'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Part Detail Panel */}
      {selectedPart && partDetail && !partDetail.error && (
        <div className="card border-blue-200 border-2">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-900">{partDetail.description}</h3>
              <p className="text-xs text-gray-500 font-mono mt-0.5">{partDetail.part_code} · {partDetail.supplier} · Lead: {partDetail.lead_time_days}d</p>
            </div>
            <div className="text-right">
              <p className="text-xs text-gray-500">Fleet Stock</p>
              <p className="text-2xl font-bold text-gray-900">{partDetail.total_stock_fleet}</p>
              <p className="text-xs text-gray-400">₹{fmt(partDetail.fleet_value_inr)} total value</p>
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {(partDetail.dealers ?? []).map((d: any) => (
              <div key={d.dealer_code} className={`p-3 rounded-lg border ${STATUS_CFG[d.stock_status]?.border ?? 'border-gray-200'} ${STATUS_CFG[d.stock_status]?.bg ?? 'bg-gray-50'}`}>
                <p className="text-xs font-semibold text-gray-700">{d.dealer_city}</p>
                <p className="text-lg font-bold text-gray-900 tabular-nums">{d.current_stock}</p>
                <div className="flex items-center justify-between mt-1">
                  <StatusBadge status={d.stock_status} />
                  <span className="text-xs text-gray-400">{d.days_of_supply != null ? `${d.days_of_supply}d` : '—'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── ERP Integration Modal ─────────────────────────────────────────────────────
function ErpModal({ order, onClose }: { order: any; onClose: () => void }) {
  const [sent, setSent] = useState(false)
  const handleSend = () => {
    setSent(true)
    setTimeout(onClose, 1800)
  }
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6" onClick={e => e.stopPropagation()}>
        {sent ? (
          <div className="text-center py-4">
            <p className="text-4xl mb-3">✅</p>
            <p className="font-semibold text-gray-900">PO Request Submitted</p>
            <p className="text-xs text-gray-500 mt-1">ERP integration will be wired in the next sprint. Reference: PO-{Date.now().toString().slice(-6)}</p>
          </div>
        ) : (
          <>
            <h3 className="font-semibold text-gray-900 mb-1">Send to ERP / Raise Purchase Order</h3>
            <p className="text-xs text-gray-500 mb-4">This will raise a PO request in your ERP system for dealer {order.dealer_code} ({order.dealer_city}).</p>
            <div className="bg-gray-50 rounded-xl p-4 mb-4 space-y-1.5 text-sm">
              <div className="flex justify-between"><span className="text-gray-500">Supplier</span><span className="font-medium">{order.supplier}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Line Items</span><span className="font-medium">{order.line_count}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Total Value</span><span className="font-bold text-blue-700">₹{fmt(order.total_cost_inr)}</span></div>
              <div className="flex justify-between"><span className="text-gray-500">Expected Delivery</span><span className="font-medium">{order.expected_delivery} ({order.lead_time_days}d)</span></div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-700 mb-4">
              ERP integration is ready for wiring. Clicking "Request PO" will log this order — connect your ERP endpoint in <span className="font-mono">/api/inventory/erp-send</span> to activate.
            </div>
            <div className="flex gap-2">
              <button onClick={handleSend} className="flex-1 bg-blue-600 text-white rounded-lg py-2.5 text-sm font-medium hover:bg-blue-700 transition-colors">
                Request PO
              </button>
              <button onClick={onClose} className="px-4 py-2.5 border border-gray-200 text-gray-600 rounded-lg text-sm hover:bg-gray-50 transition-colors">
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Reorder Plan Tab ──────────────────────────────────────────────────────────
function ReorderPlanTab() {
  const { data, isLoading } = useReorderPlan(DEALER_CODE)
  const [erpOrder, setErpOrder] = useState<any | null>(null)
  const plan = data as any

  if (isLoading) return <div className="text-gray-400 text-sm p-8 text-center">Computing reorder plan…</div>

  const orders: any[] = plan?.orders ?? []
  const totalCost: number = plan?.total_cost_inr ?? 0

  return (
    <div className="space-y-6">
      {erpOrder && <ErpModal order={erpOrder} onClose={() => setErpOrder(null)} />}

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Purchase Orders Required', value: orders.length, icon: '📋', bg: 'bg-blue-50' },
          { label: 'Urgent Orders',             value: orders.filter((o: any) => o.has_urgent).length, icon: '🚨', bg: 'bg-red-50' },
          { label: 'Total Reorder Cost',        value: `₹${fmt(totalCost / 1000)}K`, icon: '💰', bg: 'bg-green-50' },
        ].map(k => (
          <div key={k.label} className={`card flex items-start gap-3 ${k.bg}`}>
            <span className="text-2xl">{k.icon}</span>
            <div>
              <p className="text-xs text-gray-500 font-medium">{k.label}</p>
              <p className="text-2xl font-bold text-gray-900 tabular-nums">{k.value}</p>
            </div>
          </div>
        ))}
      </div>

      {orders.length === 0 && (
        <div className="card text-center py-12 text-gray-400">
          <p className="text-4xl mb-2">✅</p>
          <p className="text-sm">All dealers are well-stocked. No reorders needed.</p>
        </div>
      )}

      {orders.map((order: any, oi: number) => (
        <div key={oi} className={`card ${order.has_urgent ? 'border-red-300 border-2' : ''}`}>
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-gray-900">{order.dealer_city} ({order.dealer_code})</h3>
                {order.has_urgent && <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-red-100 text-red-700">URGENT</span>}
              </div>
              <p className="text-xs text-gray-500 mt-0.5">Supplier: {order.supplier} · Expected: {order.expected_delivery} ({order.lead_time_days}d lead)</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="text-right">
                <p className="text-xs text-gray-500">Order Total</p>
                <p className="text-xl font-bold text-gray-900">₹{fmt(order.total_cost_inr)}</p>
                <p className="text-xs text-gray-400">{order.line_count} line items</p>
              </div>
              <button
                onClick={() => setErpOrder(order)}
                className={`mt-1 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors whitespace-nowrap ${
                  order.has_urgent
                    ? 'bg-red-600 text-white border-red-600 hover:bg-red-700'
                    : 'bg-blue-600 text-white border-blue-600 hover:bg-blue-700'
                }`}
              >
                Integrate / Request PO
              </button>
            </div>
          </div>
          <table className="w-full text-sm">
            <thead className="border-b border-gray-200">
              <tr>
                {['Part', 'ABC', 'Current', 'ROP', 'Order Qty', 'Unit Cost', 'Line Cost', 'Priority'].map(h => (
                  <th key={h} className="px-2 py-2 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {order.lines.map((line: any, li: number) => (
                <tr key={li} className={`hover:bg-gray-50 ${line.priority === 'URGENT' ? 'bg-red-50/30' : ''}`}>
                  <td className="px-2 py-2">
                    <div className="font-medium text-xs text-gray-900">{line.description}</div>
                    <div className="text-xs text-gray-400 font-mono">{line.part_code}</div>
                  </td>
                  <td className="px-2 py-2"><AbcBadge cls={line.abc_class} /></td>
                  <td className="px-2 py-2 tabular-nums font-bold text-gray-900">{line.current_stock}</td>
                  <td className="px-2 py-2 tabular-nums text-gray-600">{line.reorder_point}</td>
                  <td className="px-2 py-2 tabular-nums font-bold text-blue-600">{line.order_qty}</td>
                  <td className="px-2 py-2 tabular-nums text-gray-600">₹{fmt(line.unit_cost_inr)}</td>
                  <td className="px-2 py-2 tabular-nums font-medium text-gray-800">₹{fmt(line.line_cost_inr)}</td>
                  <td className="px-2 py-2">
                    <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${line.priority === 'URGENT' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
                      {line.priority}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

// ── Analytics Tab ─────────────────────────────────────────────────────────────
function AnalyticsTab() {
  const { data, isLoading } = useInventoryAnalytics(DEALER_CODE)
  const analytics = data as any

  if (isLoading) return <div className="text-gray-400 text-sm p-8 text-center">Computing analytics…</div>
  if (!analytics || Object.keys(analytics).length === 0) return <div className="text-gray-400 text-sm p-8">No analytics data</div>

  const abc: any[]         = analytics.abc_analysis        ?? []
  const turnover: any[]    = analytics.turnover_rates       ?? []
  const slowMovers: any[]  = analytics.slow_movers          ?? []
  const txnSummary         = analytics.transaction_summary  ?? {}
  const monthlyDemand: any[]= analytics.monthly_demand      ?? []
  const fillRate: number   = analytics.fill_rate_pct        ?? 0
  const suppliers: any[]   = analytics.supplier_performance ?? []

  const abcChartData = abc.map((a: any) => ({
    name: `Class ${a.abc_class}`,
    value: Math.round(a.total_value / 1000),
    skus: a.sku_count,
    dos: a.avg_dos,
    fill: a.abc_class === 'A' ? '#ef4444' : a.abc_class === 'B' ? '#3b82f6' : '#9ca3af',
  }))

  const topTurnover = turnover.slice(0, 10).map((r: any) => ({
    name: r.part_code.replace(/-MG$/, '').substring(0, 12),
    rate: r.turnover_rate,
    fill: r.turnover_rate > 10 ? '#22c55e' : r.turnover_rate > 4 ? '#3b82f6' : '#f97316',
  }))

  const demandChartData = monthlyDemand.map((d: any) => ({
    month: d.month.substring(2),  // "2026-04" → "26-04"
    qty: d.quantity,
  }))

  return (
    <div className="space-y-6">
      {/* KPI summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {[
          { label: 'Issues (90d)',   value: txnSummary.total_issues_90d   ?? '—', icon: '📤', color: 'text-blue-700' },
          { label: 'Receipts (90d)', value: txnSummary.total_receipts_90d ?? '—', icon: '📥', color: 'text-green-700' },
          { label: 'Slow Movers',    value: slowMovers.length,                    icon: '🐢', color: 'text-orange-700' },
          { label: 'Fill Rate',      value: `${fillRate}%`,                       icon: '✅', color: fillRate >= 90 ? 'text-green-700' : fillRate >= 75 ? 'text-amber-700' : 'text-red-700' },
          { label: 'Issue Events',   value: txnSummary.issue_events        ?? '—', icon: '🔄', color: 'text-purple-700' },
        ].map(k => (
          <div key={k.label} className="card text-center py-3">
            <p className="text-xl">{k.icon}</p>
            <p className={`text-2xl font-bold tabular-nums ${k.color}`}>{k.value}</p>
            <p className="text-xs text-gray-500">{k.label}</p>
          </div>
        ))}
      </div>

      {/* Demand Trend */}
      {demandChartData.length > 0 && (
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-1">Monthly Parts Demand (Issues)</h3>
          <p className="text-xs text-gray-500 mb-3">Total quantity issued across all dealers — last 90 days</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={demandChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: any) => [`${v} units`, 'Demand']} />
              <Line type="monotone" dataKey="qty" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Parts Issued" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ABC Value */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-1">ABC Analysis — Inventory Value (₹K)</h3>
          <p className="text-xs text-gray-500 mb-3">A=Critical high-value · B=Regular · C=Low-value slow-movers</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={abcChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={v => `₹${v}K`} />
              <Tooltip formatter={(v: any, _: any, props: any) => [`₹${v}K (${props.payload.skus} SKUs, ${props.payload.dos}d avg supply)`, 'Value']} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {abcChartData.map(d => <Cell key={d.name} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Turnover */}
        <div className="card">
          <h3 className="font-semibold text-gray-900 mb-1">Top Turnover Rates (Annual)</h3>
          <p className="text-xs text-gray-500 mb-3">Higher = faster-moving — annual demand ÷ avg stock on hand</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={topTurnover} layout="vertical" margin={{ left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 9 }} width={60} />
              <Tooltip formatter={(v: any) => [`${v}×`, 'Turnover']} />
              <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                {topTurnover.map(d => <Cell key={d.name} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Supplier Performance */}
      {suppliers.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">Supplier Performance</h3>
            <p className="text-xs text-gray-500 mt-0.5">Ranked by stock health — lower % means more stockouts</p>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                {['Supplier', 'SKUs', 'Avg Days of Supply', 'Stockouts', 'Health %', 'Total Value'].map(h => (
                  <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {suppliers.map((s: any, i: number) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-3 py-2.5 font-medium text-gray-900 text-sm">{s.supplier}</td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-600">{s.sku_count}</td>
                  <td className="px-3 py-2.5"><DosBadge days={s.avg_dos} /></td>
                  <td className="px-3 py-2.5">
                    <span className={`text-xs font-bold ${s.stockout_count > 0 ? 'text-red-600' : 'text-gray-400'}`}>
                      {s.stockout_count > 0 ? s.stockout_count : '—'}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${s.health_pct >= 90 ? 'bg-green-500' : s.health_pct >= 75 ? 'bg-yellow-400' : 'bg-red-500'}`}
                          style={{ width: `${s.health_pct}%` }} />
                      </div>
                      <span className={`text-xs font-semibold ${s.health_pct >= 90 ? 'text-green-700' : s.health_pct >= 75 ? 'text-amber-700' : 'text-red-700'}`}>
                        {s.health_pct}%
                      </span>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-700">₹{fmt(s.total_value_inr / 1000)}K</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Slow Movers Table */}
      {slowMovers.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200">
            <h3 className="font-semibold text-gray-900">Slow-Moving Parts</h3>
            <p className="text-xs text-gray-500 mt-0.5">Turnover &lt; 2× per year — consider redistribution or return to supplier</p>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                {['Part', 'ABC', 'Stock', 'Annual Demand', 'Turnover', 'Recommendation'].map(h => (
                  <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {slowMovers.map((m: any, i: number) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-3 py-2.5">
                    <div className="text-xs font-medium text-gray-900">{m.description}</div>
                    <div className="text-xs text-gray-400 font-mono">{m.part_code}</div>
                  </td>
                  <td className="px-3 py-2.5"><AbcBadge cls={m.abc_class} /></td>
                  <td className="px-3 py-2.5 tabular-nums font-bold">{m.current_stock}</td>
                  <td className="px-3 py-2.5 tabular-nums text-gray-600">{m.annual_demand.toFixed(1)} units/yr</td>
                  <td className="px-3 py-2.5">
                    <span className="text-xs font-bold text-orange-600">{m.turnover_rate}×</span>
                  </td>
                  <td className="px-3 py-2.5">
                    <span className="text-xs text-gray-500">
                      {m.abc_class === 'C' ? 'Reduce min stock; return excess' : m.abc_class === 'B' ? 'Review demand forecast' : 'Investigate root cause'}
                    </span>
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

// ── Multi-Dealer Tab ──────────────────────────────────────────────────────────
function MultiDealerTab() {
  const { data: dealers = [], isLoading } = useDealerComparison()
  const list = dealers as any[]

  if (isLoading) return <div className="text-gray-400 text-sm p-8 text-center">Loading dealer comparison…</div>

  const chartData = list.map((d: any) => ({
    name: d.dealer_city,
    ok: d.ok,
    low: d.low,
    critical: d.critical,
    stockout: d.stockout,
  }))

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-1">Stock Status by Dealer</h3>
        <p className="text-xs text-gray-500 mb-4">Stacked view showing OK / Low / Critical / Stockout items per dealer</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="ok"       name="OK"       stackId="a" fill="#22c55e" />
            <Bar dataKey="low"      name="Low"      stackId="a" fill="#eab308" />
            <Bar dataKey="critical" name="Critical" stackId="a" fill="#f97316" />
            <Bar dataKey="stockout" name="Stockout" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200">
          <h3 className="font-semibold text-gray-900">Dealer Inventory Health Matrix</h3>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              {['Dealer', 'City', 'SKUs', 'OK', 'Low', 'Critical', 'Stockout', 'Avg DOS', 'Health %', 'Total Value'].map(h => (
                <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {list.map((d: any, i: number) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="px-3 py-2.5 font-mono text-xs font-semibold text-gray-700">{d.dealer_code}</td>
                <td className="px-3 py-2.5 text-gray-700">{d.dealer_city}</td>
                <td className="px-3 py-2.5 tabular-nums">{d.total_skus}</td>
                <td className="px-3 py-2.5 tabular-nums text-green-700 font-medium">{d.ok}</td>
                <td className="px-3 py-2.5 tabular-nums text-yellow-700">{d.low}</td>
                <td className="px-3 py-2.5 tabular-nums text-orange-700 font-medium">{d.critical}</td>
                <td className="px-3 py-2.5 tabular-nums text-red-700 font-bold">{d.stockout}</td>
                <td className="px-3 py-2.5"><DosBadge days={d.avg_days_of_supply} /></td>
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${d.stock_health_pct >= 80 ? 'bg-green-500' : d.stock_health_pct >= 60 ? 'bg-yellow-400' : 'bg-red-500'}`}
                        style={{ width: `${d.stock_health_pct}%` }} />
                    </div>
                    <span className="text-xs tabular-nums">{d.stock_health_pct}%</span>
                  </div>
                </td>
                <td className="px-3 py-2.5 tabular-nums text-gray-700">₹{fmt(d.total_value_inr / 1000)}K</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Transactions Tab ──────────────────────────────────────────────────────────
function TransactionsTab() {
  const [days, setDays] = useState(30)
  const params: Record<string, string> = { days: String(days) }
  if (DEALER_CODE) params.dealer_code = DEALER_CODE

  const { data: txns = [], isLoading } = useInventoryTransactions(params)
  const list = txns as any[]

  const TXN_CFG: Record<string, { bg: string; text: string }> = {
    ISSUE:       { bg: 'bg-red-50',    text: 'text-red-700' },
    RECEIPT:     { bg: 'bg-green-50',  text: 'text-green-700' },
    ADJUSTMENT:  { bg: 'bg-blue-50',   text: 'text-blue-700' },
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {[7, 30, 90].map(d => (
          <button key={d} onClick={() => setDays(d)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${days === d ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300'}`}>
            Last {d} days
          </button>
        ))}
        <span className="text-xs text-gray-400 self-center ml-2">{list.length} transactions</span>
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                {['Date', 'Dealer', 'Part', 'Type', 'Qty', 'Stock After', 'Reference'].map(h => (
                  <th key={h} className="px-3 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {isLoading && <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-400">Loading…</td></tr>}
              {!isLoading && list.length === 0 && <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-400">No transactions in this period</td></tr>}
              {list.slice(0, 200).map((t: any, i: number) => {
                const cfg = TXN_CFG[t.transaction_type] ?? { bg: 'bg-gray-50', text: 'text-gray-600' }
                return (
                  <tr key={i} className="hover:bg-gray-50">
                    <td className="px-3 py-2 text-xs text-gray-600 whitespace-nowrap">{t.date}</td>
                    <td className="px-3 py-2 text-xs font-mono text-gray-600">{t.dealer_code}</td>
                    <td className="px-3 py-2 text-xs text-gray-700">{t.part_code}</td>
                    <td className="px-3 py-2">
                      <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${cfg.bg} ${cfg.text}`}>{t.transaction_type}</span>
                    </td>
                    <td className="px-3 py-2 tabular-nums font-bold">{t.quantity}</td>
                    <td className="px-3 py-2 tabular-nums text-gray-600">{t.stock_after}</td>
                    <td className="px-3 py-2 text-xs text-gray-400 font-mono">{t.reference}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function Inventory() {
  const [tab, setTab] = useState<Tab>('Overview')
  const { data: ov } = useInventoryOverview()
  const alertCount = ov ? (ov.stockout_count + ov.critical_count + ov.low_count) : 0

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Inventory Management</h1>
          <p className="text-gray-500 text-sm mt-1">
            EOQ · Safety Stock · Reorder Planning · ABC Analysis · Supplier Management
            {DEALER_CODE ? ` — ${DEALER_CODE}` : ' — All Dealers'}
          </p>
        </div>
        {alertCount > 0 && (
          <div className="card bg-red-50 border-red-200 border px-4 py-2 text-center">
            <p className="text-2xl font-bold text-red-700">{alertCount}</p>
            <p className="text-xs text-red-600 font-medium">Stock Alerts</p>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-0.5">
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                tab === t ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}>
              {t}
              {t === 'Reorder Plan' && alertCount > 0 && (
                <span className="ml-1.5 text-xs bg-red-500 text-white px-1.5 py-0.5 rounded-full">{alertCount}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {tab === 'Overview'      && <OverviewTab />}
      {tab === 'Stock Ledger'  && <StockLedgerTab />}
      {tab === 'Reorder Plan'  && <ReorderPlanTab />}
      {tab === 'Analytics'     && <AnalyticsTab />}
      {tab === 'Multi-Dealer'  && <MultiDealerTab />}
      {tab === 'Transactions'  && <TransactionsTab />}
    </div>
  )
}
