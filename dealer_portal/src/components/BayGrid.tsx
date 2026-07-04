import type { BayStatus } from '../types'

interface Props {
  bays: BayStatus[]
  loading?: boolean
}

function BayCard({ bay }: { bay: BayStatus }) {
  const occupied = bay.status === 'occupied'
  return (
    <div
      className={`rounded-xl border-2 p-4 transition-colors ${
        occupied
          ? 'border-orange-400 bg-orange-50'
          : 'border-green-400 bg-green-50'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-gray-800">{bay.bay_id}</span>
        <span
          className={`text-xs font-bold px-2 py-0.5 rounded-full ${
            occupied
              ? 'bg-orange-200 text-orange-800'
              : 'bg-green-200 text-green-800'
          }`}
        >
          {occupied ? 'BUSY' : 'FREE'}
        </span>
      </div>

      {occupied ? (
        <div className="space-y-1">
          <p className="font-mono text-xs text-gray-600 truncate">{bay.current_vin}</p>
          <p className="text-xs text-gray-500 truncate">{bay.current_job ?? '—'}</p>
          {bay.eta_free && (
            <p className="text-xs text-orange-700 font-medium">
              Ready ~{bay.eta_free}
            </p>
          )}
        </div>
      ) : (
        <p className="text-xs text-green-600 font-medium mt-1">Available</p>
      )}
    </div>
  )
}

export function BayGrid({ bays, loading }: Props) {
  if (loading) {
    return (
      <div className="grid grid-cols-3 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="rounded-xl border-2 border-gray-200 bg-gray-50 h-28 animate-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-3 gap-3">
      {bays.map(bay => <BayCard key={bay.bay_id} bay={bay} />)}
    </div>
  )
}

export default BayGrid
