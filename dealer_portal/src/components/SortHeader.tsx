import { useState } from 'react'

export type SortDir = 'asc' | 'desc'

export function useSortable<T extends string>(defaultKey: T, defaultDir: SortDir = 'asc') {
  const [sortKey, setSortKey] = useState<T>(defaultKey)
  const [sortDir, setSortDir] = useState<SortDir>(defaultDir)

  const onSort = (k: T) => {
    if (k === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('asc') }
  }

  const sortRows = <R extends Record<string, any>>(rows: R[]): R[] =>
    [...rows].sort((a, b) => {
      let av = a[sortKey as string] ?? ''
      let bv = b[sortKey as string] ?? ''
      if (typeof av === 'string') av = av.toLowerCase()
      if (typeof bv === 'string') bv = bv.toLowerCase()
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ?  1 : -1
      return 0
    })

  return { sortKey, sortDir, onSort, sortRows }
}

export function SortTh<T extends string>({ label, col, cur, dir, onSort, className = '' }: {
  label: string; col: T; cur: T; dir: SortDir; onSort: (k: T) => void; className?: string
}) {
  const active = cur === col
  return (
    <th
      className={`px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap cursor-pointer select-none hover:text-gray-800 hover:bg-gray-100 transition-colors ${className}`}
      onClick={() => onSort(col)}
    >
      <span className="flex items-center gap-1">
        {label}
        <span className={active ? 'text-blue-500' : 'text-gray-300'}>
          {active ? (dir === 'asc' ? '↑' : '↓') : '↕'}
        </span>
      </span>
    </th>
  )
}
