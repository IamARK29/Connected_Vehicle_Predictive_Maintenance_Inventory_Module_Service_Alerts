import type { Severity } from '../types'

const CONFIG: Record<Severity, { bg: string; text: string; dot: string; border: string }> = {
  CRITICAL: { bg: 'bg-red-100',    text: 'text-red-800',    dot: 'bg-red-600',    border: 'border-red-300' },
  HIGH:     { bg: 'bg-orange-100', text: 'text-orange-800', dot: 'bg-orange-500', border: 'border-orange-300' },
  MEDIUM:   { bg: 'bg-yellow-100', text: 'text-yellow-800', dot: 'bg-yellow-500', border: 'border-yellow-300' },
  LOW:      { bg: 'bg-blue-100',   text: 'text-blue-800',   dot: 'bg-blue-400',   border: 'border-blue-300' },
}

interface Props {
  severity: Severity | string
  size?: 'sm' | 'md'
}

export function AlertBadge({ severity, size = 'sm' }: Props) {
  const key  = (severity?.toUpperCase() ?? 'LOW') as Severity
  const conf = CONFIG[key] ?? CONFIG.LOW

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-semibold ${conf.bg} ${conf.text} ${conf.border}
        ${size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'}`}
    >
      <span className={`rounded-full flex-shrink-0 ${conf.dot} ${size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2'}`} />
      {key}
    </span>
  )
}

export default AlertBadge
