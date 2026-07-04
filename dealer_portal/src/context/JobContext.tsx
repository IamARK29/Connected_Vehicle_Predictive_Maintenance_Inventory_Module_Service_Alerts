import { createContext, useContext, useState, useRef, useCallback, useEffect } from 'react'
import { getSyntheticStatus } from '../api/client'

export type Phase = 'idle' | 'generating' | 'training' | 'done'
export type ToastType = 'success' | 'error' | 'info'

export interface JobState {
  phase: Phase
  pct: number
  message: string
  jobId: string | null
}

export interface ToastMsg {
  id: number
  message: string
  type: ToastType
}

interface JobContextValue {
  job: JobState
  setJob: React.Dispatch<React.SetStateAction<JobState>>
  startPolling: (jobId: string, phase: Phase, onDone: (r: any) => void) => void
  stopJob: () => void
  toast: ToastMsg | null
  showToast: (message: string, type: ToastType) => void
  dismissToast: () => void
}

const JobContext = createContext<JobContextValue | null>(null)

export function JobProvider({ children }: { children: React.ReactNode }) {
  const [job, setJob]     = useState<JobState>({ phase: 'idle', pct: 0, message: '', jobId: null })
  const [toast, setToast] = useState<ToastMsg | null>(null)
  const intervalRef       = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }, [])

  const stopJob = useCallback(() => {
    stopPolling()
    setJob({ phase: 'idle', pct: 0, message: '', jobId: null })
  }, [stopPolling])

  const showToast = useCallback((message: string, type: ToastType) => {
    setToast({ id: Date.now(), message, type })
  }, [])

  const dismissToast = useCallback(() => setToast(null), [])

  const startPolling = useCallback((jobId: string, phase: Phase, onDone: (r: any) => void) => {
    stopPolling()
    setJob(j => ({ ...j, jobId, phase, pct: 0 }))
    intervalRef.current = setInterval(async () => {
      try {
        const d = await getSyntheticStatus(jobId)
        setJob(j => ({ ...j, pct: d.pct ?? 0, message: d.message ?? '' }))
        if ((d.pct ?? 0) >= 100) {
          stopPolling()
          onDone(d.result ?? {})
        }
      } catch { /* keep polling on transient network errors */ }
    }, 2000)
  }, [stopPolling])

  useEffect(() => () => stopPolling(), [stopPolling])

  return (
    <JobContext.Provider value={{ job, setJob, startPolling, stopJob, toast, showToast, dismissToast }}>
      {children}
    </JobContext.Provider>
  )
}

export function useJob() {
  const ctx = useContext(JobContext)
  if (!ctx) throw new Error('useJob must be used inside JobProvider')
  return ctx
}
