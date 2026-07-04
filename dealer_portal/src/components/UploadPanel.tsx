import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'

interface Props {
  accept?: string
  label?: string
  onUpload: (file: File) => Promise<{ job_id: string; ws?: string }>
  templateUrl?: string
}

type Phase = 'idle' | 'dragging' | 'uploading' | 'queued' | 'error'

export function UploadPanel({ accept = '.csv', label = 'Drop a CSV file here', onUpload, templateUrl }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [phase, setPhase]   = useState<Phase>('idle')
  const [jobId, setJobId]   = useState<string | null>(null)
  const [progress, setPct]  = useState(0)
  const [message, setMsg]   = useState('')
  const wsRef               = useRef<WebSocket | null>(null)

  const handleFile = async (file: File) => {
    setPhase('uploading')
    setMsg(`Uploading ${file.name}…`)
    try {
      const res = await onUpload(file)
      setJobId(res.job_id)
      setPhase('queued')
      setMsg('Queued — tracking progress…')
      // Connect to WS for progress updates
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/upload/${res.job_id}`
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.onmessage = evt => {
        const d = JSON.parse(evt.data)
        setPct(d.pct ?? 0)
        setMsg(d.message ?? '')
        if ((d.pct ?? 0) >= 100) { ws.close(); setPhase('idle'); setPct(0) }
      }
      ws.onerror = () => setMsg('Progress stream unavailable — job is running in background')
    } catch (err: unknown) {
      setPhase('error')
      setMsg(err instanceof Error ? err.message : 'Upload failed')
    }
  }

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setPhase('idle')
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const isDragging = phase === 'dragging'

  return (
    <div className="space-y-3">
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          isDragging
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setPhase('dragging') }}
        onDragLeave={() => setPhase('idle')}
        onDrop={onDrop}
      >
        <div className="text-4xl mb-3">📂</div>
        <p className="text-sm font-medium text-gray-700">{label}</p>
        <p className="text-xs text-gray-400 mt-1">or click to browse · {accept.replace(/\./g, '').toUpperCase()}</p>
        <input ref={inputRef} type="file" accept={accept} className="hidden" onChange={onInputChange} />
      </div>

      {templateUrl && (
        <a
          href={templateUrl}
          download
          className="inline-flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          ⬇ Download CSV template
        </a>
      )}

      {phase !== 'idle' && (
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-gray-600">
            <span>{message}</span>
            {phase === 'queued' && <span>{progress}%</span>}
          </div>
          {phase === 'queued' && (
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-2 bg-blue-600 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}
          {phase === 'error' && (
            <p className="text-xs text-red-600">{message}</p>
          )}
          {jobId && (
            <p className="text-xs text-gray-400">Job ID: <span className="font-mono">{jobId}</span></p>
          )}
        </div>
      )}
    </div>
  )
}

export default UploadPanel
