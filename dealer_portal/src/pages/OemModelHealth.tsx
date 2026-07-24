import { useState, useMemo } from 'react'
import { useOemModelHealth, useModelEda } from '../api/hooks'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const STATUS_COLORS: Record<string, string> = {
  trained:        'bg-green-100 text-green-800',
  not_trained:    'bg-gray-100 text-gray-500',
  skipped:        'bg-yellow-100 text-yellow-700',
  failed:         'bg-red-100 text-red-800',
  physics_based:  'bg-blue-100 text-blue-700',
}

const STATUS_LABELS: Record<string, string> = {
  trained:        'trained',
  not_trained:    'not trained',
  skipped:        'skipped',
  failed:         'failed',
  physics_based:  'physics-based',
}

function ConcordanceGauge({ value, label }: { value: number | null; label: string }) {
  if (value == null) return (
    <div className="flex flex-col items-center gap-1">
      <div className="w-16 h-16 rounded-full border-4 border-gray-200 flex items-center justify-center">
        <span className="text-xs text-gray-400">N/A</span>
      </div>
      <span className="text-xs text-gray-500 text-center">{label}</span>
    </div>
  )
  const color = value >= 0.7 ? '#22c55e' : value >= 0.6 ? '#eab308' : '#ef4444'
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative w-16 h-16">
        <svg viewBox="0 0 36 36" className="w-16 h-16 -rotate-90">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#f0f0f0" strokeWidth="3" />
          <circle
            cx="18" cy="18" r="15.9" fill="none"
            stroke={color} strokeWidth="3"
            strokeDasharray={`${value * 100} 100`}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-sm font-bold" style={{ color }}>
          {value.toFixed(2)}
        </span>
      </div>
      <span className="text-xs text-gray-500 text-center">{label}</span>
    </div>
  )
}

function ModelCard({ model, onSelect, selected }: { model: any; onSelect: () => void; selected: boolean }) {
  const concordance = model.metrics?.concordance_index ?? null
  const auc = model.metrics?.cv_auc ?? null
  const isNotTrained = model.status === 'not_trained'
  const isPhysicsBased = model.status === 'physics_based'

  return (
    <div
      onClick={onSelect}
      className={`rounded-xl border p-4 cursor-pointer transition-all ${
        selected
          ? isPhysicsBased
            ? 'border-blue-400 ring-2 ring-blue-100 bg-blue-50/50'
            : 'border-blue-500 ring-2 ring-blue-200 bg-blue-50'
          : 'border-gray-200 bg-white hover:border-gray-300'
      } ${isNotTrained ? 'opacity-60' : ''}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-gray-900 text-sm">{model.display_name}</h3>
          <p className="text-xs text-gray-400 mt-0.5">{model.algorithm}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[model.status] ?? 'bg-gray-100 text-gray-600'}`}>
            {STATUS_LABELS[model.status] ?? model.status}
          </span>
          {!model.artifact_exists && model.status !== 'not_trained' && (
            <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-orange-100 text-orange-700">
              no artifact
            </span>
          )}
        </div>
      </div>

      {isPhysicsBased ? (
        <div className="text-center py-3 text-xs text-blue-400 flex flex-col items-center gap-1">
          <svg className="w-5 h-5 text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Always active · No training required
        </div>
      ) : isNotTrained ? (
        <div className="text-center py-3 text-xs text-gray-400">
          Run training to generate metrics
        </div>
      ) : (concordance != null || auc != null) ? (
        <div className="flex justify-around">
          <ConcordanceGauge value={concordance} label="C-Index" />
          <ConcordanceGauge value={auc} label="AUC-ROC" />
        </div>
      ) : (
        <div className="w-full space-y-1 px-1">
          {Object.entries(model.metrics ?? {}).filter(([, v]) => v != null).slice(0, 3).map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs">
              <span className="text-gray-500 capitalize">{k.replace(/_/g, ' ')}</span>
              <span className="font-bold text-gray-800">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-2 gap-2 text-xs text-gray-500">
        <span>{model.training_samples?.toLocaleString() ?? '—'} samples</span>
        <span>{model.feature_names?.length ?? 0} features</span>
        <span className="col-span-2 truncate">
          {model.trained_at ? new Date(model.trained_at).toLocaleDateString('en-IN') : 'Not trained'}
        </span>
      </div>
    </div>
  )
}

// ── Correlation Heatmap ────────────────────────────────────────────────────────

function corrColor(v: number): string {
  const abs = Math.abs(v)
  if (v > 0.05)  return `hsl(220,${Math.round(abs * 85)}%,${Math.round(98 - abs * 38)}%)`
  if (v < -0.05) return `hsl(0,${Math.round(abs * 85)}%,${Math.round(98 - abs * 38)}%)`
  return '#f9fafb'
}

function CorrelationHeatmap({ features, matrix }: { features: string[]; matrix: number[][] }) {
  const n = features.length
  const cell = Math.max(18, Math.min(36, Math.floor(520 / Math.max(n, 1))))
  const labelW = 120
  const shortName = (f: string) => f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).slice(0, 16)

  return (
    <div className="overflow-auto max-h-[520px]" tabIndex={0}>
      {/* Column header row — vertical text avoids overlap */}
      <div className="flex" style={{ paddingLeft: labelW }}>
        {features.map((f, j) => (
          <div
            key={j}
            style={{
              width: cell, flexShrink: 0, height: 72,
              display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
              paddingBottom: 4,
            }}
          >
            <span style={{
              fontSize: 8, color: '#6b7280', whiteSpace: 'nowrap',
              writingMode: 'vertical-lr', transform: 'rotate(180deg)',
              maxHeight: 68, overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {shortName(f)}
            </span>
          </div>
        ))}
      </div>
      {/* Matrix rows */}
      <div className="flex flex-col">
        {matrix.map((row, i) => (
          <div key={i} className="flex items-center">
            <div style={{ width: labelW, fontSize: 9, color: '#6b7280', textAlign: 'right', paddingRight: 6, flexShrink: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {shortName(features[i] ?? '')}
            </div>
            {row.map((val, j) => (
              <div
                key={j}
                title={`${features[i]} × ${features[j] ?? ''}: ${typeof val === 'number' ? val.toFixed(2) : '—'}`}
                style={{ width: cell, height: cell, backgroundColor: corrColor(typeof val === 'number' ? val : 0), flexShrink: 0 }}
                className="border border-white/40"
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 mt-3 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-4 h-3 rounded" style={{ background: corrColor(-0.9) }} /> negative
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-3 rounded" style={{ background: '#f9fafb', border: '1px solid #e5e7eb' }} /> none
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-3 rounded" style={{ background: corrColor(0.9) }} /> positive
        </div>
      </div>
    </div>
  )
}

export default function OemModelHealth() {
  const { data, isLoading } = useOemModelHealth()
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [edaTab, setEdaTab] = useState<'heatmap' | 'target'>('heatmap')

  const models: any[] = data?.models ?? []
  const summary = data?.summary ?? {}
  const trainedModels = models.filter((m: any) => m.status === 'trained')
  const selected = models.find((m: any) => m.model_name === selectedModel) ?? trainedModels[0] ?? models[0]

  const { data: edaData, isLoading: edaLoading } = useModelEda(selected?.model_name ?? null)

  // All hooks must be called before any early return (Rules of Hooks)
  const featureData = useMemo(() =>
    Object.entries(selected?.feature_importances ?? {})
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 15)
      .map(([name, importance]) => ({
        name: name.replace(/_/g, ' '),
        importance: +((importance as number) * 100).toFixed(1),
      }))
  , [selected])

  const targetCorrData = useMemo(() => {
    if (!edaData?.target_correlation) return []
    return Object.entries(edaData.target_correlation as Record<string, number>)
      .map(([name, v]) => ({ name: name.replace(/_/g, ' '), corr: v, abs: Math.abs(v) }))
      .sort((a, b) => b.abs - a.abs)
      .slice(0, 15)
  }, [edaData])

  const concordanceData = useMemo(() =>
    trainedModels
      .filter((m: any) => m.metrics?.concordance_index != null)
      .map((m: any) => ({
        name: m.display_name.replace(/\s*\(.*\)/, '').trim(),
        concordance: +((m.metrics.concordance_index as number) * 100).toFixed(1),
        auc: m.metrics?.cv_auc != null ? +((m.metrics.cv_auc as number) * 100).toFixed(1) : null,
      }))
  , [trainedModels])

  if (isLoading) {
    return (
      <div className="p-6 flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-gray-400 text-sm">Loading model health…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Health Dashboard</h1>
        <p className="text-gray-400 text-sm mt-0.5">Real training metrics from models/saved/model_metrics.json</p>
      </div>

      {/* No models banner */}
      {!summary.metrics_file_exists && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
          <span className="text-amber-500 text-xl">⚠</span>
          <div>
            <p className="font-semibold text-amber-800 text-sm">No training data found</p>
            <p className="text-amber-700 text-xs mt-0.5">
              Run <code className="font-mono bg-amber-100 px-1 rounded">python -m models.train_all</code> to train models and populate metrics.
            </p>
          </div>
        </div>
      )}

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-xs text-gray-500">Total Models</p>
          <p className="text-2xl font-bold text-gray-900">{summary.total_models ?? '—'}</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-xs text-gray-500">Trained (ML)</p>
          <p className="text-2xl font-bold text-green-600">{summary.trained_count ?? 0}</p>
          {(summary.physics_count ?? 0) > 0 && (
            <p className="text-xs text-blue-500 mt-0.5">+{summary.physics_count} physics</p>
          )}
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-xs text-gray-500">Not Trained</p>
          <p className={`text-2xl font-bold ${(summary.not_trained_count ?? 0) > 0 ? 'text-orange-500' : 'text-gray-900'}`}>
            {summary.not_trained_count ?? 0}
          </p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <p className="text-xs text-gray-500">Avg Concordance</p>
          <p className="text-2xl font-bold text-blue-600">
            {summary.avg_concordance != null ? (summary.avg_concordance * 100).toFixed(0) + '%' : '—'}
          </p>
        </div>
      </div>

      {/* Concordance comparison chart */}
      {concordanceData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-1">Model Performance Comparison</h3>
          <p className="text-xs text-gray-400 mb-4">Concordance index from real training (0.5 = random baseline)</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={concordanceData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis domain={[40, 80]} tick={{ fontSize: 11 }} unit="%" />
              <Tooltip formatter={(v: any) => `${v}%`} />
              <Bar dataKey="concordance" name="C-Index" fill="#3b82f6" radius={[3, 3, 0, 0]}>
                {concordanceData.map((_: any, i: number) => (
                  <Cell key={i} fill={concordanceData[i].concordance >= 70 ? '#22c55e' : concordanceData[i].concordance >= 60 ? '#eab308' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-2 flex items-center gap-4 text-xs text-gray-400">
            <span className="flex items-center gap-1.5"><span className="w-3 h-2 bg-red-400 rounded inline-block" /> &lt;60%: poor</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-2 bg-yellow-400 rounded inline-block" /> 60-70%: acceptable</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-2 bg-green-400 rounded inline-block" /> 70%+: good</span>
          </div>
        </div>
      )}

      {/* Model cards grid + detail */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model grid — grouped by category */}
        <div className="space-y-4">
          {[
            { key: 'vehicle',     label: 'Vehicle Health Models' },
            { key: 'operational', label: 'Operational Models' },
            { key: 'vehicle_ev',  label: 'EV Physics Engines' },
          ].map(({ key: cat, label }) => {
            const group = models.filter((m: any) => m.category === cat)
            if (group.length === 0) return null
            return (
              <div key={cat}>
                <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-1">{label}</p>
                <div className="space-y-3">
                  {group.map((m: any) => (
                    <ModelCard
                      key={m.model_name}
                      model={m}
                      selected={m.model_name === (selected?.model_name)}
                      onSelect={() => setSelectedModel(m.model_name)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {/* Selected model detail */}
        {selected && (
          <div className="lg:col-span-2 space-y-4">
            {/* Warnings */}
            {selected.warnings?.length > 0 && (
              <div className="space-y-2">
                {selected.warnings.map((w: string, i: number) => (
                  <div key={i} className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2.5 flex items-start gap-2 text-xs text-amber-800">
                    <span className="mt-0.5">⚠</span>
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Model info */}
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-gray-900">{selected.display_name}</h3>
                <div className="flex gap-2">
                  {selected.fi_is_real && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-medium">
                      real importances
                    </span>
                  )}
                  {!selected.fi_is_real && Object.keys(selected.feature_importances ?? {}).length > 0 && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-gray-50 text-gray-500 border font-medium">
                      placeholder importances
                    </span>
                  )}
                </div>
              </div>
              {selected.notes && (
                <p className="text-sm text-gray-600 mb-4">{selected.notes}</p>
              )}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <p className="text-xs text-gray-400">Algorithm</p>
                  <p className="font-medium text-gray-700">{selected.algorithm}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Target</p>
                  <p className="font-medium text-gray-700 text-xs">{selected.target ?? '—'}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Training Samples</p>
                  <p className="font-medium text-gray-700">{selected.training_samples?.toLocaleString() ?? '—'}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">Trained At</p>
                  <p className="font-medium text-gray-700 text-xs">
                    {selected.trained_at ? new Date(selected.trained_at).toLocaleString('en-IN') : 'Never'}
                  </p>
                </div>
              </div>

              {/* Real metrics — all keys from store */}
              {selected.status === 'trained' && Object.keys(selected.metrics ?? {}).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  {Object.entries(selected.metrics ?? {}).map(([k, v]) => {
                    if (v == null) return null
                    const numV = typeof v === 'number' ? v : null
                    const isCIndex = k === 'concordance_index'
                    const isAuc    = k === 'cv_auc'
                    const textColor = (isCIndex || isAuc) && numV != null
                      ? numV >= 0.7 ? 'text-green-600' : numV >= 0.6 ? 'text-yellow-600' : 'text-red-500'
                      : 'text-blue-600'
                    const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                    return (
                      <div key={k}>
                        <p className="text-xs text-gray-400">{label}</p>
                        <p className={`font-bold ${textColor}`}>
                          {numV != null ? (numV % 1 === 0 ? numV.toLocaleString() : numV.toFixed(4)) : String(v)}
                        </p>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Feature importance chart */}
            {featureData.length > 0 && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900">Feature Importance</h3>
                  {!selected.fi_is_real && (
                    <span className="text-xs text-gray-400 italic">placeholders — retrain for real SHAP values</span>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={Math.max(160, featureData.length * 24)}>
                  <BarChart data={featureData} layout="vertical" barSize={14}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" tick={{ fontSize: 10 }} unit="%" domain={[0, 'auto']} />
                    <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 10 }} />
                    <Tooltip formatter={(v: any) => `${v}%`} />
                    <Bar dataKey="importance" fill="#3b82f6" radius={[0, 3, 3, 0]}>
                      {featureData.map((_: any, i: number) => (
                        <Cell key={i} fill={selected.fi_is_real ? `hsl(${220 + i * 15}, 75%, ${60 - i * 4}%)` : '#94a3b8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Feature names fallback — shown when importances are missing but feature list exists */}
            {featureData.length === 0 && (selected.feature_names?.length ?? 0) > 0 && selected.status === 'trained' && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-gray-900">Features Used ({selected.feature_names.length})</h3>
                  <span className="text-xs text-gray-400 italic">retrain to compute importances</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {selected.feature_names.slice(0, 15).map((f: string) => (
                    <span key={f} className="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded-full font-mono border border-blue-100">
                      {f.replace(/_/g, ' ')}
                    </span>
                  ))}
                  {selected.feature_names.length > 15 && (
                    <span className="text-xs px-2 py-1 bg-gray-100 text-gray-500 rounded-full">
                      +{selected.feature_names.length - 15} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* EDA — Correlation heatmap + target correlation */}
            {selected.status === 'trained' && (
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-gray-900">Feature Analysis</h3>
                    {edaData && (
                      <p className="text-xs text-gray-400 mt-0.5">{edaData.n_samples} samples · {edaData.features?.length} features</p>
                    )}
                  </div>
                  <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
                    <button
                      onClick={() => setEdaTab('heatmap')}
                      className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${edaTab === 'heatmap' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                    >
                      Correlation
                    </button>
                    <button
                      onClick={() => setEdaTab('target')}
                      className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${edaTab === 'target' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                      disabled={!edaData?.target_correlation}
                    >
                      vs Target
                    </button>
                  </div>
                </div>

                {edaLoading && (
                  <div className="flex items-center justify-center h-32 text-gray-400 text-sm">
                    <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full mr-2" />
                    Computing correlations…
                  </div>
                )}

                {!edaLoading && !edaData && (
                  <div className="text-center py-8 text-gray-400 text-sm">
                    No EDA data yet: will be generated on next retrain
                  </div>
                )}

                {!edaLoading && edaData && edaTab === 'heatmap' && (
                  <CorrelationHeatmap
                    features={edaData.features ?? []}
                    matrix={edaData.correlation ?? []}
                  />
                )}

                {!edaLoading && edaData && edaTab === 'target' && targetCorrData.length > 0 && (
                  <div>
                    <p className="text-xs text-gray-400 mb-3">
                      Pearson correlation of each feature with <span className="font-mono text-gray-600">{edaData.target_name}</span>
                    </p>
                    <ResponsiveContainer width="100%" height={Math.max(160, targetCorrData.length * 24)}>
                      <BarChart data={targetCorrData} layout="vertical" barSize={14}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis type="number" tick={{ fontSize: 10 }} domain={[-1, 1]} tickFormatter={v => v.toFixed(1)} />
                        <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 10 }} />
                        <Tooltip formatter={(v: any) => v.toFixed(3)} />
                        <Bar dataKey="corr" radius={[0, 3, 3, 0]}>
                          {targetCorrData.map((d, i) => (
                            <Cell key={i} fill={d.corr >= 0 ? '#3b82f6' : '#ef4444'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {!edaLoading && edaData && edaTab === 'target' && targetCorrData.length === 0 && (
                  <div className="text-center py-6 text-gray-400 text-sm">
                    Target correlation not available for this model type
                  </div>
                )}
              </div>
            )}

            {/* Not-trained empty state */}
            {selected.status === 'not_trained' && (
              <div className="bg-white rounded-xl border border-dashed border-gray-300 p-8 text-center">
                <p className="text-4xl mb-3">🔬</p>
                <p className="font-semibold text-gray-700 mb-1">{selected.display_name}: Not Yet Trained</p>
                <p className="text-sm text-gray-400 mb-4">No artifact found at models/saved/. Trigger training from the Retrain Control page.</p>
                <div className="text-left max-w-xs mx-auto text-xs text-gray-500 space-y-1">
                  {(selected.artifact_files ?? []).length > 0 && (
                    <p>Looking for: {selected.artifact_files?.join(', ')}</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
