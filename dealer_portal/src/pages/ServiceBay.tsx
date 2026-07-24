import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useBayStatus, useAppointments, useCreateAppointment, useUpdateAppointmentStatus } from '../api/hooks'
import { getMaintenanceCalendar } from '../api/client'
import { BayGrid } from '../components/BayGrid'
import type { AppointmentResponse, MaintenanceEvent } from '../types'

const DEALER_CODE = localStorage.getItem('ap_dealer_code') ?? 'DL001'

const STATUS_COLORS: Record<string, string> = {
  confirmed:   'bg-blue-100 text-blue-800',
  in_progress: 'bg-orange-100 text-orange-800',
  completed:   'bg-green-100 text-green-800',
  cancelled:   'bg-gray-100 text-gray-500',
}

function BookingModal({ onClose }: { onClose: () => void }) {
  const [form, setForm] = useState({ vin: '', job_type: 'GENERAL_SERVICE', date: '', time: '09:00', bay_id: 'BAY-01', notes: '' })
  const create = useCreateAppointment(DEALER_CODE)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    await create.mutateAsync(form)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md">
        <h2 className="text-lg font-bold text-gray-900 mb-4">Book Appointment</h2>
        <form onSubmit={submit} className="space-y-3">
          {[
            { label: 'VIN', key: 'vin', type: 'text', placeholder: 'MZ7X...' },
            { label: 'Date', key: 'date', type: 'date' },
            { label: 'Time', key: 'time', type: 'time' },
          ].map(({ label, key, type, placeholder }) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
              <input
                type={type}
                placeholder={placeholder}
                value={(form as any)[key]}
                onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
          ))}
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Job Type</label>
            <select
              value={form.job_type}
              onChange={e => setForm(f => ({ ...f, job_type: e.target.value }))}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {['GENERAL_SERVICE', 'BRAKE_SERVICE', 'OIL_CHANGE', 'BATTERY_CHECK', 'TYRE_SERVICE', 'DIAGNOSTIC'].map(t => (
                <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Bay</label>
            <select
              value={form.bay_id}
              onChange={e => setForm(f => ({ ...f, bay_id: e.target.value }))}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {['BAY-01', 'BAY-02', 'BAY-03', 'BAY-04', 'BAY-05', 'BAY-06'].map(b => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>
          <div className="flex gap-3 pt-2">
            <button type="button" onClick={onClose} className="flex-1 border border-gray-300 text-gray-600 rounded-lg py-2 text-sm hover:bg-gray-50">
              Cancel
            </button>
            <button
              type="submit"
              disabled={create.isPending}
              className="flex-1 bg-blue-600 text-white rounded-lg py-2 text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              {create.isPending ? 'Booking…' : 'Book'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function ServiceBay() {
  const [showModal, setShowModal] = useState(false)
  const [daysAhead, setDaysAhead] = useState(7)

  const { data: baysRaw = [], isLoading: baysLoading } = useBayStatus(DEALER_CODE)
  const { data: apptRaw = [] }                         = useAppointments(DEALER_CODE, daysAhead)
  const updateStatus                                    = useUpdateAppointmentStatus(DEALER_CODE)

  const bays  = baysRaw as any[]
  const appts = apptRaw as AppointmentResponse[]

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Service Bay</h1>
          <p className="text-gray-500 text-sm">Bay occupancy and appointment management: {DEALER_CODE}</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700"
        >
          + Book Appointment
        </button>
      </div>

      {/* Bay grid */}
      <div className="card">
        <h2 className="font-semibold text-gray-900 mb-4">Live Bay Status</h2>
        <BayGrid bays={bays} loading={baysLoading} />
      </div>

      {/* Appointments table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900">Appointments</h2>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500">Next</label>
            <select
              aria-label="Days ahead"
              value={daysAhead}
              onChange={e => setDaysAhead(Number(e.target.value))}
              className="border border-gray-300 rounded px-2 py-1 text-xs"
            >
              {[3, 7, 14, 30].map(d => <option key={d} value={d}>{d} days</option>)}
            </select>
          </div>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {['Date', 'Time', 'VIN', 'Job Type', 'Bay', 'Duration', 'Status', 'Actions'].map(h => (
                <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {appts.length === 0 && (
              <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-400">No appointments in next {daysAhead} days</td></tr>
            )}
            {appts.map(a => (
              <tr key={a.appointment_id} className="hover:bg-gray-50">
                <td className="px-4 py-2.5 text-gray-700">{a.date}</td>
                <td className="px-4 py-2.5 text-gray-700">{a.time}</td>
                <td className="px-4 py-2.5 font-mono text-xs text-blue-700">{a.vin}</td>
                <td className="px-4 py-2.5">{a.job_type?.replace(/_/g, ' ')}</td>
                <td className="px-4 py-2.5 font-medium">{a.bay_id}</td>
                <td className="px-4 py-2.5 text-gray-500">{a.duration_hours}h</td>
                <td className="px-4 py-2.5">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[a.status] ?? 'bg-gray-100 text-gray-600'}`}>
                    {a.status}
                  </span>
                </td>
                <td className="px-4 py-2.5">
                  <select
                    aria-label={`Update status for ${a.vin}`}
                    value={a.status}
                    onChange={e => updateStatus.mutate({ id: a.appointment_id, status: e.target.value })}
                    className="text-xs border border-gray-300 rounded px-1.5 py-0.5"
                  >
                    {['confirmed', 'in_progress', 'completed', 'cancelled'].map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Predicted service needs */}
      <PredictedService daysAhead={daysAhead} />

      {showModal && <BookingModal onClose={() => setShowModal(false)} />}
    </div>
  )
}

function PredictedService({ daysAhead }: { daysAhead: number }) {
  const { data: calRaw = [] } = useQuery({
    queryKey: ['fleet', 'maintenance', daysAhead],
    queryFn: () => getMaintenanceCalendar(daysAhead),
  })
  const calendar = (Array.isArray(calRaw) ? calRaw : []) as MaintenanceEvent[]

  if (calendar.length === 0) return null

  return (
    <div className="card p-0 overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-200">
        <h2 className="font-semibold text-gray-900">Predicted Service Needs</h2>
        <p className="text-xs text-gray-500 mt-0.5">ML-predicted maintenance events in the next {daysAhead} days</p>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            {['Vehicle', 'Model', 'Service Type', 'Severity', 'Days Until', 'Confidence', ''].map(h => (
              <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {calendar.map((ev, i) => (
            <tr key={i} className="hover:bg-blue-50 transition-colors">
              <td className="px-4 py-2.5 font-mono text-xs text-gray-600">{ev.vin?.slice(-8)}</td>
              <td className="px-4 py-2.5 text-gray-700">{ev.model_name}</td>
              <td className="px-4 py-2.5 capitalize">{ev.alert_type?.replace(/_/g, ' ')}</td>
              <td className="px-4 py-2.5">
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  ev.severity === 'HIGH' ? 'bg-red-100 text-red-700' :
                  ev.severity === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-gray-100 text-gray-600'
                }`}>{ev.severity}</span>
              </td>
              <td className="px-4 py-2.5">
                <span className={`font-bold text-xs ${ev.days_until <= 3 ? 'text-red-600' : ev.days_until <= 7 ? 'text-orange-600' : 'text-gray-600'}`}>
                  {ev.days_until}d
                </span>
              </td>
              <td className="px-4 py-2.5 text-xs text-gray-500">{Math.round((ev.confidence ?? 0) * 100)}%</td>
              <td className="px-4 py-2.5">
                <Link to={`/vehicles/${ev.vin}`} className="text-xs text-blue-600 hover:underline font-medium">View Vehicle</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
