import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getAdminUsers, createAdminUser, deleteAdminUser } from '../api/client'

const ROLE_BADGE: Record<string, string> = {
  ADMIN:  'bg-purple-100 text-purple-800',
  OEM:    'bg-blue-100 text-blue-800',
  DEALER: 'bg-green-100 text-green-800',
}

export default function AdminUsers() {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({ queryKey: ['admin-users'], queryFn: getAdminUsers })

  const [form, setForm] = useState({ username: '', password: '', role: 'DEALER', dealer_code: 'DL001' })
  const [formError, setFormError] = useState('')
  const [successMsg, setSuccessMsg] = useState('')

  const createMutation = useMutation({
    mutationFn: createAdminUser,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-users'] })
      setForm({ username: '', password: '', role: 'DEALER', dealer_code: 'DL001' })
      setFormError('')
      setSuccessMsg('User created successfully.')
      setTimeout(() => setSuccessMsg(''), 3000)
    },
    onError: (err: any) => {
      setFormError(err?.response?.data?.detail ?? 'Failed to create user.')
      setSuccessMsg('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteAdminUser,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin-users'] }),
  })

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault()
    setFormError('')
    createMutation.mutate({
      username: form.username.trim(),
      password: form.password,
      role: form.role,
      dealer_code: form.role === 'DEALER' ? form.dealer_code : 'ALL',
    })
  }

  const users: any[] = data?.users ?? []

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">User Management</h1>
        <p className="text-slate-500 text-sm mt-1">Create and manage portal accounts. Changes persist across server restarts.</p>
      </div>

      {/* Users table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <h2 className="font-semibold text-slate-700">Current Users</h2>
          <span className="text-xs text-slate-400">{users.length} total</span>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-slate-400 text-sm">Loading users…</div>
        ) : error ? (
          <div className="p-8 text-center text-red-500 text-sm">Failed to load users.</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-100">
              <tr>
                <th className="px-5 py-3 text-left font-medium text-slate-600">Username</th>
                <th className="px-5 py-3 text-left font-medium text-slate-600">Role</th>
                <th className="px-5 py-3 text-left font-medium text-slate-600">Dealer Code</th>
                <th className="px-5 py-3 text-right font-medium text-slate-600">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {users.map((u) => (
                <tr key={u.username} className="hover:bg-slate-50 transition-colors">
                  <td className="px-5 py-3 font-mono text-slate-800">{u.username}</td>
                  <td className="px-5 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${ROLE_BADGE[u.role] ?? 'bg-slate-100 text-slate-600'}`}>
                      {u.role}
                    </span>
                  </td>
                  <td className="px-5 py-3 text-slate-500">{u.dealer_code}</td>
                  <td className="px-5 py-3 text-right">
                    {u.username !== 'admin' ? (
                      <button
                        onClick={() => {
                          if (confirm(`Delete user "${u.username}"?`)) deleteMutation.mutate(u.username)
                        }}
                        className="text-red-500 hover:text-red-700 text-xs font-medium transition-colors"
                      >
                        Delete
                      </button>
                    ) : (
                      <span className="text-slate-300 text-xs">Protected</span>
                    )}
                  </td>
                </tr>
              ))}
              {users.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-5 py-8 text-center text-slate-400">No users found.</td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      {/* Create user form */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="font-semibold text-slate-700 mb-4">Create New User</h2>
        <form onSubmit={handleCreate} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Username</label>
            <input
              type="text"
              required
              value={form.username}
              onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
              placeholder="e.g. dealer_mumbai"
              className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Password</label>
            <input
              type="password"
              required
              value={form.password}
              onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
              placeholder="Min 6 characters"
              className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Role</label>
            <select
              value={form.role}
              onChange={e => setForm(f => ({ ...f, role: e.target.value }))}
              className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="DEALER">Dealer</option>
              <option value="OEM">OEM</option>
              <option value="ADMIN">Admin</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Dealer Code <span className="text-slate-400">(Dealer role only)</span>
            </label>
            <input
              type="text"
              value={form.dealer_code}
              onChange={e => setForm(f => ({ ...f, dealer_code: e.target.value }))}
              disabled={form.role !== 'DEALER'}
              placeholder="e.g. DL003"
              className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-50 disabled:text-slate-400"
            />
          </div>

          {formError && (
            <div className="sm:col-span-2 text-red-600 text-sm bg-red-50 px-3 py-2 rounded-lg">
              {formError}
            </div>
          )}
          {successMsg && (
            <div className="sm:col-span-2 text-green-700 text-sm bg-green-50 px-3 py-2 rounded-lg">
              {successMsg}
            </div>
          )}

          <div className="sm:col-span-2">
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="px-5 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {createMutation.isPending ? 'Creating…' : 'Create User'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
