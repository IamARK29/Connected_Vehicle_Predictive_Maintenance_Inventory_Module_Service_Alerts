import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use(config => {
  const token = localStorage.getItem('ap_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('ap_token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

// Auth
export const login = async (username: string, password: string) => {
  const res = await axios.post('/api/auth/token', { username, password })
  localStorage.setItem('ap_token', res.data.access_token)
  localStorage.setItem('ap_role', res.data.role ?? 'DEALER')
  return res.data
}

export const logout = () => {
  localStorage.removeItem('ap_token')
  localStorage.removeItem('ap_role')
}

// Fleet
export const getFleetHealth = () =>
  api.get('/fleet/health-summary').then(r => r.data)

export const getFleetAlerts = (hours = 168, severity?: string) =>
  api.get('/fleet/alerts/active', { params: { hours, severity } }).then(r => r.data)

export const getMaintenanceCalendar = (days = 90, severity?: string) =>
  api.get('/fleet/maintenance-calendar', { params: { days, severity } }).then(r => r.data)

export const getDriverScores = (limit = 50) =>
  api.get('/fleet/driver-scores', { params: { limit } }).then(r => r.data)

// Vehicles
export const getVehicles = (params?: { dealer_code?: string; fuel_type?: string; limit?: number }) =>
  api.get('/vehicles', { params }).then(r => r.data)

export const getVehicle = (vin: string) =>
  api.get(`/vehicles/${vin}`).then(r => r.data)

export const getVehicleTelemetry = (vin: string, minutes = 60, limit = 100) =>
  api.get(`/vehicles/${vin}/telemetry`, { params: { minutes, limit } }).then(r => r.data)

export const getVehiclePredictions = (vin: string) =>
  api.get(`/vehicles/${vin}/predictions`).then(r => r.data)

export const getVehicleAlerts = (vin: string, severity?: string, limit = 20) =>
  api.get(`/vehicles/${vin}/alerts`, { params: { severity, limit } }).then(r => r.data)

export const getVehicleServiceHistory = (vin: string, limit = 20) =>
  api.get(`/vehicles/${vin}/service-history`, { params: { limit } }).then(r => r.data)

export const getVehicleTrips = (vin: string, limit = 20) =>
  api.get(`/vehicles/${vin}/trips`, { params: { limit } }).then(r => r.data)

// Dealer
export const getBayStatus = (dealerCode: string) =>
  api.get(`/dealer/${dealerCode}/bay-status`).then(r => r.data)

export const getAppointments = (dealerCode: string, daysAhead = 7) =>
  api.get(`/dealer/${dealerCode}/appointments`, { params: { days_ahead: daysAhead } }).then(r => r.data)

export const createAppointment = (dealerCode: string, data: object) =>
  api.post(`/dealer/${dealerCode}/appointments`, data).then(r => r.data)

export const updateAppointmentStatus = (dealerCode: string, id: string, status: string) =>
  api.put(`/dealer/${dealerCode}/appointments/${id}/status`, { status }).then(r => r.data)

export const getInventory = (dealerCode: string) =>
  api.get(`/dealer/${dealerCode}/inventory`).then(r => r.data)

export const getDemandForecast = (dealerCode: string) =>
  api.get(`/dealer/${dealerCode}/demand-forecast`).then(r => r.data)

// Agent
export const getWorkflows = (completed = false) =>
  api.get('/agent/workflows', { params: { completed } }).then(r => r.data)

export const getWorkflowStatus = (vin: string) =>
  api.get(`/agent/workflows/${vin}`).then(r => r.data)

export const advanceWorkflow = (vin: string) =>
  api.post(`/agent/workflows/${vin}/advance`, {}).then(r => r.data)

export const triggerWorkflow = (vin: string) =>
  api.post(`/agent/trigger/${vin}`).then(r => r.data)

export const chatWithAgent = (message: string, vin?: string, chatHistory: object[] = []) =>
  api.post('/agent/chat', { message, vin, chat_history: chatHistory }).then(r => r.data)

export const getCostEstimate = (alertType: string, vin = '', modelCode = 'DEFAULT') =>
  api.get(`/agent/cost-estimate/${alertType}`, { params: { vin, model_code: modelCode } }).then(r => r.data)

export const getAvailableSlots = (dealerCode: string, jobType = 'DEFAULT', daysAhead = 7) =>
  api.get(`/agent/slots/${dealerCode}`, { params: { job_type: jobType, days_ahead: daysAhead } }).then(r => r.data)

// Upload
export const uploadTelemetryFile = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/upload/telemetry', fd, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data)
}

export const uploadTripsFile = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/upload/trips', fd, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data)
}

export const uploadServiceHistoryFile = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/upload/service-history', fd, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data)
}

export const getUploadStatus = (jobId: string) =>
  api.get(`/upload/status/${jobId}`).then(r => r.data)

// Synthetic generation
export const generateSynthetic = (payload: { num_vehicles: number; num_days: number; failure_rate: number }) =>
  api.post('/synthetic/generate', payload).then(r => r.data)

export const trainModels = () =>
  api.post('/synthetic/train').then(r => r.data)

export const getSyntheticStatus = (jobId: string) =>
  api.get(`/synthetic/status/${jobId}`).then(r => r.data)

export default api
