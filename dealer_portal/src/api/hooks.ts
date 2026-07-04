import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getFleetHealth, getFleetAlerts, getMaintenanceCalendar, getDriverScores,
  getVehicles, getVehicle, getVehiclePredictions, getVehicleAlerts,
  getVehicleServiceHistory, getVehicleTrips,
  getBayStatus, getAppointments, createAppointment, updateAppointmentStatus,
  getInventory, getDemandForecast,
  getWorkflows, getWorkflowStatus, advanceWorkflow, triggerWorkflow,
  chatWithAgent, generateSynthetic, getUploadStatus,
} from './client'

// ── Fleet ──────────────────────────────────────────────────────────────────────

export const useFleetHealth = () =>
  useQuery({ queryKey: ['fleet', 'health'], queryFn: getFleetHealth, refetchInterval: 60_000 })

export const useFleetAlerts = (hours = 168, severity?: string) =>
  useQuery({
    queryKey: ['fleet', 'alerts', hours, severity],
    queryFn: () => getFleetAlerts(hours, severity),
    refetchInterval: 30_000,
  })

export const useMaintenanceCalendar = (days = 90) =>
  useQuery({ queryKey: ['fleet', 'maintenance', days], queryFn: () => getMaintenanceCalendar(days) })

export const useDriverScores = () =>
  useQuery({ queryKey: ['fleet', 'driver-scores'], queryFn: () => getDriverScores() })

// ── Vehicles ───────────────────────────────────────────────────────────────────

export const useVehicles = (params?: { dealer_code?: string; fuel_type?: string; limit?: number }) =>
  useQuery({ queryKey: ['vehicles', params], queryFn: () => getVehicles(params), refetchInterval: 120_000 })

export const useVehicle = (vin: string) =>
  useQuery({ queryKey: ['vehicles', vin], queryFn: () => getVehicle(vin), enabled: !!vin })

export const useVehiclePredictions = (vin: string) =>
  useQuery({ queryKey: ['vehicles', vin, 'predictions'], queryFn: () => getVehiclePredictions(vin), enabled: !!vin })

export const useVehicleAlerts = (vin: string, severity?: string) =>
  useQuery({
    queryKey: ['vehicles', vin, 'alerts', severity],
    queryFn: () => getVehicleAlerts(vin, severity),
    enabled: !!vin,
    refetchInterval: 30_000,
  })

export const useVehicleServiceHistory = (vin: string) =>
  useQuery({ queryKey: ['vehicles', vin, 'service-history'], queryFn: () => getVehicleServiceHistory(vin), enabled: !!vin })

export const useVehicleTrips = (vin: string) =>
  useQuery({ queryKey: ['vehicles', vin, 'trips'], queryFn: () => getVehicleTrips(vin), enabled: !!vin })

// ── Dealer ─────────────────────────────────────────────────────────────────────

export const useBayStatus = (dealerCode: string) =>
  useQuery({
    queryKey: ['dealer', dealerCode, 'bay-status'],
    queryFn: () => getBayStatus(dealerCode),
    refetchInterval: 15_000,
    enabled: !!dealerCode,
  })

export const useAppointments = (dealerCode: string, daysAhead = 7) =>
  useQuery({
    queryKey: ['dealer', dealerCode, 'appointments', daysAhead],
    queryFn: () => getAppointments(dealerCode, daysAhead),
    enabled: !!dealerCode,
  })

export const useCreateAppointment = (dealerCode: string) => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: object) => createAppointment(dealerCode, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['dealer', dealerCode, 'appointments'] }),
  })
}

export const useUpdateAppointmentStatus = (dealerCode: string) => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      updateAppointmentStatus(dealerCode, id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['dealer', dealerCode] }),
  })
}

export const useInventory = (dealerCode: string) =>
  useQuery({ queryKey: ['dealer', dealerCode, 'inventory'], queryFn: () => getInventory(dealerCode), enabled: !!dealerCode })

export const useDemandForecast = (dealerCode: string) =>
  useQuery({ queryKey: ['dealer', dealerCode, 'demand-forecast'], queryFn: () => getDemandForecast(dealerCode), enabled: !!dealerCode })

// ── Agent ──────────────────────────────────────────────────────────────────────

export const useWorkflows = (completed = false) =>
  useQuery({ queryKey: ['agent', 'workflows', completed], queryFn: () => getWorkflows(completed), refetchInterval: 30_000 })

export const useWorkflowStatus = (vin: string) =>
  useQuery({ queryKey: ['agent', 'workflows', vin], queryFn: () => getWorkflowStatus(vin), enabled: !!vin })

export const useAdvanceWorkflow = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (vin: string) => advanceWorkflow(vin),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agent', 'workflows'] }),
  })
}

export const useTriggerWorkflow = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (vin: string) => triggerWorkflow(vin),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agent', 'workflows'] }),
  })
}

export const useChatWithAgent = () =>
  useMutation({ mutationFn: ({ message, vin, history }: { message: string; vin?: string; history?: object[] }) =>
    chatWithAgent(message, vin, history) })

// ── Upload / Synthetic ─────────────────────────────────────────────────────────

export const useGenerateSynthetic = () =>
  useMutation({ mutationFn: generateSynthetic })

export const useUploadStatus = (jobId: string | null) =>
  useQuery({
    queryKey: ['upload', 'status', jobId],
    queryFn: () => getUploadStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: 1000,
  })
