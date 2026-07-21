import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getFleetHealth, getFleetAlerts, getMaintenanceCalendar, getDriverScores,
  getVehicles, getVehicle, getVehiclePredictions, getVehicleAlerts,
  getVehicleServiceHistory, getVehicleTrips, getEVHealth,
  getBayStatus, getAppointments, createAppointment, updateAppointmentStatus,
  getInventory, getDemandForecast,
  getInventoryOverview, getInventoryStock, getInventoryAlerts,
  getReorderPlan, getInventoryAnalytics, getDealerComparison,
  getPartDetail, getInventoryTransactions,
  getWorkflows, getWorkflowStatus, advanceWorkflow, triggerWorkflow,
  chatWithAgent, generateSynthetic, getUploadStatus,
  getOemFleetOverview, getOemModelHealth, getOemEda, getModelEda, postOemWhatIf,
  getOemRetrainHistory, triggerOemRetrain, getOemRetrainStatus, stopOemRetrain,
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

export const useEVHealth = (vin: string, enabled = true) =>
  useQuery({
    queryKey: ['ev', vin, 'health'],
    queryFn:  () => getEVHealth(vin),
    enabled:  !!vin && enabled,
    staleTime: 60_000,
  })

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
  useQuery({ queryKey: ['dealer', dealerCode, 'demand-forecast'], queryFn: () => getDemandForecast(dealerCode), enabled: !!dealerCode, staleTime: 0, refetchOnMount: 'always' })

// ── Comprehensive inventory hooks ───────────────────────────────────────────
export const useInventoryOverview = () =>
  useQuery({ queryKey: ['inventory', 'overview'], queryFn: getInventoryOverview, refetchInterval: 120_000 })

export const useInventoryStock = (params?: Record<string, string>) =>
  useQuery({ queryKey: ['inventory', 'stock', params], queryFn: () => getInventoryStock(params) })

export const useInventoryAlerts = (params?: Record<string, string>) =>
  useQuery({ queryKey: ['inventory', 'alerts', params], queryFn: () => getInventoryAlerts(params), refetchInterval: 60_000 })

export const useReorderPlan = (dealerCode?: string) =>
  useQuery({ queryKey: ['inventory', 'reorder-plan', dealerCode], queryFn: () => getReorderPlan(dealerCode) })

export const useInventoryAnalytics = (dealerCode?: string) =>
  useQuery({ queryKey: ['inventory', 'analytics', dealerCode], queryFn: () => getInventoryAnalytics(dealerCode) })

export const useDealerComparison = () =>
  useQuery({ queryKey: ['inventory', 'dealers'], queryFn: getDealerComparison })

export const usePartDetail = (partCode: string | null) =>
  useQuery({ queryKey: ['inventory', 'parts', partCode], queryFn: () => getPartDetail(partCode!), enabled: !!partCode })

export const useInventoryTransactions = (params?: Record<string, string>) =>
  useQuery({ queryKey: ['inventory', 'transactions', params], queryFn: () => getInventoryTransactions(params) })

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

// ── OEM ────────────────────────────────────────────────────────────────────────

export const useOemFleetOverview = (groupBy = 'dealer_code') =>
  useQuery({ queryKey: ['oem', 'fleet-overview', groupBy], queryFn: () => getOemFleetOverview(groupBy) })

export const useOemModelHealth = (modelName?: string) =>
  useQuery({ queryKey: ['oem', 'model-health', modelName], queryFn: () => getOemModelHealth(modelName) })

export const useOemEda = (featureGroup: string) =>
  useQuery({ queryKey: ['oem', 'eda', featureGroup], queryFn: () => getOemEda(featureGroup) })

export const useModelEda = (modelName: string | null) =>
  useQuery({
    queryKey: ['oem', 'model-eda', modelName],
    queryFn: () => getModelEda(modelName!),
    enabled: !!modelName,
    staleTime: 10 * 60 * 1000,
    retry: false,
  })

export const useOemWhatIf = () =>
  useMutation({ mutationFn: (payload: object) => postOemWhatIf(payload) })

export const useOemRetrainHistory = () =>
  useQuery({ queryKey: ['oem', 'retrain', 'history'], queryFn: getOemRetrainHistory })

export const useTriggerOemRetrain = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload: object) => triggerOemRetrain(payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['oem', 'retrain'] }),
  })
}

export const useOemRetrainStatus = (jobId: string | null) =>
  useQuery({
    queryKey: ['oem', 'retrain', 'status', jobId],
    queryFn: () => getOemRetrainStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: 3000,
  })

export const useStopOemRetrain = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => stopOemRetrain(jobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['oem', 'retrain'] }),
  })
}

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
