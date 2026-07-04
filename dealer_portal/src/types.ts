export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'

export interface FleetHealth {
  total_vehicles: number
  online_now: number
  active_alerts_critical: number
  active_alerts_high: number
  active_alerts_medium: number
  vehicles_due_service: number
  fleet_avg_health_score: number
  generated_at: string
}

export interface VehicleRow {
  vin: string
  license_plate: string
  model_name: string
  fuel_type: string
  health_score: number
  status: string
  last_seen: string
  active_alert_count: number
  next_service_date?: string
  dealer_code: string
}

export interface Alert {
  vin: string
  alert_type: string
  severity: Severity
  title: string
  message_customer: string
  message_dealer?: string
  recommended_action: string
  estimated_cost_min: number
  estimated_cost_max: number
  confidence_score: number
  model_version?: string
  triggered_at: string
}

export interface MLPrediction {
  model: string
  predicted_date?: string
  days_until?: number
  severity: string
  confidence: number
  value?: number
  message?: string
  raw: Record<string, unknown>
}

export interface BayStatus {
  bay_id: string
  status: 'free' | 'occupied'
  current_vin?: string
  current_job?: string
  eta_free?: string
}

export interface AppointmentResponse {
  appointment_id: string
  vin: string
  job_type: string
  date: string
  time: string
  bay_id: string
  dealer_code: string
  status: string
  duration_hours: number
  booked_at?: string
}

export interface InventoryItem {
  part_code: string
  description: string
  in_stock: boolean
  qty: number
  reorder_qty: number
  unit_cost_inr?: number
  reorder_point?: number
  safety_stock?: number
  abc_class?: string
  lead_time_days?: number
  supplier?: string
  monthly_demand_avg?: number
  days_until_stockout?: number | null
}

export interface DemandForecast {
  part_code: string
  description: string
  demand_30d: number
  demand_90d: number
  confidence?: number
  historical_monthly_avg?: number
  alert_contribution?: number
  forecast_method?: string
  demand_trend?: string
  days_until_stockout?: number | null
}

export interface DriverScore {
  vin: string
  license_plate: string
  driver_name?: string
  score: number
  risk_category: string
  rank?: number
  percentile?: number
}

export interface Workflow {
  workflow_id: string
  vin: string
  current_stage: string
  alert_type?: string
  appointment_id?: string
  created_at: string
  updated_at: string
  completed_at?: string
  escalated: boolean
  stage_history: string[]
}

export interface MaintenanceEvent {
  vin: string
  license_plate: string
  model_name: string
  alert_type: string
  severity: string
  predicted_date: string
  days_until: number
  estimated_cost?: number
  confidence: number
}

export interface ServiceRecord {
  service_date: string
  job_type: string
  description: string
  cost: number
  dealer_code: string
}
