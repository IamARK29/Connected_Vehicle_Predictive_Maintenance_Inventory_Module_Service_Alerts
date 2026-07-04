"""Pydantic schemas for all API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Auth ───────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    role:         str
    expires_in:   int = 86400


# ── Vehicle ────────────────────────────────────────────────────────────────────

class VehicleCreate(BaseModel):
    vin:         str   = Field(..., min_length=17, max_length=17)
    make:        str   = "MG"
    model:       str
    year:        int   = Field(..., ge=2000, le=2030)
    dealer_id:   str
    driver_id:   Optional[str] = None
    odometer_km: float = 0.0
    fuel_type:   str   = "petrol"


class VehicleResponse(VehicleCreate):
    id:         str
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthScore(BaseModel):
    component:   str
    score:       float = Field(..., ge=0, le=100)
    severity:    str   = "ok"   # ok | warning | critical
    message:     str   = ""


class VehicleHealthSummary(BaseModel):
    vin:             str
    license_plate:   str = ""
    model_name:      str = ""
    fuel_type:       str = ""
    overall_score:   float = Field(0.0, ge=0, le=100)
    active_alerts:   int  = 0
    critical_alerts: int  = 0
    km_to_next_service: Optional[float] = None
    health_scores:   list[HealthScore] = []
    last_seen:       Optional[datetime] = None


class VehicleDetail(BaseModel):
    vin:              str
    license_plate:    str = ""
    model_name:       str = ""
    model_code:       str = ""
    fuel_type:        str = ""
    color:            str = ""
    manufacture_year: Optional[int]   = None
    odometer_km:      Optional[float] = None
    home_dealer_code: str = ""
    home_dealer_name: str = ""
    region:           str = ""
    health_summary:   Optional[VehicleHealthSummary] = None
    predictions:      dict[str, Any]  = {}
    active_alerts:    list[dict]      = []


# ── Telemetry ──────────────────────────────────────────────────────────────────

class TelemetryIngest(BaseModel):
    vin:                     str
    timestamp:               datetime
    speed_kmh:               float = 0.0
    engine_rpm:              float = 0.0
    engine_temp_c:           float = 90.0
    engine_oil_life_pct:     float = 100.0
    throttle_pct:            float = 0.0
    fuel_level_pct:          float = 50.0
    fuel_consumption_l100km: float = 8.0
    brake_pad_front_mm:      float = 10.0
    brake_pad_rear_mm:       float = 10.0
    brake_fluid_level_pct:   float = 100.0
    battery_12v_voltage_v:   float = 12.6
    hv_battery_soc_pct:      float = 80.0
    hv_battery_soh_pct:      float = 95.0
    hv_battery_temp_c:       float = 25.0
    tyre_pressure_fl_bar:    float = 2.3
    tyre_pressure_fr_bar:    float = 2.3
    tyre_pressure_rl_bar:    float = 2.3
    tyre_pressure_rr_bar:    float = 2.3
    odometer_km:             float = 0.0
    latitude:                float = 0.0
    longitude:               float = 0.0


class TelemetryRow(BaseModel):
    timestamp: datetime
    fields:    dict[str, Any]


# ── Predictions ────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    vin:       str
    telemetry: dict[str, Any]


class PredictionResponse(BaseModel):
    vin:          str
    component:    str
    predictions:  dict[str, Any]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class MLPrediction(BaseModel):
    model:          str
    severity:       str   = "ok"
    confidence:     float = 0.0
    predicted_date: Optional[str]  = None
    message:        str   = ""
    value:          Optional[Any]  = None
    raw:            dict[str, Any] = {}


# ── Alerts ─────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):
    vin:              str
    alert_type:       str
    severity:         str
    title:            str
    message_customer: str
    recommended_action: str
    estimated_cost_min: float = 0
    estimated_cost_max: float = 0
    confidence_score: float   = 1.0
    model_version:    str     = "rule/1.0"
    triggered_at:     str


class AlertAcknowledge(BaseModel):
    acknowledged_by: str
    note:            str = ""


# ── Fleet ──────────────────────────────────────────────────────────────────────

class FleetHealthSummary(BaseModel):
    total_vehicles:         int
    online_now:             int   = 0
    active_alerts_critical: int   = 0
    active_alerts_high:     int   = 0
    active_alerts_medium:   int   = 0
    vehicles_due_service:   int   = 0
    fleet_avg_health_score: float = 0.0
    generated_at:           datetime = Field(default_factory=datetime.utcnow)


class MaintenanceEvent(BaseModel):
    vin:              str
    license_plate:    str = ""
    model_name:       str = ""
    alert_type:       str
    severity:         str
    predicted_date:   Optional[str] = None
    days_until:       Optional[int] = None
    estimated_cost:   Optional[float] = None
    confidence:       float = 0.0


class DriverScoreEntry(BaseModel):
    vin:           str
    license_plate: str   = ""
    driver_name:   str   = ""
    score:         float = 0.0
    risk_category: str   = "medium"
    rank:          int   = 0
    percentile:    float = 0.0


# ── Dealer ─────────────────────────────────────────────────────────────────────

class AppointmentCreate(BaseModel):
    vin:       str
    job_type:  str
    date:      str
    time:      str
    bay_id:    str = "BAY-01"
    notes:     str = ""


class AppointmentResponse(BaseModel):
    appointment_id:    str
    vin:               str
    job_type:          str
    date:              str
    time:              str
    bay_id:            str
    dealer_code:       str
    status:            str = "confirmed"
    duration_hours:    float = 1.0
    booked_at:         Optional[str] = None


class AppointmentStatusUpdate(BaseModel):
    status: str   # confirmed | in_progress | completed | cancelled
    note:   str = ""


class BayStatus(BaseModel):
    bay_id:         str
    status:         str   # free | occupied | reserved
    current_vin:    Optional[str] = None
    current_job:    Optional[str] = None
    eta_free:       Optional[str] = None


class InventoryItem(BaseModel):
    part_code:   str
    description: str = ""
    in_stock:    bool = True
    qty:         int  = 0
    reorder_qty: int  = 0


class DemandForecast(BaseModel):
    part_code:     str
    description:   str   = ""
    demand_30d:    int   = 0
    demand_90d:    int   = 0
    confidence:    float = 0.5


# ── Service history & trips ────────────────────────────────────────────────────

class ServiceRecord(BaseModel):
    vin:          str
    service_type: str
    created_on:   Optional[str]   = None
    mileage:      Optional[float] = None
    dealer_code:  str = ""
    dealer_name:  str = ""
    total_value:  Optional[float] = None
    issue_type:   str = ""


class TripRecord(BaseModel):
    trip_id:       str
    vin:           str
    start_time:    Optional[str]   = None
    end_time:      Optional[str]   = None
    odometer:      Optional[float] = None
    avg_speed:     Optional[float] = None
    max_speed:     Optional[float] = None
    fuel_consumed: Optional[float] = None
    drive_score:   Optional[float] = None


# ── Agent / Workflow ───────────────────────────────────────────────────────────

class WorkflowStatus(BaseModel):
    workflow_id:    str
    vin:            str
    current_stage:  str
    alert_type:     Optional[str] = None
    appointment_id: Optional[str] = None
    created_at:     str
    updated_at:     str
    completed_at:   Optional[str] = None
    escalated:      bool = False
    stage_history:  list[dict] = []


class WorkflowAdvance(BaseModel):
    stage: Optional[str] = None
    note:  str = ""


class AgentChatRequest(BaseModel):
    message:      str
    vin:          Optional[str]  = None
    chat_history: list[dict] = []


class AgentChatResponse(BaseModel):
    response: str
    vin:      Optional[str] = None


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadJobStatus(BaseModel):
    job_id:  str
    pct:     int   = 0
    message: str   = "Queued"
    result:  dict[str, Any] = {}


# ── Maintenance schedule ───────────────────────────────────────────────────────

class MaintenanceScheduleItem(BaseModel):
    vin:                str
    component:          str
    service_type:       str
    due_km:             Optional[float] = None
    due_date:           Optional[datetime] = None
    priority:           str   = "normal"
    estimated_cost_inr: Optional[float]   = None


# ── Legacy stubs (kept for backward-compat with old routers) ───────────────────

class FleetSummary(BaseModel):
    total_vehicles:         int
    active_alerts_critical: int
    active_alerts_warning:  int
    vehicles_due_service:   int
    fleet_avg_health_score: float


class DealerCreate(BaseModel):
    name:    str
    city:    str
    state:   str
    phone:   str
    email:   str
    address: str


class DealerResponse(DealerCreate):
    id:         str
    created_at: datetime

    model_config = {"from_attributes": True}
