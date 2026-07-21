"""
AI Service Agent workflow endpoints.

GET  /api/agent/workflows              → all active workflows
GET  /api/agent/workflows/{vin}        → workflow status for VIN
POST /api/agent/workflows/{vin}/advance → manually advance stage
POST /api/agent/trigger/{vin}          → trigger new workflow for latest alerts
POST /api/agent/chat                   → conversational interface (LangChain)
GET  /api/agent/cost-estimate/{alert_type} → cost estimate for alert type
GET  /api/agent/slots/{dealer_code}    → available service slots
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user
from api.schemas import (
    WorkflowStatus, WorkflowAdvance, AgentChatRequest, AgentChatResponse,
)

router = APIRouter(prefix="/agent", tags=["AI Agent"])


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_all_workflows() -> list[dict]:
    from agent.service_agent import _LOCAL_WORKFLOWS
    result = list(_LOCAL_WORKFLOWS.values())
    try:
        import psycopg2, pandas as pd, os
        conn = psycopg2.connect(os.getenv("DATABASE_URL", ""))
        df   = pd.read_sql("SELECT * FROM service_workflows ORDER BY updated_at DESC LIMIT 500", conn)
        conn.close()
        db_rows = df.to_dict("records")
        existing_ids = {w.get("workflow_id") for w in result}
        result += [r for r in db_rows if r.get("workflow_id") not in existing_ids]
    except Exception:
        pass
    return result


def _workflow_to_schema(w: dict) -> WorkflowStatus:
    import json
    alert = w.get("alert") or {}
    if isinstance(alert, str):
        try:
            alert = json.loads(alert)
        except Exception:
            alert = {}
    hist = w.get("stage_history") or w.get("stage_history_json") or []
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except Exception:
            hist = []
    return WorkflowStatus(
        workflow_id=w.get("workflow_id", ""),
        vin=w.get("vin", ""),
        current_stage=w.get("current_stage", ""),
        alert_type=alert.get("alert_type") if isinstance(alert, dict) else w.get("alert_id"),
        appointment_id=w.get("appointment_id"),
        created_at=str(w.get("created_at", "")),
        updated_at=str(w.get("updated_at", "")),
        completed_at=str(w.get("completed_at")) if w.get("completed_at") else None,
        escalated=bool(w.get("escalated", False)),
        stage_history=hist if isinstance(hist, list) else [],
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/workflows",
    response_model=list[WorkflowStatus],
    summary="All active service workflows",
)
async def list_workflows(
    current_user: Annotated[dict, Depends(get_current_user)],
    completed: bool = Query(False, description="Include completed workflows"),
    limit:     int  = Query(50, ge=1, le=200),
):
    all_wf = _load_all_workflows()
    if not completed:
        all_wf = [w for w in all_wf if not w.get("completed_at")]
    all_wf.sort(key=lambda w: str(w.get("updated_at", "")), reverse=True)
    return [_workflow_to_schema(w) for w in all_wf[:limit]]


@router.get(
    "/workflows/{vin}",
    response_model=dict[str, Any],
    summary="Workflow status for a specific VIN",
)
async def get_workflow_status(
    vin:          str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    from agent.service_agent import ServiceAgent
    result = ServiceAgent().get_workflow_status(vin)
    return result


@router.post(
    "/workflows/{vin}/advance",
    summary="Manually advance a workflow stage (e.g. after customer approval)",
)
async def advance_workflow(
    vin:          str,
    payload:      WorkflowAdvance,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    from agent.service_agent import ServiceAgent, _load_workflow

    # Find the most recent non-completed workflow for this VIN
    wfs = _load_workflow(vin=vin)
    if not wfs:
        raise HTTPException(status_code=404, detail=f"No workflow found for VIN {vin}")

    active = [w for w in wfs if not w.get("completed_at")]
    if not active:
        raise HTTPException(status_code=400, detail="All workflows for this VIN are already completed")

    latest_wf_id = active[0].get("workflow_id", "")
    result = ServiceAgent().advance_stage(latest_wf_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post(
    "/trigger/{vin}",
    status_code=201,
    summary="Trigger a new service workflow for the latest alerts on this VIN",
)
async def trigger_workflow(
    vin:          str,
    current_user: Annotated[dict, Depends(get_current_user)],
    vehicle_profile: dict = None,
    customer_prefs:  dict = None,
):
    from alerts.rule_engine import RuleEngine
    from alerts.ml_alert_engine import MLAlertEngine
    from agent.service_agent import ServiceAgent

    # Run both engines to find active alerts
    rule_alerts = RuleEngine().evaluate(vin, {})
    ml_alerts   = MLAlertEngine().evaluate(vin)
    all_alerts  = rule_alerts + ml_alerts

    if not all_alerts:
        return {"vin": vin, "message": "No active alerts — no workflow triggered", "workflow_id": None}

    # Use highest-severity alert as workflow trigger
    sev_order   = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    top_alert   = sorted(all_alerts, key=lambda a: sev_order.get(a.severity, 9))[0]

    wf_id = ServiceAgent().handle_alert(
        top_alert,
        vehicle_profile=vehicle_profile or {"vin": vin},
        customer_prefs=customer_prefs or {},
    )
    return {
        "vin":         vin,
        "workflow_id": wf_id,
        "alert_type":  top_alert.alert_type,
        "severity":    top_alert.severity,
        "message":     "Workflow started",
    }


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Conversational AI maintenance advisor (Claude-powered)",
)
async def chat_with_agent(
    payload:      AgentChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    try:
        import os
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        system = (
            "You are AutoPredict's AI Service Agent for connected vehicle predictive maintenance. "
            "You specialize in predictive vehicle maintenance, cost estimation, "
            "and service scheduling. Be concise, helpful, and safety-first."
        )
        if payload.vin:
            system += f" You are currently analysing vehicle VIN: {payload.vin}."

            # Inject EV cost context when available so the agent can answer
            # "how much does it cost me to drive this car?" with real data.
            try:
                from features.feature_store import FeatureStore
                from features.ev_cost_features import PETROL_COST_PER_KM_INR
                driver_feats = FeatureStore().get_online(payload.vin, "driver") or {}
                cost_per_km  = driver_feats.get("cost_per_km_inr")
                if cost_per_km is not None:
                    saving     = PETROL_COST_PER_KM_INR - float(cost_per_km)
                    comparison = "cheaper" if saving > 0 else "more expensive"
                    system += (
                        f" This vehicle's current EV charging cost is ₹{cost_per_km:.2f}/km. "
                        f"Petrol benchmark: ₹{PETROL_COST_PER_KM_INR:.2f}/km "
                        f"(₹100/L petrol at 12 km/L). "
                        f"The EV is ₹{abs(saving):.2f}/km {comparison} than petrol. "
                        f"When the customer asks about driving costs, use these figures."
                    )
                    dc_premium = driver_feats.get("dc_charge_premium_inr_30d")
                    if dc_premium and float(dc_premium) > 0:
                        system += (
                            f" The customer is paying ₹{dc_premium:.0f} extra this month "
                            f"by using DC fast chargers instead of home AC charging."
                        )
            except Exception:
                pass   # Redis unavailable or not EV — proceed without cost context

        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            max_tokens=512,
        )
        messages = [SystemMessage(content=system)]
        for h in (payload.chat_history or []):
            role, content = h.get("role", "human"), h.get("content", "")
            messages.append(HumanMessage(content=content) if role == "human"
                           else SystemMessage(content=content))
        messages.append(HumanMessage(content=payload.message))

        resp = llm.invoke(messages)
        return AgentChatResponse(response=resp.content, vin=payload.vin)

    except Exception as exc:
        return AgentChatResponse(
            response=f"Agent unavailable: {exc}. Please check ANTHROPIC_API_KEY.",
            vin=payload.vin,
        )


@router.get(
    "/cost-estimate/{alert_type}",
    summary="Cost estimate for a given alert type and vehicle",
)
async def cost_estimate(
    alert_type:   str,
    current_user: Annotated[dict, Depends(get_current_user)],
    vin:          str = Query("", description="VIN for odometer / warranty check"),
    model_code:   str = Query("DEFAULT", description="Model code: HECTOR | ZSEV | GLOSTER | ASTOR"),
):
    from agent.cost_estimator import CostEstimator
    return CostEstimator().estimate(alert_type, vin, model_code)


@router.get(
    "/slots/{dealer_code}",
    summary="Available service slots at a dealer",
)
async def available_slots(
    dealer_code:  str,
    current_user: Annotated[dict, Depends(get_current_user)],
    job_type:     str = Query("DEFAULT"),
    days_ahead:   int = Query(7, ge=1, le=30),
):
    from agent.appointment_manager import AppointmentManager
    return AppointmentManager().get_available_slots(dealer_code, job_type, days_ahead)
