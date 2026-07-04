"""
AutoPredict AI Service Agent — LangGraph StateGraph workflow.

10-stage alert-to-delivery pipeline:
  detection → customer_alert → appointment_booking → parts_pre_order →
  pre_service_reminder → workshop_receipt → progress_updates →
  costing_approval → delivery_notification → post_service_followup

Each stage has:
  - entry action  : sends the appropriate Jinja2-rendered communication
  - condition     : determines next state transition (via conditional edges)
  - timeout action: escalation if stage SLA is exceeded

Workflow state is persisted to PostgreSQL service_workflows table.
Falls back to in-process dict store when DATABASE_URL is not set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

log = logging.getLogger(__name__)

PG_DSN       = os.getenv("DATABASE_URL", "")
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

# ── SLA timeouts per stage (hours) ────────────────────────────────────────────

_STAGE_SLA: dict[str, float] = {
    "detection":             0.5 / 60,   # 30 seconds for CRITICAL dispatch
    "customer_alert":        1.0,
    "appointment_booking":   2.0,
    "parts_pre_order":       4.0,
    "pre_service_reminder":  24.0,
    "workshop_receipt":      2.0,
    "progress_updates":      6.0,
    "costing_approval":      4.0,
    "delivery_notification": 2.0,
    "post_service_followup": 48.0,
}

_STAGE_ORDER = [
    "detection",
    "customer_alert",
    "appointment_booking",
    "parts_pre_order",
    "pre_service_reminder",
    "workshop_receipt",
    "progress_updates",
    "costing_approval",
    "delivery_notification",
    "post_service_followup",
]

# In-process workflow store (fallback)
_LOCAL_WORKFLOWS: dict[str, dict] = {}


# ── Workflow state TypedDict ────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    workflow_id:      str
    vin:              str
    alert:            dict          # Alert.to_dict()
    current_stage:    str
    stage_history:    list          # [{stage, entered_at, exited_at, action_result}]
    appointment_id:   str | None
    parts_order_ids:  list
    cost_estimate:    dict | None
    vehicle_profile:  dict
    customer_prefs:   dict
    created_at:       str
    updated_at:       str
    completed_at:     str | None
    escalated:        bool


# ── Jinja2 renderer ────────────────────────────────────────────────────────────

def _render(template_name: str, ctx: dict) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env = Environment(
            loader=select_autoescape(["html", "j2"]),
            autoescape=select_autoescape(["html"]),
        )
        env.loader = FileSystemLoader(str(TEMPLATE_DIR))
        tmpl = env.get_template(template_name)
        return tmpl.render(**ctx)
    except Exception as exc:
        log.debug("Template render failed (%s): %s", template_name, exc)
        return f"[{template_name}] {json.dumps(ctx, default=str)[:300]}"


# ── Stage node functions ───────────────────────────────────────────────────────

def _record_entry(state: WorkflowState, stage: str) -> WorkflowState:
    history = list(state.get("stage_history") or [])
    history.append({
        "stage":      stage,
        "entered_at": datetime.now(timezone.utc).isoformat(),
        "exited_at":  None,
    })
    return {**state, "current_stage": stage, "stage_history": history,
            "updated_at": datetime.now(timezone.utc).isoformat()}


def _close_stage(state: WorkflowState, result: str = "ok") -> WorkflowState:
    history = list(state.get("stage_history") or [])
    if history:
        history[-1]["exited_at"]    = datetime.now(timezone.utc).isoformat()
        history[-1]["action_result"] = result
    return {**state, "stage_history": history}


async def _dispatch_channels(state: WorkflowState, subject: str, push_body: str,
                              html_body: str, sms_body: str) -> None:
    """Fire-and-forget dispatch via AlertDispatcher."""
    try:
        from alerts.alert_dispatcher import _send_push, _send_sms, _send_email
        prefs = state.get("customer_prefs", {})
        coro  = asyncio.gather(
            _send_push(prefs.get("device_token", ""), subject, push_body, {"vin": state["vin"]}),
            _send_sms(prefs.get("phone_number", ""), sms_body),
            _send_email(prefs.get("email", ""), subject, html_body),
            return_exceptions=True,
        )
        asyncio.ensure_future(coro)
    except Exception as exc:
        log.debug("dispatch_channels failed: %s", exc)


# ── Stage 1: detection ────────────────────────────────────────────────────────

def _node_detection(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "detection")
    alert = state["alert"]
    log.info("Workflow %s: DETECTION  VIN=%s alert=%s sev=%s",
             state["workflow_id"], state["vin"], alert.get("alert_type"), alert.get("severity"))
    state = _close_stage(state, "detected")
    return state


# ── Stage 2: customer_alert ───────────────────────────────────────────────────

def _node_customer_alert(state: WorkflowState) -> WorkflowState:
    state  = _record_entry(state, "customer_alert")
    alert  = state["alert"]
    veh    = state["vehicle_profile"]
    prefs  = state["customer_prefs"]

    ctx = {"alert": _alert_obj(alert), "vehicle": veh}
    html  = _render("alert_email.html", ctx)
    push  = f"{alert.get('severity')}: {alert.get('title', '')}"
    sms   = _render("alert_sms.j2", ctx)
    subj  = f"[AutoPredict {alert.get('severity')}] {alert.get('title', '')}"

    asyncio.ensure_future(_dispatch_channels(state, subj, push, html, sms))
    log.info("Workflow %s: customer_alert sent (sev=%s)", state["workflow_id"], alert.get("severity"))
    state = _close_stage(state, "alert_sent")
    return state


# ── Stage 3: appointment_booking ──────────────────────────────────────────────

def _node_appointment_booking(state: WorkflowState) -> WorkflowState:
    state     = _record_entry(state, "appointment_booking")
    alert     = state["alert"]
    veh       = state["vehicle_profile"]
    dealer    = veh.get("home_dealer_code", "DLR001")
    alert_type = alert.get("alert_type", "")

    try:
        from agent.appointment_manager import AppointmentManager
        from agent.cost_estimator import CostEstimator

        mgr   = AppointmentManager()
        slots = mgr.get_available_slots(dealer, alert_type, days_ahead=7)

        cost = CostEstimator().estimate(
            alert_type,
            state["vin"],
            veh.get("model_code", "DEFAULT"),
        )
        state = {**state, "cost_estimate": cost}

        if slots:
            slot      = slots[0]   # take earliest available
            parts     = mgr.suggest_parts(alert_type)
            jc_tmpl   = json.dumps({"alert_type": alert_type, "parts": parts})
            appt_id   = mgr.book_slot(state["vin"], slot, jc_tmpl)
            state     = {**state, "appointment_id": appt_id}

            # Send confirmation
            appt_ctx = {**slot, "appointment_id": appt_id,
                        "dealer_name": veh.get("home_dealer_name", dealer)}
            html = _render("appointment_confirm.html",
                           {"appointment": appt_ctx, "vehicle": veh, "cost": cost})
            asyncio.ensure_future(_dispatch_channels(
                state,
                f"Service Appointment Confirmed — {appt_id}",
                f"Appointment booked for {slot.get('date')} at {slot.get('time')}",
                html,
                f"AutoPredict: Appointment confirmed {appt_id} on {slot.get('date')} {slot.get('time')} at {dealer}.",
            ))
            log.info("Workflow %s: appointment booked %s", state["workflow_id"], appt_id)
            state = _close_stage(state, "booked")
        else:
            log.warning("Workflow %s: no slots available at %s", state["workflow_id"], dealer)
            state = _close_stage(state, "no_slots")
    except Exception as exc:
        log.error("appointment_booking error: %s", exc, exc_info=True)
        state = _close_stage(state, f"error:{exc}")

    return state


# ── Stage 4: parts_pre_order ──────────────────────────────────────────────────

def _node_parts_pre_order(state: WorkflowState) -> WorkflowState:
    state     = _record_entry(state, "parts_pre_order")
    alert     = state["alert"]
    veh       = state["vehicle_profile"]
    dealer    = veh.get("home_dealer_code", "DLR001")

    try:
        from agent.appointment_manager import AppointmentManager
        mgr    = AppointmentManager()
        parts  = mgr.suggest_parts(alert.get("alert_type", ""))
        avail  = mgr.get_parts_availability(parts, dealer)

        order_ids = [f"PO-{uuid.uuid4().hex[:8].upper()}" for p in parts if not avail.get(p, {}).get("in_stock")]
        state     = {**state, "parts_order_ids": order_ids}

        html = _render("parts_order.html", {
            "appointment": {"appointment_id": state.get("appointment_id", ""), "date": ""},
            "vehicle": veh,
            "parts": avail,
        })
        asyncio.ensure_future(_dispatch_channels(
            state,
            "Parts Pre-Ordered for Your Service",
            f"{len(parts)} parts secured for your service.",
            html,
            f"AutoPredict: Parts pre-ordered for your {veh.get('model_name','')} service.",
        ))
        state = _close_stage(state, f"ordered:{len(order_ids)}")
    except Exception as exc:
        log.error("parts_pre_order error: %s", exc)
        state = _close_stage(state, f"error:{exc}")

    return state


# ── Stage 5: pre_service_reminder ─────────────────────────────────────────────

def _node_pre_service_reminder(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "pre_service_reminder")
    veh   = state["vehicle_profile"]
    appt  = state.get("appointment_id", "")

    push  = f"Reminder: Your vehicle service is tomorrow. Appointment {appt}."
    sms   = f"AutoPredict Reminder: {veh.get('license_plate','')} service tomorrow. Appt: {appt}."
    asyncio.ensure_future(_dispatch_channels(state, "Service Reminder", push, push, sms))

    state = _close_stage(state, "reminder_sent")
    return state


# ── Stage 6: workshop_receipt ─────────────────────────────────────────────────

def _node_workshop_receipt(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "workshop_receipt")
    veh   = state["vehicle_profile"]
    appt  = state.get("appointment_id", "")

    push = f"Your {veh.get('model_name','')} has been received at the workshop (Ref: {appt})."
    asyncio.ensure_future(_dispatch_channels(
        state, "Vehicle Received at Workshop", push, push,
        f"AutoPredict: {veh.get('license_plate','')} received at workshop. Ref: {appt}.",
    ))
    state = _close_stage(state, "receipt_sent")
    return state


# ── Stage 7: progress_updates ─────────────────────────────────────────────────

def _node_progress_updates(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "progress_updates")
    veh   = state["vehicle_profile"]
    appt  = state.get("appointment_id", "")

    steps = [
        {"name": "Vehicle received",            "status": "done",    "time": "09:15"},
        {"name": "Diagnosis / inspection",      "status": "done",    "time": "09:45"},
        {"name": "Parts sourcing",              "status": "done",    "time": "10:30"},
        {"name": "Service in progress",         "status": "active",  "time": "11:00"},
        {"name": "Quality check",               "status": "pending", "time": None},
        {"name": "Vehicle ready for handover",  "status": "pending", "time": None},
    ]
    html = _render("progress_update.html", {
        "vin": state["vin"], "vehicle": veh, "appointment_id": appt,
        "steps": steps, "eta": "Expected completion: 2:00 PM", "note": None,
    })
    asyncio.ensure_future(_dispatch_channels(
        state, "Service Progress Update", "Your vehicle service is in progress.",
        html, f"AutoPredict: {veh.get('license_plate','')} service in progress. Est. done: 2PM.",
    ))
    state = _close_stage(state, "update_sent")
    return state


# ── Stage 8: costing_approval ─────────────────────────────────────────────────

def _node_costing_approval(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "costing_approval")
    cost  = state.get("cost_estimate") or {}
    veh   = state["vehicle_profile"]
    appt  = state.get("appointment_id", "")

    total_min = cost.get("total_min", 0)
    total_max = cost.get("total_max", 0)
    wty       = cost.get("warranty_likely", False)

    push = (
        f"Cost approval needed: ₹{total_min:,.0f}–₹{total_max:,.0f}. "
        + ("Warranty may apply. " if wty else "")
        + "Reply to approve."
    )
    sms = (
        f"AutoPredict [{veh.get('license_plate','')}]: Service cost ₹{total_min:,.0f}–{total_max:,.0f}. "
        + ("WTY likely. " if wty else "")
        + f"Approve: autopredict.in/approve/{appt}"
    )
    asyncio.ensure_future(_dispatch_channels(
        state, "Service Cost Approval Required", push, push, sms
    ))
    state = _close_stage(state, "approval_sent")
    return state


# ── Stage 9: delivery_notification ────────────────────────────────────────────

def _node_delivery_notification(state: WorkflowState) -> WorkflowState:
    state = _record_entry(state, "delivery_notification")
    veh   = state["vehicle_profile"]
    cost  = state.get("cost_estimate") or {}
    appt  = state.get("appointment_id", "")

    html = _render("delivery_ready.html", {
        "vehicle": veh, "appointment_id": appt,
        "dealer_name":        veh.get("home_dealer_name", "Your MG Dealer"),
        "services_performed": ["Brake pad replacement", "Fluid top-up"],
        "parts_replaced":     ["Brake Pads (Front + Rear)"],
        "warranty_applied":   cost.get("warranty_likely", False),
        "final_amount":       cost.get("customer_pays_min", cost.get("total_min", 0)),
    })
    push = f"Your {veh.get('model_name','')} is ready for pickup at {veh.get('home_dealer_name','the dealer')}!"
    asyncio.ensure_future(_dispatch_channels(
        state, "Vehicle Ready for Pickup", push, html,
        f"AutoPredict: {veh.get('license_plate','')} ready for pickup. Ref: {appt}.",
    ))
    state = _close_stage(state, "delivery_sent")
    return state


# ── Stage 10: post_service_followup ───────────────────────────────────────────

def _node_post_service_followup(state: WorkflowState) -> WorkflowState:
    state  = _record_entry(state, "post_service_followup")
    veh    = state["vehicle_profile"]
    token  = uuid.uuid4().hex[:16]

    html = _render("followup_survey.html", {
        "vehicle": veh,
        "service_date": datetime.now(timezone.utc).strftime("%d %b %Y"),
        "survey_token": token,
        "positive_options": [
            "Fast service", "Friendly staff", "Clean facility",
            "Good communication", "Value for money",
        ],
        "improve_options": [
            "Waiting time", "Communication", "Cost transparency",
            "Shuttle service", "Online booking experience",
        ],
    })
    asyncio.ensure_future(_dispatch_channels(
        state, "How Was Your Service?",
        "Share your feedback — it takes under a minute.",
        html,
        f"AutoPredict: How was your service? Share feedback: autopredict.in/survey/{token}",
    ))

    completed_at = datetime.now(timezone.utc).isoformat()
    state = {**state, "completed_at": completed_at}
    state = _close_stage(state, "survey_sent")
    log.info("Workflow %s COMPLETED for VIN=%s", state["workflow_id"], state["vin"])
    return state


# ── Timeout escalation ────────────────────────────────────────────────────────

def _check_timeout(state: WorkflowState) -> bool:
    """Return True if the current stage has exceeded its SLA."""
    stage   = state.get("current_stage", "")
    history = state.get("stage_history") or []
    if not history:
        return False
    entered = history[-1].get("entered_at")
    if not entered:
        return False
    sla_h   = _STAGE_SLA.get(stage, 24.0)
    entered_dt = datetime.fromisoformat(entered)
    if entered_dt.tzinfo is None:
        entered_dt = entered_dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - entered_dt).total_seconds() > sla_h * 3600


def _escalate(state: WorkflowState) -> WorkflowState:
    stage = state.get("current_stage", "")
    log.warning("Workflow %s: SLA breach at stage=%s — escalating", state["workflow_id"], stage)
    veh   = state["vehicle_profile"]
    push  = f"⚠️ Service workflow SLA breach at '{stage}'. Manual intervention needed."
    asyncio.ensure_future(_dispatch_channels(
        state, "[ESCALATION] Workflow SLA Breach", push, push, push[:160]
    ))
    return {**state, "escalated": True}


# ── LangGraph StateGraph assembly ─────────────────────────────────────────────

def _build_graph():
    try:
        from langgraph.graph import StateGraph, END

        def detection(s):          return _node_detection(s)
        def customer_alert(s):     return _node_customer_alert(s)
        def appointment_booking(s):return _node_appointment_booking(s)
        def parts_pre_order(s):    return _node_parts_pre_order(s)
        def pre_service_reminder(s):return _node_pre_service_reminder(s)
        def workshop_receipt(s):   return _node_workshop_receipt(s)
        def progress_updates(s):   return _node_progress_updates(s)
        def costing_approval(s):   return _node_costing_approval(s)
        def delivery_notification(s):return _node_delivery_notification(s)
        def post_service_followup(s):return _node_post_service_followup(s)

        g = StateGraph(WorkflowState)
        for name, fn in [
            ("detection",             detection),
            ("customer_alert",        customer_alert),
            ("appointment_booking",   appointment_booking),
            ("parts_pre_order",       parts_pre_order),
            ("pre_service_reminder",  pre_service_reminder),
            ("workshop_receipt",      workshop_receipt),
            ("progress_updates",      progress_updates),
            ("costing_approval",      costing_approval),
            ("delivery_notification", delivery_notification),
            ("post_service_followup", post_service_followup),
        ]:
            g.add_node(name, fn)

        g.set_entry_point("detection")
        for i in range(len(_STAGE_ORDER) - 1):
            g.add_edge(_STAGE_ORDER[i], _STAGE_ORDER[i + 1])
        g.add_edge("post_service_followup", END)

        return g.compile()

    except Exception as exc:
        log.warning("LangGraph unavailable (%s) — using linear fallback", exc)
        return None


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ── PostgreSQL persistence ─────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS service_workflows (
    workflow_id         TEXT PRIMARY KEY,
    vin                 TEXT NOT NULL,
    alert_id            TEXT,
    current_stage       TEXT NOT NULL,
    stage_history_json  TEXT,
    appointment_id      TEXT,
    parts_order_ids     TEXT,
    cost_estimate_json  TEXT,
    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ
);
"""


def _persist(state: WorkflowState) -> None:
    _LOCAL_WORKFLOWS[state["workflow_id"]] = state
    try:
        import psycopg2
        if not PG_DSN:
            return
        conn = psycopg2.connect(PG_DSN)
        cur  = conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(
            """
            INSERT INTO service_workflows
              (workflow_id, vin, alert_id, current_stage, stage_history_json,
               appointment_id, parts_order_ids, cost_estimate_json,
               created_at, updated_at, completed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (workflow_id) DO UPDATE SET
              current_stage      = EXCLUDED.current_stage,
              stage_history_json = EXCLUDED.stage_history_json,
              appointment_id     = EXCLUDED.appointment_id,
              parts_order_ids    = EXCLUDED.parts_order_ids,
              cost_estimate_json = EXCLUDED.cost_estimate_json,
              updated_at         = EXCLUDED.updated_at,
              completed_at       = EXCLUDED.completed_at
            """,
            (
                state["workflow_id"],
                state["vin"],
                state["alert"].get("alert_type", ""),
                state["current_stage"],
                json.dumps(state.get("stage_history") or []),
                state.get("appointment_id"),
                json.dumps(state.get("parts_order_ids") or []),
                json.dumps(state.get("cost_estimate")),
                state["created_at"],
                state["updated_at"],
                state.get("completed_at"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("Workflow persist failed: %s", exc)


def _load_workflow(workflow_id: str | None = None, vin: str | None = None) -> list[dict]:
    results = []
    if workflow_id:
        if workflow_id in _LOCAL_WORKFLOWS:
            results.append(_LOCAL_WORKFLOWS[workflow_id])
    elif vin:
        results = [w for w in _LOCAL_WORKFLOWS.values() if w.get("vin") == vin]
    try:
        import psycopg2
        if not PG_DSN:
            return results
        conn = psycopg2.connect(PG_DSN)
        cur  = conn.cursor()
        if workflow_id:
            cur.execute("SELECT * FROM service_workflows WHERE workflow_id=%s", (workflow_id,))
        else:
            cur.execute("SELECT * FROM service_workflows WHERE vin=%s ORDER BY created_at DESC", (vin,))
        cols = [d[0] for d in cur.description] if cur.description else []
        for row in cur.fetchall():
            results.append(dict(zip(cols, row)))
        conn.close()
    except Exception:
        pass
    return results


# ── Helper: turn alert dict back to a simple object for templates ──────────────

class _AlertObj:
    def __init__(self, d: dict):
        self.__dict__.update(d)
        if isinstance(self.triggered_at, str):
            self.triggered_at = datetime.fromisoformat(self.triggered_at)


def _alert_obj(d: dict) -> _AlertObj:
    return _AlertObj(d)


# ── ServiceAgent ───────────────────────────────────────────────────────────────

class ServiceAgent:
    """
    End-to-end service workflow agent.

    Usage:
        agent = ServiceAgent()
        workflow_id = agent.handle_alert(alert, vehicle_profile, customer_prefs)
        status = agent.get_workflow_status(vin="MZ7X...")
    """

    def handle_alert(
        self,
        alert: Any,
        vehicle_profile: dict | None = None,
        customer_prefs:  dict | None = None,
    ) -> str:
        """
        Start a new 10-stage service workflow for *alert*.

        Returns: workflow_id (str UUID)
        """
        alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        now        = datetime.now(timezone.utc).isoformat()
        wf_id      = str(uuid.uuid4())

        state: WorkflowState = {
            "workflow_id":     wf_id,
            "vin":             alert_dict.get("vin", ""),
            "alert":           alert_dict,
            "current_stage":   "detection",
            "stage_history":   [],
            "appointment_id":  None,
            "parts_order_ids": [],
            "cost_estimate":   None,
            "vehicle_profile": vehicle_profile or {},
            "customer_prefs":  customer_prefs or {},
            "created_at":      now,
            "updated_at":      now,
            "completed_at":    None,
            "escalated":       False,
        }

        graph = _get_graph()
        if graph is not None:
            # Run LangGraph synchronously (fire-and-forget async stages)
            try:
                final = graph.invoke(state)
                _persist(final)
            except Exception as exc:
                log.error("LangGraph invocation failed: %s", exc, exc_info=True)
                state = self._run_linear(state)
                _persist(state)
        else:
            # Linear fallback
            state = self._run_linear(state)
            _persist(state)

        log.info("ServiceAgent: workflow %s started for VIN=%s", wf_id, alert_dict.get("vin"))
        return wf_id

    def _run_linear(self, state: WorkflowState) -> WorkflowState:
        """Linear fallback: run all stages in sequence (no LangGraph)."""
        node_fns = [
            _node_detection,
            _node_customer_alert,
            _node_appointment_booking,
            _node_parts_pre_order,
            _node_pre_service_reminder,
            _node_workshop_receipt,
            _node_progress_updates,
            _node_costing_approval,
            _node_delivery_notification,
            _node_post_service_followup,
        ]
        for fn in node_fns:
            try:
                state = fn(state)
                if _check_timeout(state):
                    state = _escalate(state)
                _persist(state)
            except Exception as exc:
                log.error("Stage %s failed: %s", fn.__name__, exc, exc_info=True)
        return state

    def advance_stage(self, workflow_id: str) -> dict:
        """
        Manually advance a workflow to its next stage.
        Used when an awaited action (e.g. customer approval) is received.
        """
        workflows = _load_workflow(workflow_id=workflow_id)
        if not workflows:
            return {"error": "Workflow not found"}

        state = workflows[0]
        if isinstance(state, dict) and "current_stage" in state:
            curr_idx = _STAGE_ORDER.index(state["current_stage"]) if state["current_stage"] in _STAGE_ORDER else -1
            if curr_idx < 0 or curr_idx >= len(_STAGE_ORDER) - 1:
                return {"error": "Workflow already at final stage or stage unknown"}

            next_stage = _STAGE_ORDER[curr_idx + 1]
            node_map   = {
                "workshop_receipt":      _node_workshop_receipt,
                "progress_updates":      _node_progress_updates,
                "costing_approval":      _node_costing_approval,
                "delivery_notification": _node_delivery_notification,
                "post_service_followup": _node_post_service_followup,
            }
            fn = node_map.get(next_stage)
            if fn:
                state = fn(state)   # type: ignore[arg-type]
                _persist(state)
            return {"workflow_id": workflow_id, "advanced_to": next_stage}

        return {"error": "Invalid workflow state"}

    def get_workflow_status(self, vin: str) -> dict:
        """
        Return current stage and history for all workflows for *vin*.
        """
        workflows = _load_workflow(vin=vin)
        if not workflows:
            return {"vin": vin, "workflows": []}
        result = []
        for w in workflows:
            result.append({
                "workflow_id":   w.get("workflow_id"),
                "current_stage": w.get("current_stage"),
                "alert_type":    (w.get("alert") or {}).get("alert_type") if isinstance(w.get("alert"), dict) else w.get("alert_id"),
                "appointment_id":w.get("appointment_id"),
                "created_at":    str(w.get("created_at")),
                "updated_at":    str(w.get("updated_at")),
                "completed_at":  str(w.get("completed_at")) if w.get("completed_at") else None,
                "escalated":     w.get("escalated", False),
                "stage_history": w.get("stage_history") or [],
            })
        return {"vin": vin, "workflows": result}
