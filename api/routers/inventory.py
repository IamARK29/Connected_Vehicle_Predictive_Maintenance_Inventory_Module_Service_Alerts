"""
Comprehensive Inventory Management API.

GET /api/inventory/overview              → fleet-wide KPIs
GET /api/inventory/stock                 → all dealers × parts stock ledger
GET /api/inventory/alerts                → items below reorder point / stockout
GET /api/inventory/reorder-plan          → EOQ-based purchase orders needed
GET /api/inventory/analytics             → turnover, ABC, slow movers, value distribution
GET /api/inventory/dealers               → dealer comparison matrix
GET /api/inventory/parts/{part_code}     → part deep dive (all dealers + 90d trend)
GET /api/inventory/transactions          → recent transaction log
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_current_user

router = APIRouter(prefix="/inventory", tags=["Inventory"])

DATA_DIR = Path(os.getenv("DATA_DIR", "data/synthetic"))


def _load_stock(dealer_code: str | None = None) -> pd.DataFrame:
    csv = DATA_DIR / "inventory_stock.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    if dealer_code and dealer_code.upper() not in ("ALL", "NONE", ""):
        df = df[df["dealer_code"].astype(str) == dealer_code]
    return df


def _load_transactions(dealer_code: str | None = None, days: int = 90) -> pd.DataFrame:
    csv = DATA_DIR / "inventory_transactions.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]
    if dealer_code:
        df = df[df["dealer_code"].astype(str) == dealer_code]
    return df


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (f != f or abs(f) == float("inf")) else f
    except Exception:
        return default


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/overview")
async def inventory_overview(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    df = _load_stock()
    if df.empty:
        return {"error": "Inventory data not generated. Run generate_inventory.py first."}

    total_sku        = len(df)
    total_value      = _safe_float(df["inventory_value_inr"].sum())
    stockout_count   = int((df["stock_status"] == "STOCKOUT").sum())
    critical_count   = int((df["stock_status"] == "CRITICAL").sum())
    low_count        = int((df["stock_status"] == "LOW").sum())
    ok_count         = int((df["stock_status"] == "OK").sum())
    slow_movers      = int(df.get("is_slow_mover", pd.Series(dtype=bool)).sum()) if "is_slow_mover" in df.columns else 0
    value_at_risk    = _safe_float(df[df["stock_status"].isin(["STOCKOUT", "CRITICAL"])]["inventory_value_inr"].sum())
    dealers_affected = int(df[df["stock_status"].isin(["STOCKOUT", "CRITICAL"])]["dealer_code"].nunique())

    abc_value = df.groupby("abc_class")["inventory_value_inr"].sum().to_dict()
    abc_count = df.groupby("abc_class").size().to_dict()

    # Average days-of-supply across all items that have it
    dos_col = df["days_of_supply"].replace(999, np.nan)
    avg_dos = round(_safe_float(dos_col.mean(skipna=True)), 1)

    return {
        "total_sku":           total_sku,
        "total_inventory_value_inr": round(total_value, 0),
        "stockout_count":      stockout_count,
        "critical_count":      critical_count,
        "low_count":           low_count,
        "ok_count":            ok_count,
        "slow_mover_count":    slow_movers,
        "value_at_risk_inr":   round(value_at_risk, 0),
        "dealers_affected":    dealers_affected,
        "avg_days_of_supply":  avg_dos,
        "abc_value_inr":       {k: round(float(v), 0) for k, v in abc_value.items()},
        "abc_sku_count":       {k: int(v) for k, v in abc_count.items()},
        "updated_at":          datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stock")
async def inventory_stock(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code:  str | None = Query(None),
    abc_class:    str | None = Query(None),
    status:       str | None = Query(None, description="OK|LOW|CRITICAL|STOCKOUT"),
    part_code:    str | None = Query(None),
    limit:        int        = Query(500, ge=1, le=2000),
):
    df = _load_stock(dealer_code)
    if df.empty:
        return []

    if abc_class:
        df = df[df["abc_class"].astype(str).str.upper() == abc_class.upper()]
    if status:
        df = df[df["stock_status"].astype(str).str.upper() == status.upper()]
    if part_code:
        df = df[df["part_code"].astype(str) == part_code]

    df = df.sort_values(
        ["stock_status", "abc_class", "days_of_supply"],
        ascending=[True, True, True],
        key=lambda col: col.map({"STOCKOUT": 0, "CRITICAL": 1, "LOW": 2, "OK": 3}) if col.name == "stock_status" else col,
    )

    df = df.fillna({"service_types": "", "description": "", "supplier": "",
                    "dealer_city": "", "last_restocked": "", "last_sold": ""})

    records = df.head(limit).to_dict("records")
    for r in records:
        r["inventory_value_inr"] = round(_safe_float(r.get("inventory_value_inr")), 2)
        r["avg_daily_demand"]    = round(_safe_float(r.get("avg_daily_demand")), 4)
        r["stockout_prob"]       = round(_safe_float(r.get("stockout_prob")), 3)
        dos = r.get("days_of_supply", 999)
        r["days_of_supply"]      = None if dos >= 999 else int(dos)
        # Ensure no NaN leaks through for any remaining string/numeric fields
        for k, v in list(r.items()):
            if isinstance(v, float) and (v != v):  # NaN check
                r[k] = None
    return records


@router.get("/alerts")
async def inventory_alerts(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code:  str | None = Query(None),
    min_severity: str        = Query("LOW", description="LOW|CRITICAL|STOCKOUT"),
    limit:        int        = Query(100, ge=1, le=500),
):
    df = _load_stock(dealer_code)
    if df.empty:
        return []

    sev_order = {"STOCKOUT": 0, "CRITICAL": 1, "LOW": 2, "OK": 3}
    min_sev_idx = sev_order.get(min_severity.upper(), 2)
    df = df[df["stock_status"].map(sev_order) <= min_sev_idx].copy()
    df["_sev_rank"] = df["stock_status"].map(sev_order)
    df = df.sort_values(["_sev_rank", "abc_class", "days_of_supply"])

    alerts = []
    for _, row in df.head(limit).iterrows():
        status   = str(row.get("stock_status", "LOW"))
        dos      = row.get("days_of_supply", 999)
        dos_int  = None if dos >= 999 else int(dos)
        eoq      = int(row.get("eoq", 1))
        lead     = int(row.get("lead_time_days", 7))
        unit_cost= _safe_float(row.get("unit_cost_inr", 0))
        alerts.append({
            "dealer_code":        str(row.get("dealer_code", "")),
            "dealer_city":        str(row.get("dealer_city", "")),
            "part_code":          str(row.get("part_code", "")),
            "description":        str(row.get("description", "")),
            "abc_class":          str(row.get("abc_class", "B")),
            "severity":           status,
            "current_stock":      int(row.get("current_stock", 0)),
            "reorder_point":      int(row.get("reorder_point", 0)),
            "safety_stock":       int(row.get("safety_stock", 0)),
            "days_of_supply":     dos_int,
            "stockout_prob":      round(_safe_float(row.get("stockout_prob", 0)), 3),
            "recommended_qty":    eoq,
            "estimated_cost_inr": round(eoq * unit_cost, 0),
            "supplier":           str(row.get("supplier", "")),
            "lead_time_days":     lead,
            "expected_arrival":   (datetime.now(timezone.utc) + timedelta(days=lead)).date().isoformat(),
        })
    return alerts


@router.get("/reorder-plan")
async def reorder_plan(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code:  str | None = Query(None),
):
    df = _load_stock(dealer_code)
    if df.empty:
        return {"orders": [], "total_cost_inr": 0}

    needs_reorder = df[df["current_stock"] <= df["reorder_point"]].copy()

    orders = []
    total_cost = 0.0
    for dc in needs_reorder["dealer_code"].unique():
        dc_rows = needs_reorder[needs_reorder["dealer_code"] == dc]
        supplier_groups: dict[str, list] = {}

        for _, row in dc_rows.iterrows():
            supplier = str(row.get("supplier", "OEM Direct"))
            eoq      = int(row.get("eoq", 1))
            unit_cost= _safe_float(row.get("unit_cost_inr", 0))
            line_cost= round(eoq * unit_cost, 2)
            total_cost += line_cost
            supplier_groups.setdefault(supplier, []).append({
                "part_code":    str(row.get("part_code", "")),
                "description":  str(row.get("description", "")),
                "abc_class":    str(row.get("abc_class", "B")),
                "current_stock":int(row.get("current_stock", 0)),
                "reorder_point":int(row.get("reorder_point", 0)),
                "order_qty":    eoq,
                "unit_cost_inr":unit_cost,
                "line_cost_inr":line_cost,
                "lead_time_days":int(row.get("lead_time_days", 7)),
                "priority":     "URGENT" if str(row.get("stock_status","")) in ("STOCKOUT","CRITICAL") else "NORMAL",
            })

        for supplier, lines in supplier_groups.items():
            supplier_total = sum(l["line_cost_inr"] for l in lines)
            lead = max(l["lead_time_days"] for l in lines)
            orders.append({
                "dealer_code":       dc,
                "dealer_city":       str(dc_rows.iloc[0].get("dealer_city", "")),
                "supplier":          supplier,
                "lines":             lines,
                "line_count":        len(lines),
                "total_cost_inr":    round(supplier_total, 2),
                "lead_time_days":    lead,
                "expected_delivery": (datetime.now(timezone.utc) + timedelta(days=lead)).date().isoformat(),
                "has_urgent":        any(l["priority"] == "URGENT" for l in lines),
            })

    orders.sort(key=lambda o: (0 if o["has_urgent"] else 1, o["dealer_code"]))
    return {"orders": orders, "total_cost_inr": round(total_cost, 2), "order_count": len(orders)}


@router.get("/analytics")
async def inventory_analytics(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code:  str | None = Query(None),
):
    df    = _load_stock(dealer_code)
    txn   = _load_transactions(dealer_code, days=90)

    if df.empty:
        return {}

    # ABC analysis
    abc = df.groupby("abc_class").agg(
        sku_count=("part_code", "count"),
        total_value=("inventory_value_inr", "sum"),
        avg_dos=("days_of_supply", lambda x: x[x < 999].mean()),
    ).reset_index().to_dict("records")
    for row in abc:
        row["total_value"]   = round(_safe_float(row["total_value"]), 0)
        row["avg_dos"]       = round(_safe_float(row["avg_dos"]), 1)

    # Turnover rate per part (annualised: annual demand / avg stock)
    turnover_rows = []
    for _, row in df.iterrows():
        avg_daily  = _safe_float(row.get("avg_daily_demand", 0))
        annual_dem = avg_daily * 365
        avg_stock  = _safe_float(row.get("current_stock", 1))
        if avg_stock > 0 and annual_dem > 0:
            turnover = round(annual_dem / avg_stock, 2)
        else:
            turnover = 0.0
        turnover_rows.append({
            "part_code":    str(row.get("part_code", "")),
            "description":  str(row.get("description", "")),
            "abc_class":    str(row.get("abc_class", "B")),
            "turnover_rate":turnover,
            "annual_demand":round(annual_dem, 1),
            "current_stock":int(row.get("current_stock", 0)),
        })
    turnover_rows.sort(key=lambda r: r["turnover_rate"], reverse=True)

    # Slow movers: low turnover + high stock
    slow_movers = [r for r in turnover_rows if r["turnover_rate"] < 2.0 and r["current_stock"] > 0]

    # Value distribution
    value_dist = df.groupby("abc_class").agg(
        value=("inventory_value_inr", "sum")
    ).reset_index().to_dict("records")
    for v in value_dist:
        v["value"] = round(_safe_float(v["value"]), 0)

    # Status distribution per dealer
    dealer_summary = df.groupby(["dealer_code", "dealer_city", "stock_status"]).size().reset_index(name="count")
    dealer_pivot = dealer_summary.pivot_table(
        index=["dealer_code", "dealer_city"],
        columns="stock_status",
        values="count",
        fill_value=0,
    ).reset_index()
    dealer_list = dealer_pivot.to_dict("records")

    # Transaction summary (90d)
    txn_summary: dict = {}
    if not txn.empty and "transaction_type" in txn.columns and "quantity" in txn.columns:
        issues = txn[txn["transaction_type"] == "ISSUE"]
        receipts = txn[txn["transaction_type"] == "RECEIPT"]
        txn_summary = {
            "total_issues_90d":   int(issues["quantity"].sum()),
            "total_receipts_90d": int(receipts["quantity"].sum()),
            "issue_events":       int(len(issues)),
            "receipt_events":     int(len(receipts)),
        }

    # Monthly demand trend from transaction history
    monthly_demand: list[dict] = []
    if not txn.empty and "date" in txn.columns and "quantity" in txn.columns:
        issues_df = txn[txn["transaction_type"] == "ISSUE"].copy() if "transaction_type" in txn.columns else txn.copy()
        if not issues_df.empty:
            issues_df["month"] = issues_df["date"].dt.to_period("M").astype(str)
            monthly = issues_df.groupby("month")["quantity"].sum().reset_index()
            monthly_demand = [{"month": r["month"], "quantity": int(r["quantity"])} for _, r in monthly.iterrows()]

    # Fill rate: SKUs that are not stocked out
    fill_rate = 0.0
    if not df.empty:
        non_stockout = int((df["stock_status"] != "STOCKOUT").sum())
        fill_rate = round(non_stockout / max(len(df), 1) * 100, 1)

    # Supplier performance
    supplier_perf: list[dict] = []
    if "supplier" in df.columns:
        for sup, g in df.groupby("supplier"):
            dos_vals = g["days_of_supply"].replace(999, np.nan)
            stockout = int((g["stock_status"] == "STOCKOUT").sum())
            critical = int((g["stock_status"] == "CRITICAL").sum())
            supplier_perf.append({
                "supplier":       str(sup),
                "sku_count":      len(g),
                "avg_dos":        round(_safe_float(dos_vals.mean(skipna=True)), 1),
                "stockout_count": stockout,
                "critical_count": critical,
                "total_value_inr": round(_safe_float(g["inventory_value_inr"].sum()), 0),
                "health_pct":     round((len(g) - stockout) / max(len(g), 1) * 100, 1),
            })
        supplier_perf.sort(key=lambda x: x["health_pct"])

    return {
        "abc_analysis":        abc,
        "turnover_rates":      turnover_rows[:20],
        "slow_movers":         slow_movers[:15],
        "value_dist":          value_dist,
        "dealer_summary":      dealer_list,
        "transaction_summary": txn_summary,
        "monthly_demand":      monthly_demand,
        "fill_rate_pct":       fill_rate,
        "supplier_performance": supplier_perf,
    }


@router.get("/dealers")
async def dealer_comparison(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    df = _load_stock()
    if df.empty:
        return []

    result = []
    for dc, g in df.groupby("dealer_code"):
        city        = g["dealer_city"].iloc[0] if "dealer_city" in g.columns else ""
        total_val   = _safe_float(g["inventory_value_inr"].sum())
        stockout    = int((g["stock_status"] == "STOCKOUT").sum())
        critical    = int((g["stock_status"] == "CRITICAL").sum())
        low         = int((g["stock_status"] == "LOW").sum())
        ok          = int((g["stock_status"] == "OK").sum())
        dos_vals    = g["days_of_supply"].replace(999, np.nan)
        avg_dos     = round(_safe_float(dos_vals.mean(skipna=True)), 1)
        health_pct  = round(ok / max(len(g), 1) * 100, 1)
        result.append({
            "dealer_code":      str(dc),
            "dealer_city":      str(city),
            "total_skus":       len(g),
            "total_value_inr":  round(total_val, 0),
            "stockout":         stockout,
            "critical":         critical,
            "low":              low,
            "ok":               ok,
            "avg_days_of_supply": avg_dos,
            "stock_health_pct": health_pct,
            "alerts":           stockout + critical + low,
        })
    result.sort(key=lambda r: r["alerts"], reverse=True)
    return result


@router.get("/parts/{part_code}")
async def part_detail(
    part_code:    str,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    df = _load_stock()
    if df.empty:
        return {}

    rows = df[df["part_code"].astype(str) == part_code]
    if rows.empty:
        return {"error": f"Part {part_code} not found"}

    txn = _load_transactions(days=365)
    if not txn.empty and "part_code" in txn.columns:
        txn = txn[txn["part_code"].astype(str) == part_code]

    # 30-day buckets for trend
    trend: list[dict] = []
    if not txn.empty and "date" in txn.columns and "quantity" in txn.columns:
        issues = txn[txn.get("transaction_type", pd.Series()) == "ISSUE"].copy() if "transaction_type" in txn.columns else txn.copy()
        if not issues.empty:
            issues["month"] = issues["date"].dt.to_period("M")
            monthly = issues.groupby("month")["quantity"].sum().reset_index()
            for _, mr in monthly.iterrows():
                trend.append({"month": str(mr["month"]), "qty": int(mr["quantity"])})

    dealers_data = []
    for _, row in rows.iterrows():
        dos = row.get("days_of_supply", 999)
        dealers_data.append({
            "dealer_code":    str(row.get("dealer_code", "")),
            "dealer_city":    str(row.get("dealer_city", "")),
            "current_stock":  int(row.get("current_stock", 0)),
            "reorder_point":  int(row.get("reorder_point", 0)),
            "safety_stock":   int(row.get("safety_stock", 0)),
            "eoq":            int(row.get("eoq", 1)),
            "days_of_supply": None if dos >= 999 else int(dos),
            "stock_status":   str(row.get("stock_status", "OK")),
            "stockout_prob":  round(_safe_float(row.get("stockout_prob", 0)), 3),
            "inventory_value_inr": round(_safe_float(row.get("inventory_value_inr", 0)), 2),
        })

    meta = rows.iloc[0]
    return {
        "part_code":       part_code,
        "description":     str(meta.get("description", "")),
        "abc_class":       str(meta.get("abc_class", "B")),
        "supplier":        str(meta.get("supplier", "")),
        "unit_cost_inr":   _safe_float(meta.get("unit_cost_inr", 0)),
        "lead_time_days":  int(meta.get("lead_time_days", 7)),
        "avg_daily_demand":round(_safe_float(meta.get("avg_daily_demand", 0)), 4),
        "service_types":   str(meta.get("service_types", "")),
        "dealers":         dealers_data,
        "monthly_trend":   trend,
        "total_stock_fleet": int(rows["current_stock"].sum()),
        "fleet_value_inr":   round(_safe_float(rows["inventory_value_inr"].sum()), 0),
    }


@router.get("/transactions")
async def recent_transactions(
    current_user: Annotated[dict, Depends(get_current_user)],
    dealer_code:  str | None = Query(None),
    part_code:    str | None = Query(None),
    days:         int        = Query(30, ge=1, le=365),
    limit:        int        = Query(200, ge=1, le=1000),
):
    df = _load_transactions(dealer_code, days=days)
    if df.empty:
        return []

    if part_code:
        df = df[df["part_code"].astype(str) == part_code]

    df = df.sort_values("date", ascending=False)
    records = df.head(limit).to_dict("records")
    for r in records:
        if "date" in r and hasattr(r["date"], "isoformat"):
            r["date"] = r["date"].isoformat()[:10]
    return records
