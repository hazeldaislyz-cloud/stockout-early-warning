"""Suggest stock transfers between facilities in the same county, before stock expires
at one facility while another runs out.

Donors: stock that will expire within 3 months beyond what the facility is forecast to
use in that time, plus anything above 4 months of forecast use.
Receivers: facilities with high stockout risk, short of 2 months of forecast use.
Matching: highest-risk receiver first, nearest donor first, up to a distance limit,
and only for transfers of at least MIN_UNITS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_KM = 120
MIN_UNITS = 10  # a trip is not worth making for less


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def plan_transfers(latest: pd.DataFrame, risk_threshold: float = 0.3) -> pd.DataFrame:
    """latest: one row per facility x commodity for the current month, with
    closing_balance, expiring_within_3m, forecast_next, risk and coordinates."""
    s = latest.dropna(subset=["closing_balance", "forecast_next"]).copy()
    use3 = 3 * s["forecast_next"]
    s["at_risk_of_expiry"] = (s["expiring_within_3m"].fillna(0) - use3).clip(lower=0)
    s["excess"] = (s["closing_balance"] - 4 * s["forecast_next"]).clip(lower=0)
    s["can_give"] = np.maximum(s["at_risk_of_expiry"], s["excess"]).round()
    s["need"] = (2 * s["forecast_next"] - s["closing_balance"]).clip(lower=0).round()

    transfers = []
    for (county, com), g in s.groupby(["county", "commodity"]):
        donors = g[g["can_give"] > 0].copy()
        receivers = g[(g["need"] > 0) & (g["risk"] >= risk_threshold)].sort_values("risk", ascending=False)
        for r in receivers.itertuples():
            need = r.need
            if donors.empty:
                break
            donors["km"] = [haversine_km(r.lat, r.lon, a, b) for a, b in zip(donors["lat"], donors["lon"])]
            for dn in donors.sort_values("km").itertuples():
                if need <= 0 or dn.km > MAX_KM or dn.Index == r.Index:
                    continue
                qty = min(need, donors.at[dn.Index, "can_give"])
                if qty < MIN_UNITS:
                    continue
                donors.at[dn.Index, "can_give"] -= qty
                need -= qty
                expiring_part = min(qty, dn.at_risk_of_expiry)
                transfers.append({
                    "county": county, "commodity": com,
                    "from_facility": dn.facility, "to_facility": r.facility,
                    "units": int(qty), "km": round(dn.km, 1),
                    "units_that_would_otherwise_expire": int(round(expiring_part)),
                    "receiver_stockout_risk": round(float(r.risk), 2),
                    "receiver_months_of_stock_before": round(float(r.closing_balance / max(r.forecast_next, 1e-9)), 1),
                    "receiver_months_of_stock_after": round(float((r.closing_balance + qty) / max(r.forecast_next, 1e-9)), 1),
                })
    cols = ["county", "commodity", "from_facility", "to_facility", "units", "km",
            "units_that_would_otherwise_expire", "receiver_stockout_risk",
            "receiver_months_of_stock_before", "receiver_months_of_stock_after"]
    return pd.DataFrame(transfers, columns=cols)
