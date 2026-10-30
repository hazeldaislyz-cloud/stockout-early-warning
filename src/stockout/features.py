"""Turn monthly reports into model features. Everything here uses only information
a county pharmacist would have at the end of month t."""
from __future__ import annotations

import numpy as np
import pandas as pd


def panel(fac: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(fac[["facility_id", "facility", "county", "level", "lat", "lon"]], on="facility_id")
    d = d.sort_values(["facility_id", "commodity", "month_index"]).reset_index(drop=True)
    g = d.groupby(["facility_id", "commodity"], sort=False)

    for k in (1, 2, 3, 12):
        d[f"disp_lag{k - 1}"] = g["dispensed"].shift(k - 1)  # lag0 = this month
    d["amc3"] = g["dispensed"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["amc6"] = g["dispensed"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    d["months_of_stock"] = d["closing_balance"] / d["amc3"].replace(0, np.nan)
    d["usable_stock"] = d["closing_balance"] - d["expiring_within_3m"].fillna(0) * 0.5
    d["received_ratio"] = d["received"] / d["amc3"].replace(0, np.nan)
    d["stocked_out_now"] = (d["days_out_of_stock"] > 0).astype(float).where(d["reported"])

    # A county-wide signal: how much did facilities in the county receive this month,
    # relative to their usual consumption? Low values hint at a supply problem upstream.
    d["county_received_ratio"] = d.groupby(["county", "commodity", "month_index"])["received_ratio"].transform("median")

    d["stockouts_last6"] = g["days_out_of_stock"].transform(lambda s: (s > 0).astype(float).rolling(6, min_periods=1).sum())
    d["county_received_ratio_prev"] = d.groupby(["facility_id", "commodity"])["county_received_ratio"].shift(1)

    cal = pd.PeriodIndex(d["period"], freq="M").month
    d["month_sin"] = np.sin(2 * np.pi * cal / 12)
    d["month_cos"] = np.cos(2 * np.pi * cal / 12)
    # Next month's calendar position, for forecasting the next month.
    d["next_month_sin"] = np.sin(2 * np.pi * (cal % 12 + 1) / 12)
    d["next_month_cos"] = np.cos(2 * np.pi * (cal % 12 + 1) / 12)
    # Same month last year, aligned to the month being forecast (t+1 - 12 = t-11).
    d["disp_same_month_last_year"] = g["dispensed"].shift(11)

    for c in ("commodity", "level", "county"):
        d[c + "_code"] = d[c].astype("category").cat.codes

    # Targets for month t+1.
    d["next_dispensed"] = g["dispensed"].shift(-1)
    d["next_stockout"] = (g["days_out_of_stock"].shift(-1) > 0).astype(float)
    d["next_reported"] = g["reported"].shift(-1).fillna(False).astype(bool)
    d["next_true_demand"] = g["true_demand"].shift(-1)
    d.loc[~d["next_reported"], ["next_dispensed", "next_stockout"]] = np.nan
    return d


FORECAST_FEATURES = [
    "disp_lag0", "disp_lag1", "disp_lag2", "disp_same_month_last_year", "amc3", "amc6",
    "stocked_out_now", "next_month_sin", "next_month_cos", "commodity_code", "level_code", "county_code",
]

# The model we use: a short list of features a supply officer would recognise.
RISK_FEATURES = [
    "months_of_stock", "months_of_stock_forecast", "county_received_ratio", "received_ratio",
    "stockouts_last6", "commodity_code", "level_code",
]

# A larger feature set we tried first. It overfit and did worse than the simple rule;
# kept here so the comparison in the results can be reproduced.
RISK_FEATURES_WIDE = [
    "closing_balance", "amc3", "months_of_stock", "usable_stock", "received_ratio",
    "county_received_ratio", "stocked_out_now", "expiring_within_3m", "next_month_sin",
    "next_month_cos", "commodity_code", "level_code", "county_code", "forecast_next",
    "months_of_stock_forecast", "stockouts_last6", "county_received_ratio_prev",
]
