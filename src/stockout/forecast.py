"""Forecast next month's consumption and compare against the standard LMIS method.

Baseline: average monthly consumption over the last 3 months (AMC), which is how
facility orders are usually calculated.
Model: gradient-boosted trees using recent consumption, the same month last year,
seasonality and facility type.

Months where the facility ran out are excluded from the target: what was dispensed
in those months understates what patients needed (censored demand).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FORECAST_FEATURES


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=30, random_state=0)


def trainable(d: pd.DataFrame) -> pd.Series:
    """Rows usable as training/evaluation targets: next month reported, no stockout."""
    return d["next_dispensed"].notna() & (d["next_stockout"] == 0) & d["reported"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    m = _model().fit(train[FORECAST_FEATURES], np.log1p(train["next_dispensed"]))
    return np.expm1(m.predict(test[FORECAST_FEATURES])).clip(min=0)


def backtest(d: pd.DataFrame, first_origin: int = 23) -> pd.DataFrame:
    """Rolling-origin backtest: for each month t, train on everything known up to t,
    forecast t+1, and record both the model and the AMC baseline."""
    ok = trainable(d)
    out = []
    for t in range(first_origin, int(d["month_index"].max())):
        train = d[ok & (d["month_index"] < t)]  # target month <= t
        test = d[ok & (d["month_index"] == t) & d["amc3"].notna()]
        if test.empty:
            continue
        pred = fit_predict(train, test)
        out.append(test.assign(forecast_model=pred, forecast_amc=test["amc3"].values)[
            ["facility_id", "commodity", "county", "level", "month_index", "period",
             "next_dispensed", "next_true_demand", "forecast_model", "forecast_amc"]])
    return pd.concat(out, ignore_index=True)


def wape(actual: pd.Series, pred: pd.Series) -> float:
    """Weighted absolute percentage error: total absolute error / total actual."""
    return float(np.abs(actual - pred).sum() / actual.sum())


def score(bt: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "WAPE AMC baseline": wape(g["next_dispensed"], g["forecast_amc"]),
            "WAPE model": wape(g["next_dispensed"], g["forecast_model"]),
            "bias model": float((g["forecast_model"].sum() - g["next_dispensed"].sum()) / g["next_dispensed"].sum()),
            "n": len(g),
        })
    if by is None:
        return one(bt).to_frame("all").T
    return bt.groupby(by).apply(one, include_groups=False)


def add_forecast_column(d: pd.DataFrame) -> pd.DataFrame:
    """Out-of-time forecast for every row: each month is predicted by a model trained
    only on months before it (so the risk model never sees leaked information)."""
    ok = trainable(d)
    d = d.copy()
    d["forecast_next"] = d["amc3"]
    for t in range(12, int(d["month_index"].max()) + 1):
        train = d[ok & (d["month_index"] < t)]
        rows = d["month_index"] == t
        if len(train) > 500:
            d.loc[rows, "forecast_next"] = fit_predict(train, d[rows])
    d["months_of_stock_forecast"] = d["closing_balance"] / d["forecast_next"].replace(0, np.nan)
    return d
