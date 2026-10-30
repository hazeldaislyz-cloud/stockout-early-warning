"""Score each facility x commodity for the chance of a stockout next month, and
compare against the rule most supply teams use today: alert when stock covers less
than one month of average consumption.

The comparison that matters is practical: a county pharmacist can only follow up a
limited number of facilities each month. So we compare how many of the top-N alerts
turn out to be real stockouts (precision at N), as well as PR-AUC across all rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from .features import RISK_FEATURES, RISK_FEATURES_WIDE


def labelled(d: pd.DataFrame) -> pd.Series:
    return d["next_stockout"].notna() & d["reported"] & d["amc3"].notna()


def make_model(wide: bool = False) -> HistGradientBoostingClassifier:
    if wide:
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              min_samples_leaf=30, random_state=0)
    # Shallow trees and large leaves: fewer ways to memorise the training months.
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=50, random_state=0)


def fit(train: pd.DataFrame, wide: bool = False) -> HistGradientBoostingClassifier:
    feats = RISK_FEATURES_WIDE if wide else RISK_FEATURES
    return make_model(wide).fit(train[feats], train["next_stockout"].astype(int))


def baseline_score(d: pd.DataFrame) -> np.ndarray:
    """Higher = more at risk. Months of stock below 1 is the usual alert line."""
    return -d["months_of_stock"].fillna(0).to_numpy()


def precision_at(test: pd.DataFrame, score_col: str, k: int) -> float:
    tp = n = 0
    for _, g in test.groupby("month_index"):
        top = g.nlargest(k, score_col)
        tp += int(top["next_stockout"].sum())
        n += len(top)
    return tp / max(1, n)


def evaluate(d: pd.DataFrame, split_month: int = 24, budgets: tuple[int, ...] = (10, 20, 40)) -> dict:
    """Train on months before split_month, test on the months after (out of time)."""
    lab = labelled(d)
    train = d[lab & (d["month_index"] < split_month - 1)]
    test = d[lab & (d["month_index"] >= split_month - 1)].copy()

    model = fit(train)
    wide = fit(train, wide=True)
    test["risk_model"] = model.predict_proba(test[RISK_FEATURES])[:, 1]
    test["risk_wide"] = wide.predict_proba(test[RISK_FEATURES_WIDE])[:, 1]
    test["risk_rule"] = baseline_score(test)

    y = test["next_stockout"].astype(int)
    rows = {}
    for name, col in [("Rule: lowest months of stock first", "risk_rule"),
                      ("Model (7 features, shallow trees)", "risk_model"),
                      ("Model tried first (17 features)", "risk_wide")]:
        r = {"PR-AUC": average_precision_score(y, test[col])}
        for k in budgets:
            r[f"precision, top {k} a month"] = precision_at(test, col, k)
        rows[name] = r
    return {
        "model": model,
        "test": test,
        "summary": pd.DataFrame(rows).T,
        "stockouts_in_test": int(y.sum()),
        "rows_in_test": len(test),
        "base_rate": float(y.mean()),
    }
