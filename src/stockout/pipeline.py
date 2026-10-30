"""Run everything end to end and write results to outputs/.

    python -m stockout.pipeline            # from the src/ folder
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .brief import county_facts, llm_brief
from .features import RISK_FEATURES, panel
from .forecast import add_forecast_column, backtest, score
from .redistribute import plan_transfers
from .risk import evaluate, fit, labelled
from .simulate import simulate

OUT = Path(__file__).resolve().parents[2] / "outputs"
INK, MUTED, A, B = "#1f2a44", "#8a94a6", "#2f6db5", "#c77d2e"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def run(seed: int = 7) -> dict:
    OUT.mkdir(exist_ok=True)
    fac, reports = simulate(seed)
    reports.to_csv(OUT / "simulated_reports.csv", index=False)
    p = panel(fac, reports)

    # 1. Forecasting
    bt = backtest(p)
    f_all, f_com = score(bt), score(bt, "commodity")
    bt.to_csv(OUT / "forecast_backtest.csv", index=False)

    # 2. Stockout risk
    d = add_forecast_column(p)
    ev = evaluate(d)
    ev["summary"].round(3).to_csv(OUT / "risk_evaluation.csv")

    # 3. Current month: score every facility with a model trained on all history, plan transfers
    last = int(d["month_index"].max())
    model = fit(d[labelled(d)])
    latest = d[(d["month_index"] == last) & d["reported"]].copy()
    latest["risk"] = model.predict_proba(latest[RISK_FEATURES])[:, 1]
    transfers = plan_transfers(latest)
    transfers.to_csv(OUT / "suggested_transfers.csv", index=False)
    alerts = latest.sort_values("risk", ascending=False)[
        ["county", "facility", "commodity", "risk", "closing_balance", "forecast_next", "months_of_stock"]]
    alerts.round(3).to_csv(OUT / "current_alerts.csv", index=False)

    # 4. County briefs
    period = latest["period"].iloc[0]
    briefs = []
    for county in sorted(latest["county"].unique()):
        facts = county_facts(county, latest, transfers, period)
        text, source = llm_brief(facts)
        briefs.append(f"<!-- source: {source} -->\n{text}\n")
    (OUT / "county_briefs.md").write_text("\n---\n\n".join(briefs))

    _charts(f_com, ev, bt)
    results = {
        "forecast": {k: round(float(v), 3) for k, v in f_all.iloc[0].items()},
        "risk": ev["summary"].round(3).to_dict(orient="index"),
        "risk_test_rows": ev["rows_in_test"], "risk_test_stockouts": ev["stockouts_in_test"],
        "transfers": int(len(transfers)), "units_moved": int(transfers["units"].sum()),
        "units_saved_from_expiry": int(transfers["units_that_would_otherwise_expire"].sum()),
        "period": period,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    return results


def _charts(f_com: pd.DataFrame, ev: dict, bt: pd.DataFrame) -> None:
    # Forecast error by commodity
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [c.split(" (")[0].replace("Artemether-lumefantrine", "AL") for c in f_com.index]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], 100 * f_com["WAPE AMC baseline"], 0.4, color=MUTED, label="3-month average (AMC)")
    ax.bar([i + 0.2 for i in x], 100 * f_com["WAPE model"], 0.4, color=A, label="Model")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Forecast error, % (WAPE)", color=INK)
    ax.set_title("Next-month forecast error by commodity (lower is better)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout(); fig.savefig(OUT / "forecast_error.png", dpi=150); plt.close(fig)

    # Precision at alert budgets
    s = ev["summary"]
    cols = [c for c in s.columns if c.startswith("precision")]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [c.replace("precision, ", "") for c in cols]
    x = range(len(cols))
    ax.bar([i - 0.2 for i in x], 100 * s.iloc[0][cols], 0.4, color=MUTED, label="Rule: lowest months of stock")
    ax.bar([i + 0.2 for i in x], 100 * s.iloc[1][cols], 0.4, color=A, label="Model")
    ax.axhline(100 * ev["test"]["next_stockout"].mean(), color=B, linestyle="--", linewidth=1)
    ax.text(len(cols) - 0.5, 100 * ev["test"]["next_stockout"].mean() + 1, "random pick", color=B, fontsize=8, ha="right")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Alerts that were real stockouts, %", color=INK)
    ax.set_title("If a county can follow up N facility-commodities a month", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout(); fig.savefig(OUT / "alert_precision.png", dpi=150); plt.close(fig)

    # One example series
    ins = bt[bt["commodity"].str.startswith("Artemether")]
    fid = ins.groupby("facility_id")["next_dispensed"].sum().idxmax()
    e = ins[ins["facility_id"] == fid].sort_values("month_index")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(e["period"], e["next_dispensed"], color=INK, marker="o", ms=3, label="Actual (month ahead)")
    ax.plot(e["period"], e["forecast_amc"], color=MUTED, label="3-month average")
    ax.plot(e["period"], e["forecast_model"], color=A, label="Model")
    ax.set_title(f"Antimalarial (AL) at facility {fid}: forecast vs actual", color=INK, fontsize=11, loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Month the forecast was made (it predicts the following month)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout(); fig.savefig(OUT / "example_forecast.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
