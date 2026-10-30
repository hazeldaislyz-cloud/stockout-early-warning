"""Re-run the evaluation on several simulated datasets, to check the results are not
an accident of one random seed. Takes a few minutes.

    python robustness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from stockout.features import panel  # noqa: E402
from stockout.forecast import add_forecast_column, backtest, score  # noqa: E402
from stockout.risk import evaluate  # noqa: E402
from stockout.simulate import simulate  # noqa: E402

rows = []
for seed in (1, 2, 3, 7, 11, 21):
    fac, rep = simulate(seed)
    p = panel(fac, rep)
    f = score(backtest(p)).iloc[0]
    s = evaluate(add_forecast_column(p))["summary"]
    rows.append({
        "seed": seed,
        "forecast WAPE, AMC": round(f["WAPE AMC baseline"], 3),
        "forecast WAPE, model": round(f["WAPE model"], 3),
        "PR-AUC, rule": round(s.iloc[0]["PR-AUC"], 3),
        "PR-AUC, model": round(s.iloc[1]["PR-AUC"], 3),
        "top-20 precision, rule": round(s.iloc[0]["precision, top 20 a month"], 3),
        "top-20 precision, model": round(s.iloc[1]["precision, top 20 a month"], 3),
    })
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / "outputs" / "robustness.csv", index=False)
print(out.to_string(index=False))
