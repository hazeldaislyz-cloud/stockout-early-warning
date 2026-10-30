"""Compare the simulated supply pattern with the targets taken from real records.

The real records are not in this repository. The targets below are summary figures
computed from them privately (see README, "How the simulation was calibrated").

    python calibration_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stockout.simulate import simulate  # noqa: E402

# From about 1,500 real insulin issue lines, 2020 to 2025.
TARGETS = {
    "share of months a facility receives stock (median)": "12-14% from the national warehouse alone; "
                                                          "higher in total, as facilities also buy elsewhere",
    "months between orders, repeat orderers (median)": "3",
    "issue quantities in whole packs of 10": "98%",
    "shelf life left on arrival, insulin (median, months)": "13 (p10 9, p90 22)",
    "largest facility / median facility, insulin": "30-98x",
    "month-to-month swing in national insulin deliveries (CV)": "0.43-0.46",
}


def check(seed: int = 7) -> pd.DataFrame:
    fac, rep, dl = simulate(seed, with_deliveries=True)
    ins = "Insulin"
    r = rep[rep["reported"]]
    share = r.groupby(["facility_id", "commodity"])["received"].apply(lambda s: (s > 0).mean())
    orders = dl[dl["kind"] == "order"].sort_values("month_index")
    gaps = orders.groupby(["facility_id", "commodity"])["month_index"].diff().dropna()
    ins_dl = orders[orders["commodity"].str.startswith(ins)]
    w = np.repeat(ins_dl["shelf_left_months"].to_numpy(), ins_dl["units"].astype(int).to_numpy())
    size = r[r["commodity"].str.startswith(ins)].groupby("facility_id")["dispensed"].sum()
    nat = r[r["commodity"].str.startswith(ins)].groupby("period")["received"].sum()
    sim = {
        "share of months a facility receives stock (median)": f"{share.median():.0%}",
        "months between orders, repeat orderers (median)": f"{gaps.median():.0f}",
        "issue quantities in whole packs of 10": f"{(dl['units'] % 10 == 0).mean():.0%}",
        "shelf life left on arrival, insulin (median, months)":
            f"{np.median(w):.0f} (p10 {np.percentile(w, 10):.0f}, p90 {np.percentile(w, 90):.0f})",
        "largest facility / median facility, insulin": f"{size.max() / size.median():.0f}x",
        "month-to-month swing in national insulin deliveries (CV)": f"{nat.std() / nat.mean():.2f}",
    }
    return pd.DataFrame({"real (target)": TARGETS, "simulated": sim})


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 80)
    print(check().to_string())
