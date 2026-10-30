"""Build the interactive demo page (docs/index.html) from the model's outputs.

Run the pipeline first, then:

    python build_demo.py

The page is one self-contained HTML file: the data is embedded, so it opens offline,
on GitHub Pages, or anywhere else, with no Python needed by the person viewing it.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stockout.simulate import build_facilities  # noqa: E402

OUT = ROOT / "outputs"
SHORT = {
    "Insulin": "Insulin 30/70", "Metformin": "Metformin", "Artemether": "AL (antimalarial)",
    "Amoxicillin": "Amoxicillin DT", "ORS": "ORS/zinc",
}


def short(name: str) -> str:
    for k, v in SHORT.items():
        if name.startswith(k):
            return v
    return name


def briefs() -> dict:
    text = (OUT / "county_briefs.md").read_text()
    out = {}
    for block in text.split("\n---\n"):
        block = re.sub(r"<!--.*?-->", "", block).strip()
        m = re.search(r"brief: (\w+) County", block)
        if m:
            out[m.group(1)] = block
    return out


def build() -> Path:
    fac = build_facilities(7)  # the pipeline's default seed
    alerts = pd.read_csv(OUT / "current_alerts.csv")
    transfers = pd.read_csv(OUT / "suggested_transfers.csv")
    results = json.loads((OUT / "results.json").read_text())
    robust = pd.read_csv(OUT / "robustness.csv")
    outline = json.loads((ROOT / "docs" / "kenya_outline.json").read_text())

    data = {
        "period": results["period"],
        "facilities": [
            {"id": r.facility_id, "name": r.facility, "county": r.county, "level": r.level,
             "lat": round(r.lat, 4), "lon": round(r.lon, 4)}
            for r in fac.itertuples()
        ],
        "alerts": [
            {"county": r.county, "facility": r.facility, "med": short(r.commodity),
             "risk": round(float(r.risk), 3), "stock": None if pd.isna(r.closing_balance) else int(r.closing_balance),
             "forecast": None if pd.isna(r.forecast_next) else round(float(r.forecast_next), 1),
             "mos": None if pd.isna(r.months_of_stock) else round(float(r.months_of_stock), 1)}
            for r in alerts.itertuples()
        ],
        "transfers": [
            {"county": r.county, "med": short(r.commodity), "from": r.from_facility, "to": r.to_facility,
             "units": int(r.units), "km": float(r.km), "expiring": int(r.units_that_would_otherwise_expire),
             "risk": float(r.receiver_stockout_risk), "before": float(r.receiver_months_of_stock_before),
             "after": float(r.receiver_months_of_stock_after)}
            for r in transfers.itertuples()
        ],
        "results": results,
        "robustness": {
            "forecast_gain_min": round(float((robust.iloc[:, 1] - robust.iloc[:, 2]).min() * 100), 1),
            "forecast_gain_max": round(float((robust.iloc[:, 1] - robust.iloc[:, 2]).max() * 100), 1),
            "top20_gain_min": round(float((robust.iloc[:, 6] - robust.iloc[:, 5]).min() * 100)),
            "top20_gain_max": round(float((robust.iloc[:, 6] - robust.iloc[:, 5]).max() * 100)),
            "datasets": len(robust),
        },
        "briefs": briefs(),
        "outline": outline["rings"],
    }
    html = (ROOT / "docs" / "template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    path = ROOT / "docs" / "index.html"
    path.write_text(html)
    return path


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({p.stat().st_size // 1024} KB)")
