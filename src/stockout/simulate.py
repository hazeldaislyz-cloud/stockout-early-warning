"""Simulate monthly facility stock reports shaped like a Kenyan LMIS return.

Real facility data is not public, so this module generates realistic data with the
problems real data has: seasonal demand, uneven resupply, county-wide supply shocks,
push deliveries that overstock some facilities, expiry, and missing reports.

Each row is one facility x commodity x month, with the fields a facility reports:
opening balance, quantity received, quantity dispensed, losses to expiry, closing
balance and days out of stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COUNTIES = {
    # name: (lat, lon, malaria_weight)
    "Nairobi": (-1.29, 36.82, 0.1),
    "Mombasa": (-4.04, 39.67, 0.6),
    "Kisumu": (-0.09, 34.77, 1.0),
    "Kakamega": (0.28, 34.75, 1.0),
    "Nakuru": (-0.30, 36.07, 0.3),
    "Turkana": (3.12, 35.60, 0.7),
}
LEVELS = [("County Referral Hospital", 1, 6.0), ("Health Centre", 3, 2.0), ("Dispensary", 4, 1.0)]

# commodity: (base monthly units at a dispensary, seasonality type, shelf life in months, annual growth)
COMMODITIES = {
    "Insulin (soluble human, 10ml vial)": (6, "none", 24, 0.08),
    "Metformin 500mg (pack of 100)": (14, "none", 36, 0.06),
    "Artemether-lumefantrine 20/120 (pack of 24)": (40, "malaria", 24, 0.0),
    "Amoxicillin DT 250mg (pack of 10)": (30, "respiratory", 24, 0.02),
    "ORS/zinc co-pack": (25, "diarrhoea", 24, 0.0),
}

START = "2023-01"
MONTHS = 36


def _season(kind: str, month: int, malaria_weight: float) -> float:
    if kind == "malaria":  # peaks after the long rains (May-July) and short rains (Nov-Dec)
        s = 0.45 * np.cos(2 * np.pi * (month - 6) / 12) + 0.2 * np.cos(2 * np.pi * (month - 12) / 6)
        return 1 + malaria_weight * s
    if kind == "respiratory":
        return 1 + 0.25 * np.cos(2 * np.pi * (month - 7) / 12)
    if kind == "diarrhoea":
        return 1 + 0.3 * np.cos(2 * np.pi * (month - 4) / 12)
    return 1.0


def build_facilities(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for county, (lat, lon, mw) in COUNTIES.items():
        n = 0
        for level, count, scale in LEVELS:
            for _ in range(count):
                n += 1
                rows.append({
                    "facility_id": f"{county[:3].upper()}-{n:02d}",
                    "facility": f"{county} {level} {n}",
                    "county": county,
                    "level": level,
                    "lat": lat + rng.normal(0, 0.18),
                    "lon": lon + rng.normal(0, 0.18),
                    "size": scale * rng.lognormal(0, 0.25),
                    "malaria_weight": mw,
                })
    return pd.DataFrame(rows)


def simulate(seed: int = 7, missing_rate: float = 0.06) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (facilities, monthly reports)."""
    rng = np.random.default_rng(seed)
    fac = build_facilities(seed)
    months = pd.period_range(START, periods=MONTHS, freq="M")

    # County-wide supply shocks: months where the central supplier fills very little.
    shock = {(c, m): False for c in COUNTIES for m in range(MONTHS)}
    for c in COUNTIES:
        for _ in range(rng.integers(2, 6)):
            start = int(rng.integers(6, MONTHS - 2))
            for m in range(start, min(MONTHS, start + int(rng.integers(1, 4)))):
                shock[(c, m)] = True

    rows = []
    for f in fac.itertuples():
        for com, (base, kind, shelf, growth) in COMMODITIES.items():
            mean0 = base * f.size
            batches: list[list[float]] = [[mean0 * 3, shelf - 6]]  # [quantity, expiry month index]
            pending = 0.0
            late = 0.0
            hist: list[float] = []
            for m, period in enumerate(months):
                opening = sum(b[0] for b in batches)
                received = pending
                if received > 0:
                    batches.append([received, m + shelf - int(rng.integers(3, 10))])
                # Occasional "push" delivery sent without an order (overstocking risk).
                if rng.random() < 0.035:
                    push = mean0 * rng.uniform(4, 7)
                    batches.append([push, m + int(rng.integers(3, 7))])  # short-dated stock
                    received += push
                pending = 0.0

                demand = rng.poisson(
                    mean0 * (1 + growth) ** (m / 12) * _season(kind, period.month, f.malaria_weight)
                    * rng.lognormal(0, 0.12)
                )
                # Dispense first-expiry-first-out.
                batches.sort(key=lambda b: b[1])
                need, dispensed = float(demand), 0.0
                for b in batches:
                    take = min(b[0], need)
                    b[0] -= take
                    need -= take
                    dispensed += take
                    if need <= 0:
                        break
                short = demand - dispensed
                days_out = 0 if demand == 0 else int(round(30 * short / demand))
                # Remove expired stock at month end.
                expired = sum(b[0] for b in batches if b[1] <= m)
                batches = [b for b in batches if b[1] > m and b[0] > 0.5]
                closing = sum(b[0] for b in batches)
                expiring_3m = sum(b[0] for b in batches if b[1] <= m + 3)

                # Facility orders up to 3 months of average consumption. Most orders arrive
                # next month, some arrive a month late, and the supplier rarely fills them in full.
                hist.append(dispensed)
                amc = np.mean(hist[-3:])
                order = max(0.0, 3 * amc - closing - late)
                fill = rng.uniform(0.05, 0.3) if shock[(f.county, m)] else rng.uniform(0.6, 1.05)
                if rng.random() < 0.15:
                    late += order * fill
                    pending = 0.0
                else:
                    pending = order * fill + late
                    late = 0.0

                rows.append({
                    "facility_id": f.facility_id, "commodity": com, "period": str(period),
                    "month_index": m, "opening_balance": round(opening), "received": round(received),
                    "dispensed": round(dispensed), "expired": round(expired), "closing_balance": round(closing),
                    "days_out_of_stock": days_out, "expiring_within_3m": round(expiring_3m),
                    "true_demand": int(demand),
                    "county_supply_shock": shock[(f.county, m)],
                })
    df = pd.DataFrame(rows)
    # Some monthly reports are never submitted.
    miss = rng.random(len(df)) < missing_rate
    for col in ["opening_balance", "received", "dispensed", "expired", "closing_balance",
                "days_out_of_stock", "expiring_within_3m"]:
        df.loc[miss, col] = np.nan
    df["reported"] = ~miss
    return fac, df


if __name__ == "__main__":
    fac, df = simulate()
    print(fac.shape, df.shape)
    print(df.head())
