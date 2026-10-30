"""Simulate monthly facility stock reports shaped like a Kenyan LMIS return.

Real facility data is not public, so this module generates data with the problems real
data has: lumpy resupply, partial and late orders, county-wide supply shocks, push
deliveries of short-dated stock, expiry, very uneven facility sizes and missing reports.

The supply side is calibrated against real national warehouse issue records for insulin
and real distributor sales for chronic-disease medicines (not shared; see README):

- facilities order roughly once a quarter, not every month, and top up early only when
  stock runs very low
- quantities are issued in whole packs (multiples of 10)
- stock arrives with about half its shelf life left (insulin: median ~13 of 24 months)
- facility size is very uneven: a referral hospital uses about 20 times what a
  dispensary does, and size varies a lot within each level
- demand for chronic medicines is mildly seasonal: low in December, higher in January
  and March

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
# (level, facilities per county, size relative to a dispensary)
LEVELS = [("County Referral Hospital", 1, 20.0), ("Health Centre", 3, 2.5), ("Dispensary", 4, 1.0)]
SIZE_SPREAD = 1.0  # log-scale spread of size within a level

# commodity: (base monthly units at a dispensary, seasonality type, shelf life in months, annual growth)
COMMODITIES = {
    "Insulin (premixed 30/70, 10ml vial)": (6, "chronic", 24, 0.08),
    "Metformin 500mg (pack of 100)": (12, "chronic", 36, 0.06),
    "Artemether-lumefantrine 20/120 (pack of 24)": (30, "malaria", 24, 0.0),
    "Amoxicillin DT 250mg (pack of 10)": (25, "respiratory", 24, 0.02),
    "ORS/zinc co-pack": (20, "diarrhoea", 24, 0.0),
}

START = "2023-01"
MONTHS = 36
PACK = 10            # warehouse issues whole packs
REVIEW_MONTHS = 3    # routine order cycle
MAX_MONTHS = 6       # order up to this many months of average use
EMERGENCY_BELOW = 0.5  # months of stock that trigger an early top-up order
CHRONIC_SEASON = {1: 1.10, 2: 0.98, 3: 1.15, 4: 0.95, 5: 0.95, 6: 0.97,
                  7: 1.03, 8: 1.00, 9: 1.03, 10: 1.04, 11: 1.00, 12: 0.85}


def _season(kind: str, month: int, malaria_weight: float) -> float:
    if kind == "malaria":  # peaks after the long rains (May-July) and short rains (Nov-Dec)
        s = 0.45 * np.cos(2 * np.pi * (month - 6) / 12) + 0.2 * np.cos(2 * np.pi * (month - 12) / 6)
        return 1 + malaria_weight * s
    if kind == "respiratory":
        return 1 + 0.25 * np.cos(2 * np.pi * (month - 7) / 12)
    if kind == "diarrhoea":
        return 1 + 0.3 * np.cos(2 * np.pi * (month - 4) / 12)
    if kind == "chronic":
        return CHRONIC_SEASON[month]
    return 1.0


def _packs(q: float) -> float:
    """Round an order to whole packs; anything positive is at least one pack."""
    if q <= 0:
        return 0.0
    return float(max(PACK, PACK * round(q / PACK)))


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
                    "size": scale * rng.lognormal(0, SIZE_SPREAD),
                    "malaria_weight": mw,
                    "order_phase": int(rng.integers(0, REVIEW_MONTHS)),
                })
    return pd.DataFrame(rows)


def _shelf_left(rng: np.random.Generator, shelf: int) -> int:
    """Months of shelf life left when a routine delivery arrives: about half, varying."""
    return int(np.clip(round(rng.normal(0.55 * shelf, 0.2 * shelf)), 0.3 * shelf, shelf - 1))


def simulate(seed: int = 7, missing_rate: float = 0.06, with_deliveries: bool = False):
    """Return (facilities, monthly reports), plus a table of deliveries if asked."""
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

    rows, deliveries = [], []
    for f in fac.itertuples():
        for com, (base, kind, shelf, growth) in COMMODITIES.items():
            mean0 = base * f.size
            batches: list[list[float]] = [[mean0 * 3, _shelf_left(rng, shelf)]]  # [quantity, expiry month]
            pending = 0.0
            late = 0.0
            hist: list[float] = []
            for m, period in enumerate(months):
                opening = sum(b[0] for b in batches)
                received = pending
                if received > 0:
                    left = _shelf_left(rng, shelf)
                    batches.append([received, m + left])
                    deliveries.append({"facility_id": f.facility_id, "commodity": com, "month_index": m,
                                       "units": received, "shelf_left_months": left, "kind": "order"})
                # Occasional "push" delivery of short-dated stock that nobody ordered.
                if rng.random() < 0.02:
                    push = _packs(mean0 * rng.uniform(3, 6))
                    left = int(rng.integers(4, 9))
                    batches.append([push, m + left])
                    received += push
                    deliveries.append({"facility_id": f.facility_id, "commodity": com, "month_index": m,
                                       "units": push, "shelf_left_months": left, "kind": "push"})
                pending = 0.0

                demand = rng.poisson(
                    mean0 * (1 + growth) ** (m / 12) * _season(kind, period.month, f.malaria_weight)
                    * rng.lognormal(0, 0.2)
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

                # Ordering. Facilities order on a quarterly cycle, up to MAX_MONTHS of average
                # use, and top up early only when stock is nearly gone. Some routine orders are
                # never placed (no budget, no one to place it). The supplier rarely fills in
                # full, issues whole packs, and some orders arrive a month late.
                hist.append(dispensed)
                amc = max(np.mean(hist[-3:]), 0.2 * mean0)
                routine = (m % REVIEW_MONTHS == f.order_phase) and rng.random() > 0.2
                emergency = closing < EMERGENCY_BELOW * amc and rng.random() < 0.4
                order = 0.0
                if routine or emergency:
                    order = max(0.0, MAX_MONTHS * amc - closing - late)
                fill = rng.uniform(0.05, 0.3) if shock[(f.county, m)] else rng.uniform(0.6, 1.0)
                filled = _packs(order * fill) if order > 0 else 0.0
                if filled > 0 and rng.random() < 0.15:
                    late += filled
                else:
                    pending = filled + late
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
    if with_deliveries:
        return fac, df, pd.DataFrame(deliveries)
    return fac, df


if __name__ == "__main__":
    fac, df = simulate()
    print(fac.shape, df.shape)
    print(df.head())
