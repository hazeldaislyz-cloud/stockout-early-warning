import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stockout.brief import county_facts, grounded, llm_brief, template_brief  # noqa: E402
from stockout.features import panel  # noqa: E402
from stockout.forecast import wape  # noqa: E402
from stockout.redistribute import haversine_km, plan_transfers  # noqa: E402
from stockout.simulate import simulate  # noqa: E402


@pytest.fixture(scope="module")
def data():
    fac, reports = simulate(seed=3)
    return fac, reports, panel(fac, reports)


def test_stock_balances(data):
    """Closing = opening + received - dispensed - expired, for every reported month."""
    _, r, _ = data
    r = r[r["reported"]]
    gap = r["opening_balance"] + r["received"] - r["dispensed"] - r["expired"] - r["closing_balance"]
    assert gap.abs().max() <= 3  # rounding only


def test_simulation_is_reproducible():
    a = simulate(seed=5)[1]
    b = simulate(seed=5)[1]
    pd.testing.assert_frame_equal(a, b)


def test_some_reports_missing_and_some_stockouts(data):
    _, r, _ = data
    assert 0.02 < (~r["reported"]).mean() < 0.12
    assert (r["days_out_of_stock"] > 0).mean() > 0.03


def test_deliveries_are_whole_packs_and_lumpy():
    """Calibrated supply: whole packs of 10, and most facilities are not resupplied every month."""
    _, r, dl = simulate(seed=3, with_deliveries=True)
    assert (dl["units"] % 10 == 0).all()
    r = r[r["reported"]]
    share = r.groupby(["facility_id", "commodity"])["received"].apply(lambda s: (s > 0).mean())
    assert 0.15 < share.median() < 0.5


def test_targets_do_not_leak(data):
    """next_dispensed at month t must equal dispensed at t+1 for the same series."""
    _, _, d = data
    s = d[(d["facility_id"] == d["facility_id"].iloc[0]) & (d["commodity"] == d["commodity"].iloc[0])]
    s = s.sort_values("month_index")
    nxt = s["dispensed"].shift(-1)
    both = s["next_dispensed"].notna()
    assert np.allclose(s.loc[both, "next_dispensed"], nxt[both])


def test_wape():
    assert wape(pd.Series([10, 10]), pd.Series([8, 12])) == pytest.approx(0.2)


def test_haversine_nairobi_mombasa():
    assert 430 < haversine_km(-1.29, 36.82, -4.04, 39.67) < 450


def _toy():
    return pd.DataFrame({
        "county": ["X", "X", "X"], "commodity": ["Insulin"] * 3,
        "facility": ["Donor", "Receiver", "Far"], "lat": [0.0, 0.1, 5.0], "lon": [36.0, 36.1, 36.0],
        "closing_balance": [100.0, 2.0, 100.0], "expiring_within_3m": [60.0, 0.0, 60.0],
        "forecast_next": [10.0, 20.0, 10.0], "risk": [0.05, 0.8, 0.05], "months_of_stock": [10, 0.1, 10],
    })


def test_transfer_goes_to_nearest_donor_and_respects_limits():
    t = plan_transfers(_toy())
    assert len(t) == 1
    row = t.iloc[0]
    assert row["from_facility"] == "Donor" and row["to_facility"] == "Receiver"
    assert row["units"] == 38  # receiver needs 2 x 20 - 2
    assert row["km"] < 120


def test_brief_number_check():
    facts = county_facts("X", _toy(), plan_transfers(_toy()), "2025-12")
    assert grounded(template_brief(facts), facts)
    assert not grounded("Send 999 units now.", facts)


class _FakeClient:
    def __init__(self, text):
        self.text = text
        self.messages = self

    def create(self, **kwargs):
        return type("M", (), {"content": [type("C", (), {"text": self.text})()]})()


def test_llm_brief_rejects_invented_numbers():
    facts = county_facts("X", _toy(), plan_transfers(_toy()), "2025-12")
    text, source = llm_brief(facts, client=_FakeClient("Move 500 vials to Receiver."))
    assert source.startswith("template")
    text, source = llm_brief(facts, client=_FakeClient("Receiver has an 80% risk. Move 38 units from Donor."))
    assert source == "llm"
