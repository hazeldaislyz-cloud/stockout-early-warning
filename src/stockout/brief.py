"""Write a short brief for a county pharmacist from the numbers the models produced.

The language model is only allowed to rephrase facts it is given. After it writes,
every number in its text is checked against the facts; if any number is not found,
the brief is rejected and the plain template version is used instead. Without an
API key, the template version is used.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

PROMPT = """You are helping a county pharmacist in Kenya plan this month's stock follow-ups.
Write a brief of no more than 200 words using ONLY the facts in the JSON below.
- Start with the 3 facilities most at risk of running out, with the commodity and risk.
- Then list the suggested transfers, with units and distance.
- Do not add any number, facility or medicine that is not in the JSON.
- Say plainly that these are model estimates for a pharmacist to check, not instructions.

FACTS:
{facts}
"""


def county_facts(county: str, alerts: pd.DataFrame, transfers: pd.DataFrame, period: str) -> dict:
    a = alerts[alerts["county"] == county].nlargest(5, "risk")
    t = transfers[transfers["county"] == county]
    return {
        "county": county,
        "month": period,
        "top_risks": [
            {"facility": r.facility, "commodity": r.commodity, "stockout_risk_percent": int(round(100 * r.risk)),
             "months_of_stock": round(float(r.months_of_stock), 1)}
            for r in a.itertuples()
        ],
        "transfers": [
            {"from": r.from_facility, "to": r.to_facility, "commodity": r.commodity, "units": int(r.units),
             "km": float(r.km), "units_saved_from_expiry": int(r.units_that_would_otherwise_expire)}
            for r in t.itertuples()
        ],
    }


def template_brief(f: dict) -> str:
    lines = [f"Stock follow-up brief: {f['county']} County, {f['month']}", "",
             "Highest stockout risk next month:"]
    for r in f["top_risks"][:3]:
        lines.append(f"- {r['facility']}: {r['commodity']}, {r['stockout_risk_percent']}% risk, "
                     f"{r['months_of_stock']} months of stock")
    lines += ["", "Suggested transfers:"]
    if not f["transfers"]:
        lines.append("- None this month: no facility nearby holds surplus or short-dated stock.")
    for r in f["transfers"]:
        lines.append(f"- {r['units']} units of {r['commodity']} from {r['from']} to {r['to']} "
                     f"({r['km']} km; {r['units_saved_from_expiry']} would otherwise expire)")
    lines += ["", "These are model estimates for a pharmacist to check before acting."]
    return "\n".join(lines)


def numbers_in(text: str) -> set[str]:
    return {n.rstrip(".").lstrip("0") or "0" for n in re.findall(r"\d+(?:\.\d+)?", text)}


def grounded(text: str, facts: dict) -> bool:
    """True if every number in the text appears somewhere in the facts."""
    allowed = numbers_in(json.dumps(facts)) | numbers_in(facts["county"])
    return numbers_in(text) <= allowed


def llm_brief(facts: dict, client=None, model: str = "claude-sonnet-4-5") -> tuple[str, str]:
    """Return (brief, source) where source is 'llm' or 'template'."""
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return template_brief(facts), "template"
        import anthropic
        client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=500,
                                 messages=[{"role": "user", "content": PROMPT.format(facts=json.dumps(facts, indent=1))}])
    text = msg.content[0].text
    if not grounded(text, facts):
        return template_brief(facts), "template (LLM output failed the number check)"
    return text, "llm"
