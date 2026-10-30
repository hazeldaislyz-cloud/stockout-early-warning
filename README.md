# Stockout early warning for essential medicines

A prototype that helps a county pharmacist in Kenya decide, each month, which facilities to follow up before they run out of a medicine, and where stock can be moved from a facility that has too much, before it expires.

I built this because I've spent years on the other side of the problem. At Novo Nordisk, 200,000 vials of insulin at KEMSA were close to expiring while facilities were short. We used KEMSA's own stock movement data to redistribute them, and the account went on to order another 300,000. That was done by hand, with spreadsheets and phone calls. This project asks how much of it a simple, explainable model could do every month.

> **The data is simulated.** Facility-level stock data in Kenya is not public, so `simulate.py` generates monthly reports with the same fields and the same problems as a real LMIS return. The numbers below show the method works on realistic data. They are not evidence about any real county.

## What it does

1. **Forecasts next month's consumption** for each facility and medicine, and compares it with the method most facilities use today: the average of the last three months (AMC).
2. **Scores the risk of a stockout next month**, and compares it with the usual rule of following up facilities with the fewest months of stock first.
3. **Suggests transfers** between facilities in the same county: from facilities with short-dated or surplus stock to nearby facilities at high risk.
4. **Writes a one-page brief per county** in plain language. If an Anthropic API key is set, a language model writes it; every number it uses is then checked against the model outputs, and the brief is replaced with a plain template if any number doesn't match.

## The simulated data

- 48 facilities in 6 counties (Nairobi, Mombasa, Kisumu, Kakamega, Nakuru, Turkana): a county referral hospital, health centres and dispensaries in each.
- 5 tracer medicines: insulin, metformin, artemether-lumefantrine (AL), amoxicillin DT and ORS/zinc.
- 36 months, January 2023 to December 2025.
- Each report has the fields a facility submits: opening balance, received, dispensed, expired, closing balance and days out of stock.

It builds in the problems real data has:

- seasonal demand (malaria peaks after the rains, strongest around Lake Victoria)
- orders that are filled only in part, or arrive a month late
- county-wide supply shocks lasting one to three months
- "push" deliveries of short-dated stock that nobody ordered
- expiry, dispensed first-expiry-first-out
- about 6% of monthly reports never submitted

One detail matters a lot: **in a month when a facility ran out, what it dispensed is less than what patients needed.** Those months are left out of the forecasting target, so the model doesn't learn to expect low demand from facilities that simply had nothing to give.

## Results

All evaluation is out of time. The model is only ever trained on months before the ones it's tested on.

### Forecasting next month's consumption

Rolling backtest over 12 months, 2,195 forecasts. WAPE is total absolute error divided by total actual consumption.

| Method | Forecast error (WAPE) |
|---|---|
| 3-month average (current practice) | 24.6% |
| Gradient-boosted model | 18.3% |

The gain is largest for seasonal medicines (AL, amoxicillin, ORS/zinc), because the model learns the seasons and the 3-month average always lags behind them (see `outputs/example_forecast.png`). For steady medicines like insulin and metformin the gain is small.

![Forecast error by commodity](outputs/forecast_error.png)

### Which facilities to follow up

Test period: 12 months, 2,543 facility-medicine checks, 348 of which ran out the following month (14%).

A county team can only follow up so many facilities. So the question is: **if you follow up the top N each month, how many of them were really about to run out?**

| Ranking method | PR-AUC | Top 10 a month | Top 20 a month | Top 40 a month |
|---|---|---|---|---|
| Rule: fewest months of stock first | 0.44 | 50% | 46% | 45% |
| Model (7 features, shallow trees) | 0.51 | 54% | 55% | 47% |
| Model I tried first (17 features) | 0.40 | 47% | 43% | 38% |

![Alert precision](outputs/alert_precision.png)

**What didn't work, and why it matters.** My first model used 17 features and did *worse* than the simple rule. It overfit: it learned patterns from the early months that didn't hold later. Cutting it to 7 features that a supply officer would recognise, and restricting the trees, fixed that. The simple rule is a strong baseline because it is built on the right quantity (stock compared with use). The model's edge comes from adding two things the rule ignores: whether deliveries to the whole county have been short this month (a sign of trouble upstream), and the forecast, which knows the season is turning.

**Is it robust?** `robustness.py` re-runs everything on six different simulated datasets:

- Forecasting: the model beat the 3-month average in all six, by 5.7 to 6.4 percentage points.
- Stockout risk: the model beat the rule on PR-AUC in five of six. In the sixth, the rule did better, and clearly so on the top-20 list (69% against 49%).

So the risk model is a modest, fairly consistent improvement, not a large one. I'd want to see it confirmed on real data before anyone relied on it.

### Transfers, December 2025

The plan suggests 13 transfers in 4 counties, moving 724 units. 88 of those units would otherwise have expired. Every transfer is within 120 km, in the same county, and at least 10 units. See `outputs/suggested_transfers.csv` and `outputs/county_briefs.md`.

## Responsible use

- **A person decides.** Every output is framed as an estimate for a pharmacist to check, not an instruction.
- **The language model can't invent numbers.** The county brief is checked number by number against the model outputs; if anything doesn't match, the template version is used. This is tested in `tests/test_stockout.py`.
- **Explainable by design.** The risk model uses 7 features a supply officer already thinks about (months of stock, recent deliveries, recent stockouts, facility level, medicine), and the transfer logic is a few readable rules, not a black box.
- **Fairness.** Facilities that report less often have less data and could be under-alerted. With real data I would check alert rates and missed stockouts by county and by facility level before use.

## Limitations

- Simulated data. Real LMIS data will be messier: reporting errors, unit mix-ups, and stock moving in ways a report doesn't capture.
- The model doesn't know about orders in transit or planned deliveries. With real data, adding the supplier's dispatch records would likely help more than anything else.
- Transfers ignore transport cost and who would physically move the stock.
- Seasonality is learned from only two years of history.

## Next steps with real data

1. Pilot in one county with real facility reports from KHIS, and the county pharmacist's judgement as the benchmark.
2. Add supplier dispatch data, so the model knows what is already on the way.
3. A simple interface for the county team: a monthly list, a map, and a way to mark what they acted on, so the model can learn from it.

## How to run

```bash
pip install -r requirements.txt
cd src
python -m stockout.pipeline        # runs everything, about 30 seconds; writes to outputs/
cd ..
python -m pytest tests             # 9 tests
python robustness.py               # optional, a few minutes
```

To get the written county briefs from a language model rather than the template, set `ANTHROPIC_API_KEY` before running the pipeline.

## Project layout

```
src/stockout/
  simulate.py       simulated facility reports
  features.py       features built only from what is known at month end
  forecast.py       AMC baseline, gradient-boosted model, rolling backtest
  risk.py           stockout risk model and comparison with the rule
  redistribute.py   expiry-aware transfer suggestions
  brief.py          county briefs, with the number check
  pipeline.py       runs everything and writes outputs/
tests/              stock balance, leakage, transfer logic and number-check tests
robustness.py       re-runs the evaluation on six datasets
```

## Author

Daisy Mutegi. 13 years in medicine access and supply across East and Southern Africa (Novo Nordisk, Clinton Health Access Initiative). [LinkedIn](https://linkedin.com/in/daisymutegi)

Built with the help of AI coding tools, which I used the way I'd expect a team to use them: I set the problem, the design and the evaluation, and checked the results.
