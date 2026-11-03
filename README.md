# Leverage Lab — The $1 Million Margin Challenge

A single-player Streamlit game for FIN 4200 that builds on the Week 2 "Buying on Margin" case.
Students choose their leverage before each of 10 rounds, then watch a simulated market reveal whether
they get margin-called. No login, database, or live data.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Tests: `pip install pytest && python -m pytest -q`

## Deploy (Streamlit Community Cloud)
Push this repo to GitHub → https://share.streamlit.io → *New app* → pick the repo, branch `main`, file `app.py`.
Share the URL with students. Each browser session is an independent game (`st.session_state`).

## Files
| File | Role |
|---|---|
| `app.py` | UI: intro → decide → result → … → final dashboard |
| `market.py` | Regimes, Markov transitions, return model, news, whole-game path from a seed |
| `portfolio.py` | Margin-account math (rebalance, interest, margin-call price, liquidation) |
| `simulation.py` | Month-by-month round stepping, Monte-Carlo preview, drawdown, titles |
| `tests/` | Case reproduction, mechanics, determinism, and a headless full-game UI test |

## Game design in brief
**Clock.** 10 rounds × 6 months = 5 years (`N_ROUNDS`, `MONTHS_PER_ROUND` in `market.py`; set to 5 and 12 for annual rounds).
Rates are quoted per year; interest, the distribution chart and the risk estimates are scaled to the round.

**Economies** (expected annual return / volatility / loan rate): Expansion 12/15/5, Sideways 8/18/5, Recession 2/30/4,
Recovery 16/22/3, Inflation shock 5/25/9. Regime changes follow a transition matrix (`TRANSITIONS`) in which
every regime can follow every other. Recession and inflation shock have *negative* carry
(expected return below the loan rate), so the best leverage depends on the economy.

**Returns.** Monthly lognormal with an occasional crash jump (bad regimes), calibrated so the announced
expected return is the true arithmetic mean. The outlook chart and risk estimates are Monte Carlo from this same model
(separate RNG). Headlines are picked *after* the returns and never alter them.

**Margin mechanics** (constants in `portfolio.py`)
* Position = equity ÷ initial margin; loan = position − equity. Max 2× (50% margin). No new money is ever added.
* Interest is simple, `principal × rate × months/12`, accrued monthly as a liability (as in the case).
* Margin ratio = equity ÷ stock value. Tested at **every month-end**.
* Margin-call price `P* = L / (N(1−m))`, with m = 30%. The app shows it today and again after the round's interest.
* On a call, shares are sold until the ratio is back to 35% (a buffer so the account isn't re-called after every dip),
  with a 0.1% fee; proceeds repay accrued interest first, then principal.
* If equity ≤ 0 the whole position is sold and any deficit is reported as a shortfall; the game ends.
* Cash earns the regime's risk-free rate.

**Replay.** The full market (regimes, returns, headlines) is generated from one seed at game start and never
depends on decisions. *Replay the same market* reuses the seed; students can also share the **market code**.

## Essential vs. optional
Essential (built): regimes, outlook + distribution, leverage choice with consequences, guess-then-reveal margin-call price,
month-level margin-call detection, liquidation, results dashboard, both replay modes, tests.
Optional ideas: dividends, a "hindsight" overlay of other leverage levels on the same market, partial-cure options
(deposit cash), class-wide results via a shared Google Sheet.

*All market data are simulated; nothing here is investment advice.*
