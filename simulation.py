"""Round stepping, Monte-Carlo previews, and end-of-game statistics. No Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from market import MONTHS_PER_ROUND, REGIMES, START_PRICE, Market, sample_monthly_returns
from portfolio import (
    INITIAL_EQUITY, MAINTENANCE_MARGIN, Account, MarginCallEvent, accrue_month,
    check_margin, margin_call_price, rebalance,
)


@dataclass(frozen=True)
class MonthRecord:
    month: int                 # 1..MONTHS_PER_ROUND within the round
    price: float
    ratio: float | None        # equity / stock value, before any forced sale
    threshold: float | None    # margin-call price given liabilities at this month-end
    shares: float              # shares held after any forced sale
    liabilities: float         # after any forced sale
    equity: float              # after any forced sale
    called: bool


@dataclass(frozen=True)
class RoundResult:
    account: Account
    start_equity: float
    end_equity: float
    interest: float
    months: tuple[MonthRecord, ...]
    events: tuple[MarginCallEvent, ...]

    @property
    def equity_return(self) -> float:
        return self.end_equity / self.start_equity - 1.0


def simulate_round(acct: Account, start_price: float, returns, borrow: float, rf: float) -> RoundResult:
    """Step month by month. Margin is tested at every month-end, not just at the end."""
    start_equity = acct.equity(start_price)
    price = start_price
    total_interest = 0.0
    months, events = [], []
    for i, r in enumerate(returns, start=1):
        price *= 1.0 + r
        acct, interest = accrue_month(acct, borrow, rf)
        total_interest += interest
        ratio = acct.margin_ratio(price)
        threshold = margin_call_price(acct.liabilities, acct.shares)
        acct, event = check_margin(acct, price, month=i)
        if event:
            events.append(event)
        months.append(MonthRecord(i, price, ratio, threshold, acct.shares, acct.liabilities,
                                  acct.equity(price), event is not None))
    return RoundResult(acct, start_equity, acct.equity(price), total_interest, tuple(months), tuple(events))


# ---- Pre-decision preview -------------------------------------------------
@dataclass(frozen=True)
class Preview:
    mean_equity_return: float
    prob_loss_20: float
    prob_margin_call: float
    equity_returns: np.ndarray = field(repr=False)


def preview(regime_name: str, initial_margin: float, equity: float = INITIAL_EQUITY,
            n: int = 1500, seed: int = 12345) -> Preview:
    """Monte Carlo with its OWN rng (never the market's), same stepping and liquidation rules."""
    reg = REGIMES[regime_name]
    rng = np.random.default_rng(seed)
    rets = sample_monthly_returns(reg, rng, (n, MONTHS_PER_ROUND))
    start = rebalance(Account(cash=equity), START_PRICE, initial_margin)
    out = np.empty(n)
    calls = 0
    for k in range(n):
        res = simulate_round(start, START_PRICE, rets[k], reg.borrow, reg.rf)
        out[k] = res.equity_return
        calls += bool(res.events)
    return Preview(float(out.mean()), float((out < -0.20).mean()), calls / n, out)


def thresholds(acct: Account, price: float, borrow: float, months: int = MONTHS_PER_ROUND):
    """(current margin-call price, margin-call price at round end once interest has accrued)."""
    now = margin_call_price(acct.liabilities, acct.shares)
    end_liab = acct.liabilities + acct.principal * borrow * months / 12.0
    return now, margin_call_price(end_liab, acct.shares)


# ---- Whole-game helpers ---------------------------------------------------
def max_drawdown(series) -> float:
    """Largest peak-to-trough fall as a positive fraction of the peak."""
    s = np.asarray(series, dtype=float)
    peak = np.maximum.accumulate(s)
    return float(np.max((peak - s) / peak))


def run_game(market: Market, margins: list[float]) -> dict:
    """Play a full game with a fixed list of initial margins (used by tests/headless runs)."""
    acct = Account(cash=INITIAL_EQUITY)
    series = [INITIAL_EQUITY]
    results = []
    for rd, m in zip(market.rounds, margins):
        if acct.equity(rd.start_price) <= 0:
            break
        acct = rebalance(acct, rd.start_price, m)
        reg = REGIMES[rd.regime]
        res = simulate_round(acct, rd.start_price, rd.returns, reg.borrow, reg.rf)
        acct = res.account
        series.extend(mr.equity for mr in res.months)
        results.append(res)
    return {"account": acct, "series": series, "results": results,
            "final": series[-1]}


def rule_margin(regime_name: str) -> float:
    """'Smart rule': borrow when the stock out-earns the loan, in proportion to (mu - r) / sigma^2 (capped 1x-2x)."""
    r = REGIMES[regime_name]
    lev = min(2.0, max(1.0, (r.mu - r.borrow) / r.sigma**2))
    return min(1.0, max(0.5, round(1.0 / lev, 1)))


def what_if(market: Market) -> dict[str, dict]:
    """Same market, three fixed strategies (for the results graph)."""
    plans = {
        "Never borrow (1×)": [1.0] * len(market.rounds),
        "Always max (2×)": [0.5] * len(market.rounds),
        "Smart rule": [rule_margin(rd.regime) for rd in market.rounds],
    }
    return {name: run_game(market, m) for name, m in plans.items()}


def playful_title(final: float, n_calls: int, avg_leverage: float, drawdown: float,
                  benchmark_final: float) -> tuple[str, str]:
    """Entertainment label for the results screen. First match wins; thresholds below."""
    BUFFETT_X, WIZARD_X = 1.50, 1.25     # final wealth vs. the no-leverage benchmark
    TBOND_BAND = 0.03                    # "about $1M" = within +/-3% of starting wealth
    BIG_LOSS = 0.20
    ret = final / INITIAL_EQUITY - 1.0
    if final <= 0:
        return "Broker's Favorite Customer 🤝", "You owe them money. They'll remember you fondly."
    if n_calls >= 3:
        return "Margin Call Magnet 🧲", "Your broker has you on speed dial."
    if ret > 0 and final >= BUFFETT_X * benchmark_final:
        return "Hello, Mr. Buffett 🧓", "Berkshire called. They want their playbook back."
    if ret > 0 and final >= WIZARD_X * benchmark_final:
        return "Leverage Wizard 🧙", "You borrowed like a wizard and cast it right."
    if n_calls >= 1 and ret > 0 and drawdown >= 0.25:
        return "Million-Dollar Comeback 🔥", "Knocked down, got back up."
    if abs(ret) <= TBOND_BAND:
        return "Should Have Bought T-Bonds 🏦", "Five years of stress for... the same million."
    if ret < 0:
        if ret > -BIG_LOSS:
            return "Don't Quit Your Day Job 💼", "Your professor still believes in you."
        return "Buy High, Sell Low™ Champion 🏆", "Bold strategy. Let's see how it plays out."
    if avg_leverage <= 1.1:
        return "Conservative Investor 🐢", "Slow, steady, and you slept well."
    return "Market Survivor 🪖", "You are still standing. That is a result."
