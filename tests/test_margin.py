import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from market import generate_market, monthly_params, REGIMES, sample_monthly_returns
from portfolio import (
    Account, MAINTENANCE_MARGIN, RESTORE_MARGIN, accrue_month, check_margin,
    margin_call_price, rebalance,
)
from simulation import max_drawdown, preview, run_game, simulate_round, thresholds


def test_case_roi_and_margin_call_price():
    # Excel case: $1M, 50% margin, P0=141, P1=151, $1 dividend, 5%, 12 months.
    shares = (1_000_000 / 0.5) / 141
    loan = 1_000_000
    roi = (shares * (151 - 141) + shares * 1 - loan * 0.05) / 1_000_000
    assert roi == pytest.approx(0.1060, abs=1e-4)
    assert margin_call_price(loan, shares) == pytest.approx(100.71, abs=0.01)


def test_spec_thresholds_100_dollar_2x():
    acct = rebalance(Account(cash=1_000_000), 100.0, 0.5)
    assert acct.shares == pytest.approx(20_000)
    assert acct.principal == pytest.approx(1_000_000)
    now, end12 = thresholds(acct, 100.0, 0.05, months=12)
    assert now == pytest.approx(71.43, abs=0.01)
    assert end12 == pytest.approx(75.00, abs=0.01)


def test_interest_accrual_simple():
    acct = rebalance(Account(cash=1_000_000), 100.0, 0.5)
    total = 0.0
    for _ in range(12):
        acct, i = accrue_month(acct, 0.05, 0.0)
        total += i
    assert total == pytest.approx(50_000)
    assert acct.accrued == pytest.approx(50_000)
    assert acct.principal == pytest.approx(1_000_000)


def test_rebalance_uses_current_equity_and_caps_leverage():
    acct = Account(shares=20_000, principal=1_000_000)
    new = rebalance(acct, 120.0, 0.5)          # equity 1.4M
    assert new.principal == pytest.approx(1_400_000)
    assert new.shares * 120 == pytest.approx(2_800_000)
    with pytest.raises(ValueError):
        rebalance(acct, 100.0, 0.4)            # >2x not allowed
    with pytest.raises(ValueError):
        rebalance(Account(shares=1000, principal=200_000), 100.0, 0.5)  # no equity


def test_no_call_above_threshold_and_call_below():
    acct = rebalance(Account(cash=1_000_000), 100.0, 0.5)
    same, ev = check_margin(acct, 72.0)
    assert ev is None and same == acct
    new, ev = check_margin(acct, 70.0)
    assert ev is not None and not ev.full_liquidation
    assert new.margin_ratio(70.0) == pytest.approx(RESTORE_MARGIN, abs=1e-6)
    assert new.shares < acct.shares and new.liabilities < acct.liabilities
    assert ev.ratio < MAINTENANCE_MARGIN


def test_negative_equity_reports_shortfall():
    acct = rebalance(Account(cash=1_000_000), 100.0, 0.5)
    new, ev = check_margin(acct, 45.0)          # stock value 900k < loan 1M
    assert ev.full_liquidation and ev.shortfall > 0
    assert new.shares == 0 and new.equity(45.0) < 0


def test_intra_year_call_even_if_market_recovers():
    acct = rebalance(Account(cash=1_000_000), 100.0, 0.5)
    rets = [-0.15, -0.15, -0.05, 0.30, 0.30, 0.30]      # P: 85,72.25,68.6 -> call in month 3
    res = simulate_round(acct, 100.0, rets, 0.05, 0.02)
    assert [m.called for m in res.months][:3] == [False, False, True]
    unlevered_end = 100 * np.prod(1 + np.array(rets))
    assert unlevered_end > 100                            # market ended up
    # Hypothetical account that was never forced to sell: 20,000 shares, $1M loan, 6 months simple interest.
    never_called = 20_000 * unlevered_end - 1_000_000 - 1_000_000 * 0.05 * 6 / 12
    assert res.end_equity < never_called                  # forced sale locked in losses, missed the rebound


def test_market_mean_is_calibrated():
    rng = np.random.default_rng(1)
    for reg in REGIMES.values():
        r = sample_monthly_returns(reg, rng, (400_000,))
        annual = (1 + r.mean()) ** 12 - 1
        assert annual == pytest.approx(reg.mu, abs=0.01)


def test_same_seed_same_market_regardless_of_strategy():
    m1, m2 = generate_market(42), generate_market(42)
    a = run_game(m1, [0.5] * 10)
    b = run_game(m2, [1.0] * 10)
    assert m1.price_series() == m2.price_series()
    assert generate_market(43).price_series() != m1.price_series()
    assert a["final"] != b["final"]


def test_preview_is_deterministic_and_leverage_monotone():
    p1, p2 = preview("Recession", 0.5), preview("Recession", 0.5)
    assert p1.prob_margin_call == p2.prob_margin_call
    assert preview("Recession", 1.0).prob_margin_call == 0.0
    assert p1.prob_margin_call > preview("Recession", 0.8).prob_margin_call


def test_playful_title():
    from simulation import playful_title
    t = lambda final, calls=0, lev=1.5, dd=0.1, bench=1.4e6: playful_title(final, calls, lev, dd, bench)[0]
    assert "Buffett" in t(2.2e6)
    assert "Wizard" in t(1.8e6)
    assert "Comeback" in t(1.5e6, calls=1, dd=0.4)
    assert "T-Bonds" in t(1.02e6)
    assert "Day Job" in t(0.9e6)
    assert "Champion" in t(0.6e6)
    assert "Broker" in t(-5e4)
    assert "Magnet" in t(1.5e6, calls=3)
    assert "Conservative" in t(1.3e6, lev=1.0)
    assert "Survivor" in t(1.3e6)


def test_headlines_match_returns():
    from market import NEWS_AFTER
    for seed in range(1, 150):
        for rd in generate_market(seed).rounds:
            if rd.news_after in NEWS_AFTER["crash"]:
                assert rd.stock_return < 0
            if rd.news_after in NEWS_AFTER["rebound"]:
                assert rd.stock_return >= 0.10 and min(rd.returns) < -0.10
            if rd.news_after in NEWS_AFTER["scare"]:
                assert 0 <= rd.stock_return < 0.10
            if rd.news_after in NEWS_AFTER["boom"]:
                assert rd.stock_return >= 0.15


def _records(market, margins):
    from simulation import run_game
    g = run_game(market, margins)
    return [dict(round=i + 1, regime=rd.regime, margin=m, lev=1 / m, rd=rd, res=res, stock_ret=rd.stock_return,
                 guess=None, answer=None)
            for i, (rd, m, res) in enumerate(zip(market.rounds, margins, g["results"]))], g


def test_badges_and_what_if():
    from badges import BADGES, Ctx, earned
    from simulation import what_if
    mkt = generate_market(7)
    alts = what_if(mkt)
    assert set(alts) == {"Never borrow (1×)", "Always max (2×)", "Smart rule"}
    assert alts["Never borrow (1×)"]["final"] == pytest.approx(1e6 * mkt.price_series()[-1] / 100)

    recs, g = _records(mkt, [1.0] * 10)
    names = {b.name for b in earned(Ctx(recs, g["final"], 0.0, alts["Smart rule"]["final"]))}
    assert "Sleeps Like a Baby" in names and "All In, Every Time" not in names

    recs, g = _records(mkt, [0.5] * 10)
    interest = sum(r["res"].interest for r in recs)
    names = {b.name for b in earned(Ctx(recs, g["final"], interest, alts["Smart rule"]["final"]))}
    assert "All In, Every Time" in names and "Sleeps Like a Baby" not in names

    # guess badge and replay badges
    recs[0]["guess"], recs[0]["answer"] = 71.0, 71.43
    c = Ctx(recs, 1_000_000, 0.0, 9e9, prior_final=800_000)
    names = {b.name for b in earned(c)}
    assert {"Nostradamus of Margin", "Time Traveler"} <= names and "Groundhog Day" not in names
    assert len({b.name for b in BADGES}) == len(BADGES)


def test_max_drawdown():
    assert max_drawdown([100, 120, 60, 90]) == pytest.approx(0.5)
