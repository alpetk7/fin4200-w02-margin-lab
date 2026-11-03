"""Simulated market: economic regimes, regime transitions, stock-price paths, news.

Everything here is a pure function of a seed. The market path is generated once,
up front, and never looks at what the student does -- so "Replay the Same Market"
really is the same market.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- Game clock -----------------------------------------------------------
N_ROUNDS = 10            # number of decisions ("steps")
MONTHS_PER_ROUND = 6     # 10 rounds x 6 months = 5 years. Set to (5, 12) for annual rounds.
START_PRICE = 100.0

# Monthly crash-jump size (log return) used in the bad regimes.
JUMP_MEAN = -0.10
JUMP_SD = 0.04


@dataclass(frozen=True)
class Regime:
    name: str
    mu: float          # expected annual stock return (arithmetic mean)
    sigma: float       # annual volatility
    borrow: float      # margin-loan rate (annual)
    rf: float          # risk-free rate on idle cash (annual)
    jump_prob: float   # per-month probability of a crash jump
    sentiment: str
    icon: str
    color: str


REGIMES: dict[str, Regime] = {
    r.name: r
    for r in [
        Regime("Expansion", 0.12, 0.15, 0.05, 0.04, 0.005, "Optimistic", "📈", "#16a34a"),
        Regime("Recession", 0.02, 0.30, 0.04, 0.02, 0.060, "Fearful", "📉", "#dc2626"),
        Regime("Recovery", 0.16, 0.22, 0.03, 0.015, 0.020, "Cautiously hopeful", "🌱", "#0891b2"),
        Regime("Inflation shock", 0.05, 0.25, 0.09, 0.05, 0.050, "Nervous", "🔥", "#d97706"),
        Regime("Sideways", 0.08, 0.18, 0.05, 0.03, 0.010, "Indifferent", "😐", "#64748b"),
    ]
}
REGIME_ORDER = list(REGIMES)

# Transition probabilities per round (rows = from, columns = to, in REGIME_ORDER).
# Persistent but never deterministic: every regime can follow every other one.
TRANSITIONS = np.array(
    [
        # Exp    Rec    Recov  Infl   Side
        [0.60, 0.08, 0.02, 0.15, 0.15],   # from Expansion
        [0.04, 0.46, 0.34, 0.06, 0.10],   # from Recession
        [0.42, 0.08, 0.30, 0.03, 0.17],   # from Recovery
        [0.12, 0.27, 0.08, 0.40, 0.13],   # from Inflation shock
        [0.30, 0.15, 0.10, 0.10, 0.35],   # from Sideways
    ]
)
INITIAL_REGIME_PROBS = np.array([0.40, 0.10, 0.10, 0.15, 0.25])


# ---- Return model ---------------------------------------------------------
def monthly_params(regime: Regime) -> tuple[float, float, float]:
    """Return (a, sd, p): monthly log-return = a + sd*Z + jump*Bernoulli(p).

    Calibrated so that the *arithmetic* annual mean equals regime.mu exactly and
    total annual volatility is close to regime.sigma.
    """
    p = regime.jump_prob
    jump_var = p * (JUMP_MEAN**2 + JUMP_SD**2)
    var_m = max(regime.sigma**2 / 12 - jump_var, 0.25 * regime.sigma**2 / 12)
    sd = float(np.sqrt(var_m))
    e_jump = 1 - p + p * np.exp(JUMP_MEAN + JUMP_SD**2 / 2)
    a = np.log1p(regime.mu) / 12 - sd**2 / 2 - np.log(e_jump)
    return float(a), sd, p


def sample_monthly_returns(regime: Regime, rng: np.random.Generator, size) -> np.ndarray:
    """Simple monthly stock returns, shape `size` (int or tuple)."""
    a, sd, p = monthly_params(regime)
    z = rng.standard_normal(size)
    hit = rng.random(size) < p
    jump = rng.normal(JUMP_MEAN, JUMP_SD, size) * hit
    return np.maximum(np.exp(a + sd * z + jump) - 1.0, -0.60)


# ---- Whole-game path ------------------------------------------------------
@dataclass(frozen=True)
class RoundData:
    index: int
    regime: str
    returns: tuple[float, ...]      # monthly simple returns
    start_price: float
    prices: tuple[float, ...]       # month-end prices
    news_before: str
    news_after: str

    @property
    def end_price(self) -> float:
        return self.prices[-1]

    @property
    def stock_return(self) -> float:
        return self.end_price / self.start_price - 1.0


@dataclass(frozen=True)
class Market:
    seed: int
    rounds: tuple[RoundData, ...]

    def price_series(self) -> list[float]:
        out = [START_PRICE]
        for r in self.rounds:
            out.extend(r.prices)
        return out


NEWS_BEFORE = {
    "Expansion": [
        "Hiring is strong and earnings are growing. Investors are in a buying mood.",
        "Consumer spending is healthy and the central bank sees no reason to panic.",
        "Corporate profits are beating forecasts. Optimism is high -- sometimes too high.",
    ],
    "Recession": [
        "Factories are cutting shifts and layoffs are rising. Credit is tightening.",
        "Surveys point to a contraction. Volatility is climbing as investors head for the exits.",
        "Profit warnings are piling up. Nobody is sure how deep this goes.",
    ],
    "Recovery": [
        "Rate cuts and stimulus spending are working their way through the economy.",
        "Early data surprise on the upside. Cheap money is fueling a rebound.",
        "Companies are rehiring and borrowing is cheap, but the rebound is still fragile.",
    ],
    "Inflation shock": [
        "Prices are surging and the central bank is raising rates. Borrowing just got expensive.",
        "An energy price spike is feeding inflation. Bond yields jump and stocks wobble.",
        "Wage and price pressures force the central bank to turn hawkish.",
    ],
    "Sideways": [
        "Nothing much is happening: growth is modest, inflation is tame, and investors are waiting for a catalyst.",
        "A quiet economy: no big news, no big fear. Markets drift while analysts argue about what comes next.",
        "Steady but unexciting. Earnings are fine, rates are stable, and the headlines are boring.",
    ],
}
TRANSITION_NEWS = {
    ("Recession", "Recovery"): "The central bank slashed rates and the government launched a stimulus package. The worst may be over.",
    ("Expansion", "Recession"): "The boom has cooled fast: credit conditions tightened and profits are slipping.",
    ("Expansion", "Inflation shock"): "Overheating economy: inflation jumps and the central bank pivots to rate hikes.",
    ("Inflation shock", "Recession"): "Rate hikes bite: growth stalls as high borrowing costs squeeze companies.",
    ("Inflation shock", "Expansion"): "Inflation cools without a crash -- a soft landing. Markets cheer.",
    ("Recovery", "Expansion"): "The rebound has matured into a broad expansion.",
    ("Expansion", "Sideways"): "The boom has lost steam. Growth is positive but unremarkable.",
    ("Recession", "Sideways"): "The slump has stopped, but there is no rebound yet. The economy is just... flat.",
    ("Sideways", "Expansion"): "A catalyst at last: strong data and falling uncertainty kick off an expansion.",
    ("Sideways", "Recession"): "The calm breaks: layoffs and weak data tip the economy into contraction.",
}
NEWS_AFTER = {
    "crash": [
        "MARKET CRASH: a sudden sell-off wiped out months of gains in days.",
        "SELL-OFF DEEPENS: panic selling pushed indices sharply lower, and they stayed there.",
    ],
    "rebound": [
        "V-SHAPED ROLLERCOASTER: stocks plunged in a panic, then roared back to finish the round sharply higher.",
        "WILD RIDE: a sell-off scare reversed hard, and buyers pushed the market well above where it started.",
    ],
    "scare": [
        "PANIC SELLING: indices plunged amid a liquidity scare before steadying.",
        "FLASH CRASH: stocks dropped sharply mid-round, then clawed most of it back.",
    ],
    "bust": [
        "Stocks slid as weak data and shaky earnings weighed on sentiment.",
        "A grinding sell-off: bad news kept coming and buyers stayed away.",
    ],
    "flat": [
        "Stocks went nowhere: good news and bad news cancelled out.",
        "A sideways market: investors waited for clearer signals.",
    ],
    "up": [
        "Stocks climbed steadily on solid earnings and resilient demand.",
        "A healthy advance: earnings growth beat expectations.",
    ],
    "boom": [
        "RALLY: stocks surged as earnings growth far exceeded forecasts.",
        "BREAKOUT YEAR: investors piled in and the market ripped higher.",
    ],
}


def _bucket(ret: float, worst_month: float) -> str:
    if ret < -0.20 or (worst_month < -0.10 and ret < 0):
        return "crash"
    if worst_month < -0.10:      # a plunge that was recovered by the end of the round
        return "rebound" if ret >= 0.10 else "scare"
    if ret < -0.04:
        return "bust"
    if ret < 0.04:
        return "flat"
    if ret < 0.15:
        return "up"
    return "boom"


def generate_market(seed: int) -> Market:
    """Pre-generate the entire path. Draw order is fixed, so same seed => same market."""
    rng = np.random.default_rng(seed)
    state = rng.choice(len(REGIME_ORDER), p=INITIAL_REGIME_PROBS)
    prev_name: str | None = None
    price = START_PRICE
    rounds = []
    for i in range(N_ROUNDS):
        if i > 0:
            state = rng.choice(len(REGIME_ORDER), p=TRANSITIONS[state])
        name = REGIME_ORDER[state]
        rets = sample_monthly_returns(REGIMES[name], rng, MONTHS_PER_ROUND)
        pre_pick = int(rng.integers(3))
        post_pick = int(rng.integers(2))
        prices = price * np.cumprod(1.0 + rets)
        bucket = _bucket(prices[-1] / price - 1.0, float(rets.min()))
        before = TRANSITION_NEWS.get((prev_name, name), NEWS_BEFORE[name][pre_pick])
        rounds.append(
            RoundData(
                index=i,
                regime=name,
                returns=tuple(float(x) for x in rets),
                start_price=float(price),
                prices=tuple(float(x) for x in prices),
                news_before=before,
                news_after=NEWS_AFTER[bucket][post_pick],
            )
        )
        price = float(prices[-1])
        prev_name = name
    return Market(seed=seed, rounds=tuple(rounds))


def horizon_return_samples(regime: Regime, n: int, seed: int = 7) -> np.ndarray:
    """Simulated distribution of a round's stock return (for the outlook chart)."""
    rng = np.random.default_rng(seed)
    r = sample_monthly_returns(regime, rng, (n, MONTHS_PER_ROUND))
    return np.prod(1.0 + r, axis=1) - 1.0
