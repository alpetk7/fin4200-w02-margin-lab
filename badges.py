"""End-of-game badges: entertainment only. Pure functions over the list of round records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from portfolio import INITIAL_EQUITY

BAD_ECONOMIES = ("Recession", "Inflation shock")


@dataclass(frozen=True)
class Ctx:
    recs: list              # one dict per round played (see app.py)
    final: float
    interest: float
    rule_final: float       # what the "smart rule" finished with on this market
    prior_final: float | None = None   # your last result on this same market, if any


@dataclass(frozen=True)
class Badge:
    emoji: str
    name: str
    desc: str
    check: Callable[[Ctx], bool]


def _lev_in(c: Ctx, pick) -> float | None:
    if len(c.recs) < 2:
        return None
    return pick(c.recs, key=lambda r: r["stock_ret"])["lev"]


def _timing(c: Ctx) -> float:
    """Leverage in the best round minus leverage in the worst round (positive = good timing)."""
    best, worst = _lev_in(c, max), _lev_in(c, min)
    return 0.0 if best is None else best - worst


BADGES: list[Badge] = [
    Badge("🎯", "Nostradamus of Margin", "Guessed a margin-call price within $1.",
          lambda c: any(r.get("guess") and r.get("answer") and abs(r["guess"] - r["answer"]) <= 1.0 for r in c.recs)),
    Badge("🧘", "Iron Stomach", "Held 1.5×+ leverage through a 15% drop inside a round and never got called.",
          lambda c: any(r["lev"] >= 1.5 and not r["res"].events
                        and min(m.price for m in r["res"].months) <= 0.85 * r["rd"].start_price for r in c.recs)),
    Badge("🛡️", "Teflon Trader", "Used 1.6×+ leverage in 4 or more rounds and never got a margin call.",
          lambda c: sum(r["lev"] >= 1.6 for r in c.recs) >= 4 and not any(r["res"].events for r in c.recs)),
    Badge("👶", "Sleeps Like a Baby", "Never borrowed a single dollar.",
          lambda c: len(c.recs) >= 5 and all(r["margin"] == 1.0 for r in c.recs)),
    Badge("🎰", "All In, Every Time", "Took maximum 2× leverage in every round.",
          lambda c: len(c.recs) >= 5 and all(r["margin"] == 0.5 for r in c.recs)),
    Badge("🪂", "Parachute Packer", "Skipped leverage in every recession or inflation round (at least two).",
          lambda c: (lambda b: len(b) >= 2 and all(r["margin"] == 1.0 for r in b))(
              [r for r in c.recs if r["regime"] in BAD_ECONOMIES])),
    Badge("🐂", "Bull in a China Shop", "Borrowed heavily (1.6×+) in a recession or inflation shock.",
          lambda c: any(r["regime"] in BAD_ECONOMIES and r["lev"] >= 1.6 for r in c.recs)),
    Badge("🔮", "Crystal Ball", "Used much more leverage in the best round than in the worst one.",
          lambda c: _timing(c) >= 0.5),
    Badge("🤡", "Perfectly Wrong", "Used much more leverage in the worst round than in the best one.",
          lambda c: _timing(c) <= -0.5),
    Badge("🦋", "Commitment Issues", "Changed your leverage in at least 7 rounds.",
          lambda c: sum(a["margin"] != b["margin"] for a, b in zip(c.recs, c.recs[1:])) >= 7),
    Badge("😭", "Sold the Bottom", "Got margin-called in a round where the stock ended up anyway.",
          lambda c: any(r["res"].events and r["rd"].end_price >= r["rd"].start_price for r in c.recs)),
    Badge("🧨", "Full Liquidation Club", "Your broker sold the entire position.",
          lambda c: any(e.full_liquidation for r in c.recs for e in r["res"].events)),
    Badge("🏦", "Bank's Favorite Student", "Paid $100,000+ in total loan interest.",
          lambda c: c.interest >= 100_000),
    Badge("💰", "Double or Nothing", "Doubled your money.",
          lambda c: c.final >= 2 * INITIAL_EQUITY),
    Badge("🤖", "Beat the Machine", "Finished ahead of the smart rule on this same market.",
          lambda c: c.final > c.rule_final),
    Badge("⏪", "Time Traveler", "Replayed the same market and did better the second time.",
          lambda c: c.prior_final is not None and c.final > c.prior_final + 1),
    Badge("🦫", "Groundhog Day", "Replayed the same market and ended no better off.",
          lambda c: c.prior_final is not None and c.final <= c.prior_final + 1),
]


def earned(c: Ctx) -> list[Badge]:
    return [b for b in BADGES if b.check(c)]
