"""Margin-account mechanics. Pure functions, no Streamlit, no randomness.

Conventions (all documented in the README):
* Interest is simple: principal * rate * months/12, accrued monthly and carried as a
  separate liability. Liabilities = principal + accrued interest.
* Margin ratio = equity / market value of stock (the "E/A" of the case).
* Maintenance is tested at every month-end (after the price move and interest accrual).
* A margin call forces the broker to sell shares until the ratio is back to
  RESTORE_MARGIN (a small buffer above maintenance, so the account is not
  re-called on the very next small dip). Sale proceeds repay accrued interest first,
  then principal. A small fee is charged on forced sales.
* If equity <= 0 the whole position is sold; any remaining deficit is reported as a
  shortfall (negative cash) and the game ends.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

INITIAL_EQUITY = 1_000_000.0
MAINTENANCE_MARGIN = 0.30
RESTORE_MARGIN = 0.35
MIN_INITIAL_MARGIN = 0.50          # => maximum 2x leverage
LIQUIDATION_FEE = 0.001            # 0.1% of the value of forced sales

EPS = 1e-9


@dataclass(frozen=True)
class Account:
    shares: float = 0.0
    cash: float = 0.0
    principal: float = 0.0
    accrued: float = 0.0

    @property
    def liabilities(self) -> float:
        return self.principal + self.accrued

    def stock_value(self, price: float) -> float:
        return self.shares * price

    def equity(self, price: float) -> float:
        return self.shares * price + self.cash - self.liabilities

    def margin_ratio(self, price: float) -> float | None:
        v = self.stock_value(price)
        return None if v <= 0 else self.equity(price) / v


@dataclass(frozen=True)
class MarginCallEvent:
    month: int
    price: float
    stock_value: float
    liabilities: float
    equity: float
    ratio: float
    shares_sold: float
    proceeds: float
    fee: float
    full_liquidation: bool
    shortfall: float
    ratio_after: float | None


def leverage(initial_margin: float) -> float:
    return 1.0 / initial_margin


def margin_call_price(liabilities: float, shares: float, maintenance: float = MAINTENANCE_MARGIN) -> float | None:
    """Price at which equity/stock value hits `maintenance`:  L / (N * (1 - m))."""
    if shares <= 0 or liabilities <= 0:
        return None
    return liabilities / (shares * (1.0 - maintenance))


def rebalance(acct: Account, price: float, initial_margin: float) -> Account:
    """Reset the account to the chosen initial margin using *current* equity only.

    No new money is ever added. position = E / m, loan = position - E, so the
    leverage cap (1 / MIN_INITIAL_MARGIN) can never be exceeded.
    Accrued interest is settled (it is already netted out of equity).
    """
    if initial_margin < MIN_INITIAL_MARGIN - EPS or initial_margin > 1.0 + EPS:
        raise ValueError("initial margin must be between 50% and 100%")
    equity = acct.equity(price)
    if equity <= 0:
        raise ValueError("cannot rebalance an account with non-positive equity")
    position = equity / initial_margin
    return Account(shares=position / price, cash=0.0, principal=position - equity, accrued=0.0)


def accrue_month(acct: Account, borrow_rate: float, rf_rate: float) -> tuple[Account, float]:
    """One month of interest. Returns (new account, interest charged this month)."""
    interest = acct.principal * borrow_rate / 12.0
    cash = acct.cash * (1.0 + rf_rate / 12.0) if acct.cash > 0 else acct.cash
    return replace(acct, accrued=acct.accrued + interest, cash=cash), interest


def check_margin(acct: Account, price: float, month: int = 0) -> tuple[Account, MarginCallEvent | None]:
    """Test maintenance margin at `price`; liquidate if breached."""
    v = acct.stock_value(price)
    if acct.liabilities <= 0 or v <= 0:
        return acct, None
    equity = acct.equity(price)
    ratio = equity / v
    if ratio >= MAINTENANCE_MARGIN - EPS:
        return acct, None

    fee = LIQUIDATION_FEE
    target = RESTORE_MARGIN
    # Sell value s so that (E - s*fee) / (V - s) = target  =>  s = (target*V - E) / (target - fee)
    sell_value = v if equity <= 0 else min(v, (target * v - equity) / (target - fee))
    full = sell_value >= v - EPS
    if full:
        sell_value = v
    shares_sold = sell_value / price
    fee_paid = sell_value * fee
    proceeds = sell_value - fee_paid

    if full:
        remaining = proceeds - acct.liabilities     # may be negative => shortfall
        new = Account(shares=0.0, cash=acct.cash + remaining, principal=0.0, accrued=0.0)
        shortfall = max(0.0, -new.equity(price))
        ratio_after = None
    else:
        pay_interest = min(proceeds, acct.accrued)
        pay_principal = proceeds - pay_interest
        new = Account(
            shares=acct.shares - shares_sold,
            cash=acct.cash,
            principal=acct.principal - pay_principal,
            accrued=acct.accrued - pay_interest,
        )
        shortfall = 0.0
        ratio_after = new.margin_ratio(price)

    event = MarginCallEvent(
        month=month, price=price, stock_value=v, liabilities=acct.liabilities, equity=equity,
        ratio=ratio, shares_sold=shares_sold, proceeds=proceeds, fee=fee_paid,
        full_liquidation=full, shortfall=shortfall, ratio_after=ratio_after,
    )
    return new, event
