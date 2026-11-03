"""Leverage Lab -- The $1 Million Margin Challenge. A margin-trading game for FIN 4200."""
from __future__ import annotations

import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from market import (MONTHS_PER_ROUND, N_ROUNDS, REGIMES, START_PRICE, generate_market,
                    horizon_return_samples)
from portfolio import (INITIAL_EQUITY, MAINTENANCE_MARGIN, RESTORE_MARGIN, Account, leverage,
                       rebalance)
from badges import BADGES, Ctx, earned
from simulation import max_drawdown, playful_title, preview, simulate_round, thresholds, what_if

MARGINS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
BLUE, ORANGE, GRAY, RED, GREEN = "#2563eb", "#ea580c", "#6b7280", "#dc2626", "#16a34a"
ANIMATE = not os.environ.get("MARGIN_GAME_NO_ANIM")

st.set_page_config(page_title="Leverage Lab: The $1 Million Margin Challenge", page_icon="📉", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1400px;}
div.stButton > button[kind="primary"] {height: 3.8rem; font-size: 1.25rem; font-weight: 800; letter-spacing: .02em;}
.news {border-left: 6px solid #ea580c; background:#fff7ed; padding:.7rem 1rem; border-radius:6px; font-size:1.05rem;}
.regime {display:inline-block; padding:.25rem .8rem; border-radius:999px; color:white; font-weight:700;}
.callbox {border:3px solid #dc2626; background:#fef2f2; padding:.8rem 1.1rem; border-radius:8px; color:#7f1d1d;}
.callbox h3 {margin:0 0 .3rem 0; color:#dc2626;}
.tag {font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:#6b7280; font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def esc(s: str) -> str:
    """Streamlit markdown treats $...$ as LaTeX; escape dollar signs in prose that has several."""
    return s.replace("$", r"\$")


def pct(x: float, d: int = 1) -> str:
    return f"{x:+.{d}%}"


# ---- State ----------------------------------------------------------------
def new_game(seed: int, start: bool = True) -> None:
    ss = st.session_state
    ss.seed = int(seed)
    ss.market = generate_market(ss.seed)
    ss.r = 0
    ss.acct = Account(cash=INITIAL_EQUITY)
    ss.series = [INITIAL_EQUITY]
    ss.records = []
    ss.anim = False
    ss.logged, ss.prior_final = False, None
    for k in [k for k in ss if str(k).startswith(("rev_", "guess_"))]:
        del ss[k]
    ss.phase = "decide" if start else "intro"


def fresh_seed() -> int:
    return int(np.random.default_rng().integers(1, 999_999))


if "phase" not in st.session_state:
    st.session_state.phase = "intro"
    st.session_state.seed = fresh_seed()
    st.session_state.past_finals = {}     # market code -> your last final wealth (this browser session only)

ss = st.session_state


@st.cache_data(show_spinner=False)
def cached_preview(regime: str, margin: float, equity: float):
    return preview(regime, margin, equity)


@st.cache_data(show_spinner=False)
def cached_samples(regime: str) -> np.ndarray:
    return horizon_return_samples(REGIMES[regime], 20000)


def months_label(r: int) -> str:
    return f"Months {r * MONTHS_PER_ROUND + 1}–{(r + 1) * MONTHS_PER_ROUND}"


# ---- Charts ---------------------------------------------------------------
def style(fig: go.Figure, h: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
                      font=dict(size=13))
    return fig


def dist_chart(regime: str, margin: float, pv) -> go.Figure:
    bins = np.linspace(-0.8, 1.2, 81)
    mid = (bins[:-1] + bins[1:]) / 2 * 100
    fig = go.Figure()
    d, _ = np.histogram(np.clip(cached_samples(regime), -0.8, 1.2), bins=bins, density=True)
    fig.add_trace(go.Scatter(x=mid, y=d, name="Stock return", fill="tozeroy", line=dict(color=BLUE, width=2),
                             fillcolor="rgba(37,99,235,.18)", hovertemplate="%{x:.0f}%"))
    if margin < 1.0:
        e, _ = np.histogram(np.clip(pv.equity_returns, -0.8, 1.2), bins=bins, density=True)
        fig.add_trace(go.Scatter(x=mid, y=e, name=f"Your equity return ({1 / margin:.2f}×)",
                                 line=dict(color=ORANGE, width=3), hovertemplate="%{x:.0f}%"))
    fig.add_vline(x=0, line=dict(color=GRAY, dash="dot"))
    fig.update_xaxes(title=f"Return over the next {MONTHS_PER_ROUND} months (%)", range=[-80, 120])
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Possible outcomes (simulated, not a forecast)")
    return style(fig, 300)


def round_chart(rd, res, upto: int) -> go.Figure:
    xs = list(range(upto + 1))
    prices = [rd.start_price] + [m.price for m in res.months[:upto]]
    ths = [m.threshold for m in res.months[:upto]]
    ys_all = [rd.start_price] + [m.price for m in res.months] + [t for m in res.months if (t := m.threshold)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=prices, name="Stock price", mode="lines+markers",
                             line=dict(color=BLUE, width=3), marker=dict(size=9),
                             hovertemplate="$%{y:.2f}"))
    tx = [i + 1 for i, t in enumerate(ths) if t]
    ty = [t for t in ths if t]
    if ty:
        fig.add_trace(go.Scatter(x=tx, y=ty, name="Margin-call price", mode="lines+markers",
                                 line=dict(color=RED, dash="dash", width=2), marker=dict(size=7),
                                 hovertemplate="$%{y:.2f}"))
    cx = [m.month for m in res.months[:upto] if m.called]
    if cx:
        fig.add_trace(go.Scatter(x=cx, y=[res.months[c - 1].price for c in cx], name="Margin call",
                                 mode="markers", marker=dict(symbol="x", size=18, color=RED,
                                                             line=dict(width=3, color="white")),
                                 hovertemplate="MARGIN CALL at $%{y:.2f}"))
    fig.update_xaxes(title="Month of this round", dtick=1, range=[-0.2, MONTHS_PER_ROUND + 0.2])
    fig.update_yaxes(title="Price ($)", range=[min(ys_all) * 0.93, max(ys_all) * 1.07])
    fig.update_layout(title=f"Stock price, month by month ({months_label(rd.index)})")
    return style(fig, 340)


# ---- Screens --------------------------------------------------------------
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📉 Leverage Lab")
        if ss.phase != "intro":
            st.caption(f"Market code: **{ss.seed}**  \nShare it so classmates can play the same market.")
            st.divider()
            acct, r = ss.acct, min(ss.r, N_ROUNDS - 1)
            p = ss.market.rounds[r].start_price if ss.phase == "decide" else ss.market.rounds[r].end_price
            st.metric("Your equity", money(acct.equity(p)), pct(acct.equity(p) / INITIAL_EQUITY - 1))
            st.metric("Stock price", f"${p:,.2f}")
            st.metric("Loan (principal + interest)", money(acct.liabilities))
            st.divider()
        with st.expander("Rules in 5 lines"):
            st.markdown(esc(
                f"- Start with **{money(INITIAL_EQUITY)}**, stock at **${START_PRICE:.0f}**.\n"
                f"- Each round you pick your **initial margin** (50%–100%, i.e. up to 2×). No new money.\n"
                f"- Your broker checks **every month-end**. If equity/stock value < **{MAINTENANCE_MARGIN:.0%}** → margin call.\n"
                f"- Forced sales restore the ratio to {RESTORE_MARGIN:.0%} (0.1% fee). Interest accrues monthly.\n"
                "- Everything is simulated. Expected returns are *not* guaranteed returns."
            ))
        if ss.phase != "intro" and st.button("↺ Restart (new market)", width="stretch"):
            new_game(fresh_seed())
            st.rerun()


def intro() -> None:
    st.title("📉 Leverage Lab")
    st.subheader("The $1 Million Margin Challenge")
    st.caption("Same market. Your leverage. Your fate.")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(
            f"""
You manage **$1,000,000** of your own money for **{N_ROUNDS} rounds** ({N_ROUNDS * MONTHS_PER_ROUND // 12} simulated years).
Before each round you read the **economic outlook**, then choose how much to **borrow on margin**.

* Borrow more → bigger gains **and** bigger losses.
* Your broker watches the account **every month**. Fall below **30% equity** and you get a **margin call**.
* The market may recover by December… but you may be **sold out in March**.

There is no secret winning strategy. This isn't graded, so take risks, then play again.
"""
        )
        code = st.text_input("Market code (optional – to play the same market as a classmate)", value="",
                             placeholder="leave blank for a new random market")
        if st.button("▶ START THE GAME", type="primary", width="stretch"):
            seed = int(code) if code.strip().isdigit() else fresh_seed()
            new_game(seed)
            st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("**The five economies you might meet**")
            for r in REGIMES.values():
                st.markdown(
                    f"{r.icon} **{r.name}** – expected return {r.mu:+.0%}, volatility {r.sigma:.0%}, "
                    f"loan rate {r.borrow:.0%}"
                )
            st.caption("Notice: in some economies borrowing costs *more* than the stock is expected to earn.")


def decide() -> None:
    r = ss.r
    rd = ss.market.rounds[r]
    reg = REGIMES[rd.regime]
    price = rd.start_price
    equity = ss.acct.equity(price)

    h1, h2 = st.columns([3, 1])
    h1.markdown(f"## Round {r + 1} of {N_ROUNDS} &nbsp;·&nbsp; <span style='color:{GRAY}'>{months_label(r)}</span>",
                unsafe_allow_html=True)
    h2.markdown(f"<div style='text-align:right'><span class='tag'>Your equity</span><br>"
                f"<span style='font-size:1.8rem;font-weight:800'>{money(equity)}</span></div>", unsafe_allow_html=True)
    st.progress(r / N_ROUNDS)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("<span class='tag'>Known before you invest</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='regime' style='background:{reg.color}'>{reg.icon} {reg.name}</span>"
                        f" &nbsp; Market mood: **{reg.sentiment}**", unsafe_allow_html=True)
            st.markdown(f"<div class='news'>📰 {rd.news_before}</div>", unsafe_allow_html=True)
            m = st.columns(4)
            m[0].metric("Expected return / yr", f"{reg.mu:+.0%}")
            m[1].metric("Volatility / yr", f"{reg.sigma:.0%}")
            m[2].metric("Margin loan rate", f"{reg.borrow:.0%}")
            m[3].metric("Risk-free rate", f"{reg.rf:.1%}")
            st.caption("Expected ≠ guaranteed. Volatility means a typical year lands anywhere from "
                       f"{reg.mu - reg.sigma:+.0%} to {reg.mu + reg.sigma:+.0%}, and worse is possible.")
        chart_slot = st.empty()

    with right:
        with st.container(border=True):
            st.markdown("<span class='tag'>Your decision</span>", unsafe_allow_html=True)
            margin = st.select_slider("Initial margin (equity as % of your stock position)",
                                      options=MARGINS, value=ss.get("last_margin", 0.5),
                                      format_func=lambda m: f"{m:.0%}", key=f"margin_{r}")
            ss.last_margin = margin
            lev = leverage(margin)
            trial = rebalance(ss.acct, price, margin)
            pv = cached_preview(rd.regime, margin, round(equity, 0))
            st.markdown(f"**{lev:.2f}× leverage**" + (" – no borrowing, no margin call possible" if margin == 1 else ""))
            a = st.columns(3)
            a[0].metric("Stock position", money(trial.shares * price))
            a[1].metric("Borrowed", money(trial.principal))
            a[2].metric("Your equity", money(equity))
            b = st.columns(3)
            b[0].metric("Interest this round", money(trial.principal * reg.borrow * MONTHS_PER_ROUND / 12))
            ann = (1 + pv.mean_equity_return) ** (12 / MONTHS_PER_ROUND) - 1
            b[1].metric(f"Expected equity return ({MONTHS_PER_ROUND} mo)", pct(pv.mean_equity_return),
                        f"{ann:+.0%} a year", delta_color="off",
                        help="Average over 1,500 simulated rounds in this economy; the small number under it is the same return annualized")
            b[2].metric("Chance of losing >20%", f"{pv.prob_loss_20:.0%}")
            st.metric("Estimated chance of a margin call this round", f"{pv.prob_margin_call:.0%}")

        guess = None
        if margin < 1.0:
            with st.container(border=True):
                st.markdown("<span class='tag'>Challenge (optional, ungraded)</span>", unsafe_allow_html=True)
                st.markdown(f"**What stock price do you think will trigger a margin call?** "
                            f"(Today: ${price:,.2f}, maintenance margin {MAINTENANCE_MARGIN:.0%})")
                g1, g2 = st.columns([2, 1])
                guess = g1.number_input("My guess ($/share)", min_value=0.0, value=None, step=1.0,
                                        placeholder="e.g. 70", key=f"guess_{r}", label_visibility="collapsed")
                if g2.button("Reveal", key=f"btn_rev_{r}", width="stretch"):
                    ss[f"rev_{r}"] = True
                if ss.get(f"rev_{r}"):
                    now, end = thresholds(trial, price, reg.borrow)
                    if guess:
                        st.markdown(esc(f"You guessed **${guess:,.2f}**; the answer is **${now:,.2f}** "
                                        f"({'off by ' + f'${abs(guess - now):,.2f}' if abs(guess - now) > 0.005 else 'spot on!'})."))
                    c = st.columns(2)
                    c[0].metric("Margin-call price today", f"${now:,.2f}", f"{now / price - 1:+.1%} from ${price:,.2f}", delta_color="off")
                    c[1].metric(f"…after {MONTHS_PER_ROUND} months of interest", f"${end:,.2f}",
                                f"{end / price - 1:+.1%}", delta_color="off")
                    st.caption("Interest accrues on the loan, so the trigger price creeps **up** over time.")
                    with st.expander("Show the math"):
                        st.latex(r"\frac{\text{Equity}}{\text{Stock value}}=\frac{N\,P-L}{N\,P}=m"
                                 r"\;\Rightarrow\;P^{*}=\frac{L}{N\,(1-m)}")
                        st.markdown(esc(f"N = {trial.shares:,.0f} shares, L = {money(trial.liabilities)}, m = {MAINTENANCE_MARGIN:.0%}  \n"
                                        f"Today: {money(trial.liabilities)} / ({trial.shares:,.0f} × {1 - MAINTENANCE_MARGIN:.2f}) = **${now:,.2f}**  \n"
                                        f"After interest, L grows to {money(trial.liabilities + trial.principal * reg.borrow * MONTHS_PER_ROUND / 12)}"
                                        f" → **${end:,.2f}**"))

        if st.button("NEXT ROUND — REVEAL WHAT HAPPENS! 🎲", type="primary", width="stretch"):
            acct = rebalance(ss.acct, price, margin)
            res = simulate_round(acct, price, rd.returns, reg.borrow, reg.rf)
            answer = thresholds(acct, price, reg.borrow)[0]
            ss.records.append(dict(round=r + 1, regime=rd.regime, margin=margin, lev=lev, start_equity=equity,
                                   borrowed=acct.principal, rd=rd, res=res, stock_ret=rd.stock_return,
                                   guess=guess, answer=answer))
            ss.acct = res.account
            ss.series.extend(m.equity for m in res.months)
            ss.phase, ss.anim = "result", True
            st.rerun()

    chart_slot.plotly_chart(dist_chart(rd.regime, margin, pv), width="stretch")


def result() -> None:
    rec = ss.records[-1]
    rd, res = rec["rd"], rec["res"]
    reg = REGIMES[rd.regime]
    r = rd.index
    st.markdown(f"## Round {r + 1} of {N_ROUNDS} results &nbsp;·&nbsp; <span style='color:{GRAY}'>"
                f"{months_label(r)}</span>", unsafe_allow_html=True)
    st.progress((r + 1) / N_ROUNDS)
    st.markdown("<span class='tag'>Revealed after the market moved</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='news'>📰 {rd.news_after}</div>", unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        slot = st.empty()
        if ss.anim and ANIMATE:
            for k in range(0, MONTHS_PER_ROUND + 1):
                slot.plotly_chart(round_chart(rd, res, k), width="stretch")
                time.sleep(0.45)
        slot.plotly_chart(round_chart(rd, res, MONTHS_PER_ROUND), width="stretch", key=f"final_chart_{r}")
        ss.anim = False
    end_acct = res.account
    wiped = res.end_equity <= 0
    with right:
        a = st.columns(2)
        a[0].metric("Stock return", pct(rd.stock_return), f"${rd.start_price:,.2f} → ${rd.end_price:,.2f}", delta_color="off")
        a[1].metric(f"Your return ({rec['lev']:.2f}×)", pct(res.equity_return))
        b = st.columns(2)
        b[0].metric("Portfolio value", money(end_acct.shares * rd.end_price + end_acct.cash))
        b[1].metric("Debt outstanding", money(end_acct.liabilities))
        c = st.columns(2)
        c[0].metric("Interest expense", money(res.interest))
        c[1].metric("Ending equity", money(res.end_equity), money(res.end_equity - res.start_equity))
        if rec["lev"] > 1:
            unlev = rec["stock_ret"]
            st.caption(f"Without borrowing you would have made {pct(unlev)} on the stock "
                       f"(before cash/risk-free effects). Leverage {'helped' if res.equity_return > unlev else 'hurt'} this round.")

    if res.events:
        ev = res.events[0]
        sold = sum(e.shares_sold for e in res.events)
        st.markdown(
            f"<div class='callbox'><h3>🚨 MARGIN CALL{'S' if len(res.events) > 1 else ''}! "
            f"({len(res.events)} in this round)</h3>"
            f"The stock price fell below your maintenance threshold in month {ev.month}. "
            f"Your broker sold <b>{sold:,.0f} shares</b> to reduce your loan"
            f"{' – the whole position' if any(e.full_liquidation for e in res.events) else ''}.</div>",
            unsafe_allow_html=True)
        if not ss.get(f"toast_{r}"):
            st.toast("Margin call! Your broker sold your shares.", icon="🚨")
            ss[f"toast_{r}"] = True
        for e in res.events:
            with st.expander(f"How the month-{e.month} margin call worked"):
                st.markdown(esc(
                    f"| Stock price | Stock value (assets) | Liabilities | Equity | Equity ÷ assets |\n|---|---|---|---|---|\n"
                    f"| ${e.price:,.2f} | {money(e.stock_value)} | {money(e.liabilities)} | {money(e.equity)} | "
                    f"**{e.ratio:.1%}** (< {MAINTENANCE_MARGIN:.0%}) |"))
                if e.full_liquidation:
                    st.markdown(esc(f"Equity was too low to restore the margin, so **all {e.shares_sold:,.0f} shares** were sold "
                                    f"for {money(e.proceeds)} (after a {money(e.fee)} fee) to repay {money(e.liabilities)}."))
                else:
                    st.markdown(esc(f"The broker sold **{e.shares_sold:,.0f} shares** (${e.shares_sold * e.price:,.0f}), "
                                    f"paid a {money(e.fee)} fee, and used {money(e.proceeds)} to repay interest first, then principal. "
                                    f"Equity ÷ assets is back to **{e.ratio_after:.1%}** (target {RESTORE_MARGIN:.0%})."))
                    st.caption("Shares sold: (target × V − E) ÷ (target − fee) ÷ price")
                if e.shortfall > 0:
                    st.error(f"The price fell so far between month-ends that your account went negative. "
                             f"Shortfall owed to the broker: {money(e.shortfall)}.")
        if res.months[-1].price > rd.start_price * 0.999 and rec["lev"] > 1:
            st.info("Notice the lesson: the stock may have recovered by the end of the round, "
                    "but a forced sale locks in the loss. Recoveries only help the shares you still own.")

    if wiped:
        st.error(f"Your account is wiped out. Final equity: {money(res.end_equity)}. The game ends here.")
    last = wiped or r == N_ROUNDS - 1
    if st.button("🏁 SEE FINAL RESULTS" if last else f"Continue to round {r + 2} →", type="primary", width="stretch"):
        if last:
            ss.phase = "done"
        else:
            ss.r += 1
            ss.phase = "decide"
        st.rerun()


def done() -> None:
    mkt, recs = ss.market, ss.records
    series = ss.series
    final = series[-1]
    n_calls = sum(len(x["res"].events) for x in recs)
    interest = sum(x["res"].interest for x in recs)
    avg_lev = float(np.mean([x["lev"] for x in recs]))
    dd = max_drawdown(series)
    prices = mkt.price_series()
    bench = [INITIAL_EQUITY * p / START_PRICE for p in prices]
    title, blurb = playful_title(final, n_calls, avg_lev, dd, bench[-1])

    st.markdown(f"## {title}")
    st.caption(f"{blurb} (just for fun – not an assessment)")
    m = st.columns(6)
    m[0].metric("Starting wealth", money(INITIAL_EQUITY))
    m[1].metric("Final wealth", money(final), pct(final / INITIAL_EQUITY - 1))
    m[2].metric("No-leverage benchmark", money(bench[-1]), pct(bench[-1] / INITIAL_EQUITY - 1), delta_color="off",
                help="Buy $1M of stock on day 1 and hold, using this same market.")
    m[3].metric("Max drawdown", f"{dd:.0%}")
    m[4].metric("Margin calls", n_calls)
    m[5].metric("Total borrowing cost", money(interest))
    if final < 0:
        st.error(f"Your account ended with a shortfall of {money(-final)}: you owe this to the broker.")

    alts = what_if(mkt)
    rule_final = alts["Smart rule"]["final"]
    if not ss.logged:                      # once per finished game (reruns must not repeat this)
        ss.prior_final = ss.past_finals.get(ss.seed)
        ss.past_finals[ss.seed] = final
        ss.logged = True
        if title.startswith(("Hello, Mr. Buffett", "Leverage Wizard")):
            st.balloons()

    c1, c2 = st.columns(2)
    x_all = list(range(len(prices)))
    f1 = go.Figure()
    alt_style = {"Never borrow (1×)": (GRAY, "dash"), "Always max (2×)": ("#7c3aed", "dot"),
                 "Smart rule": ("#0891b2", "dashdot")}
    for name, g in alts.items():
        col, dash = alt_style[name]
        f1.add_trace(go.Scatter(x=list(range(len(g["series"]))), y=g["series"], line=dict(color=col, width=2, dash=dash),
                                name=f"{name}: {money(g['final'])}"))
    f1.add_trace(go.Scatter(x=list(range(len(series))), y=series, name=f"YOU: {money(final)}",
                            line=dict(color=ORANGE, width=4)))
    for x in recs:
        for e in x["res"].events:
            mo = x["rd"].index * MONTHS_PER_ROUND + e.month
            f1.add_trace(go.Scatter(x=[mo], y=[series[mo]], mode="markers", showlegend=False,
                                    marker=dict(symbol="x", size=14, color=RED, line=dict(width=2, color="white")),
                                    hovertemplate="Margin call"))
    f1.add_hline(y=INITIAL_EQUITY, line=dict(color="#d1d5db", dash="dot"))
    f1.update_xaxes(title="Month")
    f1.update_yaxes(title="Equity ($)", tickformat="$,.0f")
    c1.plotly_chart(style(f1), width="stretch")
    better = [n for n, g in alts.items() if g["final"] > final]
    if not better:
        c1.success("You beat every strategy on this market. Can you do it again on a new one?")
    else:
        c1.info(f"{len(better)} of 3 strategies beat you on this exact market. "
                "Hit **Replay the SAME market** and try to beat them. Their leverage is the only difference.")
    c1.caption("Smart rule: 2× in expansion/recovery, no leverage in recession, inflation shock and sideways "
               "(borrow only when the stock's expected return clearly beats the loan rate).")

    f2 = go.Figure(go.Scatter(x=x_all, y=prices, name="Stock price", line=dict(color=BLUE, width=3), hovertemplate="$%{y:.2f}"))
    for rd in mkt.rounds:
        f2.add_vrect(x0=rd.index * MONTHS_PER_ROUND, x1=(rd.index + 1) * MONTHS_PER_ROUND,
                     fillcolor=REGIMES[rd.regime].color, opacity=0.10, line_width=0)
    f2.update_xaxes(title="Month (shading = economy: green expansion, red recession, teal recovery, amber inflation, gray sideways)")
    f2.update_yaxes(title="Price ($)")
    f2.update_layout(title="The stock you traded")
    c2.plotly_chart(style(f2), width="stretch")

    import pandas as pd

    # Ex-ante: average of the announced outlooks over the rounds you played.
    # A position with leverage L earns ~ L*mu - (L-1)*borrow with volatility L*sigma.
    regs = [REGIMES[x["regime"]] for x in recs]
    levs = [x["lev"] for x in recs]
    exp_stock = np.mean([g.mu for g in regs])
    vol_stock = np.mean([g.sigma for g in regs])
    exp_you = np.mean([L * g.mu - (L - 1) * g.borrow for L, g in zip(levs, regs)])
    vol_you = np.mean([L * g.sigma for L, g in zip(levs, regs)])

    def realized(vals):
        v = np.asarray(vals, dtype=float)
        ok = v[:-1] > 0
        r = (v[1:] / np.where(ok, v[:-1], 1) - 1)[ok]
        return (r.mean() * 12, r.std(ddof=1) * np.sqrt(12)) if len(r) > 2 else (np.nan, np.nan)

    played = len(series)
    rs, vs = realized(prices[:played])
    ry, vy = realized(series)
    rf_avg = np.mean([g.rf for g in regs])

    def sharpe(ret, vol):
        return f"{(ret - rf_avg) / vol:.2f}" if vol and np.isfinite(vol) and np.isfinite(ret) else "n/a"

    st.markdown("##### Risk and return: expected vs. what happened")
    st.dataframe(pd.DataFrame({
        "": ["Stock, no leverage", "Your position"],
        "Expected return / yr": [f"{exp_stock:.1%}", f"{exp_you:.1%}"],
        "Expected volatility / yr": [f"{vol_stock:.1%}", f"{vol_you:.1%}"],
        "Expected Sharpe": [sharpe(exp_stock, vol_stock), sharpe(exp_you, vol_you)],
        "Realized return / yr": [f"{rs:.1%}", f"{ry:.1%}"],
        "Realized volatility / yr": [f"{vs:.1%}", f"{vy:.1%}"],
        "Realized Sharpe": [sharpe(rs, vs), sharpe(ry, vy)],
    }), hide_index=True, width="stretch")
    st.caption("Expected = average of the outlooks you were shown (your leverage as chosen each round; "
               "loan cost included). Realized = annualized from this game's monthly results. "
               f"Sharpe = (return − risk-free rate) ÷ volatility, using the average risk-free rate in the outlooks ({rf_avg:.1%}).")

    got = earned(Ctx(recs=recs, final=final, interest=interest, rule_final=rule_final, prior_final=ss.prior_final))
    st.markdown(f"##### 🏅 Your badges ({len(got)} of {len(BADGES)})")
    if got:
        cols = st.columns(4)
        for i, b in enumerate(got):
            with cols[i % 4].container(border=True):
                st.markdown(f"### {b.emoji}")
                st.markdown(f"**{b.name}**")
                st.caption(esc(b.desc))
    else:
        st.write("No badges this time. Your play style was hard to categorize.")
    st.caption(f"{len(BADGES) - len(got)} badges still locked. Replay with a different style to find them.")

    df = pd.DataFrame([{
        "Round": x["round"], "Economy": f"{REGIMES[x['regime']].icon} {x['regime']}",
        "Margin": f"{x['margin']:.0%}", "Leverage": f"{x['lev']:.2f}×",
        "Stock return": pct(x["stock_ret"]), "Your return": pct(x["res"].equity_return),
        "Margin calls": len(x["res"].events), "Interest": money(x["res"].interest),
        "Ending equity": money(x["res"].end_equity)} for x in recs])
    with st.expander("Your decisions round by round", expanded=False):
        st.dataframe(df, hide_index=True, width="stretch")

    b1, b2 = st.columns(2)
    if b1.button("🔁 Replay the SAME market (try a different strategy)", type="primary", width="stretch"):
        new_game(ss.seed)
        st.rerun()
    if b2.button("🎲 Play a NEW market", width="stretch"):
        new_game(fresh_seed())
        st.rerun()
    st.caption(f"Market code {ss.seed}: classmates can type it on the start screen to play exactly this market.")


sidebar()
{"intro": intro, "decide": decide, "result": result, "done": done}[ss.phase]()
st.markdown("<div style='text-align:center;color:#6b7280;font-size:.85rem;margin-top:1.5rem'>"
            "FIN4200 Applied Finance Lab | University of Denver</div>", unsafe_allow_html=True)
