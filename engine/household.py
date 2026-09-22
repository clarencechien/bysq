"""v4 §1 floor + upside family (S1-S4), with mortality, pension floors,
bond ladders, deferred annuities, spending smile, LTC shocks and behavioural
capitulation — the household-level simulator.

Units: total wealth at retirement = 1.0. Spending, income, ladder and
annuity amounts are REAL per-year fractions of that. Ladder cost, annuity
premium and the 勞退 account are carved out of the 1.0 at t=0; the rest is
the investable portfolio.

Income streams
  勞保老年年金  lifetime, nominal amount stepped up by the realised CPI each
               time cumulative CPI since the last step reaches +5% (勞保條例
               §65-4; never stepped down). Starts at lab_start_age (65).
  勞退月退休金  NOT a life annuity (v4 §0.2): the individual account is paid
               out over the remaining life expectancy at the claim age (勞保局
               table: 23y at 60), at the guaranteed rate, in NOMINAL terms,
               recomputed every 3 years; nothing after the term.
  ladder       S2: N rungs of the floor gap bought at t=0. kind='real' pays a
               CPI-indexed rung at an assumed real yield (TIPS-like; for TWD
               this is the 定存實質利率 assumption — hypothetical, flagged);
               kind='nominal' buys nominal zero-coupons at the GS10 yield of
               the path's first month with rungs grown at trailing-10y
               inflation, and the real payout erodes with realised CPI.
  annuity      S3: deferred life annuity bought at t=0, paying from
               ann_start_age until death; priced actuarially fair on the
               population table shifted +3y (annuitant selection) at
               ann_rate, times (1 + load). kind='nominal' (what Taiwan sells)
               or 'real' (hypothetical, no CPI-linked annuity exists in TWD).
Spending rule on the portfolio: 'VPW' (amortise to age 100 with horizon
extensions 90->110, 100->120 so the end-point cliff of §3.3 never bites),
'GK' (guardrails on the residual draw with the floor uncuttable) or 'FIX'.
Every rule tops the draw up so total spending never falls below the floor
while the portfolio lasts (the 'V-floor' of §3.1).
All decisions use information through t-1.
"""
import csv
import os
import numpy as np

from .mortality import load_qx, survival_curve, sample_death_years

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw")

LR_TERM = {60: 23, 61: 22, 62: 22, 63: 21, 64: 20, 65: 19}   # 勞保局 月退 平均餘命 (yrs)
LR_RATE = 0.011473                                            # 保證收益率 3y avg (勞保局)


def load_shiller_extra(dates):
    """GS10 yield (decimal) and CAPE aligned to a us_monthly date list."""
    import xlrd
    sh = xlrd.open_workbook(os.path.join(RAW, "shiller.xls")).sheet_by_name("Data")
    gs, cape = {}, {}
    for r in range(8, sh.nrows):
        d = sh.cell_value(r, 0)
        if not isinstance(d, float):
            continue
        y = int(d); m = int(round((d - y) * 100))
        if not 1 <= m <= 12:
            continue
        key = f"{y}-{m:02d}"
        g = sh.cell_value(r, 6); c = sh.cell_value(r, 12)
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan
        gs[key] = num(g) / 100
        cape[key] = num(c)
    return (np.array([gs.get(d, np.nan) for d in dates]),
            np.array([cape.get(d, np.nan) for d in dates]))


def trailing_inflation(panel, years=10):
    """Annualised CPI inflation over the trailing `years` ending at each row
    (uses rows <= t only)."""
    cpi = panel["cpi"]
    n = years * 12
    out = np.full(len(cpi), np.nan)
    out[n:] = (cpi[n:] / cpi[:-n]) ** (1 / years) - 1
    # early rows: use what is available (>= 5y) else 2%
    for t in range(60, n):
        out[t] = (cpi[t] / cpi[0]) ** (12 / t) - 1
    out[:60] = 0.02
    return out


class HH:
    """Household spec. See module doc."""
    def __init__(self, retire_age=65, sex="all", joint=False, spend=0.04,
                 floor_ratio=0.5, widow_factor=0.75, survivor_share=0.68,
                 pension=0.0, pension_mult=1.0, lab_start_age=65,
                 lr_account=0.0, lr_claim_age=60,
                 ladder_years=0, ladder_kind="real", ladder_real_rate=0.01,
                 ann_start_age=None, ann_load=0.2, ann_kind="nominal", ann_rate=None,
                 stock_w=1.0, rule="VPW", vpw_rate=0.03, gk_rate=None,
                 smile=None, behavior=None, fee=0.0,
                 ltc=None, horizon_fixed=None, longevity_shift=0):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        if ann_rate is None:
            self.ann_rate = 0.0175 if ann_kind == "nominal" else 0.01
        parts = []
        if pension:
            parts.append(f"P{pension*100:g}x{pension_mult:g}")
        if lr_account:
            parts.append(f"LR{lr_account*100:g}")
        if ladder_years:
            parts.append(f"L{ladder_years}{ladder_kind[0]}")
        if ann_start_age:
            parts.append(f"A{ann_start_age}{ann_kind[0]}{int(ann_load*100)}")
        parts.append(f"{int(stock_w*100)}/{int(round((1-stock_w)*100))}")
        parts.append(rule if rule != "VPW" else f"VPW{vpw_rate*100:g}")
        if smile:
            parts.append("smile")
        if behavior:
            parts.append(f"beh{behavior[2]:g}")
        self.name = " ".join(parts)


def annuity_factor(qx, age0, start_age, rate, shift=3, max_age=110):
    """PV at age0 of 1/yr paid annually from start_age for life (population
    table shifted by `shift` years to mimic annuitant selection)."""
    S = survival_curve(qx, age0, shift, max_age)
    k = np.arange(len(S))
    v = (1 + rate) ** -k
    return float((S * v)[k >= (start_age - age0)].sum())


def simulate_household(idx, panel, hh, death_m, cash_col="bill", gs10=None,
                       infl_e=None, qx=None, seed=0):
    """idx (P,T) row indices; death_m (P,) month of death (>= T -> survives
    the window) — for joint households pass a tuple (death_m1, death_m2).
    Returns per-path metrics + annual real spending matrix (units of W0)."""
    P, T = idx.shape
    years = T // 12
    fee_m = hh.fee / 12.0
    stock_r = panel["stock"][idx] - fee_m
    bond_r = panel["bond"][idx] - fee_m
    cash_r = panel[cash_col][idx] - fee_m
    infl_m = panel["infl"][idx]
    price = np.ones((P, T))
    lvl = np.ones(P)
    for t in range(1, T):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl
    rng = np.random.default_rng(20_000 + seed)
    if qx is None:
        qx = load_qx(114)
    qx_all = qx["all"]

    joint = isinstance(death_m, tuple)
    if joint:
        d1, d2 = death_m
        d_last = np.maximum(d1, d2)
    else:
        d1 = d2 = d_last = death_m

    age0 = hh.retire_age
    s0 = hh.spend
    floor0 = hh.floor_ratio * s0
    pension0 = hh.pension * hh.pension_mult                # real/yr, household

    # ---------------- t=0 carve-outs ----------------
    invest = np.ones(P)
    # 勞退 account (nominal, guaranteed rate), term from claim age
    lr_bal = np.full(P, hh.lr_account)
    invest -= hh.lr_account
    lr_term_m = LR_TERM.get(hh.lr_claim_age, 23) * 12
    lr_start_m = max(hh.lr_claim_age - age0, 0) * 12
    lr_pay = np.zeros(P)
    # ladder covering the floor gap not met by the lifetime pension
    gap = max(floor0 - (pension0 if age0 >= hh.lab_start_age else 0.0), 0.0)
    N = hh.ladder_years
    ladder_real = np.zeros((P, years))          # real payout schedule
    if N > 0 and gap > 0:
        if hh.ladder_kind == "real":
            r = hh.ladder_real_rate
            cost = gap * sum((1 + r) ** -k for k in range(1, N + 1))
            invest -= cost
            ladder_real[:, :N] = gap
            ladder_cost = np.full(P, cost)
        else:                                   # nominal zero-coupon ladder
            y0 = gs10[idx[:, 0]]
            pe = infl_e[idx[:, 0]]
            ks = np.arange(1, N + 1)
            nom = gap * (1 + pe[:, None]) ** ks[None, :]           # (P,N)
            cost = (nom * (1 + y0[:, None]) ** -ks[None, :]).sum(axis=1)
            invest -= cost
            ladder_cost = cost
            # real payout in year k = nominal / price level at start of year k
            # (price known only path-by-path as we go) -> store nominal, deflate later
            ladder_nom = np.zeros((P, years)); ladder_nom[:, :N] = nom
            ladder_real = None
    else:
        ladder_cost = np.zeros(P)
        ladder_nom = None
    # deferred annuity buying the floor gap from ann_start_age
    ann_real0 = 0.0
    ann_prem = 0.0
    if hh.ann_start_age:
        ann_real0 = gap if gap > 0 else floor0
        af = annuity_factor(qx_all, age0, hh.ann_start_age, hh.ann_rate, shift=3)
        ann_prem = ann_real0 * af * (1 + hh.ann_load)
        invest -= ann_prem
    feasible = invest > 0.02
    invest = np.maximum(invest, 0.0)

    bal = np.stack([invest * hh.stock_w, invest * (1 - hh.stock_w), np.zeros(P)], axis=1)
    ruined = np.zeros(P, bool)
    ruin_m = np.full(P, -1)
    annual_spend = np.zeros((P, years))
    annual_floor = np.zeros((P, years))
    annual_income = np.zeros((P, years))
    draw_real = np.zeros(P)
    gk_mult = np.ones(P)
    need0 = None
    lab_base = np.ones(P)            # price level at last 勞保 CPI step
    stock_index = np.ones(P); stock_hwm = np.ones(P)
    in_cash = np.zeros(P, bool); dd_months = np.zeros(P, int)
    capitulations = np.zeros(P, int)
    lr_recalc_m = lr_start_m

    for t in range(T):
        y = t // 12
        age = age0 + t / 12.0
        alive1 = t < d1; alive2 = t < d2
        alive = t < d_last
        both = alive1 & alive2 if joint else alive
        one = alive & ~both if joint else np.zeros(P, bool)
        wf = np.where(one, hh.widow_factor, 1.0)
        # ---- spending shape (smile) ----
        sm = 1.0
        if hh.smile:
            g1, a_turn, g2 = hh.smile
            yrs = t / 12.0
            sm = (1 + g1) ** min(yrs, max(a_turn - age0, 0))
            if age > a_turn:
                sm *= (1 + g2) ** (age - a_turn)
        target = s0 * sm * wf
        floor = floor0 * sm * wf
        # ---- income (real) ----
        inc = np.zeros(P)
        if age >= hh.lab_start_age and pension0 > 0:
            step = price[:, t] / lab_base >= 1.05
            lab_base = np.where(step, price[:, t], lab_base)
            lab_real = pension0 * lab_base / price[:, t]      # nominal fixed between steps
            inc += np.where(both, lab_real, np.where(one, lab_real * hh.survivor_share, 0.0))
        if hh.lr_account > 0 and t >= lr_start_m and t < lr_start_m + lr_term_m:
            if t == lr_recalc_m:
                n_left = lr_start_m + lr_term_m - t
                g = (1 + LR_RATE) ** (1 / 12) - 1
                lr_pay = lr_bal * g / (1 - (1 + g) ** -n_left)
                lr_recalc_m += 36
            lr_pay_eff = np.minimum(lr_pay, lr_bal)
            lr_bal = (lr_bal - lr_pay_eff) * (1 + LR_RATE) ** (1 / 12)
            inc += np.where(alive, lr_pay_eff / price[:, t] * 12, 0.0)
        if N > 0 and gap > 0 and y < N:
            if ladder_real is not None:
                inc += ladder_real[:, y]
            else:
                inc += ladder_nom[:, y] / price[:, t]
        if hh.ann_start_age and age >= hh.ann_start_age:
            if hh.ann_kind == "real":
                inc += np.where(alive1 if not joint else alive, ann_real0, 0.0)
            else:
                inc += np.where(alive1 if not joint else alive, ann_real0 / price[:, t], 0.0)
        # ---- annual draw decision ----
        if t % 12 == 0:
            wealth = bal.sum(axis=1)
            wealth_real = wealth / price[:, t]
            need = np.maximum(target - inc, 0.0)
            if hh.rule == "FIX":
                draw_real = need
            elif hh.rule == "GK":
                if t == 0:
                    need0 = np.maximum(need, 1e-9)
                    w_inv0 = np.maximum(invest, 1e-9)
                else:
                    wr = np.where(wealth > 0, draw_real * price[:, t] / np.maximum(wealth, 1e-12), np.inf)
                    wr0 = need0 / w_inv0
                    gk_mult = np.where(wr > 1.2 * wr0, gk_mult * 0.9, gk_mult)
                    gk_mult = np.where(wr < 0.8 * wr0, gk_mult * 1.1, gk_mult)
                    gk_mult = np.clip(gk_mult, 0.0, 1.5)
                draw_real = need * gk_mult
            else:  # VPW to 100 with extensions 90->110, 100->120
                h = 100 - age0
                if y >= 100 - age0: h = 120 - age0
                elif y >= 90 - age0: h = 110 - age0
                n_left = max(h - y, 1)
                r = hh.vpw_rate
                f = r / (1 - (1 + r) ** -n_left) if n_left > 1 else 1.0
                draw_real = wealth_real * f
            # floor top-up (V-floor), and never more than the balance
            draw_real = np.maximum(draw_real, np.maximum(floor - inc, 0.0))
            draw_real = np.minimum(draw_real, wealth_real)
            # annual rebalance
            if True:
                w_now = bal.sum(axis=1)
                tgt_stock = np.where(in_cash, 0.0, w_now * hh.stock_w)
                bal[:, 0] = tgt_stock
                bal[:, 1] = np.where(in_cash, 0.0, w_now * (1 - hh.stock_w))
                bal[:, 2] = np.where(in_cash, w_now, 0.0)
        # ---- monthly withdrawal ----
        # monthly floor top-up against the CURRENT real income (nominal streams
        # erode inside the year); draw itself is CPI-indexed within the year
        want_real = np.maximum(draw_real, np.maximum(floor - inc, 0.0))
        want = np.where(alive & ~ruined, want_real * price[:, t] / 12.0, 0.0)
        wealth = bal.sum(axis=1)
        take = np.minimum(want, wealth)
        frac = np.where(wealth > 0, take / np.maximum(wealth, 1e-12), 0.0)
        bal *= (1.0 - frac)[:, None]
        newly = alive & ~ruined & (take < want - 1e-9)
        ruined |= newly
        ruin_m = np.where(newly & (ruin_m < 0), t, ruin_m)
        spend_real = np.where(alive, inc / 12.0 + take / price[:, t], 0.0)
        annual_spend[:, y] += spend_real
        annual_floor[:, y] += np.where(alive, floor / 12.0, 0.0)
        annual_income[:, y] += np.where(alive, inc / 12.0, 0.0)
        # ---- behaviour: capitulation ----
        if hh.behavior:
            X, Y, q, m = hh.behavior
            dd = stock_index / np.maximum(stock_hwm, 1e-12)
            dd_months = np.where(dd < 1 - X, dd_months + 1, 0)
            trig = (~in_cash) & (dd_months == Y + 1) & (rng.random(P) < q)   # one draw per episode
            back = in_cash & (rng.random(P) < m)
            if trig.any():
                w_now = bal.sum(axis=1)
                bal[trig, 2] = w_now[trig]; bal[trig, 0] = 0; bal[trig, 1] = 0
                in_cash |= trig; capitulations += trig
                dd_months = np.where(trig, 0, dd_months)
            if back.any():
                w_now = bal.sum(axis=1)
                bal[back, 0] = w_now[back] * hh.stock_w
                bal[back, 1] = w_now[back] * (1 - hh.stock_w); bal[back, 2] = 0
                in_cash &= ~back
        # ---- returns ----
        bal[:, 0] *= (1 + stock_r[:, t]); bal[:, 1] *= (1 + bond_r[:, t]); bal[:, 2] *= (1 + cash_r[:, t])
        bal = np.maximum(bal, 0.0)
        stock_index *= (1 + stock_r[:, t]); stock_hwm = np.maximum(stock_hwm, stock_index)

    dy = np.minimum(d_last // 12, years)                              # FULL years alive
    yrs = np.arange(years)[None, :]
    alive_y = yrs < dy[:, None]
    ruin_alive = ruined & (ruin_m < d_last)
    below = (annual_spend < annual_floor * 0.98) & alive_y      # 2% tolerance: rounding, not a breach
    run = np.zeros(P); c = np.zeros(P)
    for yy in range(years):
        run = np.where(below[:, yy], run + 1, 0); c = np.maximum(c, run)
    life = (annual_spend * alive_y).sum(axis=1)
    avg = life / np.maximum(dy, 1)
    sa = np.maximum(annual_spend, 0.0005)[alive_y]
    ce4 = float(np.mean(sa ** -3) ** (-1 / 3)) if sa.size else float("nan")
    # spending smoothness on alive years (median across paths)
    with np.errstate(divide="ignore", invalid="ignore"):
        chg = np.diff(annual_spend, axis=1) / np.maximum(annual_spend[:, :-1], 1e-9)
    chg = np.where(alive_y[:, 1:], chg, np.nan)
    spend_vol = float(np.nanmedian(np.nanstd(chg, axis=1)))
    peak = np.maximum.accumulate(np.where(alive_y, annual_spend, 0), axis=1)
    ddn = np.where(alive_y, 1 - annual_spend / np.maximum(peak, 1e-9), 0)
    bequest = bal.sum(axis=1) / price[:, -1]
    return dict(
        name=hh.name, feasible_pct=round(100 * feasible.mean(), 1),
        invest0=round(float(np.median(invest)), 3),
        ladder_cost=round(float(np.median(ladder_cost)), 3), ann_prem=round(ann_prem, 3),
        ruin_alive_pct=round(100 * float(ruin_alive.mean()), 2),
        breach1_pct=round(100 * float(below.any(axis=1).mean()), 2),
        breach3_pct=round(100 * float((c >= 3).mean()), 2),
        life_p5=round(float(np.percentile(life, 5)), 3),
        life_med=round(float(np.median(life)), 3),
        avg_p5=round(float(np.percentile(avg[dy > 0], 5)), 4),
        avg_med=round(float(np.median(avg[dy > 0])), 4),
        ce_g4=round(ce4, 4),
        spend_vol=round(spend_vol, 3),
        spend_dd_max=round(float(np.median(ddn.max(axis=1))), 3),
        spend_dd_yrs=float(np.median((ddn > 0.2).sum(axis=1))),
        bequest_med=round(float(np.median(bequest)), 3),
        capit_pct=round(100 * float((capitulations > 0).mean()), 1),
        annual_spend=annual_spend, annual_floor=annual_floor, alive_y=alive_y,
        ruin_alive=ruin_alive, death_years=dy,
    )
