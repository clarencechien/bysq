"""Simulation engine.

Design constraints (per HANDOFF):
  - All series sampled on ONE shared time axis: a bootstrap draw picks a row
    index, and stock/bond/bill/CPI(/FX) of that month move together, so the
    return-inflation covariance structure survives (fixes fatal flaw #1/#2).
  - Withdrawals at the START of the month (fixes flaw #5).
  - Every decision uses information through month t-1 only. Decisions are
    functions of current balances (which embed returns through t-1) and the
    price level through t-1. Nothing reads returns[t] before applying it
    (verified by tests/test_no_lookahead.py; fixes flaw #3).
  - Cash inside a bucket earns the same stochastic bill series as everywhere
    else (fixes flaw #6).
  - Bucket refills look at the water level AND the market state, not "did it
    go up this year" (fixes flaw #7).
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "processed")


# ---------------------------------------------------------------- data ------

def load_panel(which):
    """Return dict of aligned numpy arrays of monthly data.

    which='us':  stock, bond, bill nominal USD returns + US CPI level.
                 bill is NaN before 1934-02.
    which='tw':  stock, bond (TWD-converted), usd_bill (TWD-converted),
                 tw CPI level. A synthetic 'twd_cash' column is added by
                 callers who need it (assumed real rate — flagged in report).
    """
    if which == "us":
        path, cols = os.path.join(PROC, "us_monthly.csv"), \
            ["stock_nom", "bond_nom", "bill_nom", "us_cpi"]
    else:
        path, cols = os.path.join(PROC, "tw_monthly.csv"), \
            ["stock_twd_nom", "bond_twd_nom", "usdbill_twd_nom", "tw_cpi"]
    dates, data = [], {c: [] for c in cols}
    with open(path) as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
            for c in cols:
                v = row[c]
                data[c].append(float(v) if v not in ("", None) else np.nan)
    out = {c.replace("_twd_nom", "").replace("_nom", ""): np.array(data[c])
           for c in cols}
    out["cpi"] = out.pop("us_cpi", None) if which == "us" else out.pop("tw_cpi")
    if which == "us":
        pass
    out["infl"] = np.empty_like(out["cpi"])
    out["infl"][0] = 0.0
    out["infl"][1:] = out["cpi"][1:] / out["cpi"][:-1] - 1.0
    out["dates"] = dates
    return out


def restrict(panel, first_date=None, last_date=None):
    """Slice panel to rows first_date <= date <= last_date.
    v4 §9.4: v1-v3 results used last_date='2023-07'; the rebuilt panels run
    to 2026-07, so pass last_date explicitly to reproduce the older tables."""
    d = np.array(panel["dates"])
    mask = np.ones(len(d), bool)
    if first_date:
        mask &= d >= first_date
    if last_date:
        mask &= d <= last_date
    keys = [k for k in panel if k not in ("dates",)]
    out = {k: panel[k][mask] for k in keys}
    out["dates"] = list(d[mask])
    return out


# ----------------------------------------------------------- sampling ------

def stationary_bootstrap_indices(t_src, t_out, n_paths, mean_block, rng):
    """Politis-Romano stationary bootstrap over ROW indices (shared axis)."""
    p = 1.0 / mean_block
    jump = rng.random((n_paths, t_out)) < p
    jump[:, 0] = True
    starts = rng.integers(0, t_src, (n_paths, t_out))
    idx = np.empty((n_paths, t_out), dtype=np.int64)
    cur = starts[:, 0].copy()
    for t in range(t_out):
        if t > 0:
            cur = np.where(jump[:, t], starts[:, t], (cur + 1) % t_src)
        idx[:, t] = cur
    return idx


def historical_indices(t_src, t_out):
    """All overlapping windows, monthly starts. No model assumptions."""
    n = t_src - t_out + 1
    if n <= 0:
        return np.empty((0, t_out), dtype=np.int64)
    return np.arange(n)[:, None] + np.arange(t_out)[None, :]


# --------------------------------------------------------- strategies ------

class Alloc:
    """Static stock/bond mix, annually rebalanced. weights=(stock, bond)."""
    def __init__(self, stock_w):
        self.stock_w = stock_w
        self.name = f"{int(stock_w*100)}/{int((1-stock_w)*100)}"


class BucketAlloc:
    """100% stock + cash bucket of `years` years of current annual spending.

    Rules (all computed from information through t-1):
      - stock drawdown dd = stock total-return index / its running max
      - dd <  spend_thresh : withdraw from cash bucket while it lasts
      - dd >= refill_thresh: withdraw from stock and refill bucket to target
      - otherwise          : withdraw from stock, leave bucket alone
    """
    def __init__(self, years, spend_thresh=0.90, refill_thresh=0.95):
        self.years = years
        self.spend_thresh = spend_thresh
        self.refill_thresh = refill_thresh
        self.name = f"BYSQ-{years}y"


class FixedReal:
    """Bengen: initial rate, then CPI-adjusted every year. No reaction."""
    def __init__(self, rate):
        self.rate = rate
        self.name = f"FIX{rate*100:g}"


class GuytonKlinger:
    """Full four-rule Guyton-Klinger (2006), annual application.

    - Withdrawal (inflation) rule: raise spending by realized CPI, capped at
      6%/yr; the raise is SKIPPED in a year following a negative portfolio
      return if the current withdrawal rate exceeds the initial rate.
    - Capital preservation rule: if current WR > 1.2 x initial WR, cut
      nominal spending 10%. Not applied in the final 15 years.
    - Prosperity rule: if current WR < 0.8 x initial WR, raise spending 10%.
    - Portfolio management rule: withdrawals are funded from the overweight
      asset first (equivalent, in a 2-asset annually-rebalanced portfolio,
      to rebalancing at the withdrawal date), and following a negative-stock
      year the withdrawal is taken from bonds first.
    """
    def __init__(self, rate, cap=0.06, cut=0.10, raise_=0.10,
                 upper=1.2, lower=0.8, cpr_off_years=15, floor_ratio=0.0):
        self.rate, self.cap, self.cut, self.raise_ = rate, cap, cut, raise_
        self.upper, self.lower, self.cpr_off_years = upper, lower, cpr_off_years
        self.floor_ratio = floor_ratio   # uncuttable share of initial spending
        self.name = f"GK{rate*100:g}" + (f"-fl{int(floor_ratio*100)}" if floor_ratio else "")


class VPW:
    """W4 amortization: each year spend = wealth x annuity factor over the
    remaining horizon at an assumed real return. Two parameters, both from
    an identity rather than a backtest fit.

    v4 additions (all default off):
      horizon     planning years for the annuity factor (default = sim years).
                  Once y >= horizon the factor is 1.0: the rule spends the
                  whole balance (the 'fixed end-point' cliff of v4 §3.3).
      extensions  list of (trigger_year, new_horizon): when the path reaches
                  trigger_year still alive, the planning horizon jumps to
                  new_horizon (v4 §3.3 'plan to 100, extend to 110 at 90').
      smooth      None | ('cap', up, down): annual REAL change clamped to
                  [-down, +up] vs last year's real spending (Vanguard-style)
                  | ('yale', alpha): spend = alpha x last real + (1-alpha) x VPW
      floor_abs   real spending floor in units of W0 (V-floor; only meaningful
                  when an external floor exists, v4 §3.1) — spending is topped
                  up to it while wealth lasts.
      life_table  (qx array indexed by age, retire_age, buffer): Taiwan RMD
                  (v4 §3.2): horizon each year = remaining life expectancy at
                  the current age from the table + buffer, instead of a fixed
                  end-point. qx must be the population table; the rule reads
                  only e_x, which is deterministic (no lookahead issue)."""
    def __init__(self, expected_real=0.03, horizon=None, extensions=(),
                 smooth=None, floor_abs=0.0, life_table=None, cape_rate=None):
        self.expected_real = expected_real
        self.horizon = horizon
        self.extensions = list(extensions)
        self.smooth = smooth
        self.floor_abs = floor_abs
        self.life_table = life_table
        self.cape_rate = cape_rate      # True: expected_real = clip(1/CAPE at t-1, 1%, 8%)
        self.rate = None
        tag = f"VPW{expected_real*100:g}" if not cape_rate else "VPWcape"
        if life_table is not None:
            tag = f"TWRMD{expected_real*100:g}+{life_table[2]}"
        if horizon:
            tag += f"h{horizon}"
        if extensions:
            tag += "x" + "/".join(f"{a}-{b}" for a, b in extensions)
        if smooth:
            tag += "-" + (f"cap{smooth[1]*100:g}/{smooth[2]*100:g}" if smooth[0] == "cap"
                          else f"yale{smooth[1]:g}")
        if floor_abs:
            tag += f"-fl{floor_abs*100:g}"
        self.name = tag

    def years_left(self, y, sim_years):
        """Planning years remaining at the START of year y (info: age only)."""
        if self.life_table is not None:
            qx, retire_age, buffer = self.life_table
            age = retire_age + y
            surv = np.cumprod(1 - qx[age:])
            ex = surv.sum() + 0.5
            return max(int(round(ex + buffer)), 1)
        h = self.horizon or sim_years
        for trig, new_h in self.extensions:
            if y >= trig:
                h = max(h, new_h)
        return max(h - y, 1)


class RMD:
    """W5: spend = wealth / remaining years. Zero parameters."""
    def __init__(self):
        self.rate = None
        self.name = "RMD"


class FundedRatio:
    """W6 (v4 §3.4): fixed-real spending with a funded-ratio guardrail.
    funded_ratio = wealth / PV(remaining real spending at `discount`);
    < lower -> cut 10%, > upper -> raise 10%, at most once a year."""
    def __init__(self, rate, discount=0.02, lower=0.8, upper=1.2, cut=0.10, raise_=0.10):
        self.rate, self.discount = rate, discount
        self.lower, self.upper, self.cut, self.raise_ = lower, upper, cut, raise_
        self.name = f"W6-FR{rate*100:g}"


class RiskGuardrail:
    """W7 (v4 §3.4): probability-of-success guardrail (the US-practitioner
    successor to GK). Each year estimate P(portfolio funds the current real
    spending for the remaining years) under FIXED assumed real drift mu and
    vol sigma via the Milevsky-Robinson (2005) reciprocal-Gamma
    approximation of the stochastic present value; cut 10% when below p_cut,
    raise 10% when above p_raise. Uses no sampled returns at all."""
    def __init__(self, rate, mu=0.04, sigma=0.12, p_cut=0.70, p_raise=0.95,
                 cut=0.10, raise_=0.10):
        self.rate, self.mu, self.sigma = rate, mu, sigma
        self.p_cut, self.p_raise, self.cut, self.raise_ = p_cut, p_raise, cut, raise_
        self.name = f"W7-PoS{rate*100:g}"

    def p_success(self, wr, n_left):
        """P(SPV of n_left years of 1/yr real spending < 1/wr).
        Milevsky-Robinson: SPV ~ reciprocal Gamma(alpha, beta) with
        alpha = (2 mu + 4 lambda)/(sigma^2 + lambda) - 1, beta = (sigma^2 + lambda)/2,
        lambda = 1/n_left (finite-horizon correction via mortality-like hazard)."""
        from math import lgamma
        lam = 1.0 / max(n_left, 1)
        mu, s2 = self.mu, self.sigma ** 2
        alpha = (2 * mu + 4 * lam) / (s2 + lam) - 1.0
        beta = (s2 + lam) / 2.0
        if alpha <= 0:
            return np.zeros_like(wr)
        # P(SPV < 1/wr) = P(1/SPV > wr) = 1 - GammaCDF(wr; alpha, scale=beta)
        # -> regularized lower incomplete gamma via series/continued fraction
        x = np.asarray(wr, float) / beta
        return 1.0 - _gammainc(alpha, x)


def _gammainc(a, x):
    """Regularized lower incomplete gamma P(a, x), vectorized in x
    (Numerical Recipes series + Lentz continued fraction)."""
    from math import lgamma
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    gln = lgamma(a)
    ser = x < a + 1.0
    # series
    xs = x[ser]
    if xs.size:
        ap = a
        s = np.full_like(xs, 1.0 / a)
        d = np.full_like(xs, 1.0 / a)
        for _ in range(500):
            ap += 1.0
            d = d * xs / ap
            s = s + d
            if np.all(np.abs(d) < np.abs(s) * 1e-12):
                break
        out[ser] = s * np.exp(-xs + a * np.log(np.maximum(xs, 1e-300)) - gln)
    # continued fraction
    xc = x[~ser]
    if xc.size:
        tiny = 1e-300
        b = xc + 1.0 - a
        c = np.full_like(xc, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        for i in range(1, 500):
            an = -i * (i - a)
            b = b + 2.0
            d = an * d + b
            d = np.where(np.abs(d) < tiny, tiny, d)
            c = b + an / c
            c = np.where(np.abs(c) < tiny, tiny, c)
            d = 1.0 / d
            dl = d * c
            h = h * dl
            if np.all(np.abs(dl - 1.0) < 1e-12):
                break
        out[~ser] = 1.0 - np.exp(-xc + a * np.log(np.maximum(xc, 1e-300)) - gln) * h
    return np.clip(out, 0.0, 1.0)


class VanguardDynamic:
    """Vanguard dynamic spending: pct-of-portfolio with annual change
    clamped to [-floor, +ceiling] vs prior year's nominal spending,
    then compared in real terms for metrics."""
    def __init__(self, rate, ceiling=0.05, floor=0.025):
        self.rate, self.ceiling, self.floor = rate, ceiling, floor
        self.name = f"VG{rate*100:g}"


# --------------------------------------------------------- simulator -------

def simulate(idx, panel, alloc, rule, cash_col="bill", w0=1.0, fee=0.0,
             extra_spend=None):
    """Vectorized across paths. Returns per-path metric dict.

    idx: (P, T) row indices into panel arrays. T = years*12.
    Wealth starts at w0 (scalar or per-path array; v4 §5 hands the whole
    accumulation-phase terminal distribution in here). Spending rules that
    take a `rate` interpret it as a fraction of the REFERENCE wealth 1.0,
    i.e. an absolute real spending target, so per-path w0 != 1 changes the
    effective withdrawal rate path by path. Withdrawals happen at the start
    of each month (annual amount / 12). Annual decisions at month t use
    balances after returns through t-1 and the CPI level through t-1.
    fee: annual expense/tax drag subtracted from every asset's return
    (v4 §9.2), applied monthly as fee/12.
    extra_spend: optional (P, years) matrix of ADDITIONAL real annual
    spending (units of W0) forced on top of the rule — v4 §6 long-term-care
    and one-off shocks. It is withdrawn monthly (amount/12) and counted in
    annual_real_spend; the rule itself never sees it (a shock is not a
    decision).
    """
    P, T = idx.shape
    years = T // 12
    stock_r = panel["stock"][idx] - fee / 12.0          # (P, T)
    bond_r = panel["bond"][idx] - fee / 12.0
    cash_r = panel[cash_col][idx] - fee / 12.0
    infl_m = panel["infl"][idx]
    w0 = np.broadcast_to(np.asarray(w0, float), (P,)).copy()

    # price level at the START of month t (before month-t inflation accrues)
    price = np.ones((P, T))
    lvl = np.ones(P)
    for t in range(1, T):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    is_bucket = isinstance(alloc, BucketAlloc)
    if is_bucket:
        target_cash_years = alloc.years
        cw = min((rule.rate or 0.04) * target_cash_years, 0.5)
        bal = np.stack([w0 * (1.0 - cw),                # stock
                        np.zeros(P),                    # bond unused
                        w0 * cw], axis=1)               # cash
        stock_index = np.ones(P)
        stock_hwm = np.ones(P)
    else:
        bal = np.stack([w0 * alloc.stock_w,
                        w0 * (1.0 - alloc.stock_w),
                        np.zeros(P)], axis=1)

    if isinstance(rule, VPW):
        r_e = rule.expected_real
        if rule.cape_rate:
            cape_m = panel["cape"][idx]
            r_e = np.clip(1.0 / cape_m[:, 0], 0.01, 0.08)
        n0 = rule.years_left(0, years)
        f0 = r_e / (1 - (1 + r_e) ** -n0) if n0 > 1 else 1.0
        first = w0 * f0
        if rule.floor_abs:
            first = np.maximum(first, rule.floor_abs)
    elif isinstance(rule, RMD):
        first = w0 / years
    elif isinstance(rule, (FundedRatio, RiskGuardrail)):
        first = np.full(P, rule.rate)
    else:
        first = np.full(P, rule.rate)
    spend_nom = np.array(first, float) * np.ones(P)   # this year's nominal spending
    initial_rate = spend_nom / w0           # per-path initial withdrawal rate
    last_real_spend = spend_nom.copy()      # for VPW smoothing layers
    alive = np.ones(P, bool)
    annual_real_spend = np.zeros((P, years))
    prev_year_port_ret_neg = np.zeros(P, bool)
    prev_year_stock_ret_neg = np.zeros(P, bool)
    ruin_year = np.full(P, -1)

    for t in range(T):
        y = t // 12
        if t % 12 == 0:
            wealth = bal.sum(axis=1)
            if t > 0:
                # ---- annual spending decision (info through t-1 only) ----
                yr_infl = price[:, t] / price[:, t - 12] - 1.0
                if isinstance(rule, FixedReal):
                    spend_nom = spend_nom * (1.0 + yr_infl)
                elif isinstance(rule, GuytonKlinger):
                    cur_wr = np.where(wealth > 0, spend_nom / np.maximum(wealth, 1e-12), np.inf)
                    adj = np.minimum(yr_infl, rule.cap)
                    skip = prev_year_port_ret_neg & (cur_wr > initial_rate)
                    spend_nom = spend_nom * (1.0 + np.where(skip, 0.0, np.maximum(adj, 0.0)))
                    cur_wr = np.where(wealth > 0, spend_nom / np.maximum(wealth, 1e-12), np.inf)
                    if y < years - rule.cpr_off_years:
                        hit = cur_wr > rule.upper * initial_rate
                        spend_nom = np.where(hit, spend_nom * (1 - rule.cut), spend_nom)
                    low = (spend_nom / np.maximum(wealth, 1e-12)) < rule.lower * initial_rate
                    spend_nom = np.where(low & (wealth > 0), spend_nom * (1 + rule.raise_), spend_nom)
                    if rule.floor_ratio > 0:
                        # the uncuttable share of initial REAL spending: no rule
                        # may push spending below it (v3 §5)
                        floor_nom = rule.floor_ratio * rule.rate * price[:, t]
                        spend_nom = np.maximum(spend_nom, floor_nom)
                elif isinstance(rule, VPW):
                    n_left = rule.years_left(y, years)
                    r_e = rule.expected_real
                    if rule.cape_rate:
                        r_e = np.clip(1.0 / cape_m[:, t - 1], 0.01, 0.08)
                    f = r_e / (1 - (1 + r_e) ** -n_left) if n_left > 1 else 1.0
                    target_real = wealth * f / price[:, t]
                    if rule.smooth is not None:
                        if rule.smooth[0] == "cap":
                            up, down = rule.smooth[1], rule.smooth[2]
                            target_real = np.clip(target_real,
                                                  last_real_spend * (1 - down),
                                                  last_real_spend * (1 + up))
                        elif rule.smooth[0] == "yale":
                            a = rule.smooth[1]
                            target_real = a * last_real_spend + (1 - a) * target_real
                        # smoothing can never ask for more than the balance
                        target_real = np.minimum(target_real, wealth / price[:, t])
                    if rule.floor_abs:
                        target_real = np.maximum(target_real, rule.floor_abs)
                    spend_nom = target_real * price[:, t]
                elif isinstance(rule, FundedRatio):
                    # W6: funded ratio = wealth / PV(remaining real spending
                    # at the rule's discount rate); cut/raise 10% outside band
                    n_left = years - y
                    r_d = rule.discount
                    pv_f = (1 - (1 + r_d) ** -n_left) / r_d if r_d > 0 else n_left
                    spend_nom = spend_nom * (1.0 + yr_infl)             # CPI-adjust
                    fr = wealth / np.maximum(spend_nom * pv_f, 1e-12)
                    spend_nom = np.where(fr < rule.lower, spend_nom * (1 - rule.cut), spend_nom)
                    spend_nom = np.where(fr > rule.upper, spend_nom * (1 + rule.raise_), spend_nom)
                elif isinstance(rule, RiskGuardrail):
                    # W7: probability-of-success guardrail. The success
                    # probability is a closed-form lognormal/Gamma approx
                    # (Milevsky-Robinson 2005 stochastic present value) with
                    # FIXED assumed real drift/vol — it never reads the
                    # sampled returns, so it trivially passes no-lookahead.
                    n_left = years - y
                    spend_nom = spend_nom * (1.0 + yr_infl)
                    wr = spend_nom / np.maximum(wealth, 1e-12)
                    p_ok = rule.p_success(wr, n_left)
                    spend_nom = np.where(p_ok < rule.p_cut, spend_nom * (1 - rule.cut), spend_nom)
                    spend_nom = np.where(p_ok > rule.p_raise, spend_nom * (1 + rule.raise_), spend_nom)
                elif isinstance(rule, RMD):
                    n_left = years - y
                    spend_nom = wealth / max(n_left, 1)
                elif isinstance(rule, VanguardDynamic):
                    tgt = rule.rate * wealth
                    hi = spend_nom * (1 + rule.ceiling)
                    lo = spend_nom * (1 - rule.floor)
                    spend_nom = np.clip(tgt, lo, hi)

            # ---- annual rebalance / bucket management ----
            if is_bucket:
                dd = stock_index / np.maximum(stock_hwm, 1e-12)
                target_cash = target_cash_years * spend_nom
                refill = (dd >= alloc.refill_thresh) & (bal[:, 2] < target_cash)
                move = np.where(refill,
                                np.minimum(np.maximum(target_cash - bal[:, 2], 0.0), bal[:, 0]),
                                0.0)
                bal[:, 0] -= move
                bal[:, 2] += move
            else:
                wealth = bal.sum(axis=1)
                bal[:, 0] = wealth * alloc.stock_w
                bal[:, 1] = wealth * (1.0 - alloc.stock_w)
                bal[:, 2] = 0.0

        # ---- start-of-month withdrawal ----
        w_month = np.where(alive, spend_nom / 12.0, 0.0)
        if extra_spend is not None:
            w_month = w_month + np.where(alive, extra_spend[:, y] * price[:, t] / 12.0, 0.0)
        if is_bucket:
            dd = stock_index / np.maximum(stock_hwm, 1e-12)
            from_cash = (dd < alloc.spend_thresh) & (bal[:, 2] > 0)
            take_cash = np.where(from_cash, np.minimum(w_month, bal[:, 2]), 0.0)
            rest = w_month - take_cash
            take_stock = np.minimum(rest, bal[:, 0])
            shortfall_cash = np.minimum(rest - take_stock, bal[:, 2])
            bal[:, 2] -= take_cash + shortfall_cash
            bal[:, 0] -= take_stock
            got = take_cash + take_stock + shortfall_cash
        else:
            if isinstance(rule, GuytonKlinger):
                # PMR: after a negative stock year fund withdrawals from
                # bonds first; otherwise proportionally (annual rebalance
                # restores targets each January).
                order = np.where(prev_year_stock_ret_neg[:, None],
                                 np.array([[1, 0, 2]]),
                                 np.array([[0, 1, 2]]))
                got = np.zeros(P)
                need = w_month.copy()
                for k in range(3):
                    a = order[:, k]
                    avail = bal[np.arange(P), a]
                    take = np.minimum(need, avail)
                    bal[np.arange(P), a] = avail - take
                    got += take
                    need -= take
            else:
                wealth = np.maximum(bal.sum(axis=1), 1e-15)
                frac = np.minimum(w_month / wealth, 1.0)
                got = wealth * frac
                bal *= (1.0 - frac)[:, None]
        newly_ruined = alive & (got < w_month - 1e-12)
        ruin_year = np.where(newly_ruined & (ruin_year < 0), y, ruin_year)
        alive = alive & ~newly_ruined
        annual_real_spend[:, y] += got / price[:, t]
        if t % 12 == 11:
            last_real_spend = annual_real_spend[:, y]

        # ---- apply month-t returns ----
        bal[:, 0] *= (1.0 + stock_r[:, t])
        bal[:, 1] *= (1.0 + bond_r[:, t])
        bal[:, 2] *= (1.0 + cash_r[:, t])
        bal = np.maximum(bal, 0.0)
        if is_bucket:
            stock_index = stock_index * (1.0 + stock_r[:, t])
            stock_hwm = np.maximum(stock_hwm, stock_index)

        if (t + 1) % 12 == 0 and t >= 11:
            # record year-over-year return signs for GK rules; uses months
            # <= t, consumed only by decisions at t+1 (no lookahead)
            yr_slice = slice(t - 11, t + 1)
            sw = 1.0 if is_bucket else alloc.stock_w
            r_stock_y = (1 + stock_r[:, yr_slice]).prod(axis=1) - 1
            r_bond_y = (1 + bond_r[:, yr_slice]).prod(axis=1) - 1
            prev_year_stock_ret_neg = r_stock_y < 0
            prev_year_port_ret_neg = (sw * r_stock_y + (1 - sw) * r_bond_y) < 0

    final_real_wealth = bal.sum(axis=1) / (price[:, -1] * (1 + infl_m[:, -1]))
    s1 = annual_real_spend[:, 0]
    rel = annual_real_spend / np.maximum(s1[:, None], 1e-12)
    below70 = rel < 0.70
    # 3+ consecutive years below 70% of initial real spending
    c = np.zeros(P)
    run = np.zeros(P)
    for yy in range(years):
        run = np.where(below70[:, yy], run + 1, 0)
        c = np.maximum(c, run)
    # v4 §0.1: 'ruined' = assets exhausted with MORE than one year of the plan
    # left. Exhaustion inside the final plan year is 'planned_depletion'
    # (amortizing rules spend the whole balance in year N by design).
    return dict(
        total_real_spend=annual_real_spend.sum(axis=1),
        ruined=(ruin_year >= 0) & (ruin_year < years - 1),
        planned_depletion=(ruin_year == years - 1),
        exhausted=(ruin_year >= 0),
        ruin_year=ruin_year,
        spend_fail=(c >= 3),
        yrs_below80=(rel < 0.80).sum(axis=1),
        final_real_wealth=final_real_wealth,
        annual_real_spend=annual_real_spend,
    )


def spend_fail_abs(annual_real_spend, floor_abs, min_years=3):
    """v3 §1 scoring fix: failure = real spending below an ABSOLUTE floor
    (units of initial wealth, inflation-adjusted by construction) for
    >= min_years consecutive years. Same ruler for every strategy and
    every starting rate — the old 70%-of-own-initial denominator let low
    starting rates lower their own bar."""
    P, years = annual_real_spend.shape
    below = annual_real_spend < floor_abs
    c = np.zeros(P)
    run = np.zeros(P)
    for yy in range(years):
        run = np.where(below[:, yy], run + 1, 0)
        c = np.maximum(c, run)
    return c >= min_years


def ce_spending(annual_real_spend, gamma, floor_abs=0.0005):
    """CRRA certainty-equivalent annual real spending across paths+years.
    Ruin years are floored at an ABSOLUTE 0.05% of initial wealth (v3: no
    strategy-relative bars anywhere in the scoring)."""
    s = np.maximum(annual_real_spend, floor_abs)
    if gamma == 1:
        u = np.log(s)
        return float(np.exp(u.mean()))
    u = s ** (1 - gamma) / (1 - gamma)
    m = u.mean()
    return float((m * (1 - gamma)) ** (1 / (1 - gamma)))
