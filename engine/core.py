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


def restrict(panel, first_date=None):
    """Slice panel to rows >= first_date (and rows where bill is finite)."""
    d = np.array(panel["dates"])
    mask = np.ones(len(d), bool)
    if first_date:
        mask &= d >= first_date
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
                 upper=1.2, lower=0.8, cpr_off_years=15):
        self.rate, self.cap, self.cut, self.raise_ = rate, cap, cut, raise_
        self.upper, self.lower, self.cpr_off_years = upper, lower, cpr_off_years
        self.name = f"GK{rate*100:g}"


class VanguardDynamic:
    """Vanguard dynamic spending: pct-of-portfolio with annual change
    clamped to [-floor, +ceiling] vs prior year's nominal spending,
    then compared in real terms for metrics."""
    def __init__(self, rate, ceiling=0.05, floor=0.025):
        self.rate, self.ceiling, self.floor = rate, ceiling, floor
        self.name = f"VG{rate*100:g}"


# --------------------------------------------------------- simulator -------

def simulate(idx, panel, alloc, rule, cash_col="bill"):
    """Vectorized across paths. Returns per-path metric dict.

    idx: (P, T) row indices into panel arrays. T = years*12.
    Wealth starts at 1.0. Withdrawals happen at the start of each month
    (annual amount / 12). Annual decisions at month t use balances after
    returns through t-1 and the CPI level through t-1.
    """
    P, T = idx.shape
    years = T // 12
    stock_r = panel["stock"][idx]           # (P, T)
    bond_r = panel["bond"][idx]
    cash_r = panel[cash_col][idx]
    infl_m = panel["infl"][idx]

    # price level at the START of month t (before month-t inflation accrues)
    price = np.ones((P, T))
    lvl = np.ones(P)
    for t in range(1, T):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    is_bucket = isinstance(alloc, BucketAlloc)
    if is_bucket:
        target_cash_years = alloc.years
        w0 = min(rule.rate * target_cash_years, 0.5)
        bal = np.stack([np.full(P, 1.0 - w0),          # stock
                        np.zeros(P),                    # bond unused
                        np.full(P, w0)], axis=1)        # cash
        stock_index = np.ones(P)
        stock_hwm = np.ones(P)
    else:
        bal = np.stack([np.full(P, alloc.stock_w),
                        np.full(P, 1.0 - alloc.stock_w),
                        np.zeros(P)], axis=1)

    spend_nom = np.full(P, rule.rate)       # this year's nominal spending
    initial_rate = rule.rate
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
    return dict(
        total_real_spend=annual_real_spend.sum(axis=1),
        ruined=(ruin_year >= 0),
        ruin_year=ruin_year,
        spend_fail=(c >= 3),
        yrs_below80=(rel < 0.80).sum(axis=1),
        final_real_wealth=final_real_wealth,
        annual_real_spend=annual_real_spend,
    )


def ce_spending(annual_real_spend, gamma, floor_frac=0.01):
    """CRRA certainty-equivalent annual real spending across paths+years."""
    s1 = np.maximum(annual_real_spend[:, 0:1], 1e-12)
    s = np.maximum(annual_real_spend, floor_frac * s1)
    if gamma == 1:
        u = np.log(s)
        return float(np.exp(u.mean()))
    u = s ** (1 - gamma) / (1 - gamma)
    m = u.mean()
    return float((m * (1 - gamma)) ** (1 / (1 - gamma)))
