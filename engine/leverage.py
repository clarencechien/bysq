"""v2 leverage engine: borrow-against-portfolio withdrawal (質借提領).

Implements handoff-v2:
  - Spending funded by borrowing instead of selling (P6*), vs sell baseline (P8)
  - Opportunistic bucket refill (§3): borrow to fill an N-year cash bucket only
    when the position is near its high-water mark; spend from the bucket in
    drawdowns; three fallbacks when the bucket runs dry (F1 borrow anyway /
    F2 sell / F3 cut spending)
  - Maintenance liquidation (§2): forced partial sale + slippage when
    LTV = D/W crosses the kill threshold
  - Credit-line review (§4): annual, stress-dependent probability of the line
    being cut to beta x collateral-implied line; shortfall repaid from bucket
    first, then forced sale
  - All decisions use information through t-1 (no-lookahead, §8.8)
  - Path B borrow rate rides the sampled bill series (+spread) on the SAME
    time axis (§8.9); Path A/C constant rates are flagged sensitivity-grade

Wealth unit: initial portfolio = 1.0. Spending is real (CPI-indexed), like
the v1 FixedReal rule.
"""
import numpy as np


class LevSpec:
    def __init__(self, rate=0.04,                 # initial spending rate
                 bucket_years=0,                  # N; 0 = borrow every month
                 alpha=0.95,                      # refill when W >= alpha*peak
                 ltv_max=0.30,                    # soft cap for refill borrowing
                 ltv_kill=0.77,                   # maintenance liquidation line
                 ltv_reset=0.50,                  # post-liquidation LTV target
                 fallback="F1",                   # F1 borrow / F2 sell / F3 cut
                 borrow_mode="path_c",            # path_b: bill+spread, path_c: const
                 borrow_const=0.03, borrow_spread=0.015,
                 slippage=0.01,
                 line_p0=0.02, line_k=5.0, line_beta=0.7, line_ltv=0.50,
                 line_review=True,
                 cut_frac=0.10,                   # F3: cut real spending 10%
                 peak_window=None):               # None=all-time peak, else months
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.name = (f"{'P6b' if borrow_mode=='path_b' else 'P6c'}"
                     f"-N{bucket_years}-a{int(alpha*100)}-L{int(ltv_max*100)}-{fallback}")


def simulate_leverage(idx, panel, spec, cash_col="twd_cash", seed=0):
    """Vectorized across paths. Returns per-path metric dict.

    Collateral = 100% stock position W (never sold except forced events and F2).
    Debt D accrues monthly interest. Bucket C earns the cash series.
    LTV is computed on the position only (banks count pledged shares).
    """
    P, T = idx.shape
    years = T // 12
    stock_r = panel["stock"][idx]
    cash_r = panel[cash_col][idx]
    infl_m = panel["infl"][idx]
    if spec.borrow_mode == "path_b":
        # bill total-return series back to an annualized rate + spread
        borrow_m = (1.0 + panel["bill"][idx]) * (1.0 + spec.borrow_spread) ** (1 / 12) - 1.0
    else:
        borrow_m = np.full((P, T), (1.0 + spec.borrow_const) ** (1 / 12) - 1.0)

    price = np.ones((P, T))
    lvl = np.ones(P)
    for t in range(1, T):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    rng = np.random.default_rng(10_000 + seed)

    W = np.ones(P)                    # stock position (collateral)
    D = np.zeros(P)                   # debt
    C = np.zeros(P)                   # cash bucket
    peak = np.ones(P)                 # W high-water mark (uses t-1 info)
    spend_nom = np.full(P, spec.rate)
    line = np.full(P, np.inf)         # credit line; refreshed at review
    line_cut_mult = np.ones(P)

    alive = np.ones(P, bool)          # not wiped out
    last_cut = np.full(P, -12)        # F3: cut at most once per 12 months
    annual_real_spend = np.zeros((P, years))
    forced_liq = np.zeros(P, bool)
    line_event_forced = np.zeros(P, bool)
    first_hit_year = np.full(P, -1)
    peak_ltv = np.zeros(P)
    cum_interest = np.zeros(P)
    networth_path_min = np.full(P, np.inf)
    networth_peak = np.full(P, 1e-12)
    trail12 = np.zeros((P, 12))       # trailing monthly stock returns

    def register_hit(mask, y):
        nonlocal first_hit_year
        first_hit_year = np.where(mask & (first_hit_year < 0), y, first_hit_year)

    for t in range(T):
        y = t // 12
        if t % 12 == 0 and t > 0:
            yr_infl = price[:, t] / price[:, t - 12] - 1.0
            spend_nom = spend_nom * (1.0 + np.maximum(yr_infl, 0.0))

            # ---- annual credit-line review (Path A/C only) ----
            if spec.line_review and spec.borrow_mode != "path_b":
                r12 = (1.0 + trail12).prod(axis=1) - 1.0
                stress = r12 < -0.20
                p_cut = spec.line_p0 * np.where(stress, spec.line_k, 1.0)
                cut = rng.random(P) < p_cut
                line_cut_mult = np.where(cut, spec.line_beta, 1.0)
                line = spec.line_ltv * W * line_cut_mult
                short = np.maximum(D - line, 0.0)
                use_c = np.minimum(short, C)
                C -= use_c
                D -= use_c
                short -= use_c
                must_sell = short > 1e-12
                sell = np.minimum(short * (1 + spec.slippage), W)
                W = np.where(must_sell, W - sell, W)
                D = np.where(must_sell, D - sell / (1 + spec.slippage), D)
                line_event_forced |= must_sell & alive
                register_hit(must_sell & alive, y)

        # ---- monthly spending (uses W from t-1, peak through t-1) ----
        w_month = np.where(alive, spend_nom / 12.0, 0.0)
        if spec.bucket_years > 0:
            take_c = np.minimum(w_month, C)
            C -= take_c
            rest = w_month - take_c
        else:
            rest = w_month
        # v3 §2 orthogonal design: the fallback is the DRAWDOWN response,
        # independent of whether a bucket exists. In good times (position at
        # or near its high-water mark) unfunded spending is always borrowed;
        # the fallback governs what happens when the bucket (if any) is dry
        # AND the position is in drawdown. bucket=OFF x fallback is therefore
        # a meaningful cell: no pre-funding, drawdown response only.
        need = rest > 1e-15
        in_dd = W < spec.alpha * peak          # t-1 info
        fb = need & in_dd
        good = need & ~in_dd
        D += np.where(good, rest, 0.0)
        funded_rest = np.where(good, rest, 0.0)
        if spec.fallback == "F1":              # borrow anyway
            D += np.where(fb, rest, 0.0)
            funded_rest = np.where(fb, rest, funded_rest)
        elif spec.fallback == "F2":            # sell at the low
            sell = np.where(fb, np.minimum(rest, W), 0.0)
            W -= sell
            shortD = np.where(fb, rest - sell, 0.0)
            D += np.where(shortD > 1e-15, shortD, 0.0)
            funded_rest = np.where(fb, rest, funded_rest)
        else:                                  # F3: cut 10% (once/12m), borrow rest
            do_cut = fb & (t - last_cut >= 12)
            last_cut = np.where(do_cut, t, last_cut)
            spend_nom = np.where(do_cut, spend_nom * (1 - spec.cut_frac), spend_nom)
            fr = np.where(fb, rest * np.where(do_cut, 1 - spec.cut_frac, 1.0), 0.0)
            D += fr
            funded_rest = funded_rest + fr
        spent = (take_c if spec.bucket_years > 0 else 0.0) + funded_rest
        annual_real_spend[:, y] += np.where(alive, spent / price[:, t], 0.0)

        # ---- opportunistic refill (t-1 info only: W, peak) ----
        if spec.bucket_years > 0:
            cond = W >= spec.alpha * peak
            target = spec.bucket_years * spend_nom
            head = np.maximum(spec.ltv_max * W - D, 0.0)
            borrow = np.where(cond & alive, np.minimum(np.maximum(target - C, 0.0), head), 0.0)
            D += borrow
            C += borrow

        # ---- interest & returns ----
        interest = D * borrow_m[:, t]
        D += interest
        cum_interest += np.where(alive, interest / price[:, t], 0.0)
        C *= (1.0 + cash_r[:, t])
        W *= (1.0 + stock_r[:, t])
        peak = np.maximum(peak, W)

        # ---- maintenance check (end of month, on realized values) ----
        ltv = np.where(W > 1e-12, D / np.maximum(W, 1e-12), np.inf)
        peak_ltv = np.maximum(peak_ltv, np.where(alive, np.minimum(ltv, 10.0), peak_ltv))
        hit = alive & (ltv > spec.ltv_kill)
        if hit.any():
            sell_gross = np.where(hit, np.maximum(
                (D - spec.ltv_reset * W) / (1 - spec.ltv_reset), 0.0), 0.0)
            sell_gross = np.minimum(sell_gross * (1 + spec.slippage), W)
            W = np.where(hit, W - sell_gross, W)
            D = np.where(hit, D - sell_gross / (1 + spec.slippage), D)
            forced_liq |= hit
            register_hit(hit, y)
        wiped = alive & ((W + C - D) <= 0)
        alive = alive & ~wiped

        nw = W + C - D
        networth_peak = np.maximum(networth_peak, nw)
        networth_path_min = np.minimum(networth_path_min,
                                       nw / np.maximum(networth_peak, 1e-12))

    final_nw_real = (W + C - D) / (price[:, -1] * (1 + infl_m[:, -1]))
    s1 = annual_real_spend[:, 0]
    rel = annual_real_spend / np.maximum(s1[:, None], 1e-12)
    below70 = rel < 0.70
    c = np.zeros(P)
    run = np.zeros(P)
    for yy in range(years):
        run = np.where(below70[:, yy], run + 1, 0)
        c = np.maximum(c, run)
    # stock max drawdown per path (for conditional analysis)
    si = np.cumprod(1.0 + stock_r, axis=1)
    smax = np.maximum.accumulate(si, axis=1)
    stock_mdd = (si / smax - 1.0).min(axis=1)
    return dict(
        final_nw_real=final_nw_real,
        wiped=~alive,
        forced_liq=forced_liq,
        line_forced=line_event_forced,
        first_hit_year=first_hit_year,
        peak_ltv=peak_ltv,
        cum_interest_real=cum_interest,
        total_real_spend=annual_real_spend.sum(axis=1),
        annual_real_spend=annual_real_spend,
        spend_fail=(c >= 3),
        nw_mdd=networth_path_min - 1.0,
        stock_mdd=stock_mdd,
    )


def deterministic_check():
    """§8.11: annual hand-calc replication. 1e8 TWD -> unit 1.0; spend 2.5%/yr
    growing 3%/yr; borrow at start of year; debt 3%/yr; asset 7%/yr nominal.
    Must reproduce: LTV peak 24.1% @ year 26; D(30)=0.18204e2; NW(30)=5.7918."""
    W, D = 1.0, 0.0
    peak_ltv, peak_year = 0.0, -1
    spend0 = 0.025
    cum_spend = 0.0
    out = {}
    for year in range(41):
        ltv = D / W                      # measured at the start of the year,
        if ltv > peak_ltv:               # before this year's borrow
            peak_ltv, peak_year = ltv, year
        if year in (26, 30, 40):
            out[year] = dict(W=W, D=D, NW=W - D, LTV=ltv, cum_spend=cum_spend)
        borrow = spend0 * 1.03 ** year
        D += borrow
        cum_spend += borrow
        W *= 1.07
        D *= 1.03
    return peak_ltv, peak_year, out
