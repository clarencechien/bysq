"""v4 §5: accumulation phase (A1-A5 glide paths, H4 income interruption,
Coast FIRE) handed as a FULL terminal-wealth distribution to the
withdrawal engine.

python run_v4_accum.py -> results/v4_accum.csv        (terminal W_T tables)
                         results/v4_accum_joint.csv  (accum glide x withdrawal rule)
                         results/v4_accum_cont.csv   (fresh vs continuous draw)

Units: S = planned annual real retirement spending = 1.0. Contributions c
are in S/yr; W_T in units of S; 25 x S is the 4%-rule target.
The withdrawal engine's reference wealth is 1.0 with rate 0.04 = S, so a
path with W_T is handed in as w0 = W_T / 25 and its lifetime real spending
(engine units) is divided by 0.04 to get units of S.
"""
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices,
                         simulate, Alloc, FixedReal, GuytonKlinger, VPW,
                         spend_fail_abs, ce_spending)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
BLOCK = 36
S_RATE = 0.04            # engine reference: spending S == 4% of reference wealth 1.0
TARGET = 25.0            # 4%-rule target in units of S
COAST_G = 0.05           # assumed real growth used ONLY to draw the coast line
COAST_RESUME = 0.8       # resume contributions when W < 0.8 x coast line
H0 = 0.015               # base job-loss hazard per year
LOSS_COST = 0.5          # living cost drawn from portfolio while unemployed, S/yr


# ------------------------------------------------------------- panels ------

def us_panel():
    return restrict(load_panel("us"), "1934-02")


def tw_panel():
    p = load_panel("tw")
    p["bill"] = p.pop("usdbill")
    p["twd_cash"] = (1 + p["infl"]) - 1
    return p


# ------------------------------------------------------------- glides ------

def glide_share(glide, horizon, year):
    """Stock share for plan-year `year` (0-based) of an accumulation lasting
    `horizon` years. yrs_left = years to retirement at the START of the year."""
    yrs_left = horizon - year
    if glide == "A1":
        return 0.6
    if glide == "A2":
        return 0.8
    if glide == "A3":
        return 1.0
    if glide == "A4":      # bond tent: 100/0 until 10y out, linear to 60/40
        if yrs_left >= 10:
            return 1.0
        return 1.0 - 0.4 * (10 - yrs_left) / 10.0
    if glide == "A5":      # age-based: 100/0 until 20y out, linear to 40/60
        if yrs_left >= 20:
            return 1.0
        return 1.0 - 0.6 * (20 - yrs_left) / 20.0
    raise ValueError(glide)


GLIDES = ["A1", "A2", "A3", "A4", "A5"]
GLIDE_DESC = {"A1": "60/40", "A2": "80/20", "A3": "100/0",
              "A4": "tent 100->60/40 last 10y", "A5": "100->40/60 last 20y"}


# ---------------------------------------------------------- simulator ------

def simulate_accum(idx, panel, glide, c, h0=H0, k=1.0, draw_on_loss=True,
                   loss_cost=LOSS_COST, coast_theta=None, coast_g=COAST_G,
                   coast_target=TARGET, coast_resume=COAST_RESUME,
                   shock_seed=0, trace=False):
    """Vectorized monthly accumulation. idx: (P, T) shared-axis row indices.

    Every decision at month t (contribute? crash state? coast?) uses balances
    and prices through t-1 only. Contributions c/12 real (CPI-indexed) are
    added at the START of the month, then month-t returns apply.
    H4: while employed, monthly hazard hm = 1-(1-h0)^(1/12), multiplied by k
    in months where the path's own stock TR index (through t-1) is >20%
    below its trailing 12-month high. A loss lasts U{6..18} months with
    contributions stopped and, if draw_on_loss, loss_cost/yr real drawn
    from the portfolio. k=0 switches H4 off.
    Coast: line_t = coast_target / (1+g)^{years_left}; stop contributing
    when real wealth >= theta x line_t; resume when < coast_resume x line_t.
    """
    P, T = idx.shape
    horizon = T // 12
    stock_r = panel["stock"][idx]
    bond_r = panel["bond"][idx]
    infl_m = panel["infl"][idx]

    # price level at the START of month t (inflation through t-1)
    cum = np.cumprod(1.0 + infl_m, axis=1)
    price = np.concatenate([np.ones((P, 1)), cum[:, :-1]], axis=1)
    deflator_T = cum[:, -1]

    # H4 crash state, all through t-1: the path's own stock TR index vs its
    # trailing 12-month high. Padded with 12 months at level 1.0 so month 0
    # uses only the (flat) pre-history. hit[:, t] = a loss WOULD start at t
    # if the saver is employed; shock draws are pre-drawn so tampering
    # returns after t cannot change anything before t.
    hm = 1.0 - (1.0 - h0) ** (1.0 / 12.0) if (h0 > 0 and k > 0) else 0.0
    if hm > 0:
        lvl = np.cumprod(1.0 + stock_r, axis=1)
        padded = np.concatenate([np.ones((P, 12)), lvl], axis=1)
        # trailing 12-month running max by doubling: m[j] = max(padded[j-11..j])
        m = padded.copy()
        for s in (1, 2, 4, 4):               # 1+1+2+4+4 = 12 (windows overlap)
            m[:, s:] = np.maximum(m[:, s:], m[:, :-s])
        trailing_high = m[:, 11:T + 11]      # max of levels through t-1
        in_crash = padded[:, 11:T + 11] < 0.8 * trailing_high
        del m
        rng = np.random.default_rng(10_000 + shock_seed)
        u_loss = rng.random((P, T))
        dur = rng.integers(6, 19, (P, T))
        hit = u_loss < hm * np.where(in_crash, float(k), 1.0)
        del lvl, padded, trailing_high, u_loss
    else:
        hit = None

    bal_s = np.zeros(P)
    bal_b = np.zeros(P)
    unemp_left = np.zeros(P, dtype=np.int64)
    unemployed = np.zeros(P, bool)
    coasting = np.zeros(P, bool)
    ever_coast = np.zeros(P, bool)
    ever_resume = np.zeros(P, bool)
    first_coast_m = np.full(P, -1)
    months_contrib = np.zeros(P)
    months_unemp = np.zeros(P)
    unfunded = np.zeros(P)             # living cost the portfolio could not pay
    if trace:
        tr_contrib = np.zeros((P, T))
        tr_coast = np.zeros((P, T), bool)
        tr_unemp = np.zeros((P, T), bool)
    log_g = np.log1p(coast_g)

    for t in range(T):
        y = t // 12
        sw = glide_share(glide, horizon, y)
        if t % 12 == 0:
            w = bal_s + bal_b
            bal_s = w * sw
            bal_b = w * (1.0 - sw)

        # ---- H4 state (info through t-1) ----
        if hit is not None:
            new_loss = (unemp_left == 0) & hit[:, t]
            unemp_left = np.where(new_loss, dur[:, t], unemp_left)
            unemployed = unemp_left > 0

        # ---- coast decision (info through t-1) ----
        if coast_theta is not None:
            w_real = (bal_s + bal_b) / price[:, t]
            line = coast_target * np.exp(-log_g * (T - t) / 12.0)
            start = ~coasting & (w_real >= coast_theta * line)
            resume = coasting & (w_real < coast_resume * line)
            first_coast_m = np.where(start & (first_coast_m < 0), t, first_coast_m)
            ever_coast |= start
            ever_resume |= resume
            coasting = (coasting | start) & ~resume

        # ---- start-of-month contribution ----
        active = ~(unemployed | coasting)
        contrib_nom = np.where(active, c / 12.0, 0.0) * price[:, t]
        bal_s = bal_s + contrib_nom * sw
        bal_b = bal_b + contrib_nom * (1.0 - sw)
        months_contrib += active
        months_unemp += unemployed

        # ---- living cost while unemployed ----
        if draw_on_loss and hit is not None:
            need = np.where(unemployed, loss_cost / 12.0 * price[:, t], 0.0)
            w = bal_s + bal_b
            frac = np.minimum(need / np.maximum(w, 1e-15), 1.0)
            unfunded += np.maximum(need - w, 0.0) / price[:, t]
            bal_s = bal_s * (1.0 - frac)
            bal_b = bal_b * (1.0 - frac)

        if trace:
            tr_contrib[:, t] = np.where(active, c / 12.0, 0.0)
            tr_coast[:, t] = coasting
            tr_unemp[:, t] = unemployed

        # ---- apply month-t returns ----
        bal_s = bal_s * (1.0 + stock_r[:, t])
        bal_b = bal_b * (1.0 + bond_r[:, t])
        if hit is not None:
            unemp_left = np.maximum(unemp_left - 1, 0)

    w_T = (bal_s + bal_b) / deflator_T
    out = dict(w_T=w_T, months_contrib=months_contrib, months_unemp=months_unemp,
               ever_coast=ever_coast, ever_resume=ever_resume,
               first_coast_year=np.where(first_coast_m >= 0, first_coast_m / 12.0, np.nan),
               unfunded=unfunded)
    if trace:
        out.update(contrib=tr_contrib, coasting=tr_coast, unemployed=tr_unemp)
    return out


# ------------------------------------------------------------ scoring ------

def accum_row(res, **keys):
    w = res["w_T"]
    row = dict(keys)
    for q in (5, 25, 50, 75, 95):
        row[f"p{q}"] = round(float(np.percentile(w, q)), 2)
    row["lt25_pct"] = round(100 * float((w < 25).mean()), 2)
    row["lt33_pct"] = round(100 * float((w < 33).mean()), 2)
    row["contrib_yrs_med"] = round(float(np.median(res["months_contrib"])) / 12, 2)
    row["unemp_months_mean"] = round(float(res["months_unemp"].mean()), 2)
    row["coast_pct"] = round(100 * float(res["ever_coast"].mean()), 2)
    row["resume_pct"] = round(100 * float(res["ever_resume"].mean()), 2)
    fc = res["first_coast_year"]
    row["coast_year_med"] = round(float(np.nanmedian(fc)), 1) if np.isfinite(fc).any() else ""
    return row


def joint_row(res, **keys):
    s = res["annual_real_spend"] / S_RATE          # units of S
    tot = res["total_real_spend"] / S_RATE
    row = dict(keys)
    row["ruin_pct"] = round(100 * float(res["ruined"].mean()), 2)
    row["spend_p5"] = round(float(np.percentile(tot, 5)), 2)
    row["spend_med"] = round(float(np.median(tot)), 2)
    row["sfabs50"] = round(100 * float(spend_fail_abs(s, 0.5).mean()), 2)
    row["sfabs80"] = round(100 * float(spend_fail_abs(s, 0.8).mean()), 2)
    row["ce_g4"] = round(ce_spending(s, 4, floor_abs=0.0005 / S_RATE), 3)
    row["endw_med"] = round(float(np.median(res["final_real_wealth"])) / S_RATE, 1)
    return row


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


# --------------------------------------------------------------- runs ------

ENGINES = {"US1934": us_panel, "TW1983": tw_panel}
HORIZONS = (20, 30, 40)
CS = (0.5, 1.0)
KS = (0, 1, 2, 4)                 # 0 = H4 off
THETAS = (1.0, 1.25, 1.5, 2.0)


def _accum_job(args):
    """One (engine, accumulation horizon) block: every glide x c x k x coast."""
    ename, H, n_paths = args
    panel = ENGINES[ename]()
    t_src = len(panel["stock"])
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(t_src, H * 12, n_paths, BLOCK, rng)
    rows, store = [], {}
    t0 = time.time()
    for glide in GLIDES:
        for c in CS:
            for k in KS:
                res = simulate_accum(idx, panel, glide, c, k=k)
                store[(ename, glide, H, c, k)] = res["w_T"]
                rows.append(accum_row(res, engine=ename, glide=glide, horizon=H,
                                      c=c, k=k, draw_on_loss=1, coast_theta=""))
            # draw_on_loss OFF control at k=4 (contributions stop, no selling)
            res = simulate_accum(idx, panel, glide, c, k=4, draw_on_loss=False)
            rows.append(accum_row(res, engine=ename, glide=glide, horizon=H,
                                  c=c, k=4, draw_on_loss=0, coast_theta=""))
            # Coast FIRE variants, k in {1, 4}
            for k in (1, 4):
                for th in THETAS:
                    res = simulate_accum(idx, panel, glide, c, k=k, coast_theta=th)
                    rows.append(accum_row(res, engine=ename, glide=glide, horizon=H,
                                          c=c, k=k, draw_on_loss=1, coast_theta=th))
        print(f"accum {ename} {H}y {glide} done ({time.time()-t0:.0f}s)", flush=True)
    return rows, store


def _joint_job(args):
    """One (engine, withdrawal horizon) block with a FRESH seed-1 draw."""
    ename, Hw, n_paths, store = args
    panel = ENGINES[ename]()
    t_src = len(panel["stock"])
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(t_src, Hw * 12, n_paths, BLOCK, rng)
    rows = []
    t0 = time.time()
    for Ha in HORIZONS:
        for c in CS:
            for k in (1, 4):
                for glide in GLIDES:
                    w_T = store[(ename, glide, Ha, c, k)]
                    w0 = np.maximum(w_T / TARGET, 1e-9)
                    for aw in (0.6, 0.8):
                        for rule in (FixedReal(S_RATE), GuytonKlinger(S_RATE), VPW(0.03)):
                            res = simulate(idx, panel, Alloc(aw), rule, w0=w0)
                            rows.append(joint_row(
                                res, engine=ename, acc_glide=glide, acc_horizon=Ha,
                                c=c, k=k, wT_p5=round(float(np.percentile(w_T, 5)), 2),
                                wT_med=round(float(np.median(w_T)), 2),
                                wd_alloc=Alloc(aw).name, rule=rule.name,
                                wd_horizon=Hw, draw="fresh"))
            print(f"joint {ename} wd{Hw} acc{Ha} c{c} done ({time.time()-t0:.0f}s)", flush=True)
    return rows


def _cont_job(args):
    """Same accumulation prefix; withdrawal window either a FRESH seed-1 draw
    or the CONTINUATION of the same bootstrap index sequence (regime
    persistence across the retirement date)."""
    ename, n_paths = args
    Ha, Hw, c = 30, 30, 1.0
    panel = ENGINES[ename]()
    t_src = len(panel["stock"])
    rng = np.random.default_rng(0)
    idx_long = stationary_bootstrap_indices(t_src, (Ha + Hw) * 12, n_paths, BLOCK, rng)
    idx_acc, idx_cont = idx_long[:, :Ha * 12], idx_long[:, Ha * 12:]
    rng = np.random.default_rng(1)
    idx_fresh = stationary_bootstrap_indices(t_src, Hw * 12, n_paths, BLOCK, rng)
    rows = []
    for k in (1, 4):
        for glide in ("A1", "A3", "A5"):
            acc = simulate_accum(idx_acc, panel, glide, c, k=k)
            w_T = acc["w_T"]
            w0 = np.maximum(w_T / TARGET, 1e-9)
            for draw, idx in (("fresh", idx_fresh), ("continuous", idx_cont)):
                for rule in (FixedReal(S_RATE), GuytonKlinger(S_RATE), VPW(0.03)):
                    res = simulate(idx, panel, Alloc(0.6), rule, w0=w0)
                    rows.append(joint_row(
                        res, engine=ename, acc_glide=glide, acc_horizon=Ha, c=c, k=k,
                        wT_p5=round(float(np.percentile(w_T, 5)), 2),
                        wT_med=round(float(np.median(w_T)), 2),
                        wd_alloc="60/40", rule=rule.name, wd_horizon=Hw, draw=draw))
    print(f"continuous {ename} done", flush=True)
    return rows


def run_all(n_paths=N_PATHS, workers=4):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as pool:
        acc = pool.map(_accum_job, [(e, H, n_paths) for e in ENGINES for H in HORIZONS])
    rows, store = [], {}
    for r, s in acc:
        rows += r
        store.update(s)
    write("v4_accum.csv", rows)
    print(f"accum phase {time.time()-t0:.0f}s", flush=True)
    with Pool(workers) as pool:
        jj = pool.map(_joint_job, [(e, Hw, n_paths, store) for e in ENGINES for Hw in (30, 50)])
        cc = pool.map(_cont_job, [(e, n_paths) for e in ENGINES])
    write("v4_accum_joint.csv", [r for rows in jj for r in rows])
    write("v4_accum_cont.csv", [r for rows in cc for r in rows])
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_PATHS
    run_all(n)
