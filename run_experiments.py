"""Experiment runner. python run_experiments.py → results/*.csv"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, simulate, historical_indices,
                         stationary_bootstrap_indices, Alloc, BucketAlloc,
                         FixedReal, GuytonKlinger, VanguardDynamic, ce_spending)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

N_PATHS = 10_000
SEEDS = [0, 1, 2]
BLOCKS_MAIN = 36
BLOCKS_SENS = [12, 60]


def us_panel():
    return restrict(load_panel("us"), "1934-02")  # bills exist from here


def tw_panel(twd_cash_real=0.0):
    p = load_panel("tw")
    p["bill"] = p.pop("usdbill")          # USD bills converted to TWD
    m = (1 + p["infl"]) * (1 + twd_cash_real) ** (1 / 12.0) - 1
    p["twd_cash"] = m                     # ASSUMPTION: TWD cash earns CPI + r
    return p


def strategy_matrix():
    out = []
    for w, r in [(0.6, 0.04), (0.8, 0.04), (1.0, 0.04)]:
        out.append((Alloc(w), FixedReal(r), "bill"))
    out.append((Alloc(0.6), GuytonKlinger(0.04), "bill"))
    out.append((Alloc(0.6), GuytonKlinger(0.05), "bill"))
    out.append((Alloc(0.8), GuytonKlinger(0.05), "bill"))
    out.append((Alloc(0.6), VanguardDynamic(0.04), "bill"))
    out.append((BucketAlloc(3), FixedReal(0.04), "bill"))
    out.append((BucketAlloc(5), FixedReal(0.04), "bill"))
    out.append((BucketAlloc(3), FixedReal(0.05), "bill"))
    out.append((BucketAlloc(3), GuytonKlinger(0.05), "bill"))
    return out


def summarize(res, gamma=4):
    n = len(res["ruined"])
    ruin = res["ruined"].mean()
    sf = res["spend_fail"].mean()
    return dict(
        n=n,
        ruin_pct=round(100 * ruin, 2),
        ruin_se=round(100 * np.sqrt(ruin * (1 - ruin) / n), 2),
        spendfail_pct=round(100 * sf, 2),
        spendfail_se=round(100 * np.sqrt(sf * (1 - sf) / n), 2),
        spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
        spend_med=round(float(np.median(res["total_real_spend"])), 3),
        yrs_below80_med=float(np.median(res["yrs_below80"])),
        yrs_below80_p90=float(np.percentile(res["yrs_below80"], 90)),
        final_wealth_med=round(float(np.median(res["final_real_wealth"])), 3),
        ce_g4=round(ce_spending(res["annual_real_spend"], gamma), 4),
    )


def run_all():
    rows = []
    engines = {"US1934": us_panel(), "TW1983": tw_panel()}
    for ename, panel in engines.items():
        t_src = len(panel["stock"])
        for horizon in (30, 50):
            T = horizon * 12
            samplers = []
            hidx = historical_indices(t_src, T)
            if hidx.shape[0] >= 50:
                samplers.append(("hist", None, hidx))
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                samplers.append((f"boot{BLOCKS_MAIN}", seed,
                                 stationary_bootstrap_indices(t_src, T, N_PATHS, BLOCKS_MAIN, rng)))
            for bl in BLOCKS_SENS:
                rng = np.random.default_rng(0)
                samplers.append((f"boot{bl}", 0,
                                 stationary_bootstrap_indices(t_src, T, N_PATHS, bl, rng)))
            for sname, seed, idx in samplers:
                for alloc, rule, cash in strategy_matrix():
                    res = simulate(idx, panel, alloc, rule, cash_col=cash)
                    row = dict(engine=ename, sampler=sname, seed=seed,
                               horizon=horizon, alloc=alloc.name, rule=rule.name,
                               bucket_ccy="USD" if isinstance(alloc, BucketAlloc) else "")
                    row.update(summarize(res))
                    rows.append(row)
                    print(f"{ename} {sname} s{seed} {horizon}y {alloc.name:8s} {rule.name:6s} "
                          f"ruin {row['ruin_pct']:5.1f}% sf {row['spendfail_pct']:5.1f}% "
                          f"p5 {row['spend_p5']:.2f} med {row['spend_med']:.2f}")
                # TWD-cash bucket variant (TW engine only, main sampler only)
                if ename == "TW1983" and sname == f"boot{BLOCKS_MAIN}":
                    for real in (-0.01, 0.0, 0.01):
                        p2 = tw_panel(real)
                        alloc = BucketAlloc(3)
                        res = simulate(idx, p2, alloc, FixedReal(0.04), cash_col="twd_cash")
                        row = dict(engine=ename, sampler=sname, seed=seed,
                                   horizon=horizon, alloc=alloc.name, rule="FIX4",
                                   bucket_ccy=f"TWD(r={real:+.0%})")
                        row.update(summarize(res))
                        rows.append(row)
    with open(os.path.join(RESULTS, "main.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/main.csv ({len(rows)} rows)")


def run_swr_sweep():
    """Max initial rate such that ruin<=5% AND spend-fail<=5%, 50y bootstrap."""
    rows = []
    for ename, panel in {"US1934": us_panel(), "TW1983": tw_panel()}.items():
        t_src = len(panel["stock"])
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(t_src, 600, N_PATHS, BLOCKS_MAIN, rng)
        for build in [lambda r: (Alloc(0.6), FixedReal(r)),
                      lambda r: (Alloc(1.0), FixedReal(r)),
                      lambda r: (Alloc(0.6), GuytonKlinger(r)),
                      lambda r: (BucketAlloc(3), FixedReal(r))]:
            for rate in np.arange(0.025, 0.0575, 0.0025):
                alloc, rule = build(rate)
                res = simulate(idx, panel, alloc, rule)
                rows.append(dict(engine=ename, alloc=alloc.name, rule=rule.name,
                                 rate=round(rate, 4),
                                 ruin_pct=round(100 * res["ruined"].mean(), 2),
                                 spendfail_pct=round(100 * res["spend_fail"].mean(), 2),
                                 spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
                                 spend_med=round(float(np.median(res["total_real_spend"])), 3)))
                print(f"sweep {ename} {alloc.name:8s} {rule.name:8s} {rate:.3f} "
                      f"ruin {rows[-1]['ruin_pct']:5.1f} sf {rows[-1]['spendfail_pct']:5.1f}")
    with open(os.path.join(RESULTS, "swr_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/swr_sweep.csv ({len(rows)} rows)")


if __name__ == "__main__":
    run_all()
    run_swr_sweep()
