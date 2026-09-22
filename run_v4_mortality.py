"""v4 §2 + §10.2: mortality into the main conclusions.
Runs the core engine to age 110 for retirement ages {40,50,60,65}, then
applies the life table post hoc (single 'all' / joint m+f; longevity_shift
0/+3/+6) and a fixed-50-year reference on the same paths.
python run_v4_mortality.py [--quick] -> results/v4_mortality.csv"""
import csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, GuytonKlinger, VPW, RMD, FundedRatio,
                         RiskGuardrail, spend_fail_abs, ce_spending)
from engine.mortality import load_qx, sample_death_years, joint_last_survivor, alive_metrics, survival_curve

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000
CUTOFF = "2023-07"


def panels():
    us = restrict(load_panel("us"), "1934-02", CUTOFF)
    tw = restrict(load_panel("tw"), None, CUTOFF); tw["bill"] = tw.pop("usdbill")
    return [("US1934", us), ("TW1983", tw)]


def strategies(age):
    h100 = 100 - age
    ext = [(90 - age, 110 - age), (100 - age, 120 - age)]
    return [
        ("FIX4 60/40", Alloc(0.6), FixedReal(0.04)),
        ("FIX4 100/0", Alloc(1.0), FixedReal(0.04)),
        ("FIX3 60/40", Alloc(0.6), FixedReal(0.03)),
        ("GK4 60/40", Alloc(0.6), GuytonKlinger(0.04)),
        ("GK5 80/20", Alloc(0.8), GuytonKlinger(0.05)),
        ("GK4-fl70 60/40", Alloc(0.6), GuytonKlinger(0.04, floor_ratio=0.7)),
        ("VPW3 60/40", Alloc(0.6), VPW(0.03, horizon=h100, extensions=ext)),
        ("VPW3 80/20", Alloc(0.8), VPW(0.03, horizon=h100, extensions=ext)),
        ("VPW3 100/0", Alloc(1.0), VPW(0.03, horizon=h100, extensions=ext)),
        ("RMD 60/40", Alloc(0.6), RMD()),
        ("W6-FR4 60/40", Alloc(0.6), FundedRatio(0.04)),
        ("W7-PoS4 60/40", Alloc(0.6), RiskGuardrail(0.04)),
    ]


def main():
    qx = load_qx(114)
    rows = []
    t0 = time.time()
    for ename, panel in panels():
        t_src = len(panel["stock"])
        for age in (40, 50, 60, 65):
            T = (110 - age) * 12
            rng = np.random.default_rng(0)
            idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
            # death draws (independent of markets), per variant
            variants = {}
            for shift in (0, 3, 6):
                drng = np.random.default_rng(100 + shift)
                variants[("single", shift)] = sample_death_years(qx["all"], age, N_PATHS, drng, shift)
                _, _, dj = joint_last_survivor(qx["m"], qx["f"], age, age, N_PATHS, drng, shift)
                variants[("joint", shift)] = dj
            fixed50 = np.full(N_PATHS, min(50, 110 - age))
            variants[("fixed50", 0)] = fixed50
            S = survival_curve(qx["all"], age)
            p90 = int(np.searchsorted(-S, -0.10))
            for tag, alloc, rule in strategies(age):
                res = simulate(idx, panel, alloc, rule)
                for (mode, shift), dy in variants.items():
                    m = alive_metrics(res, dy, 0.02)
                    long = dy >= p90
                    row = dict(engine=ename, age=age, mortality=mode, shift=shift, strategy=tag,
                               e_years=round(float(dy.mean()), 1), **m)
                    row["p90_ruin_pct"] = round(100 * float(((res["ruin_year"] >= 0) & (res["ruin_year"] < dy))[long].mean()), 2) if long.sum() > 50 else ""
                    rows.append(row)
                print(f"{ename} age{age} {tag}: fixed50 ruin {rows[-1]['ruin_alive_pct'] if False else ''}"
                      f" single ruin {[r for r in rows if r['strategy']==tag and r['engine']==ename and r['age']==age and r['mortality']=='single' and r['shift']==0][0]['ruin_alive_pct']}"
                      f" ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(RESULTS, "v4_mortality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote results/v4_mortality.csv", len(rows))


if __name__ == "__main__":
    main()
