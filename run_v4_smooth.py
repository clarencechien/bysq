"""v4 §3: VPW smoothing layers, Taiwan RMD, horizon-extension cliff, W6/W7.
Fixed 50-year horizon (v3-comparable) + age-65 mortality post hoc.
python run_v4_smooth.py [--quick] -> results/v4_smooth.csv"""
import csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, GuytonKlinger, VPW, RMD, FundedRatio,
                         RiskGuardrail, VanguardDynamic, spend_fail_abs, ce_spending)
from engine.mortality import load_qx, sample_death_years, alive_metrics
from engine.household import load_shiller_extra

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000
CUTOFF = "2023-07"
AGE = 65


def smooth_metrics(s):
    with np.errstate(divide="ignore", invalid="ignore"):
        chg = np.diff(s, axis=1) / np.maximum(s[:, :-1], 1e-9)
    peak = np.maximum.accumulate(s, axis=1)
    dd = 1 - s / np.maximum(peak, 1e-9)
    return dict(spend_vol=round(float(np.median(np.std(chg, axis=1))), 4),
                spend_dd_max=round(float(np.median(dd.max(axis=1))), 3),
                spend_dd_yrs=float(np.median((dd > 0.2).sum(axis=1))))


def score(res):
    s = res["annual_real_spend"]
    row = dict(ruin_pct=round(100 * res["ruined"].mean(), 2),
               planned_dep_pct=round(100 * res["planned_depletion"].mean(), 2),
               spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
               spend_med=round(float(np.median(res["total_real_spend"])), 3),
               endw_med=round(float(np.median(res["final_real_wealth"])), 2),
               sfabs15=round(100 * spend_fail_abs(s, 0.015).mean(), 2),
               sfabs20=round(100 * spend_fail_abs(s, 0.02).mean(), 2),
               sfabs25=round(100 * spend_fail_abs(s, 0.025).mean(), 2),
               spend_y1=round(float(np.median(s[:, 0])), 4),
               spend_y30_med=round(float(np.median(s[:, 29])), 4),
               spend_y30_p10=round(float(np.percentile(s[:, 29], 10)), 4))
    for g in (2, 4, 8):
        row[f"ce_g{g}"] = round(ce_spending(s, g), 4)
    row.update(smooth_metrics(s))
    return row


def strategies(qx, has_cape):
    L = []
    for w in (0.6, 0.8):
        a = Alloc(w)
        L += [("GK4", a, GuytonKlinger(0.04)), ("GK5", a, GuytonKlinger(0.05)),
              ("VPW3", a, VPW(0.03)), ("VPW5", a, VPW(0.05)),
              ("V-cap 5/2.5", a, VPW(0.03, smooth=("cap", 0.05, 0.025))),
              ("V-cap 5/5", a, VPW(0.03, smooth=("cap", 0.05, 0.05))),
              ("V-cap 10/5", a, VPW(0.03, smooth=("cap", 0.10, 0.05))),
              ("V-yale 0.5", a, VPW(0.03, smooth=("yale", 0.5))),
              ("V-yale 0.7", a, VPW(0.03, smooth=("yale", 0.7))),
              ("V-yale 0.8", a, VPW(0.03, smooth=("yale", 0.8))),
              ("V-floor fl2", a, VPW(0.03, floor_abs=0.02)),
              ("VG4", a, VanguardDynamic(0.04)),
              ("W6-FR4", a, FundedRatio(0.04)), ("W7-PoS4", a, RiskGuardrail(0.04)),
              ("RMD", a, RMD())]
        for buf in (0, 5, 10):
            L.append((f"TW-RMD r5 +{buf}", a, VPW(0.05, life_table=(qx["all"], AGE, buf))))
            if has_cape:
                L.append((f"TW-RMD 1/CAPE +{buf}", a, VPW(0.05, life_table=(qx["all"], AGE, buf), cape_rate=True)))
        # §3.3 cliff test: plan to 100 (35y from 65), extend when reached vs not
        L += [("VPW3 plan100 no-ext", a, VPW(0.03, horizon=35)),
              ("VPW3 plan100 ext90/100/110", a, VPW(0.03, horizon=35, extensions=[(25, 45), (35, 55), (45, 65)])),
              ("VPW3 plan110 no-ext", a, VPW(0.03, horizon=45))]
    return L


def main():
    qx = load_qx(114)
    us = restrict(load_panel("us"), "1934-02", CUTOFF)
    us["gs10"], us["cape"] = load_shiller_extra(us["dates"])
    tw = restrict(load_panel("tw"), None, CUTOFF); tw["bill"] = tw.pop("usdbill")
    rows = []
    t0 = time.time()
    for ename, panel in [("US1934", us), ("TW1983", tw)]:
        t_src = len(panel["stock"])
        T = 50 * 12
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
        drng = np.random.default_rng(100)
        death = sample_death_years(qx["all"], AGE, N_PATHS, drng)
        for tag, alloc, rule in strategies(qx, ename == "US1934"):
            res = simulate(idx, panel, alloc, rule)
            row = dict(engine=ename, alloc=alloc.name, strategy=tag, rule=rule.name)
            row.update(score(res))
            m = alive_metrics(res, death, 0.02)
            row.update({f"m65_{k}": v for k, v in m.items() if k in ("ruin_alive_pct", "breach_alive_pct", "ce_g4_alive", "life_spend_med")})
            rows.append(row)
            print(f"{ename} {alloc.name} {tag:28s} ruin {row['ruin_pct']:5.2f} sf20 {row['sfabs20']:5.2f} "
                  f"p5 {row['spend_p5']:.3f} med {row['spend_med']:.3f} vol {row['spend_vol']:.3f} "
                  f"ddmax {row['spend_dd_max']:.2f} ce4 {row['ce_g4']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(RESULTS, "v4_smooth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote results/v4_smooth.csv", len(rows))


if __name__ == "__main__":
    main()
