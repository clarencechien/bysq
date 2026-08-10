"""US stress-regime check for the v2 leverage strategies (handoff-v2 H6/H7).

The TW engine sample (1983-2023) contains no sustained inflation/high-rate
regime; this run repeats the core comparison on the US 1934-2023 panel where
the margin rate rides the REAL bill series (incl. the 1970s-80s, bills >15%).
python run_v2_us_stress.py -> results/v2_us_stress.csv
"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices,
                         simulate, Alloc, FixedReal)
from engine.leverage import LevSpec, simulate_leverage

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

us = restrict(load_panel("us"), "1934-02")
us["twd_cash"] = us["bill"]
t_src = len(us["stock"])
rows = []
for horizon in (30, 40):
    T = horizon * 12
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(t_src, T, 10_000, 36, rng)
    p8 = simulate(idx, us, Alloc(1.0), FixedReal(0.025), cash_col="bill")
    p8nw = p8["final_real_wealth"]
    rows.append(dict(strategy="P8-sell-2.5", horizon=horizon,
                     wiped_pct=round(p8["ruined"].mean() * 100, 2),
                     forced_pct=0, peak_ltv_p95=0,
                     nw_med=round(float(np.median(p8nw)), 2),
                     nw_p5=round(float(np.percentile(p8nw, 5)), 2),
                     beats_sell_pct="", crash_cond_pct="",
                     interest_over_spend=0.0,
                     spend_fail_pct=round(p8["spend_fail"].mean() * 100, 2)))
    for name, spec in [
            ("P6-margin-N0", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_b",
                                     borrow_spread=0.015, ltv_kill=0.75, line_review=False)),
            ("P6-margin-N3-F3", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                        ltv_max=0.30, fallback="F3", borrow_mode="path_b",
                                        borrow_spread=0.015, ltv_kill=0.75, line_review=False)),
            ("P6-const3-N0", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                                     borrow_const=0.03, line_review=False)),
            ("P6-margin-N0-4pc", LevSpec(rate=0.04, bucket_years=0, borrow_mode="path_b",
                                         borrow_spread=0.015, ltv_kill=0.75, line_review=False)),
    ]:
        r = simulate_leverage(idx, us, spec, cash_col="twd_cash", seed=0)
        crash = r["stock_mdd"] <= -0.40
        adv = r["final_nw_real"] - p8nw
        rows.append(dict(strategy=name, horizon=horizon,
                         wiped_pct=round(r["wiped"].mean() * 100, 2),
                         forced_pct=round((r["forced_liq"] | r["line_forced"]).mean() * 100, 2),
                         peak_ltv_p95=round(float(np.percentile(r["peak_ltv"], 95)) * 100, 1),
                         nw_med=round(float(np.median(r["final_nw_real"])), 2),
                         nw_p5=round(float(np.percentile(r["final_nw_real"], 5)), 2),
                         beats_sell_pct=round((adv > 0).mean() * 100, 1),
                         crash_cond_pct=round((adv > 0)[crash].mean() * 100, 1),
                         interest_over_spend=round(float(np.median(
                             r["cum_interest_real"] / np.maximum(r["total_real_spend"], 1e-9))), 2),
                         spend_fail_pct=round(r["spend_fail"].mean() * 100, 2)))
        print(rows[-1])
with open(os.path.join(RESULTS, "v2_us_stress.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote results/v2_us_stress.csv")
