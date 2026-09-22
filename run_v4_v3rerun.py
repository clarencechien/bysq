"""v4 §0.1: re-emit the v3 main-table VPW/RMD rows under the corrected
`ruined` accounting (exhausted with >1 plan-year left; in-plan depletion is a
separate column). Same seeds/paths/panel cutoff as run_v3.py.
python run_v4_v3rerun.py -> results/v4_v3rerun.csv"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, GuytonKlinger, VPW, RMD, spend_fail_abs, ce_spending)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000


def main():
    us = restrict(load_panel("us"), "1934-02", "2023-07")
    tw = restrict(load_panel("tw"), None, "2023-07"); tw["bill"] = tw.pop("usdbill")
    rows = []
    for ename, panel in [("US1934", us), ("TW1983", tw)]:
        for horizon in (30, 50):
            rng = np.random.default_rng(0)
            idx = stationary_bootstrap_indices(len(panel["stock"]), horizon * 12, N_PATHS, 36, rng)
            for alloc, rule in [(Alloc(0.6), FixedReal(0.04)), (Alloc(0.6), GuytonKlinger(0.04)),
                                (Alloc(0.8), GuytonKlinger(0.05)),
                                (Alloc(0.6), VPW(0.03)), (Alloc(0.8), VPW(0.03)),
                                (Alloc(0.6), RMD()), (Alloc(0.8), RMD())]:
                res = simulate(idx, panel, alloc, rule)
                s = res["annual_real_spend"]
                row = dict(engine=ename, horizon=horizon, strategy=f"{alloc.name} {rule.name}",
                           ruin_pct=round(100 * res["ruined"].mean(), 2),
                           planned_depletion_pct=round(100 * res["planned_depletion"].mean(), 2),
                           exhausted_any_pct=round(100 * res["exhausted"].mean(), 2),
                           spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
                           spend_med=round(float(np.median(res["total_real_spend"])), 3))
                for fl in (0.015, 0.02, 0.025):
                    row[f"sfabs{int(fl*1000)}"] = round(100 * spend_fail_abs(s, fl).mean(), 2)
                for g in (2, 4, 8):
                    row[f"ce_g{g}"] = round(ce_spending(s, g), 4)
                rows.append(row)
                print(row)
    with open(os.path.join(RESULTS, "v4_v3rerun.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote results/v4_v3rerun.csv")


if __name__ == "__main__":
    main()
