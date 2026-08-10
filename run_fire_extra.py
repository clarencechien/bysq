"""Extra FIRE cells: the 100M tier stress band (250k/300k/350k per month,
i.e. 3.0%/3.6%/4.2% withdrawal) that the first pass never tested.
python run_fire_extra.py -> merges into results/fire_case.json"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices
import run_fire_case as F

tw = load_panel("tw")
tw["bill"] = tw.pop("usdbill")
us = restrict(load_panel("us"), "1934-02")
rng = np.random.default_rng(7)
death_h = F.death_months_from50(F.N_PATHS, 18.2, 0.105, rng)
death_w = F.death_months_from50(F.N_PATHS, 21.9, 0.110, rng)

path = os.path.join(F.RESULTS, "fire_case.json")
out = json.load(open(path))
for ename, panel in [("US_STRESS", us), ("TW", tw)]:
    rng2 = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(len(panel["stock"]), F.T, F.N_PATHS, 36, rng2)
    for pname, (peach, prem) in F.PENSION.items():
        for spend in (250_000.0, 300_000.0, 350_000.0):
            for strat in ("DEPOSIT", "FIX", "GK", "VPW"):
                r = F.simulate(idx, panel, strat, 100_000_000.0, peach, prem,
                               spend, death_h, death_w)
                r.pop("annual")
                key = f"{ename}|100M|{pname}|{int(spend/1000)}k|{strat}"
                out[key] = r
                print(f"{key:40s} ruin {r['ruin_alive_pct']:5.1f}% "
                      f"(<65 {r['ruin_before65_pct']:4.1f}) life {r['lifetime_med']:6.0f} "
                      f"p5 {r['lifetime_p5']:6.0f}"
                      + (f" @{r['ruin_age_med']}" if r['ruin_age_med'] else ""))
with open(path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("merged into results/fire_case.json")
