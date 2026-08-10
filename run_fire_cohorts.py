"""FIRE-50 三種人生: the DINK couple (NT$50M, 15萬/mo, union pension top-up)
dropped into 1966 / 1973 / 1982 openings. Husband dies 84, wife 91.
python run_fire_cohorts.py"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict
import run_fire_case as F

us = restrict(load_panel("us"), "1934-02")
T = 41 * 12                       # age 50 -> 91
death_h = np.array([(84 - 50) * 12])
death_w = np.array([(91 - 50) * 12])
peach, prem = F.PENSION["union"]

out = {}
for start in ("1966-01", "1973-01", "1982-06"):
    i = us["dates"].index(start)
    idx = np.arange(i, i + T)[None, :]
    for strat in ("FIX", "GK", "VPW"):
        r = F.simulate(idx, us, strat, 50_000_000.0, peach, prem, 150_000.0,
                       death_h, death_w)
        yearly = (r.pop("annual")[0] / 12 / 1e4).round(2)
        out[f"{start}|{strat}"] = dict(
            yearly_monthly_avg_wan=yearly.tolist(),
            ruin_age=r["ruin_age_med"], lifetime_wan=r["lifetime_med"],
            bequest_wan=r["bequest_med"],
            min_spend_wan=float(yearly[yearly > 0].min()),
            max_spend_wan=float(yearly.max()))
        print(f"{start} {strat:4s} life {r['lifetime_med']:6.0f}萬 "
              f"min {out[f'{start}|{strat}']['min_spend_wan']:5.2f} "
              f"max {out[f'{start}|{strat}']['max_spend_wan']:5.2f} 萬/月 "
              f"beq {r['bequest_med']:7.0f}萬 "
              + (f"RUIN@{r['ruin_age_med']}" if r['ruin_age_med'] else ""))
with open(os.path.join(F.RESULTS, "fire_cohorts.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote results/fire_cohorts.json")
