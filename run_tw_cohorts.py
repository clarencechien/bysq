"""三種人生: the same 65-year-old couple (NT$10M, full pension, own home)
dropped into the 1966 / 1973 / 1982 openings (US real data as the stress
history Taiwan has not yet had). Husband dies at 84, wife at 91 (medians
from the calibrated life table). python run_tw_cohorts.py"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict
import run_tw_case as C

us = restrict(load_panel("us"), "1934-02")
T = 26 * 12                      # to wife's death at 91
death_h = np.array([(84 - 65) * 12])
death_w = np.array([(91 - 65) * 12])

out = {}
for start in ("1966-01", "1973-01", "1982-08"):
    i = us["dates"].index(start)
    idx = np.arange(i, i + T)[None, :]
    for strat in ("FIX", "GK", "VPW", "DEPOSIT"):
        r = C.simulate_case(idx, us, strat, 1.0, "own", death_h, death_w)
        yearly = (r.pop("annual_total_real")[0] / 12 / 1e4).round(2)  # 萬/月 avg
        out[f"{start}|{strat}"] = dict(
            yearly_monthly_avg_wan=yearly.tolist(),
            ruin=r["ruin_alive_pct"] > 0, ruin_age=r["ruin_age_med"],
            lifetime_wan=r["lifetime_med"], bequest_wan=r["bequest_med"],
            floor_breach=r["floor_breach_pct"] > 0,
            min_spend_wan=float(yearly[yearly > 0].min()),
            max_spend_wan=float(yearly.max()))
        print(f"{start} {strat:8s} life {r['lifetime_med']:5.0f}萬 "
              f"min {out[f'{start}|{strat}']['min_spend_wan']:.2f} "
              f"max {out[f'{start}|{strat}']['max_spend_wan']:.2f} 萬/月 "
              f"bequest {r['bequest_med']:6.0f}萬 "
              + (f"RUIN@{r['ruin_age_med']}" if r['ruin_age_med'] else ""))
with open(os.path.join(C.RESULTS, "tw_cohorts.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote results/tw_cohorts.json")
