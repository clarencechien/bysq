"""v4 §0.2: 勞退 monthly pension modelled as a fixed-term nominal payout
(to life expectancy at claim, recomputed every 3 years) vs the earlier
'lump sum folded into investable assets' treatment. 10M-type household:
勞保 5% of W0 (42k/mo on 10M), 勞退 account 20% of W0, floor 70%.
python run_v4_lr.py -> results/v4_lr.csv"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices
from engine.mortality import load_qx, sample_death_years
from engine.household import HH, simulate_household

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000


def main():
    qx = load_qx(114)
    us = restrict(load_panel("us"), "1934-02", "2023-07")
    tw = restrict(load_panel("tw"), None, "2023-07"); tw["bill"] = tw.pop("usdbill")
    rows = []
    for ename, panel in [("US1934", us), ("TW1983", tw)]:
        T = 45 * 12
        idx = stationary_bootstrap_indices(len(panel["stock"]), T, N_PATHS, 36, np.random.default_rng(0))
        drng = np.random.default_rng(7)
        death = sample_death_years(qx["all"], 65, N_PATHS, drng) * 12 + drng.integers(0, 12, N_PATHS)
        for pm in (1.0, 0.7, 0.0):
            for w, rule in [(0.8, "VPW"), (0.8, "GK"), (0.8, "FIX"), (0.6, "VPW")]:
                for lr_mode in ("lump_sum_invested", "monthly_term", "monthly_lifetime(舊模型)"):
                    kw = dict(retire_age=65, spend=0.06, floor_ratio=0.7, pension=0.05, pension_mult=pm,
                              stock_w=w, rule=rule)
                    if lr_mode == "monthly_term":
                        kw["lr_account"] = 0.2; kw["lr_claim_age"] = 65
                    elif lr_mode.startswith("monthly_lifetime"):
                        # old treatment: the same account bought a lifetime real annuity at a fair price
                        from engine.household import annuity_factor
                        af = annuity_factor(qx["all"], 65, 65, 0.01, shift=0)
                        kw["pension"] = 0.05 * pm + 0.2 / af / max(pm, 1e-9) if pm > 0 else 0.2 / af
                        kw["pension_mult"] = pm if pm > 0 else 1.0
                        kw["_note"] = "lr as lifetime real annuity"
                    kw.pop("_note", None)
                    hh = HH(**kw)
                    if lr_mode == "lump_sum_invested":
                        pass  # 0.2 stays inside the investable 1.0
                    r = simulate_household(idx, panel, hh, death, qx=qx)
                    rows.append(dict(engine=ename, pension_mult=pm, alloc=f"{int(w*100)}/{int(round((1-w)*100))}",
                                     rule=rule, lr_mode=lr_mode,
                                     **{k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k != "name"}))
                    print(rows[-1]["engine"], pm, rule, lr_mode, "ruin", rows[-1]["ruin_alive_pct"],
                          "breach1", rows[-1]["breach1_pct"], "avg_med", rows[-1]["avg_med"], flush=True)
    with open(os.path.join(RESULTS, "v4_lr.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("wrote results/v4_lr.csv", len(rows))


if __name__ == "__main__":
    main()
