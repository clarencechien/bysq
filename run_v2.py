"""v2 experiments: borrow-to-spend vs sell, three paths, opportunistic bucket.
python run_v2.py -> results/v2_main.csv, results/v2_h6.csv"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal)
from engine.leverage import LevSpec, simulate_leverage

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
SEEDS = [0, 1, 2]


def tw_panel(twd_cash_real=0.0):
    p = load_panel("tw")
    p["bill"] = p.pop("usdbill")
    p["twd_cash"] = (1 + p["infl"]) * (1 + twd_cash_real) ** (1 / 12.0) - 1
    return p


def pct(x):
    return round(100 * float(np.mean(x)), 2)


def summarize(r, p8_nw, name, seed, horizon, extra=None):
    crash = r["stock_mdd"] <= -0.40
    row = dict(strategy=name, seed=seed, horizon=horizon,
               wiped_pct=pct(r["wiped"]),
               forced_liq_pct=pct(r["forced_liq"]),
               line_forced_pct=pct(r["line_forced"]),
               any_forced_pct=pct(r["forced_liq"] | r["line_forced"] | r["wiped"]),
               peak_ltv_med=round(float(np.median(r["peak_ltv"])) * 100, 1),
               peak_ltv_p95=round(float(np.percentile(r["peak_ltv"], 95)) * 100, 1),
               peak_ltv_p99=round(float(np.percentile(r["peak_ltv"], 99)) * 100, 1),
               first_hit_med=float(np.median(r["first_hit_year"][r["first_hit_year"] >= 0]))
               if (r["first_hit_year"] >= 0).any() else -1,
               nw_med=round(float(np.median(r["final_nw_real"])), 2),
               nw_p5=round(float(np.percentile(r["final_nw_real"], 5)), 2),
               interest_over_spend=round(float(np.median(
                   r["cum_interest_real"] / np.maximum(r["total_real_spend"], 1e-9))), 3),
               nw_mdd_med=round(float(np.median(r["nw_mdd"])) * 100, 1),
               spend_fail_pct=pct(r["spend_fail"]),
               beats_sell_pct=pct(r["final_nw_real"] > p8_nw) if p8_nw is not None else "",
               beats_sell_crash_pct=pct((r["final_nw_real"] > p8_nw)[crash])
               if p8_nw is not None and crash.any() else "",
               crash_frac_pct=pct(crash))
    if extra:
        row.update(extra)
    return row


def main():
    os.makedirs(RESULTS, exist_ok=True)
    panel = tw_panel()
    t_src = len(panel["stock"])
    rows = []
    for horizon in (30, 40):
        T = horizon * 12
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)

            # P8 baseline: sell 100% stock, fixed real 2.5%
            p8 = simulate(idx, panel, Alloc(1.0), FixedReal(0.025), cash_col="bill")
            p8_nw = p8["final_wealth_med"] if False else p8["final_real_wealth"]
            rows.append(dict(strategy="P8-sell-2.5", seed=seed, horizon=horizon,
                             wiped_pct=pct(p8["ruined"]), forced_liq_pct=0,
                             line_forced_pct=0, any_forced_pct=pct(p8["ruined"]),
                             peak_ltv_med=0, peak_ltv_p95=0, peak_ltv_p99=0,
                             first_hit_med=-1,
                             nw_med=round(float(np.median(p8["final_real_wealth"])), 2),
                             nw_p5=round(float(np.percentile(p8["final_real_wealth"], 5)), 2),
                             interest_over_spend=0.0, nw_mdd_med="",
                             spend_fail_pct=pct(p8["spend_fail"]),
                             beats_sell_pct="", beats_sell_crash_pct="", crash_frac_pct=""))

            configs = [
                ("P6c-N0", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                                   borrow_const=0.03)),
                ("P6c-N3-F3", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                      ltv_max=0.30, fallback="F3",
                                      borrow_mode="path_c", borrow_const=0.03)),
                ("P6c-N3-F1", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                      ltv_max=0.30, fallback="F1",
                                      borrow_mode="path_c", borrow_const=0.03)),
                ("P6c-N3-F2", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                      ltv_max=0.30, fallback="F2",
                                      borrow_mode="path_c", borrow_const=0.03)),
                ("P6c-N5-F3", LevSpec(rate=0.025, bucket_years=5, alpha=0.95,
                                      ltv_max=0.30, fallback="F3",
                                      borrow_mode="path_c", borrow_const=0.03)),
                ("P6c-N0-r2.5", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                                        borrow_const=0.025)),
                ("P6c-N0-r3.5", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                                        borrow_const=0.035)),
                ("P6c-N0-noline", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                                          borrow_const=0.03, line_review=False)),
                ("P6c-N3-noline", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                          ltv_max=0.30, fallback="F3",
                                          borrow_mode="path_c", borrow_const=0.03,
                                          line_review=False)),
                ("P6b-N0", LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_b",
                                   borrow_spread=0.015, ltv_kill=0.75,
                                   line_review=False)),
                ("P6b-N3-F3", LevSpec(rate=0.025, bucket_years=3, alpha=0.95,
                                      ltv_max=0.30, fallback="F3",
                                      borrow_mode="path_b", borrow_spread=0.015,
                                      ltv_kill=0.75, line_review=False)),
                ("P6c-N0-4pc", LevSpec(rate=0.04, bucket_years=0, borrow_mode="path_c",
                                       borrow_const=0.03)),
            ]
            for name, spec in configs:
                r = simulate_leverage(idx, panel, spec, cash_col="twd_cash", seed=seed)
                rows.append(summarize(r, p8_nw, name, seed, horizon))
                print(f"{horizon}y s{seed} {name:15s} wiped {rows[-1]['wiped_pct']:5.2f}% "
                      f"forced {rows[-1]['any_forced_pct']:5.2f}% "
                      f"pkLTV {rows[-1]['peak_ltv_med']:5.1f}/{rows[-1]['peak_ltv_p95']:5.1f} "
                      f"NW {rows[-1]['nw_med']:6.2f} beats-sell {rows[-1]['beats_sell_pct']}%"
                      f" (crash {rows[-1]['beats_sell_crash_pct']}%)")
    with open(os.path.join(RESULTS, "v2_main.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/v2_main.csv ({len(rows)} rows)")

    # H6: advantage window by horizon (paired, same idx per horizon)
    h6 = []
    for horizon in (10, 15, 20, 25, 30, 35, 40):
        T = horizon * 12
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
        p8 = simulate(idx, panel, Alloc(1.0), FixedReal(0.025), cash_col="bill")
        spec = LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c", borrow_const=0.03)
        r = simulate_leverage(idx, panel, spec, cash_col="twd_cash", seed=0)
        crash_late = r["stock_mdd"] <= -0.40
        adv = r["final_nw_real"] - p8["final_real_wealth"]
        h6.append(dict(horizon=horizon,
                       p_behind=pct(adv < 0),
                       p_behind_crash=pct((adv < 0)[crash_late]) if crash_late.any() else "",
                       adv_med=round(float(np.median(adv)), 3),
                       adv_p5=round(float(np.percentile(adv, 5)), 3),
                       forced_pct=pct(r["forced_liq"] | r["line_forced"])))
        print(f"H6 {horizon}y: P(lev behind sell) {h6[-1]['p_behind']}% "
              f"(crash-cond {h6[-1]['p_behind_crash']}%) adv med {h6[-1]['adv_med']}")
    with open(os.path.join(RESULTS, "v2_h6.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(h6[0].keys()))
        w.writeheader()
        w.writerows(h6)
    print("wrote results/v2_h6.csv")


if __name__ == "__main__":
    main()
