"""v4 §10.1: per-strategy bias sensitivity. The Taiwan 1983-2023 sample is
optimistic in a non-uniform way: sellers are flattered by returns only,
borrowers by returns AND the rate spread. Pull the TW panel toward the US
stress panel along two separate dials (lambda_ret, lambda_rate in [0,1]) and
find where the sell-vs-pledge ranking flips.
  lambda_ret : monthly stock/bond returns shifted so the TW mean real return
               moves lambda of the way to the US1934 mean real return
  lambda_rate: borrowing cost moves from the v2 constant 3% (Path C) toward
               the real US T-bill series + 1.5% (Path B margin, same time axis)
python run_v4_bias.py [--quick] -> results/v4_bias.csv"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, VPW, spend_fail_abs)
from engine.leverage import LevSpec, simulate_leverage

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000


def main():
    us = restrict(load_panel("us"), "1934-02", "2023-07")
    tw = restrict(load_panel("tw"), None, "2023-07"); tw["bill"] = tw.pop("usdbill")
    tw["twd_cash"] = tw["infl"]
    # monthly real-return means
    def real_mean(p, k):
        return float(np.mean((1 + p[k]) / (1 + p["infl"]) - 1))
    d_stock = real_mean(tw, "stock") - real_mean(us, "stock")
    d_bond = real_mean(tw, "bond") - real_mean(us, "bond")
    print(f"monthly real mean gap TW-US: stock {d_stock*12*100:.2f} pp/yr, bond {d_bond*12*100:.2f} pp/yr")
    rows = []
    T = 360
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(len(tw["stock"]), T, N_PATHS, 36, rng)
    for lam_ret in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = dict(tw)
        p["stock"] = tw["stock"] - lam_ret * d_stock
        p["bond"] = tw["bond"] - lam_ret * d_bond
        # sellers (no rate dial)
        for tag, alloc, rule in [("P8 sell FIX2.5 100/0", Alloc(1.0), FixedReal(0.025)),
                                 ("sell FIX4 60/40", Alloc(0.6), FixedReal(0.04)),
                                 ("sell VPW3 80/20", Alloc(0.8), VPW(0.03))]:
            r = simulate(idx, p, alloc, rule)
            rows.append(dict(lambda_ret=lam_ret, lambda_rate="", strategy=tag,
                             ruin_pct=round(100 * r["ruined"].mean(), 2), forced_pct="",
                             nw_med=round(float(np.median(r["final_real_wealth"])), 2),
                             nw_p5=round(float(np.percentile(r["final_real_wealth"], 5)), 2),
                             spend_med=round(float(np.median(r["total_real_spend"])), 3),
                             sfabs20=round(100 * spend_fail_abs(r["annual_real_spend"], 0.02).mean(), 2)))
        sell_nw = simulate(idx, p, Alloc(1.0), FixedReal(0.025))["final_real_wealth"]   # path-wise, same idx
        for lam_rate in (0.0, 0.5, 1.0):
            # borrow cost: blend constant 3% with (bill + 1.5%) path
            if lam_rate == 0.0:
                spec = LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c", borrow_const=0.03,
                               ltv_kill=0.77, line_review=True)
                r = simulate_leverage(idx, p, spec, cash_col="twd_cash", seed=0)
            else:
                # emulate the blend by shifting the bill series: borrow = bill*lam + const*(1-lam) + spread*lam
                q = dict(p)
                const_m = (1.03) ** (1 / 12) - 1
                q["bill"] = lam_rate * ((1 + p["bill"]) * 1.015 ** (1 / 12) - 1) + (1 - lam_rate) * const_m
                spec = LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c_var", borrow_spread=0.0,
                               ltv_kill=0.77, line_review=True)
                r = simulate_leverage(idx, q, spec, cash_col="twd_cash", seed=0)
            nw = r["final_nw_real"]
            rows.append(dict(lambda_ret=lam_ret, lambda_rate=lam_rate, strategy="P6 pledge 2.5% N0",
                             ruin_pct=round(100 * r["wiped"].mean(), 2),
                             forced_pct=round(100 * (r["forced_liq"] | r["line_forced"]).mean(), 2),
                             nw_med=round(float(np.median(nw)), 2), nw_p5=round(float(np.percentile(nw, 5)), 2),
                             spend_med=round(float(np.median(r["total_real_spend"])), 3),
                             sfabs20=round(100 * spend_fail_abs(r["annual_real_spend"], 0.02).mean(), 2),
                             beats_sell_pct=round(100 * float((nw > sell_nw).mean()), 1),
                             interest_over_spend=round(float(np.median(r["cum_interest_real"] / np.maximum(r["total_real_spend"], 1e-9))), 2)))
            print(rows[-1], flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(RESULTS, "v4_bias.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote results/v4_bias.csv", len(rows))


if __name__ == "__main__":
    main()
