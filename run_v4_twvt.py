"""台灣人持有 VT + BND（含匯率）簡易版：annual JST engine.
Equity = USD-denominated, real-GDP-weighted 16-country equity (≈ VT);
bonds = US bonds (≈ BND); price level = US CPI; then a per-path REAL
TWD/USD overlay drawn (iid by year) from the 1985-2025 history
(data/processed/twd_real_fx_annual.csv, mean +0.7%/yr, sd 5.5%).
Ages 40..65, rates 3/3.5/4%, allocs 80/20 & 100/0, rules FIX & VPW3,
life-table mortality; bequest = real wealth at the start of the death year.
python run_v4_twvt.py -> results/v4_twvt.csv"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.annual import (load_jst, global_usd_series, us_series, single_series_indices,
                           simulate_annual, FixedReal, VPW)
from engine.mortality import load_qx, sample_death_years, alive_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
N = 10_000


def load_fx():
    with open(os.path.join(HERE, "data", "processed", "twd_real_fx_annual.csv")) as f:
        return np.array([float(r["real_fx_overlay"]) for r in csv.DictReader(f)])


def main():
    qx = load_qx(114)
    C = load_jst()
    fx_hist = load_fx()
    samples = {"global_usd_fx": (global_usd_series(C), "hist"),
               "global_usd_fx0": (global_usd_series(C), "demeaned"),   # overlay with its +0.7%/yr mean removed
               "global_usd_nofx": (global_usd_series(C), None),
               "us_usd_fx": (us_series(C), "hist")}
    rows = []
    for sname, (panel, use_fx) in samples.items():
        for age in (40, 45, 50, 55, 60, 65):
            years = 110 - age
            idx = single_series_indices(panel, years, N, 10, np.random.default_rng(0))
            src = None if use_fx is None else (fx_hist if use_fx == "hist" else fx_hist - fx_hist.mean())
            fx = np.random.default_rng(5).choice(src, size=(N, years)) if src is not None else None
            death = sample_death_years(qx["all"], age, N, np.random.default_rng(100))
            dy = np.minimum(death, years - 1)
            for sw in (0.8, 1.0):
                for rate in (0.03, 0.035, 0.04):
                    h = 100 - age; ext = [(90 - age, 110 - age), (100 - age, 120 - age)]
                    for tag, rule in (("FIX", FixedReal(rate)), ("VPW", VPW(0.03, horizon=h, extensions=ext))):
                        if tag == "VPW" and rate != 0.04:
                            continue
                        res = simulate_annual(idx, panel, sw, rule, fx_overlay=fx)
                        m = alive_metrics(res, death, 0.02)
                        s = res["annual_real_spend"]
                        beq = res["wealth_path"][np.arange(N), dy]
                        alive_y = np.arange(years)[None, :] < death[:, None]
                        avg = (s * alive_y).sum(axis=1) / np.maximum(np.minimum(death, years), 1)
                        rows.append(dict(sample=sname, age=age, alloc=f"{int(sw*100)}/{int(round((1-sw)*100))}",
                                         rule=tag, rate=rate, ruin_alive=m["ruin_alive_pct"],
                                         breach_alive=m["breach_alive_pct"],
                                         bequest_p10=round(float(np.percentile(beq, 10)), 3),
                                         bequest_p50=round(float(np.median(beq)), 3),
                                         bequest_p90=round(float(np.percentile(beq, 90)), 3),
                                         avg_spend_p10=round(float(np.percentile(avg, 10)), 4),
                                         avg_spend_med=round(float(np.median(avg)), 4),
                                         spend_y20_p10=round(float(np.percentile(s[:, 19], 10)), 4),
                                         spend_y20_med=round(float(np.median(s[:, 19])), 4)))
                        print(rows[-1], flush=True)
    with open(os.path.join(RESULTS, "v4_twvt.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
