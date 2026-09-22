"""v4 FIRE example for the HTML report: no 勞保/勞退, fixed real spending,
retire at 40 / 50 / 65, AUM 1000/3000/5000萬. Same withdrawal RATE gives the
same probabilities regardless of AUM (engine is scale-free); AUM matters only
through the long-term-care shock (absolute NT$).
python run_v4_fire50.py -> results/v4_fire50.csv"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, GuytonKlinger, VPW)
from engine.mortality import load_qx, sample_death_years, alive_metrics, survival_curve

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N = 10_000
RATES = [0.03, 0.035, 0.04]
AUM = {"1000萬": 1000, "3000萬": 3000, "5000萬": 5000}
LTC_COST, LTC_YEARS, P_LTC = 120, 5, 0.5      # 萬/年, 年, 機率


def strategies(age):
    h = 100 - age; ext = [(90 - age, 110 - age), (100 - age, 120 - age)]
    return lambda r: [("固定提領 60/40", Alloc(0.6), FixedReal(r)),
                      ("固定提領 80/20", Alloc(0.8), FixedReal(r)),
                      ("固定提領 100/0", Alloc(1.0), FixedReal(r)),
                      ("GK 護欄 60/40", Alloc(0.6), GuytonKlinger(r)),
                      ("VPW 60/40", Alloc(0.6), VPW(0.03, horizon=h, extensions=ext)),
                      ("VPW 平滑 0.7 60/40", Alloc(0.6), VPW(0.03, horizon=h, extensions=ext, smooth=("yale", 0.7))),
                      ("VPW 平滑 0.7 80/20", Alloc(0.8), VPW(0.03, horizon=h, extensions=ext, smooth=("yale", 0.7)))]


def main():
    qx = load_qx(114)
    us = restrict(load_panel("us"), "1934-02", "2023-07")
    tw = restrict(load_panel("tw"), None, "2023-07"); tw["bill"] = tw.pop("usdbill")
    rows = []
    for ename, panel in [("US1934", us), ("TW1983", tw)]:
        for age in (40, 50, 65):
            T = (110 - age) * 12; years = T // 12
            idx = stationary_bootstrap_indices(len(panel["stock"]), T, N, 36, np.random.default_rng(0))
            death = sample_death_years(qx["all"], age, N, np.random.default_rng(100))
            S = survival_curve(qx["all"], age)
            p_alive50 = float(S[min(50, len(S) - 1)])
            fixed50 = np.full(N, min(50, years))
            for r in RATES:
                for tag, alloc, rule in strategies(age)(r):
                    res = simulate(idx, panel, alloc, rule)
                    s = res["annual_real_spend"]
                    m = alive_metrics(res, death, 0.02)
                    m50 = alive_metrics(res, fixed50, 0.02)
                    ry = res["ruin_year"]
                    ruin50 = float(((ry >= 0) & (ry < min(50, years) - 1)).mean()) * 100
                    row = dict(engine=ename, age=age, rate=r, strategy=tag, p_alive_50y=round(100 * p_alive50, 1),
                               ruin_fixed50=round(ruin50, 2), ruin_alive=m["ruin_alive_pct"],
                               breach_alive=m["breach_alive_pct"], ce_alive=m["ce_g4_alive"],
                               spend_y10_p10=round(float(np.percentile(s[:, 9], 10)) / r, 2),
                               spend_y20_p10=round(float(np.percentile(s[:, 19], 10)) / r, 2),
                               spend_y30_p10=round(float(np.percentile(s[:, 29], 10)) / r, 2),
                               spend_med_ratio=round(float(np.median(s[:, :30].mean(axis=1))) / r, 2),
                               ruin_year_med=round(float(np.median(ry[ry >= 0])), 1) if (ry >= 0).any() else "")
                    # LTC at 4% only: cost/AUM differs per AUM
                    if r == 0.04:
                        for aum_name, aum in AUM.items():
                            ex = np.zeros((N, years)); need = np.random.default_rng(7).random(N) < P_LTC
                            onset = np.maximum(death - LTC_YEARS, 1)
                            yy = np.arange(years)[None, :]
                            mask = need[:, None] & (yy >= onset[:, None]) & (yy < (onset + LTC_YEARS)[:, None])
                            ex[mask] = LTC_COST / aum
                            res2 = simulate(idx, panel, alloc, rule, extra_spend=ex)
                            m2 = alive_metrics(res2, death, 0.02)
                            row[f"ruin_alive_ltc_{aum_name}"] = m2["ruin_alive_pct"]
                    rows.append(row)
                    print(ename, age, r, tag, "fixed50", row["ruin_fixed50"], "alive", row["ruin_alive"], flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(os.path.join(RESULTS, "v4_fire50.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote results/v4_fire50.csv", len(rows))


if __name__ == "__main__":
    main()
