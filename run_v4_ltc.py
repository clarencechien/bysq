"""v4 §6: long-term-care and one-off shocks, incl. the conditional case where
the shock lands inside a >30% market drawdown; tests whether the cash bucket
finally earns its keep. Core engine + extra_spend, age-65 mortality.
python run_v4_ltc.py [--quick] -> results/v4_ltc.csv"""
import csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, BucketAlloc, FixedReal, GuytonKlinger, VPW, spend_fail_abs)
from engine.mortality import load_qx, sample_death_years, alive_metrics

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000
CUTOFF = "2023-07"
AGE = 65
# LTC cost tiers: annual real cost / W0. 60萬 and 120萬 per year for
# households with W0 = 1000萬 / 3000萬 / 5000萬.
TIERS = {"W10M": (0.06, 0.12), "W30M": (0.02, 0.04), "W50M": (0.012, 0.024)}
P_LTC = 0.5      # lifetime probability of needing LTC (assumption, flagged)


def ltc_matrix(death_years, years, dur, cost, rng, cond=None):
    """Extra real spending (P, years): LTC in the last `dur` years of life
    (onset = death - dur, min year 1) for a random P_LTC share of paths.
    cond: optional (P, years) bool 'in >30% drawdown' matrix -> the onset is
    moved to the first drawdown year >= 5 for paths that have one; paths
    without a qualifying drawdown keep the end-of-life onset."""
    P = len(death_years)
    need = rng.random(P) < P_LTC
    onset = np.maximum(death_years - dur, 1)
    if cond is not None:
        has = cond[:, 5:].any(axis=1)
        first = 5 + np.argmax(cond[:, 5:], axis=1)
        onset = np.where(has, first, onset)
    ex = np.zeros((P, years))
    yrs = np.arange(years)[None, :]
    mask = need[:, None] & (yrs >= onset[:, None]) & (yrs < (onset + dur)[:, None])
    ex[mask] = cost
    return ex, onset, need


def main():
    qx = load_qx(114)
    us = restrict(load_panel("us"), "1934-02", CUTOFF)
    tw = restrict(load_panel("tw"), None, CUTOFF); tw["bill"] = tw.pop("usdbill")
    rows = []
    t0 = time.time()
    T = (110 - AGE) * 12; years = T // 12
    for ename, panel in [("US1934", us), ("TW1983", tw)]:
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(len(panel["stock"]), T, N_PATHS, 36, rng)
        death = sample_death_years(qx["all"], AGE, N_PATHS, np.random.default_rng(100))
        # drawdown matrix from the sampled stock path (info through the year start)
        si = np.cumprod(1 + panel["stock"][idx], axis=1)
        hwm = np.maximum.accumulate(si, axis=1)
        dd_m = si / hwm
        dd_y = dd_m[:, ::12] < 0.70                       # start-of-year drawdown > 30%
        strategies = [("FIX4 60/40", Alloc(0.6), FixedReal(0.04)),
                      ("GK4 60/40", Alloc(0.6), GuytonKlinger(0.04)),
                      ("VPW3 60/40", Alloc(0.6), VPW(0.03, horizon=35, extensions=[(25, 45), (35, 55)])),
                      ("VPW3 80/20", Alloc(0.8), VPW(0.03, horizon=35, extensions=[(25, 45), (35, 55)])),
                      ("VPW3 100/0", Alloc(1.0), VPW(0.03, horizon=35, extensions=[(25, 45), (35, 55)])),
                      ("BYSQ-3y FIX4", BucketAlloc(3), FixedReal(0.04)),
                      ("BYSQ-3y GK4", BucketAlloc(3), GuytonKlinger(0.04)),
                      ("BYSQ-5y FIX4", BucketAlloc(5), FixedReal(0.04))]
        for tier, costs in TIERS.items():
            for cost in costs:
                for dur in (3, 5, 8):
                    for cond_name, cond in (("end_of_life", None), ("in_crash", dd_y)):
                        srng = np.random.default_rng(7)
                        ex, onset, need = ltc_matrix(death, years, dur, cost, srng, cond)
                        for tag, alloc, rule in strategies:
                            res = simulate(idx, panel, alloc, rule, extra_spend=ex)
                            m = alive_metrics(res, death, 0.02)
                            base = alive_metrics(res, death, 0.02)
                            # ruin among the paths that actually had the shock
                            ra = ((res["ruin_year"] >= 0) & (res["ruin_year"] < death))
                            rows.append(dict(engine=ename, tier=tier, cost=cost, dur=dur, cond=cond_name,
                                             strategy=tag, ruin_alive_pct=m["ruin_alive_pct"],
                                             ruin_shocked_pct=round(100 * float(ra[need].mean()), 2),
                                             ruin_unshocked_pct=round(100 * float(ra[~need].mean()), 2),
                                             breach_alive_pct=m["breach_alive_pct"],
                                             life_spend_med=m["life_spend_med"], ce_g4_alive=m["ce_g4_alive"]))
                        print(f"{ename} {tier} c{cost} d{dur} {cond_name}: " +
                              " ".join(f"{r['strategy']}={r['ruin_shocked_pct']}" for r in rows[-len(strategies):]) +
                              f" ({time.time()-t0:.0f}s)", flush=True)
        # no-shock baseline
        for tag, alloc, rule in strategies:
            res = simulate(idx, panel, alloc, rule)
            m = alive_metrics(res, death, 0.02)
            rows.append(dict(engine=ename, tier="none", cost=0, dur=0, cond="none", strategy=tag,
                             ruin_alive_pct=m["ruin_alive_pct"], ruin_shocked_pct="", ruin_unshocked_pct="",
                             breach_alive_pct=m["breach_alive_pct"], life_spend_med=m["life_spend_med"],
                             ce_g4_alive=m["ce_g4_alive"]))
    with open(os.path.join(RESULTS, "v4_ltc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote results/v4_ltc.csv", len(rows))


if __name__ == "__main__":
    main()
