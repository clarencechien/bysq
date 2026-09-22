"""v4 §9 realism modules + §9.4 sample extension.
  9.1 behavioural capitulation (household sim)      -> results/v4_behavior.csv
  9.2 fee / withholding drag (core, fee=)           -> results/v4_fees.csv
  9.3 spending smile (household sim)                -> results/v4_smile.csv
  9.4 sample to 2026-07 vs 2023-07 cutoff (core)    -> results/v4_sample.csv
python run_v4_realism.py [--quick]"""
import csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices, simulate,
                         Alloc, FixedReal, GuytonKlinger, VPW, RMD, spend_fail_abs, ce_spending)
from engine.mortality import load_qx, sample_death_years, alive_metrics
from engine.household import HH, simulate_household

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000
AGE = 65


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote results/" + name, len(rows), flush=True)


def score(res, extra=None):
    s = res["annual_real_spend"]
    row = dict(ruin_pct=round(100 * res["ruined"].mean(), 2),
               spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
               spend_med=round(float(np.median(res["total_real_spend"])), 3),
               sfabs20=round(100 * spend_fail_abs(s, 0.02).mean(), 2),
               sfabs25=round(100 * spend_fail_abs(s, 0.025).mean(), 2),
               ce_g4=round(ce_spending(s, 4), 4))
    if extra is not None:
        row.update(extra)
    return row


def panels(cutoff):
    us = restrict(load_panel("us"), "1934-02", cutoff)
    tw = restrict(load_panel("tw"), None, cutoff); tw["bill"] = tw.pop("usdbill")
    return [("US1934", us), ("TW1983", tw)]


CORE_STRATS = [("FIX4 60/40", Alloc(0.6), FixedReal(0.04)), ("FIX4 100/0", Alloc(1.0), FixedReal(0.04)),
               ("GK4 60/40", Alloc(0.6), GuytonKlinger(0.04)), ("GK5 80/20", Alloc(0.8), GuytonKlinger(0.05)),
               ("VPW3 60/40", Alloc(0.6), VPW(0.03)), ("VPW3 80/20", Alloc(0.8), VPW(0.03)),
               ("VPW3 100/0", Alloc(1.0), VPW(0.03))]


def run_fees():
    rows = []
    for ename, panel in panels("2023-07"):
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(len(panel["stock"]), 600, N_PATHS, 36, rng)
        for fee in (0.0, 0.003, 0.007, 0.012):
            for tag, alloc, rule in CORE_STRATS:
                res = simulate(idx, panel, alloc, rule, fee=fee)
                rows.append(dict(engine=ename, horizon=50, fee=fee, strategy=tag, **score(res)))
            print("fees", ename, fee, flush=True)
    write("v4_fees.csv", rows)


def run_sample():
    rows = []
    for cutoff in ("2023-07", None):
        for ename, panel in panels(cutoff):
            for horizon in (30, 50):
                rng = np.random.default_rng(0)
                idx = stationary_bootstrap_indices(len(panel["stock"]), horizon * 12, N_PATHS, 36, rng)
                for tag, alloc, rule in CORE_STRATS + [("RMD 60/40", Alloc(0.6), RMD())]:
                    res = simulate(idx, panel, alloc, rule)
                    rows.append(dict(engine=ename, sample_end=cutoff or panel["dates"][-1], horizon=horizon,
                                     n_months=len(panel["stock"]), strategy=tag, **score(res)))
            print("sample", cutoff, ename, flush=True)
    write("v4_sample.csv", rows)


def run_behavior_and_smile():
    qx = load_qx(114)
    rows_b, rows_s = [], []
    for ename, panel in panels("2023-07"):
        T = (110 - AGE) * 12
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(len(panel["stock"]), T, N_PATHS, 36, rng)
        drng = np.random.default_rng(7)
        death = sample_death_years(qx["all"], AGE, N_PATHS, drng) * 12 + drng.integers(0, 12, N_PATHS)
        # 9.1 behaviour: X=30% drawdown lasting > Y=6 months, capitulate with prob q, return m/month
        for fr in (0.5, 0.7):
            for (w, rule) in [(1.0, "VPW"), (0.8, "VPW"), (0.6, "VPW"), (0.6, "GK"), (0.8, "GK"), (0.6, "FIX")]:
                for beh in (None, (0.3, 6, 0.3, 0.05), (0.3, 6, 0.6, 0.05), (0.3, 6, 0.6, 0.02), (0.2, 3, 0.6, 0.05)):
                    hh = HH(retire_age=AGE, spend=0.04, floor_ratio=fr, stock_w=w, rule=rule, behavior=beh)
                    r = simulate_household(idx, panel, hh, death, qx=qx)
                    rows_b.append(dict(engine=ename, floor_ratio=fr, alloc=f"{int(w*100)}/{int(round((1-w)*100))}", rule=rule,
                                       behavior="none" if beh is None else f"X{beh[0]:g} Y{beh[1]} q{beh[2]:g} m{beh[3]:g}",
                                       **{k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k != "name"}))
            print("behavior", ename, fr, flush=True)
        # 9.3 smile: real spending -1%/yr to 75 then +2%/yr (and variants)
        for fr in (0.5, 0.7):
            for (w, rule) in [(0.6, "FIX"), (0.6, "GK"), (0.6, "VPW"), (0.8, "VPW"), (1.0, "VPW")]:
                for smile in (None, (-0.01, 75, 0.02), (-0.01, 80, 0.02), (-0.02, 75, 0.03), (0.0, 75, 0.02)):
                    hh = HH(retire_age=AGE, spend=0.04, floor_ratio=fr, stock_w=w, rule=rule, smile=smile)
                    r = simulate_household(idx, panel, hh, death, qx=qx)
                    rows_s.append(dict(engine=ename, floor_ratio=fr, alloc=f"{int(w*100)}/{int(round((1-w)*100))}", rule=rule,
                                       smile="flat" if smile is None else f"{smile[0]*100:g}%/{smile[1]}/{smile[2]*100:+g}%",
                                       **{k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k != "name"}))
            print("smile", ename, fr, flush=True)
    write("v4_behavior.csv", rows_b)
    write("v4_smile.csv", rows_s)


if __name__ == "__main__":
    run_fees()
    run_sample()
    run_behavior_and_smile()
