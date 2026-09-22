"""v4 §1: floor + upside family (S1-S4) vs dynamic rules, with mortality.
python run_v4_floor.py [--quick] -> results/v4_floor.csv"""
import csv, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices
from engine.mortality import load_qx, sample_death_years, survival_curve
from engine.household import HH, simulate_household, load_shiller_extra, trailing_inflation

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
QUICK = "--quick" in sys.argv
N_PATHS = 2000 if QUICK else 10_000
CUTOFF = "2023-07"


def us_panel():
    p = restrict(load_panel("us"), "1934-02", CUTOFF)
    p["gs10"], p["cape"] = load_shiller_extra(p["dates"])
    p["infl_e"] = trailing_inflation(p)
    return p


def tw_panel():
    p = restrict(load_panel("tw"), None, CUTOFF)
    p["bill"] = p.pop("usdbill")
    return p


def strategies(engine):
    S = []
    def add(tag, family, **kw):
        S.append((tag, family, kw))
    add("VPW3 60/40", "dynamic", stock_w=0.6, rule="VPW")
    add("VPW3 80/20", "dynamic", stock_w=0.8, rule="VPW")
    add("GK 60/40", "dynamic", stock_w=0.6, rule="GK")
    add("GK 80/20", "dynamic", stock_w=0.8, rule="GK")
    add("FIX 60/40", "fixed", stock_w=0.6, rule="FIX")
    add("S1 VPW3 100/0", "floor", stock_w=1.0, rule="VPW")
    for n in (10, 15, 20):
        add(f"S2 ladder{n}r + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ladder_years=n, ladder_kind="real")
    add("S2 ladder15r + VPW3 80/20", "floor", stock_w=0.8, rule="VPW", ladder_years=15, ladder_kind="real")
    add("S2 ladder15r(0%) + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ladder_years=15, ladder_kind="real", ladder_real_rate=0.0)
    if engine == "US1934":
        add("S2 ladder15n + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ladder_years=15, ladder_kind="nominal")
    for load in (0.1, 0.2, 0.3):
        add(f"S3 ann80n load{int(load*100)} + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ann_start_age=80, ann_load=load, ann_kind="nominal")
    add("S3 ann85n load20 + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ann_start_age=85, ann_load=0.2, ann_kind="nominal")
    add("S3 ann80r load20 + VPW3 100/0 (假設性)", "floor", stock_w=1.0, rule="VPW", ann_start_age=80, ann_load=0.2, ann_kind="real")
    add("S4 ladder15r + ann85n + VPW3 100/0", "floor", stock_w=1.0, rule="VPW", ladder_years=15, ann_start_age=85, ann_load=0.2, ann_kind="nominal")
    return S


def summarize(r, p90_years):
    row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
    long = r["death_years"] >= p90_years
    if long.sum() > 50:
        s = r["annual_spend"][long]; fl = r["annual_floor"][long]; al = r["alive_y"][long]
        below = (s < fl * 0.98) & al
        life = (s * al).sum(axis=1) / np.maximum(al.sum(axis=1), 1)
        row["p90_ruin_pct"] = round(100 * float(r["ruin_alive"][long].mean()), 2)
        row["p90_breach1_pct"] = round(100 * float(below.any(axis=1).mean()), 2)
        row["p90_avg_p5"] = round(float(np.percentile(life, 5)), 4)
        row["p90_avg_med"] = round(float(np.median(life)), 4)
    return row


def main():
    qx = load_qx(114)
    rows = []
    t_start = time.time()
    for ename, panel in [("US1934", us_panel()), ("TW1983", tw_panel())]:
        t_src = len(panel["stock"])
        for age in (65, 50):
            T = (110 - age) * 12
            rng = np.random.default_rng(0)
            idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
            drng = np.random.default_rng(7)
            death = sample_death_years(qx["all"], age, N_PATHS, drng) * 12 + drng.integers(0, 12, N_PATHS)
            S = survival_curve(qx["all"], age)
            p90_years = int(np.searchsorted(-S, -0.10))       # years until only 10% survive
            for fr in (0.35, 0.5, 0.7, 1.0):
                for pension, mult in [(0.0, 1.0), (0.01, 1.0), (0.01, 0.7), (0.025, 1.0), (0.025, 0.7)]:
                    for tag, family, kw in strategies(ename):
                        hh = HH(retire_age=age, spend=0.04, floor_ratio=fr, pension=pension,
                                pension_mult=mult, **kw)
                        r = simulate_household(idx, panel, hh, death, cash_col="bill",
                                               gs10=panel.get("gs10"), infl_e=panel.get("infl_e"), qx=qx)
                        row = dict(engine=ename, age=age, floor_ratio=fr, pension=pension,
                                   pension_mult=mult, pension_eff=round(pension * mult, 4),
                                   strategy=tag, family=family)
                        row.update(summarize(r, p90_years))
                        row.pop("name", None)
                        rows.append(row)
                    print(f"{ename} age{age} fr{fr} P{pension}x{mult} done "
                          f"({len(rows)} rows, {time.time()-t_start:.0f}s)", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RESULTS, "v4_floor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print("wrote results/v4_floor.csv", len(rows))


if __name__ == "__main__":
    main()
