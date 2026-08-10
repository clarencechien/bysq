"""v3 experiments: scoring fix, rate rescan, bucket factorial, floor scan, H8.
python run_v3.py -> results/v3_rates.csv, v3_factorial.csv, v3_floor.csv, v3_h8.csv"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices,
                         simulate, Alloc, BucketAlloc, FixedReal, GuytonKlinger,
                         VanguardDynamic, VPW, RMD, spend_fail_abs, ce_spending)
from engine.leverage import LevSpec, simulate_leverage

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
FLOORS = [0.015, 0.020, 0.025, 0.030]


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


def us_panel():
    return restrict(load_panel("us"), "1934-02")


def tw_panel():
    p = load_panel("tw")
    p["bill"] = p.pop("usdbill")
    p["twd_cash"] = (1 + p["infl"]) - 1
    return p


def score(res, strategy, rate, engine, horizon):
    s = res["annual_real_spend"]
    row = dict(engine=engine, horizon=horizon, strategy=strategy,
               rate="" if rate is None else round(rate, 4),
               ruin_pct=round(100 * res["ruined"].mean(), 2),
               spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
               spend_med=round(float(np.median(res["total_real_spend"])), 3),
               endw_med=round(float(np.median(res["final_real_wealth"])), 2))
    for fl in FLOORS:
        row[f"sfabs{int(fl*1000)}"] = round(100 * spend_fail_abs(s, fl).mean(), 2)
    for g in (2, 4, 8):
        row[f"ce_g{g}"] = round(ce_spending(s, g), 4)
    return row


def run_rates():
    """v3 §1: withdrawal-rate scan 2.0-5.0% on the ABSOLUTE ruler."""
    rows = []
    for ename, panel, horizons in [("US1934", us_panel(), (30, 50)),
                                   ("TW1983", tw_panel(), (30, 50))]:
        t_src = len(panel["stock"])
        for horizon in horizons:
            T = horizon * 12
            rng = np.random.default_rng(0)
            idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
            for rate in np.arange(0.020, 0.0525, 0.0025):
                for build in [lambda r: (Alloc(0.6), FixedReal(r)),
                              lambda r: (Alloc(1.0), FixedReal(r)),
                              lambda r: (Alloc(0.6), GuytonKlinger(r)),
                              lambda r: (Alloc(0.8), GuytonKlinger(r))]:
                    alloc, rule = build(rate)
                    res = simulate(idx, panel, alloc, rule)
                    rows.append(score(res, f"{alloc.name} {rule.name}", rate,
                                      ename, horizon))
            # parameter-light rules, one cell each
            for alloc, rule in [(Alloc(0.6), VPW(0.03)), (Alloc(0.8), VPW(0.03)),
                                (Alloc(0.6), RMD()), (Alloc(0.8), RMD()),
                                (Alloc(0.6), VanguardDynamic(0.04))]:
                res = simulate(idx, panel, alloc, rule)
                rows.append(score(res, f"{alloc.name} {rule.name}", None,
                                  ename, horizon))
            print(f"rates {ename} {horizon}y done")
    write("v3_rates.csv", rows)


def run_factorial():
    """v3 §2: bucket x fallback orthogonal design, incl. bucket=OFF cells."""
    rows = []
    for ename, panel in [("TW1983", tw_panel()), ("US1934", us_panel())]:
        if ename == "US1934":
            panel = dict(panel)
            panel["twd_cash"] = panel["bill"]
        t_src = len(panel["stock"])
        T = 30 * 12
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
        mode = "path_c" if ename == "TW1983" else "path_b"
        for n in (0, 2, 3, 5):
            for fb in ("F1", "F2", "F3"):
                spec = LevSpec(rate=0.025, bucket_years=n, alpha=0.95,
                               ltv_max=0.30, fallback=fb, borrow_mode=mode,
                               borrow_const=0.03, borrow_spread=0.015,
                               ltv_kill=0.77 if mode == "path_c" else 0.75,
                               line_review=(mode == "path_c"))
                r = simulate_leverage(idx, panel, spec, cash_col="twd_cash", seed=0)
                s = r["total_real_spend"]
                rows.append(dict(
                    engine=ename, bucket=f"N{n}" if n else "OFF", fallback=fb,
                    wiped_pct=round(100 * r["wiped"].mean(), 2),
                    forced_pct=round(100 * (r["forced_liq"] | r["line_forced"]).mean(), 2),
                    peak_ltv_med=round(float(np.median(r["peak_ltv"])) * 100, 1),
                    peak_ltv_p95=round(float(np.percentile(r["peak_ltv"], 95)) * 100, 1),
                    nw_med=round(float(np.median(r["final_nw_real"])), 2),
                    nw_p5=round(float(np.percentile(r["final_nw_real"], 5)), 2),
                    spend_med=round(float(np.median(s)), 3),
                    sfabs20=round(100 * spend_fail_abs(
                        r["annual_real_spend"], 0.020).mean(), 2)))
                print(f"fact {ename} N{n} {fb}: forced {rows[-1]['forced_pct']}% "
                      f"pkLTV {rows[-1]['peak_ltv_med']}/{rows[-1]['peak_ltv_p95']} "
                      f"NW {rows[-1]['nw_med']} sf20 {rows[-1]['sfabs20']}%")
    write("v3_factorial.csv", rows)


def run_floor():
    """v3 §5: GK with an uncuttable floor share; find where ruin returns."""
    rows = []
    panel = us_panel()
    t_src = len(panel["stock"])
    T = 50 * 12
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(t_src, T, N_PATHS, 36, rng)
    for alloc_w, rate in [(0.6, 0.04), (0.8, 0.05), (0.6, 0.05)]:
        for fr in (0.0, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
            rule = GuytonKlinger(rate, floor_ratio=fr)
            res = simulate(idx, panel, Alloc(alloc_w), rule)
            s = res["annual_real_spend"]
            rows.append(dict(alloc=f"{int(alloc_w*100)}/{int(round((1-alloc_w)*100))}",
                             rate=rate, floor_ratio=fr,
                             ruin_pct=round(100 * res["ruined"].mean(), 2),
                             sfabs20=round(100 * spend_fail_abs(s, 0.020).mean(), 2),
                             sfabs25=round(100 * spend_fail_abs(s, 0.025).mean(), 2),
                             spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
                             spend_med=round(float(np.median(res["total_real_spend"])), 3)))
            print(f"floor {rows[-1]['alloc']} GK{rate*100:g} fl={fr}: "
                  f"ruin {rows[-1]['ruin_pct']}% sf20 {rows[-1]['sfabs20']}%")
    write("v3_floor.csv", rows)


def run_h8():
    """v3 §9.3: Path C with FX frozen vs historical FX (same time axis)."""
    tw = tw_panel()
    us = load_panel("us")
    # rebuild TW-dated panel with FX frozen: stock return in TWD == USD return,
    # deflated by TW CPI (same months as tw panel)
    ud = {d: i for i, d in enumerate(us["dates"])}
    ti = [i for i, d in enumerate(tw["dates"]) if d in ud]
    ui = [ud[tw["dates"][i]] for i in ti]
    frozen = dict(stock=us["stock"][ui], bond=us["bond"][ui],
                  bill=us["bill"][ui], infl=tw["infl"][ti],
                  cpi=tw["cpi"][ti], dates=[tw["dates"][i] for i in ti])
    frozen["twd_cash"] = frozen["infl"]
    rows = []
    for label, panel in [("historical_fx", tw), ("frozen_fx", frozen)]:
        t_src = len(panel["stock"])
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(t_src, 360, N_PATHS, 36, rng)
        spec = LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                       borrow_const=0.03)
        r = simulate_leverage(idx, panel, spec, cash_col="twd_cash", seed=0)
        rows.append(dict(fx=label,
                         wiped_pct=round(100 * r["wiped"].mean(), 2),
                         forced_pct=round(100 * (r["forced_liq"] | r["line_forced"]).mean(), 2),
                         peak_ltv_med=round(float(np.median(r["peak_ltv"])) * 100, 1),
                         peak_ltv_p95=round(float(np.percentile(r["peak_ltv"], 95)) * 100, 1),
                         nw_med=round(float(np.median(r["final_nw_real"])), 2)))
        print(f"h8 {label}: forced {rows[-1]['forced_pct']}% "
              f"pkLTV p95 {rows[-1]['peak_ltv_p95']}%")
    write("v3_h8.csv", rows)


if __name__ == "__main__":
    run_rates()
    run_factorial()
    run_floor()
    run_h8()
