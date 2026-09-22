"""v4 §7: return stacking (RSSB + RSST five-five) vs VT / 60-40 / 80-20.
python run_v4_stack.py -> results/v4_stack.csv, v4_stack_regimes.csv, v4_stack_sens.csv

Conventions (run_v3.py): US1934 panel, 10k paths, seed 0, stationary
bootstrap mean block 36 months, absolute-floor scoring. Two sample ends
(2026-07 = full; 2023-07 = v1-v3 cutoff, spec §9.4). Stacked portfolios are
a single synthetic 'stock' series run with Alloc(1.0) (engine/stack.py).
"""
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, stationary_bootstrap_indices,
                         simulate, Alloc, FixedReal, GuytonKlinger, VPW,
                         spend_fail_abs, ce_spending)
from engine.stack import (load_trend, align_trend, stacked_panel, control_panel,
                          window_real_path, max_drawdown, spending_smoothness,
                          FEE_RSSB, FEE_RSST, FEE_VT, FEE_BND)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
BLOCK = 36
FLOORS = [0.015, 0.020, 0.025, 0.030]
SAMPLES = [("2026-07", None), ("2023-07", "2023-07")]
HORIZONS = (30, 50)
STACKED = [("RSST", "RSST"), ("RSSB", "RSSB"), ("5050 RSSB+RSST", "FIVEFIVE")]
CONTROLS = [("100/0 VT", 1.0), ("80/20", 0.8), ("60/40", 0.6)]
REGIMES = [("1966-1982", "1966-01", "1982-12"), ("1973-1974", "1973-01", "1974-12"),
           ("2000-2002", "2000-01", "2002-12"), ("2008-2009Q1", "2008-01", "2009-03"),
           ("2022", "2022-01", "2022-12")]


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


def score(res):
    s = res["annual_real_spend"]
    row = dict(ruin_pct=round(100 * res["ruined"].mean(), 2),
               spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
               spend_med=round(float(np.median(res["total_real_spend"])), 3),
               endw_med=round(float(np.median(res["final_real_wealth"])), 2))
    for fl in FLOORS:
        row[f"sfabs{int(fl*1000)}"] = round(100 * spend_fail_abs(s, fl).mean(), 2)
    for g in (2, 4, 8):
        row[f"ce_g{g}"] = round(ce_spending(s, g), 4)
    for k, v in spending_smoothness(s).items():
        row[k] = round(v, 4)
    return row


def rules_for(portfolio):
    r = [FixedReal(0.04), VPW(0.03), VPW(0.05)]
    if portfolio in ("60/40", "80/20"):
        r.insert(1, GuytonKlinger(0.04))
    return r


def portfolios(base, trend, spread=0.0, haircut=1.0, which="all"):
    """Yield (label, panel, alloc). Controls carry ETF fees (VT 0.07%, BND
    0.03%); stacked funds carry their own fees inside the series."""
    if which in ("all", "controls"):
        cp = control_panel(base, FEE_VT, FEE_BND)
        for label, w in CONTROLS:
            yield label, cp, Alloc(w)
    if which in ("all", "stacked"):
        for label, kind in STACKED:
            yield label, stacked_panel(base, trend, kind, spread=spread, haircut=haircut), Alloc(1.0)


def run_main():
    rows = []
    trend = load_trend("trend_excess")
    proxy_only = load_trend("proxy_scaled")
    for sample_end, last in SAMPLES:
        base = restrict(load_panel("us"), "1934-02", last)
        t_src = len(base["stock"])
        for horizon in HORIZONS:
            T = horizon * 12
            rng = np.random.default_rng(0)
            idx = stationary_bootstrap_indices(t_src, T, N_PATHS, BLOCK, rng)
            t0 = time.time()
            for label, panel, alloc in portfolios(base, trend):
                for rule in rules_for(label):
                    res = simulate(idx, panel, alloc, rule)
                    rows.append(dict(engine="US1934", sample_end=sample_end, horizon=horizon,
                                     portfolio=label, strategy=rule.name, spread=0.0,
                                     haircut=1.0, trend_src="spliced", **score(res)))
            # robustness: in-house proxy over the WHOLE period (no AQR)
            if last is None:
                for label, panel, alloc in portfolios(base, proxy_only, which="stacked"):
                    if label == "RSSB":
                        continue
                    for rule in (FixedReal(0.04), VPW(0.03)):
                        res = simulate(idx, panel, alloc, rule)
                        rows.append(dict(engine="US1934", sample_end=sample_end, horizon=horizon,
                                         portfolio=label, strategy=rule.name, spread=0.0,
                                         haircut=1.0, trend_src="proxy_only", **score(res)))
            print(f"main {sample_end} {horizon}y done ({time.time()-t0:.0f}s)")
    write("v4_stack.csv", rows)
    return rows


def run_sens():
    """Spread x trend-haircut grid, 50y, full sample, FIX4 + VPW3; 80/20
    control rows repeated so each grid cell can be ranked against it.
    haircut 0.25 is an extra point (not in the spec) to locate where the
    ranking vs 80/20 flips."""
    rows = []
    trend = load_trend("trend_excess")
    base = restrict(load_panel("us"), "1934-02")
    t_src = len(base["stock"])
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(t_src, 600, N_PATHS, BLOCK, rng)
    ref = {}
    cp = control_panel(base)
    for rule in (FixedReal(0.04), VPW(0.03)):
        res = simulate(idx, cp, Alloc(0.8), rule)
        ref[rule.name] = score(res)
        rows.append(dict(portfolio="80/20", strategy=rule.name, spread=0.0, haircut=1.0,
                         **ref[rule.name], d_ce_g2=0.0, d_ce_g4=0.0, d_ce_g8=0.0,
                         d_sfabs20=0.0, d_spend_med=0.0, d_spend_p5=0.0))
    t0 = time.time()
    for spread in (0.0, 0.005, 0.01):
        for haircut in (1.0, 0.5, 0.25, 0.0):
            for label, panel, alloc in portfolios(base, trend, spread, haircut, which="stacked"):
                for rule in (FixedReal(0.04), VPW(0.03)):
                    sc = score(simulate(idx, panel, alloc, rule))
                    r0 = ref[rule.name]
                    rows.append(dict(portfolio=label, strategy=rule.name, spread=spread,
                                     haircut=haircut, **sc,
                                     d_ce_g2=round(sc["ce_g2"] - r0["ce_g2"], 4),
                                     d_ce_g4=round(sc["ce_g4"] - r0["ce_g4"], 4),
                                     d_ce_g8=round(sc["ce_g8"] - r0["ce_g8"], 4),
                                     d_sfabs20=round(sc["sfabs20"] - r0["sfabs20"], 2),
                                     d_spend_med=round(sc["spend_med"] - r0["spend_med"], 3),
                                     d_spend_p5=round(sc["spend_p5"] - r0["spend_p5"], 3)))
            print(f"sens spread={spread} haircut={haircut} done ({time.time()-t0:.0f}s)")
    write("v4_stack_sens.csv", rows)
    return rows


def run_regimes():
    """Historical windows (no bootstrap): cumulative REAL return and max real
    drawdown of every portfolio, plus the nominal leg decomposition."""
    rows = []
    trend = load_trend("trend_excess")
    base = restrict(load_panel("us"), "1934-02")
    tr = align_trend(base, trend)
    d = np.array(base["dates"])
    for name, first, last in REGIMES:
        m = (d >= first) & (d <= last)
        for label, panel, alloc in portfolios(base, trend):
            path = window_real_path(panel, first, last, alloc.stock_w)
            rows.append(dict(window=name, portfolio=label,
                             cum_real=round(float(path[-1] - 1.0), 4),
                             max_dd_real=round(max_drawdown(path), 4),
                             months=int(m.sum())))
        # leg decomposition (nominal, compounded, no fees)
        legs = dict(stock=base["stock"][m], bond_minus_bill=base["bond"][m] - base["bill"][m],
                    trend=tr[m], bill=base["bill"][m], cpi=base["infl"][m])
        for leg, r in legs.items():
            rows.append(dict(window=name, portfolio=f"leg:{leg}",
                             cum_real=round(float(np.prod(1 + r) - 1.0), 4),
                             max_dd_real=round(max_drawdown(np.cumprod(1 + r)), 4),
                             months=int(m.sum())))
    write("v4_stack_regimes.csv", rows)
    # 2022 recovery arithmetic for the notes
    m = (d >= "2022-01") & (d <= "2022-12")
    s = float(np.sum(base["stock"][m]))
    bb = float(np.sum(base["bond"][m] - base["bill"][m]))
    t = float(np.sum(tr[m]))
    print(f"2022 arithmetic monthly sums: stock {s:+.4f}, bond-bill {bb:+.4f}, trend {t:+.4f}")
    print(f"  RSST : stock loss {s:+.4f}, trend leg {t:+.4f} -> recovered {100*t/abs(s):.0f}% of the stock loss")
    print(f"  5050 : stock+0.5*bond loss {s+0.5*bb:+.4f}, 0.5*trend {0.5*t:+.4f} -> recovered {100*0.5*t/abs(s+0.5*bb):.0f}%")
    print(f"  RSSB : stock+bond loss {s+bb:+.4f}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    run_main()
    run_sens()
    run_regimes()
    print(f"total {time.time()-t0:.0f}s")
