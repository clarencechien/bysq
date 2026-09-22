"""v4 §4 (H2 rescue): the core strategy subset on the JST 16-country annual
panel, five samples. python run_v4_jst.py ->
  results/v4_jst.csv          main table (sample x horizon x alloc x strategy)
  results/v4_jst_summary.csv  source-sample and bootstrap-path return stats
  results/v4_jst_swr.csv      FIX withdrawal-rate scan 2.0-5.0% (ruin per sample)
  results/v4_jst_vpw_sens.csv VPW expected_real sensitivity (pooled sample)
Samples: (a) pooled16_mb10 (b) pooled16_mb5 (c) global_rgdp (d) us_annual
(e) pooled_exUSA_mb10 (f) pooled16_exHyper_mb10 = (a) minus the 21
country-years with inflation > 50% (sensitivity, not in the spec).
All results are the domestic-investor view (local nominal / local CPI)."""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.annual import (load_jst, stack_panel, global_series, us_series, filter_countries,
                           pooled_block_indices, single_series_indices,
                           simulate_annual, score, real_stats,
                           FixedReal, GuytonKlinger, VPW, FloorVPW, LadderVPW)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
SEED = 0
ALLOCS = [0.6, 0.8, 1.0]


def write(name, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


def alloc_name(sw):
    return f"{int(round(sw*100))}/{int(round((1-sw)*100))}"


def build_samples():
    C = load_jst()
    P16 = stack_panel(C)
    P15 = stack_panel(C, exclude=("USA",))
    G = global_series(C, weights="rgdp")
    U = us_series(C)
    # sensitivity: drop the 21 country-years with inflation > 50% (WWI/WWII
    # hyperinflations: DEU 1920/22/24, JPN 1945, ITA 1943-47, FIN 1917/18, ...)
    PXH = stack_panel(filter_countries(C, lambda c, i: c["infl"][i] <= 0.5))
    # name -> (panel, sampler, mean_block, horizons)
    return {
        "pooled16_mb10": (P16, "pooled", 10, (30, 50)),
        "pooled16_mb5": (P16, "pooled", 5, (50,)),
        "global_rgdp": (G, "single", 10, (30, 50)),
        "us_annual": (U, "single", 10, (30, 50)),
        "pooled_exUSA_mb10": (P15, "pooled", 10, (30, 50)),
        "pooled16_exHyper_mb10": (PXH, "pooled", 10, (30, 50)),
    }


def draw(panel, sampler, mean_block, horizon):
    rng = np.random.default_rng(SEED)
    if sampler == "pooled":
        return pooled_block_indices(panel, horizon, N_PATHS, mean_block, rng)
    return single_series_indices(panel, horizon, N_PATHS, mean_block, rng)


def strategies():
    out = []
    for sw in ALLOCS:
        for rule in (FixedReal(0.03), FixedReal(0.04), GuytonKlinger(0.04),
                     GuytonKlinger(0.05), VPW(0.03), VPW(0.05)):
            out.append((sw, rule))
    out.append((1.0, FloorVPW(0.02, 0.03)))          # S1
    out.append((1.0, LadderVPW(15, 0.02, 0.01, 0.03)))  # S2-lite
    return out


def path_stats(panel, idx):
    """Return stats realised inside the bootstrap paths (they differ from
    the source when segments are re-drawn / blocks are short)."""
    d = 1.0 + panel["infl"][idx]
    req = (1 + panel["eq"][idx]) / d - 1
    geo = np.exp(np.mean(np.log1p(np.maximum(req, -0.999999)), axis=1)) - 1
    return dict(path_req_mean=round(float(req.mean()), 4),
                path_req_geo_med=round(float(np.median(geo)), 4),
                path_req_geo_p5=round(float(np.percentile(geo, 5)), 4),
                path_infl_mean=round(float(panel["infl"][idx].mean()), 4))


def main():
    samples = build_samples()
    main_rows, summ_rows, swr_rows, sens_rows = [], [], [], []
    for sname, (panel, sampler, mb, horizons) in samples.items():
        for horizon in horizons:
            idx = draw(panel, sampler, mb, horizon)
            summ_rows.append(dict(sample=sname, horizon=horizon, mean_block=mb,
                                  **real_stats(panel), **path_stats(panel, idx)))
            for sw, rule in strategies():
                res = simulate_annual(idx, panel, sw, rule)
                main_rows.append(dict(sample=sname, horizon=horizon,
                                      alloc=alloc_name(sw), strategy=rule.name,
                                      rate="" if rule.rate is None else round(rule.rate, 4),
                                      **score(res)))
            for sw in ALLOCS:
                for rate in np.arange(0.020, 0.05001, 0.0025):
                    res = simulate_annual(idx, panel, sw, FixedReal(rate))
                    s = score(res)
                    swr_rows.append(dict(sample=sname, horizon=horizon, alloc=alloc_name(sw),
                                         rate=round(float(rate), 4), ruin_pct=s["ruin_pct"],
                                         sfabs20=s["sfabs20"], spend_p5=s["spend_p5"],
                                         spend_med=s["spend_med"]))
            if sname in ("pooled16_mb10", "us_annual", "pooled_exUSA_mb10"):
                st = real_stats(panel)
                for sw in ALLOCS:
                    for er, tag in [(0.03, "3%"), (0.04, "4%"), (0.05, "5% (VPW default)"),
                                    (st["req_geo"], "pooled geometric mean"),
                                    (st["req_med"], "pooled median (arithmetic)")]:
                        res = simulate_annual(idx, panel, sw, VPW(er))
                        s = score(res)
                        sens_rows.append(dict(sample=sname, horizon=horizon, alloc=alloc_name(sw),
                                              expected_real=round(float(er), 4), label=tag,
                                              first_year_rate=round(float(res["annual_real_spend"][:, 0].mean()), 4),
                                              **s))
            print(f"{sname} {horizon}y done")
    write("v4_jst.csv", main_rows)
    write("v4_jst_summary.csv", summ_rows)
    write("v4_jst_swr.csv", swr_rows)
    write("v4_jst_vpw_sens.csv", sens_rows)


if __name__ == "__main__":
    main()
