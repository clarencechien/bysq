"""FIRE example across SAMPLES (for the HTML report): what changes if the
investor's world is not the US? Annual JST engine + life-table mortality.
Samples: us_annual (American), global_rgdp (own the world, everyone's local
inflation averaged ~ a VT holder without FX), pooled16 (you are a random
developed country's domestic investor), pooled16 ex-hyperinflation.
python run_v4_fire50_intl.py -> results/v4_fire50_intl.csv"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.annual import simulate_annual, FixedReal, VPW
from engine.mortality import load_qx, sample_death_years, alive_metrics
from run_v4_jst import build_samples, draw

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N = 10_000


def main():
    qx = load_qx(114)
    samples = build_samples()
    rows = []
    for sname in ("us_annual", "global_rgdp", "pooled16_mb10", "pooled16_exHyper_mb10"):
        panel, sampler, mb, _ = samples[sname]
        for age in (40, 50, 65):
            years = 110 - age
            idx = draw(panel, sampler, mb, years)
            death = sample_death_years(qx["all"], age, N, np.random.default_rng(100))
            for rate in (0.03, 0.035, 0.04):
                for sw in (0.6, 0.8, 1.0):
                    for tag, rule in (("固定提領", FixedReal(rate)), ("VPW", VPW(0.03, horizon=years))):
                        if tag == "VPW" and rate != 0.04:
                            continue
                        res = simulate_annual(idx, panel, sw, rule)
                        m = alive_metrics(res, death, 0.02)
                        ry = res["ruin_year"]
                        rows.append(dict(sample=sname, age=age, rate=rate,
                                         alloc=f"{int(sw*100)}/{int(round((1-sw)*100))}", strategy=tag,
                                         ruin_fixed50=round(100 * float(((ry >= 0) & (ry < min(50, years) - 1)).mean()), 2),
                                         ruin_alive=m["ruin_alive_pct"], breach_alive=m["breach_alive_pct"],
                                         life_spend_med=m["life_spend_med"]))
                        print(rows[-1], flush=True)
    with open(os.path.join(RESULTS, "v4_fire50_intl.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
