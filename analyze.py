"""Summarize results/main.csv into report tables. python analyze.py"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, "results", name)) as f:
        return list(csv.DictReader(f))


def fixname(a):
    return "80/20" if a == "80/19" else a


def main():
    rows = load("main.csv")
    # seed dispersion + main table: average across seeds for boot36
    key = lambda r: (r["engine"], r["horizon"], fixname(r["alloc"]), r["rule"], r["bucket_ccy"])
    boot = defaultdict(list)
    hist = {}
    sens = defaultdict(dict)
    for r in rows:
        if r["sampler"] == "boot36":
            boot[key(r)].append(r)
        elif r["sampler"] == "hist":
            hist[key(r)] = r
        elif r["sampler"] in ("boot12", "boot60"):
            sens[key(r)][r["sampler"]] = r

    print("=" * 118)
    print(f"{'engine':7s} {'hz':3s} {'alloc':9s} {'rule':6s} {'ccy':11s} "
          f"{'ruin% (seeds)':16s} {'spendfail%':12s} {'p5 spend':9s} {'med spend':9s} "
          f"{'yr<80 med/p90':13s} {'endW med':8s} {'CE(g4)':7s} {'hist ruin/sf':12s}")
    for k in sorted(boot, key=lambda k: (k[0], int(k[1]), k[3], k[2])):
        rs = boot[k]
        ruins = [float(r["ruin_pct"]) for r in rs]
        sfs = [float(r["spendfail_pct"]) for r in rs]
        p5 = sum(float(r["spend_p5"]) for r in rs) / len(rs)
        med = sum(float(r["spend_med"]) for r in rs) / len(rs)
        yb = rs[0]["yrs_below80_med"], rs[0]["yrs_below80_p90"]
        fw = sum(float(r["final_wealth_med"]) for r in rs) / len(rs)
        ce = sum(float(r["ce_g4"]) for r in rs) / len(rs)
        h = hist.get(k)
        hs = f"{h['ruin_pct']}/{h['spendfail_pct']}" if h else "-"
        print(f"{k[0]:7s} {k[1]:3s} {k[2]:9s} {k[3]:6s} {k[4]:11s} "
              f"{min(ruins):4.1f}-{max(ruins):4.1f}      {min(sfs):4.1f}-{max(sfs):4.1f}   "
              f"{p5:7.2f}   {med:7.2f}   {yb[0]:>4s}/{yb[1]:>5s}    {fw:6.2f}   {ce:5.3f}  {hs:12s}")

    print("\n--- block length sensitivity (seed 0): ruin% | spendfail% at boot12 / boot36 / boot60 ---")
    for k in sorted(sens, key=lambda k: (k[0], int(k[1]), k[3], k[2])):
        if k[1] != "50" or k[0] != "US1934":
            continue
        b36 = [r for r in boot[k] if r["seed"] == "0"]
        parts = []
        for s in ("boot12", "boot36", "boot60"):
            r = sens[k].get(s) or (b36[0] if b36 else None)
            parts.append(f"{r['ruin_pct']:>5s}|{r['spendfail_pct']:>5s}" if r else "  -  ")
        print(f"{k[0]} {k[1]}y {k[2]:9s} {k[3]:6s}  " + "   ".join(parts))

    print("\n--- SWR sweep: max initial rate with ruin<=5% and spendfail<=10% (50y, boot36 seed0) ---")
    sweep = load("swr_sweep.csv")
    bykey = defaultdict(list)
    for r in sweep:
        bykey[(r["engine"], fixname(r["alloc"]), r["rule"].rstrip("0123456789.").rstrip())].append(r)
    for k, rs in sorted(bykey.items()):
        ok = [r for r in rs if float(r["ruin_pct"]) <= 5.0 and float(r["spendfail_pct"]) <= 10.0]
        best = max(ok, key=lambda r: float(r["rate"])) if ok else None
        if best:
            print(f"{k[0]:7s} {k[1]:9s} {k[2]:4s} max rate {float(best['rate'])*100:.2f}%  "
                  f"(ruin {best['ruin_pct']}%, sf {best['spendfail_pct']}%, p5 {best['spend_p5']}, med {best['spend_med']})")
        else:
            print(f"{k[0]:7s} {k[1]:9s} {k[2]:4s} none satisfies both constraints")


if __name__ == "__main__":
    main()
