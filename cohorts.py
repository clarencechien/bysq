"""Concrete historical cohorts for the report narrative. python cohorts.py"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, simulate, Alloc, BucketAlloc,
                         FixedReal, GuytonKlinger)


def window(panel, start_date, years):
    i = panel["dates"].index(start_date)
    T = years * 12
    if i + T > len(panel["stock"]):
        return None
    return np.arange(i, i + T)[None, :]


def show(panel, start, years, strategies):
    print(f"\n=== retiree cohort {start}, {years}y ===")
    for alloc, rule in strategies:
        idx = window(panel, start, years)
        if idx is None:
            print("  (window exceeds sample)")
            return
        r = simulate(idx, panel, alloc, rule)
        s = r["annual_real_spend"][0]
        rel = s / s[0]
        ruin = r["ruin_year"][0]
        yrs_low = int((rel < 0.8).sum())
        trough = rel.min()
        print(f"  {alloc.name:9s} {rule.name:6s} lifetime {s.sum():.2f}  "
              f"trough {trough*100:4.0f}%  yrs<80% {yrs_low:2d}  "
              f"ruin@{ruin if ruin >= 0 else '-'}  endW {r['final_real_wealth'][0]:.2f}")


us = restrict(load_panel("us"), "1934-02")
strats = [(Alloc(0.6), FixedReal(0.04)), (Alloc(1.0), FixedReal(0.04)),
          (BucketAlloc(3), FixedReal(0.04)), (Alloc(0.6), GuytonKlinger(0.05)),
          (BucketAlloc(3), FixedReal(0.05)), (Alloc(0.6), GuytonKlinger(0.04))]
for cohort in ["1966-01", "1973-01", "1929-02", "2000-01", "1982-08"]:
    if cohort in us["dates"]:
        show(us, cohort, 30, strats)

tw = load_panel("tw")
tw["bill"] = tw.pop("usdbill")
for cohort in ["1989-10", "1993-07"]:
    show(tw, cohort, 30, strats)
