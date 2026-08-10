"""Report figures: Pareto frontier + 1966 cohort spending paths."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import (load_panel, restrict, simulate, Alloc, BucketAlloc,
                         FixedReal, GuytonKlinger)

HERE = os.path.dirname(os.path.abspath(__file__))


def pareto():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "main.csv")))
            if r["engine"] == "US1934" and r["horizon"] == "50"
            and r["sampler"] == "boot36" and r["seed"] == "0"
            and not r["bucket_ccy"].startswith("TWD")]
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        x, y = float(r["spend_med"]), float(r["spend_p5"])
        ruin = float(r["ruin_pct"])
        sf = float(r["spendfail_pct"])
        label = f'{("80/20" if r["alloc"]=="80/19" else r["alloc"])} {r["rule"]}'
        ax.scatter(x, y, s=60 + 25 * ruin, alpha=0.75,
                   c=[plt.cm.RdYlGn_r(sf / 50)], edgecolors="k", zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xlabel("median lifetime real spending (x initial wealth)")
    ax.set_ylabel("5th percentile lifetime real spending")
    ax.set_title("US engine 1934-2023, 50y retirement, block bootstrap (36m)\n"
                 "bubble size = ruin probability; color = spending-failure probability")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "pareto_us50.png"), dpi=130)
    print("wrote results/pareto_us50.png")


def cohort_paths():
    us = restrict(load_panel("us"), "1934-02")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, start, title in [(axes[0], "1966-01", "1966 retiree (stagflation)"),
                             (axes[1], "1982-08", "1982 retiree (bull era)")]:
        i = us["dates"].index(start)
        idx = np.arange(i, i + 360)[None, :]
        for alloc, rule, style in [
                (Alloc(0.6), FixedReal(0.04), dict(color="#1f77b4", lw=2)),
                (Alloc(0.6), GuytonKlinger(0.05), dict(color="#d62728", lw=2)),
                (Alloc(0.6), GuytonKlinger(0.04), dict(color="#ff9896", lw=1.5, ls="--")),
                (BucketAlloc(3), FixedReal(0.05), dict(color="#2ca02c", lw=1.5, ls=":")),
        ]:
            s = simulate(idx, us, alloc, rule)["annual_real_spend"][0]
            ax.plot(range(1, 31), 100 * s / s[0], label=f"{alloc.name} {rule.name}", **style)
        ax.axhline(70, color="gray", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("retirement year")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("real spending, % of initial")
    axes[0].legend(fontsize=8)
    fig.suptitle("Real spending paths: what the guardrails actually do", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "cohort_paths.png"), dpi=130)
    print("wrote results/cohort_paths.png")


if __name__ == "__main__":
    pareto()
    cohort_paths()
