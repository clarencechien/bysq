"""FIRE-50 DINK case: tech couple, both 50, no kids / no property / no car,
renters, FIRE with NT$50M or NT$100M.

Pension scenarios (勞保老年年金, claimable at 65 with >=15y 年資):
  leave: stop insuring at 50. 年資 25y each at ceiling 45,800:
         45,800 x 25 x 1.55% = 17,750/mo each -> household 35,500 from 65.
  union: join 職業工會 and self-pay premiums 50->65 (~3,500/mo/person at the
         ceiling, modeled as 7,000/mo household extra outflow, real),
         年資 40y -> 28,400 each -> 56,800/mo from 65.
  zero:  no pension ever (reform stress, not a forecast).
Survivor keeps one share. Spending: total target in {100k,150k,200k}/mo real,
uncuttable floor 65k (rent 40k + essentials 25k), widow factor 0.8.
Portfolio 80/20, horizon to 105 (55y). Mortality: same Gompertz calibration,
survival integrated from age 50. python run_fire_case.py"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
START_AGE = 50
HORIZON_Y = 55
T = HORIZON_Y * 12
PENSION_AGE_M = (65 - START_AGE) * 12
STOCK_W = 0.8
FLOOR = 65_000.0
WIDOW = 0.8
PENSION = {"leave": (17_750.0, 0.0), "union": (28_400.0, 7_000.0), "zero": (0.0, 0.0)}
SPEND_LEVELS = [100_000.0, 150_000.0, 200_000.0]
WEALTH_LEVELS = {"50M": 50_000_000.0, "100M": 100_000_000.0}


def gompertz_A(e65_target, B):
    ages = np.arange(65, 111, 1 / 12)

    def e65(A):
        mu = A * np.exp(B * ages)
        return np.exp(-np.cumsum(mu) / 12).sum() / 12
    lo, hi = 1e-8, 1e-2
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if e65(mid) > e65_target else (lo, mid)
    return np.sqrt(lo * hi)


def death_months_from50(n, e65_target, B, rng):
    A = gompertz_A(e65_target, B)
    ages = np.arange(START_AGE, 111, 1 / 12)
    mu = A * np.exp(B * ages)
    surv = np.exp(-np.cumsum(mu) / 12)
    u = rng.random(n)
    return np.searchsorted(-surv, -u)


def simulate(idx, panel, strategy, w0, pension_each, premium, total_target,
             death_h, death_w):
    P, Tn = idx.shape
    stock_r = panel["stock"][idx]
    bond_r = panel["bond"][idx]
    infl_m = panel["infl"][idx]
    price = np.ones((P, Tn))
    lvl = np.ones(P)
    for t in range(1, Tn):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    bal = np.stack([np.full(P, w0 * STOCK_W), np.full(P, w0 * (1 - STOCK_W))], axis=1)
    ruined = np.zeros(P, bool)
    ruin_month = np.full(P, -1)
    floor_breach = np.zeros(P, bool)
    lifetime_real = np.zeros(P)
    annual = np.zeros((P, Tn // 12))
    gk_mult = np.ones(P)
    draw_nom = np.zeros(P)
    need_init = None

    for t in range(Tn):
        y = t // 12
        both = (t < death_h) & (t < death_w)
        one = ((t < death_h) | (t < death_w)) & ~both
        alive = both | one
        target = np.where(both, total_target, np.where(one, total_target * WIDOW, 0.0))
        floor = np.where(both, FLOOR, np.where(one, FLOOR * WIDOW, 0.0))
        pension = np.where(t >= PENSION_AGE_M,
                           np.where(both, 2 * pension_each, np.where(one, pension_each, 0.0)),
                           0.0)
        prem = np.where((t < PENSION_AGE_M) & alive, premium, 0.0)

        wealth = bal.sum(axis=1)
        if t % 12 == 0:
            need = np.maximum(target - pension, 0.0) + prem     # real monthly
            if strategy in ("DEPOSIT", "FIX"):
                draw_real = need
            elif strategy == "GK":
                if need_init is None:
                    need_init = need.copy()
                    wr0 = need_init * 12 / w0
                else:
                    wr = np.where(wealth > 0, draw_nom * 12 / np.maximum(wealth, 1), np.inf)
                    gk_mult = np.where(wr > 1.2 * wr0, gk_mult * 0.9, gk_mult)
                    gk_mult = np.where(wr < 0.8 * wr0, gk_mult * 1.1, gk_mult)
                    gk_mult = np.clip(gk_mult, 0.0, 1.5)
                draw_real = need * gk_mult
                min_draw = np.maximum(floor - pension, 0.0) + prem
                draw_real = np.maximum(draw_real, np.minimum(min_draw, need))
            elif strategy == "VPW":
                n_left = max((100 - START_AGE) - y, 1)
                r = 0.03
                f = r / (1 - (1 + r) ** -n_left) if n_left > 1 else 1.0
                draw_real = (wealth / price[:, t]) * f / 12.0
                min_draw = np.maximum(floor - pension, 0.0) + prem
                draw_real = np.maximum(draw_real, min_draw)
                cap = np.maximum((target - pension) * 2.0, min_draw) + prem
                draw_real = np.minimum(draw_real, cap)
            draw_nom = draw_real * price[:, t]
            if strategy != "DEPOSIT":
                bal[:, 0] = wealth * STOCK_W
                bal[:, 1] = wealth * (1 - STOCK_W)

        want = np.where(alive & ~ruined, draw_nom, 0.0)
        wealth = bal.sum(axis=1)
        take = np.minimum(want, wealth)
        frac = np.where(wealth > 0, take / np.maximum(wealth, 1e-9), 0.0)
        bal *= (1.0 - frac)[:, None]
        newly = alive & ~ruined & (take < want - 1e-6)
        ruined |= newly
        ruin_month = np.where(newly & (ruin_month < 0), t, ruin_month)

        spend_real = np.where(alive, pension + take / price[:, t] - np.where(ruined, 0, prem), 0.0)
        spend_real = np.maximum(spend_real, np.where(alive, pension, 0.0))
        lifetime_real += spend_real
        annual[:, y] += spend_real
        floor_breach |= alive & (spend_real < floor - 1)

        if strategy == "DEPOSIT":
            bal *= (1.0 + infl_m[:, t])[:, None]
        else:
            bal[:, 0] *= (1.0 + stock_r[:, t])
            bal[:, 1] *= (1.0 + bond_r[:, t])

    bequest = bal.sum(axis=1) / price[:, -1]
    return dict(
        ruin_alive_pct=round(100 * float(ruined.mean()), 2),
        ruin_age_med=round(START_AGE + float(np.median(ruin_month[ruined])) / 12, 1)
        if ruined.any() else None,
        ruin_before65_pct=round(100 * float((ruined & (ruin_month < PENSION_AGE_M)).mean()), 2),
        floor_breach_pct=round(100 * float(floor_breach.mean()), 2),
        lifetime_med=round(float(np.median(lifetime_real)) / 1e4, 0),
        lifetime_p5=round(float(np.percentile(lifetime_real, 5)) / 1e4, 0),
        bequest_med=round(float(np.median(bequest)) / 1e4, 0),
        annual=annual,
    )


def main():
    tw = load_panel("tw")
    tw["bill"] = tw.pop("usdbill")
    us = restrict(load_panel("us"), "1934-02")
    rng = np.random.default_rng(7)
    death_h = death_months_from50(N_PATHS, 18.2, 0.105, rng)
    death_w = death_months_from50(N_PATHS, 21.9, 0.110, rng)
    out = {"mortality": dict(
        h_med=round(START_AGE + float(np.median(death_h)) / 12, 1),
        w_med=round(START_AGE + float(np.median(death_w)) / 12, 1),
        joint_med=round(START_AGE + float(np.median(np.maximum(death_h, death_w))) / 12, 1),
        p_any95=round(100 * float((np.maximum(death_h, death_w) > (95 - 50) * 12).mean()), 1))}
    print("mortality:", out["mortality"])
    for ename, panel in [("TW", tw), ("US_STRESS", us)]:
        rng2 = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(len(panel["stock"]), T, N_PATHS, 36, rng2)
        for wname, w0 in WEALTH_LEVELS.items():
            for pname, (peach, prem) in PENSION.items():
                for spend in SPEND_LEVELS:
                    for strat in ("DEPOSIT", "FIX", "GK", "VPW"):
                        r = simulate(idx, panel, strat, w0, peach, prem, spend,
                                     death_h, death_w)
                        r.pop("annual")
                        key = f"{ename}|{wname}|{pname}|{int(spend/1000)}k|{strat}"
                        out[key] = r
                        print(f"{key:38s} ruin {r['ruin_alive_pct']:5.1f}% "
                              f"(<65: {r['ruin_before65_pct']:4.1f}%) "
                              f"life {r['lifetime_med']:6.0f} p5 {r['lifetime_p5']:6.0f} "
                              f"beq {r['bequest_med']:7.0f}"
                              + (f" @{r['ruin_age_med']}" if r['ruin_age_med'] else ""))
    with open(os.path.join(RESULTS, "fire_case.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote results/fire_case.json")


if __name__ == "__main__":
    main()
