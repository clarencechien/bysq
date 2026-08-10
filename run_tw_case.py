"""Taiwan household case study: 65-year-old same-age couple, NT$10M at
retirement, Labor Insurance pension floor, latest-life-table mortality.

Assumptions (all stated in the report):
  - Couple, both 65 in year 0. Deaths independent, Gompertz force of
    mortality calibrated to the 2023 MOI abridged life table remaining life
    expectancy at 65 (M 18.2y, F 21.9y).
  - Pension (勞保年金, CPI-indexed by law when cumulative CPI>5% — treated
    as fully real, optimistic):
      husband: insured-salary ceiling 45,800 x 40 years x 1.55% = 28,400/mo
      wife:    35,000 x 25y x 1.55%                            = 13,600/mo
      household full = 42,000/mo; scenarios: full / x0.7 / zero.
      After first death the survivor keeps 28,400 (simplification: the
      higher pension survives).
  - 勞退 (labor pension individual accounts) taken as lump sum and already
    included in the NT$10M investable assets.
  - Household spending target real 60,000/mo while both alive; x0.75 after
    the first death. Housing profiles change total & uncuttable floor:
      own_nomortgage: total 60k, floor 30k  (50%)
      renter:         total 70k, floor 49k  (70%)
      mortgage_to75:  total 75k (to age 75) then 60k; floor 45k then 30k
  - Portfolio: 80% global stock / 20% bonds (TWD-converted), annual
    rebalance, monthly sim on the joint-sampled panels.
  - Strategies fund (spending - pension) from the portfolio:
      DEPOSIT: assets in cash at 0% real (定存想像), fixed real draw
      FIX:     fixed real draw (Bengen-style, never adjusts)
      GK:      Guyton-Klinger-style guardrails on the draw, cuts never push
               total spending below the floor
      VPW:     amortize over years remaining to age 100 at 3% real;
               forced top-up to the floor if the amortized draw is below it
  - Ruin = assets exhausted while at least one spouse alive; afterwards the
    household lives on the pension alone (spending = max(pension, 0)).
python run_tw_case.py -> results/tw_case.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
HORIZON_Y = 40          # to age 105, capped
W0 = 10_000_000.0       # NT$
STOCK_W = 0.8

PENSION_H, PENSION_W = 28_400.0, 13_600.0
PENSION_SCEN = {"full": 1.0, "cut30": 0.7, "zero": 0.0}
HOUSING = {
    "own":      dict(total=60_000.0, floor=30_000.0, total2=60_000.0, floor2=30_000.0, sw_age=999),
    "renter":   dict(total=70_000.0, floor=49_000.0, total2=70_000.0, floor2=49_000.0, sw_age=999),
    "mortgage": dict(total=75_000.0, floor=45_000.0, total2=60_000.0, floor2=30_000.0, sw_age=75),
}
WIDOW_FACTOR = 0.75


# ---------------------------------------------------------------- mortality --

def gompertz_calibrate(e65_target, B):
    """Find A in mu(x)=A*exp(B*x) matching remaining life expectancy at 65."""
    ages = np.arange(65, 111, 1 / 12)

    def e65(A):
        mu = A * np.exp(B * ages)
        surv = np.exp(-np.cumsum(mu) / 12)
        return surv.sum() / 12

    lo, hi = 1e-8, 1e-2
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if e65(mid) > e65_target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def death_months(n, e65, B, rng):
    """Sample months-from-65 until death (Gompertz, monthly grid)."""
    A = gompertz_calibrate(e65, B)
    ages = np.arange(65, 111, 1 / 12)
    mu = A * np.exp(B * ages)
    surv = np.exp(-np.cumsum(mu) / 12)
    u = rng.random(n)
    return np.searchsorted(-surv, -u)          # first month where surv < u


# ---------------------------------------------------------------- simulator --

def simulate_case(idx, panel, strategy, pension_mult, housing, death_h, death_w):
    P, T = idx.shape
    stock_r = panel["stock"][idx]
    bond_r = panel["bond"][idx]
    infl_m = panel["infl"][idx]
    price = np.ones((P, T))
    lvl = np.ones(P)
    for t in range(1, T):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    hcfg = HOUSING[housing]
    bal = np.stack([np.full(P, W0 * STOCK_W), np.full(P, W0 * (1 - STOCK_W))], axis=1)
    ruined = np.zeros(P, bool)
    ruin_month = np.full(P, -1)
    floor_breach = np.zeros(P, bool)
    lifetime_real = np.zeros(P)
    draw_nom = None
    gk_mult = np.ones(P)                      # GK cumulative adjustment
    yrs = T // 12
    annual_total_real = np.zeros((P, yrs))

    for t in range(T):
        y = t // 12
        age = 65 + t / 12.0
        both = (t < death_h) & (t < death_w)
        one = ((t < death_h) | (t < death_w)) & ~both
        alive_any = both | one

        total_target = np.where(age < hcfg["sw_age"], hcfg["total"], hcfg["total2"])
        floor = np.where(age < hcfg["sw_age"], hcfg["floor"], hcfg["floor2"])
        total_target = np.where(one, total_target * WIDOW_FACTOR, total_target)
        floor = np.where(one, floor * WIDOW_FACTOR, floor)
        pension = np.where(both, (PENSION_H + PENSION_W) * pension_mult,
                           np.where(one, PENSION_H * pension_mult, 0.0))

        wealth = bal.sum(axis=1)
        if t % 12 == 0:
            # ---- annual draw decision (info through t-1) ----
            need = np.maximum(total_target - pension, 0.0)     # real, monthly
            if strategy in ("DEPOSIT", "FIX"):
                draw_real = need
            elif strategy == "GK":
                if t > 0:
                    wr = np.where(wealth > 0, draw_nom * 12 / np.maximum(wealth, 1), np.inf)
                    wr0 = need_init * 12 / W0
                    gk_mult = np.where(wr > 1.2 * wr0, gk_mult * 0.9, gk_mult)
                    gk_mult = np.where(wr < 0.8 * wr0, gk_mult * 1.1, gk_mult)
                    gk_mult = np.clip(gk_mult, 0.0, 1.5)
                else:
                    need_init = need.copy()
                draw_real = need * gk_mult
                # cuts may not push total spending below the floor
                min_draw = np.maximum(floor - pension, 0.0)
                draw_real = np.maximum(draw_real, np.minimum(min_draw, need))
            elif strategy == "VPW":
                n_left = max((100 - 65) - y, 1)
                r = 0.03
                f = r / (1 - (1 + r) ** -n_left) if n_left > 1 else 1.0
                draw_real = (wealth / price[:, t]) * f / 12.0
                min_draw = np.maximum(floor - pension, 0.0)
                draw_real = np.maximum(draw_real, min_draw)     # floor top-up
                draw_real = np.minimum(draw_real, np.maximum(need * 2.5, min_draw))
            draw_nom = draw_real * price[:, t]
            # rebalance annually (skip for DEPOSIT: all cash conceptually)
            if strategy != "DEPOSIT":
                bal[:, 0] = wealth * STOCK_W
                bal[:, 1] = wealth * (1 - STOCK_W)

        # ---- monthly withdrawal ----
        want = np.where(alive_any & ~ruined, draw_nom, 0.0)
        wealth = bal.sum(axis=1)
        take = np.minimum(want, wealth)
        frac = np.where(wealth > 0, take / np.maximum(wealth, 1e-9), 0.0)
        bal *= (1.0 - frac)[:, None]
        newly = alive_any & ~ruined & (take < want - 1e-6)
        ruined |= newly
        ruin_month = np.where(newly & (ruin_month < 0), t, ruin_month)

        spend_real = np.where(alive_any, pension + take / price[:, t], 0.0)
        lifetime_real += spend_real
        annual_total_real[:, y] += spend_real
        floor_breach |= alive_any & (spend_real < floor - 1)

        # ---- returns ----
        if strategy == "DEPOSIT":
            bal[:, 0] *= (1.0 + infl_m[:, t])    # cash holds real value (0% real)
            bal[:, 1] *= (1.0 + infl_m[:, t])
        else:
            bal[:, 0] *= (1.0 + stock_r[:, t])
            bal[:, 1] *= (1.0 + bond_r[:, t])

    last_death = np.maximum(death_h, death_w)
    bequest_real = bal.sum(axis=1) / price[:, -1]
    return dict(
        ruin_alive_pct=round(100 * float(ruined.mean()), 2),
        ruin_age_med=round(65 + float(np.median(ruin_month[ruined])) / 12, 1) if ruined.any() else None,
        floor_breach_pct=round(100 * float(floor_breach.mean()), 2),
        lifetime_med=round(float(np.median(lifetime_real)) / 1e4, 0),
        lifetime_p5=round(float(np.percentile(lifetime_real, 5)) / 1e4, 0),
        bequest_med=round(float(np.median(bequest_real)) / 1e4, 0),
        annual_total_real=annual_total_real,
    )


def main():
    tw = load_panel("tw")
    tw["bill"] = tw.pop("usdbill")
    us = restrict(load_panel("us"), "1934-02")
    T = HORIZON_Y * 12
    rng = np.random.default_rng(7)
    death_h = death_months(N_PATHS, 18.2, 0.105, rng)
    death_w = death_months(N_PATHS, 21.9, 0.110, rng)
    print(f"mortality check: median death age H {65+np.median(death_h)/12:.1f} "
          f"W {65+np.median(death_w)/12:.1f} "
          f"joint last-survivor median {65+np.median(np.maximum(death_h,death_w))/12:.1f} "
          f"P(either alive at 95) {(np.maximum(death_h,death_w)>360).mean()*100:.0f}%")

    out = {"mortality": dict(
        h_med=round(65 + float(np.median(death_h)) / 12, 1),
        w_med=round(65 + float(np.median(death_w)) / 12, 1),
        joint_med=round(65 + float(np.median(np.maximum(death_h, death_w))) / 12, 1),
        p_any95=round(100 * float((np.maximum(death_h, death_w) > 360).mean()), 1))}

    for ename, panel in [("TW", tw), ("US_STRESS", us)]:
        rng2 = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(len(panel["stock"]), T, N_PATHS, 36, rng2)
        for pname, pmult in PENSION_SCEN.items():
            for hname in HOUSING:
                for strat in ("DEPOSIT", "FIX", "GK", "VPW"):
                    r = simulate_case(idx, panel, strat, pmult, hname, death_h, death_w)
                    r.pop("annual_total_real", None)
                    key = f"{ename}|{pname}|{hname}|{strat}"
                    out[key] = r
                    print(f"{key:34s} ruin {r['ruin_alive_pct']:5.1f}% "
                          f"floor-breach {r['floor_breach_pct']:5.1f}% "
                          f"life {r['lifetime_med']:6.0f}萬 p5 {r['lifetime_p5']:6.0f} "
                          f"bequest {r['bequest_med']:6.0f}萬"
                          + (f" ruin@{r['ruin_age_med']}" if r['ruin_age_med'] else ""))
    with open(os.path.join(RESULTS, "tw_case.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote results/tw_case.json")


if __name__ == "__main__":
    main()
