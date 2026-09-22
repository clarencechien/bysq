"""v4 §2: mortality layer.

Life table: 內政部 簡易生命表 (data/processed/tw_lifetable_<roc>.csv, built by
data/build_lifetable.py from the official ODS; ages 0-84 verbatim, 85+ closed
with a Gompertz tail calibrated to the published e85).

Everything here is independent of market returns, so mortality can be applied
AFTER a simulation: P(ruin & alive) = P(ruin_year < death_year), etc. That
keeps the no-lookahead engine untouched and makes the mortality layer a pure
post-processing step (v4 §10.2 re-checks v1-v3 tables the same way).
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "processed")


def load_qx(roc_year=114):
    """Return dict sex -> qx array indexed by age 0..110."""
    out = {"all": [], "m": [], "f": []}
    with open(os.path.join(PROC, f"tw_lifetable_{roc_year}.csv")) as f:
        for row in csv.DictReader(f):
            out["all"].append(float(row["qx_all"]))
            out["m"].append(float(row["qx_m"]))
            out["f"].append(float(row["qx_f"]))
    return {k: np.array(v) for k, v in out.items()}


def survival_curve(qx, age0, shift=0, max_age=110):
    """S[k] = P(alive at age0+k | alive at age0), k = 0..max_age-age0.
    shift: longevity_shift in years — a person aged x is given the mortality
    of a person aged x-shift (period-table -> cohort/high-SES correction,
    v4 §2.2)."""
    ages = np.arange(age0, max_age + 1) - shift
    ages = np.clip(ages, 0, len(qx) - 1)
    q = qx[ages].copy()
    q[-1] = 1.0
    S = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
    return S


def life_expectancy(qx, age0, shift=0):
    S = survival_curve(qx, age0, shift)
    return S[1:].sum() + 0.5


def sample_death_years(qx, age0, n, rng, shift=0, max_age=110):
    """Whole years lived after age0 (death occurs during year k -> returns k,
    meaning the person is alive for years 0..k-1 and dies in year k).
    Uniform within-year timing is applied by callers that need months."""
    S = survival_curve(qx, age0, shift, max_age)
    # P(death in year k) = S[k] - S[k+1]
    pk = S[:-1] - S[1:]
    pk = np.append(pk, S[-1])          # everyone left dies at max_age
    pk = pk / pk.sum()
    return rng.choice(len(pk), size=n, p=pk)


def joint_last_survivor(qx_m, qx_f, age_m, age_f, n, rng, shift=0):
    dm = sample_death_years(qx_m, age_m, n, rng, shift)
    df = sample_death_years(qx_f, age_f, n, rng, shift)
    return dm, df, np.maximum(dm, df)


def alive_metrics(res, death_year, floor_abs, min_years=3):
    """Post-hoc mortality metrics on a core.simulate result.

    death_year: per-path years lived after retirement (person alive in years
    0..death_year-1). Metrics:
      ruin_alive     assets exhausted in a year the person is still alive
      breach_alive   real spending < floor_abs for >= min_years consecutive
                     years, all of them while alive
      breach1_alive  at least one alive year below the floor
      life_spend     lifetime real spending while alive (units of W0)
      ce_g4_alive    CRRA(4) certainty-equivalent over alive path-years
    """
    s = res["annual_real_spend"]
    P, years = s.shape
    dy = np.minimum(death_year, years)
    yrs = np.arange(years)[None, :]
    alive = yrs < dy[:, None]
    ruin_alive = (res["ruin_year"] >= 0) & (res["ruin_year"] < dy)
    below = (s < floor_abs) & alive
    run = np.zeros(P)
    c = np.zeros(P)
    for y in range(years):
        run = np.where(below[:, y], run + 1, 0)
        c = np.maximum(c, run)
    life_spend = (s * alive).sum(axis=1)
    sa = np.maximum(s, 0.0005)[alive]
    ce4 = float((np.mean(sa ** -3) * -3 / -3) ** (-1 / 3))  # gamma=4 CE
    return dict(
        ruin_alive_pct=round(100 * float(ruin_alive.mean()), 2),
        breach_alive_pct=round(100 * float((c >= min_years).mean()), 2),
        breach1_alive_pct=round(100 * float(below.any(axis=1).mean()), 2),
        life_spend_p5=round(float(np.percentile(life_spend, 5)), 3),
        life_spend_med=round(float(np.median(life_spend)), 3),
        ce_g4_alive=round(ce4, 4),
        yrs_med=float(np.median(dy)),
    )
