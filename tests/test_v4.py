"""v4 anti-self-deception tests: mortality layer, household simulator,
ladder/annuity pricing identities, no-lookahead of the floor family.
Run: python tests/test_v4.py"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.mortality import load_qx, life_expectancy, sample_death_years, alive_metrics
from engine.household import HH, simulate_household, annuity_factor
from engine.core import simulate, Alloc, VPW, FixedReal


def synthetic_panel(T, r=0.002, infl=0.002):
    cpi = np.cumprod(np.full(T, 1 + infl))
    return dict(stock=np.full(T, r), bond=np.full(T, r), bill=np.full(T, r), cpi=cpi,
                infl=np.full(T, infl), dates=[str(i) for i in range(T)])


def test_life_table_matches_published():
    qx = load_qx(114)
    # 內政部 114 年簡易生命表 published e0/e65 (全體 81.27 / 20.88; 男 77.93 / 18.77; 女 84.75 / 22.90)
    for k, e0, e65 in (("all", 81.27, 20.88), ("m", 77.93, 18.77), ("f", 84.75, 22.90)):
        assert abs(life_expectancy(qx[k], 0) - e0) < 0.05, (k, life_expectancy(qx[k], 0))
        assert abs(life_expectancy(qx[k], 65) - e65) < 0.05, (k, life_expectancy(qx[k], 65))
    d = sample_death_years(qx["all"], 65, 200_000, np.random.default_rng(0))
    assert abs(d.mean() + 0.5 - 20.88) < 0.15
    print("PASS life table: e0/e65 reproduce the published values; sampled e65 =", round(d.mean() + 0.5, 2))


def test_household_zero_vol_fixed_rule():
    """Zero vol, nominal return == inflation, no income, FIX rule at 4%:
    the portfolio must last 25 years exactly (start-of-month, CPI-indexed)."""
    T = 12 * 45
    p = synthetic_panel(T)
    idx = np.arange(T)[None, :]
    hh = HH(retire_age=65, spend=0.04, floor_ratio=0.5, stock_w=0.6, rule="FIX")
    death = np.array([T + 1])
    r = simulate_household(idx, p, hh, death)
    s = r["annual_spend"][0]
    assert abs(s[:25].sum() - 1.0) < 0.01, s[:25].sum()
    assert s[26] < 1e-9 and r["ruin_alive_pct"] == 100.0
    print(f"PASS household zero-vol: 25 years of 4% real, lifetime {s.sum():.4f}, ruin flagged")


def test_ladder_cost_identity_and_payout():
    """A real ladder of N rungs must cost gap * annuity factor and pay the gap
    for exactly N years; with zero-vol returns and a FIX rule the total
    spending equals the floor gap from the ladder plus the portfolio draw."""
    T = 12 * 30
    p = synthetic_panel(T)
    idx = np.arange(T)[None, :]
    hh = HH(retire_age=65, spend=0.04, floor_ratio=0.7, stock_w=1.0, rule="FIX",
            ladder_years=10, ladder_kind="real", ladder_real_rate=0.01)
    r = simulate_household(idx, p, hh, np.array([T + 1]))
    gap = 0.028
    cost = gap * sum(1.01 ** -k for k in range(1, 11))
    assert abs(r["ladder_cost"] - cost) < 2e-3   # csv-rounded to 3 dp
    s = r["annual_spend"][0]
    assert abs(s[0] - 0.04) < 1e-6 and abs(s[9] - 0.04) < 1e-6     # ladder + draw = target
    assert abs(r["invest0"] - (1 - cost)) < 2e-3
    print(f"PASS ladder: cost {cost:.4f} = identity, pays 10 years, spend on target")


def test_annuity_factor_sanity():
    qx = load_qx(114)
    af80 = annuity_factor(qx["all"], 65, 80, 0.0175, shift=3)
    af65 = annuity_factor(qx["all"], 65, 65, 0.0175, shift=3)
    assert 3.0 < af80 < 9.0 and af65 > af80 * 2, (af65, af80)
    # deferred annuity must cost less than the lifetime one and less than a
    # 15-year ladder of the same amount (mortality credits)
    assert af80 < sum(1.0175 ** -k for k in range(15, 30))
    print(f"PASS annuity factor: immediate {af65:.2f}, deferred-to-80 {af80:.2f} per 1/yr")


def test_household_no_lookahead():
    rng = np.random.default_rng(0)
    T = 12 * 40
    base = dict(stock=rng.normal(0.006, 0.04, T), bond=rng.normal(0.003, 0.015, T),
                bill=np.full(T, 0.002), infl=rng.normal(0.002, 0.003, T))
    base["cpi"] = np.cumprod(1 + base["infl"]); base["dates"] = [str(i) for i in range(T)]
    tam = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    cut = 12 * 20
    tam["stock"][cut:] = -0.5; tam["bond"][cut:] = -0.5
    idx = np.arange(T)[None, :]
    death = np.array([T + 1])
    for kw in (dict(stock_w=1.0, rule="VPW", pension=0.01), dict(stock_w=0.6, rule="GK"),
               dict(stock_w=1.0, rule="VPW", ladder_years=15), dict(stock_w=1.0, rule="VPW", ann_start_age=80),
               dict(stock_w=1.0, rule="VPW", behavior=(0.3, 6, 1.0, 0.05))):
        hh = HH(retire_age=65, spend=0.04, floor_ratio=0.7, **kw)
        a = simulate_household(idx, base, hh, death)["annual_spend"][0]
        b = simulate_household(idx, tam, hh, death)["annual_spend"][0]
        assert np.allclose(a[:cut // 12], b[:cut // 12]), f"lookahead in {hh.name}"
    print("PASS household no-lookahead: VPW/GK/ladder/annuity/behaviour unchanged before tamper")


def test_mortality_layer_bounds():
    """P(ruin & alive) <= P(ruin); with death at year 0 for all, everything is 0."""
    T = 12 * 30
    rng = np.random.default_rng(1)
    p = dict(stock=rng.normal(0.004, 0.05, T), bond=rng.normal(0.002, 0.02, T),
             bill=np.full(T, 0.001), infl=np.full(T, 0.003))
    p["cpi"] = np.cumprod(1 + p["infl"]); p["dates"] = [str(i) for i in range(T)]
    idx = np.tile(np.arange(T), (200, 1))
    idx = (idx + rng.integers(0, T, (200, 1))) % T
    res = simulate(idx, p, Alloc(0.6), FixedReal(0.06))
    m_all = alive_metrics(res, np.full(200, 30), 0.02)
    m_half = alive_metrics(res, np.full(200, 15), 0.02)
    m_zero = alive_metrics(res, np.full(200, 0), 0.02)
    assert m_half["ruin_alive_pct"] <= m_all["ruin_alive_pct"]
    assert m_zero["ruin_alive_pct"] == 0.0 and m_zero["breach_alive_pct"] == 0.0
    assert abs(m_all["ruin_alive_pct"] - 100 * res["exhausted"].mean()) < 1e-9
    print(f"PASS mortality bounds: ruin 30y {m_all['ruin_alive_pct']}% >= 15y {m_half['ruin_alive_pct']}% >= 0y 0")


if __name__ == "__main__":
    test_life_table_matches_published()
    test_household_zero_vol_fixed_rule()
    test_ladder_cost_identity_and_payout()
    test_annuity_factor_sanity()
    test_household_no_lookahead()
    test_mortality_layer_bounds()
    print("ALL V4 TESTS PASSED")
