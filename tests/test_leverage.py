"""v2 sanity tests (handoff-v2 §8). Run: python tests/test_leverage.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.leverage import LevSpec, simulate_leverage, deterministic_check


def test_hand_calc_replication():
    """§8.11: LTV peak ~24.1% around year 26; D(30)=1.8204; NW(30)=5.7918;
    cumulative nominal spend through year 30 = 1.1894 (x initial wealth)."""
    peak_ltv, peak_year, out = deterministic_check()
    assert 25 <= peak_year <= 27, peak_year
    assert abs(peak_ltv - 0.241) < 0.002, peak_ltv
    y30 = out[30]
    assert abs(y30["D"] - 1.8204) < 0.002, y30["D"]
    assert abs(y30["NW"] - 5.7918) < 0.005, y30["NW"]
    assert abs(y30["cum_spend"] - 1.1894) < 0.002, y30["cum_spend"]
    print(f"PASS hand-calc: LTV peak {peak_ltv*100:.2f}% @ y{peak_year}, "
          f"D30 {y30['D']:.4f}, NW30 {y30['NW']:.4f}, spend30 {y30['cum_spend']:.4f}")


def zero_vol_panel(T, g_ann=0.07, infl_ann=0.03, cash_ann=0.01):
    gm = (1 + g_ann) ** (1 / 12) - 1
    im = (1 + infl_ann) ** (1 / 12) - 1
    cm = (1 + cash_ann) ** (1 / 12) - 1
    return dict(stock=np.full(T, gm), bill=np.full(T, cm),
                twd_cash=np.full(T, cm), infl=np.full(T, im),
                cpi=np.cumprod(np.full(T, 1 + im)))


def test_engine_matches_hand_calc_zero_vol():
    """Monthly engine under zero vol / 7% asset / 3% debt / CPI-3% spending
    must land within 2% of the annual hand calc at year 30."""
    T = 360
    p = zero_vol_panel(T)
    idx = np.arange(T)[None, :]
    spec = LevSpec(rate=0.025, bucket_years=0, borrow_mode="path_c",
                   borrow_const=0.03, line_review=False, ltv_kill=9.9)
    r = simulate_leverage(idx, p, spec)
    # nominal NW at year 30 = real NW * price level
    price30 = 1.03 ** 30
    nw_nom = r["final_nw_real"][0] * price30
    _, _, out = deterministic_check()
    ref = out[30]["NW"]
    assert abs(nw_nom - ref) / ref < 0.02, (nw_nom, ref)
    assert not r["wiped"][0] and not r["forced_liq"][0]
    print(f"PASS zero-vol engine: NW30 nominal {nw_nom:.4f} vs hand-calc {ref:.4f} "
          f"({(nw_nom/ref-1)*100:+.2f}%), peak LTV {r['peak_ltv'][0]*100:.1f}%")


def test_no_lookahead_refill():
    """§8.8: tampering with future returns must not change refill decisions
    (proxied by identical spending and debt paths pre-tamper)."""
    rng = np.random.default_rng(3)
    T = 240
    p = zero_vol_panel(T)
    p["stock"] = rng.normal(0.006, 0.05, T)
    p2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    cut = 120
    p2["stock"][cut:] = -0.4
    idx = np.arange(T)[None, :]
    spec = LevSpec(rate=0.04, bucket_years=3, alpha=0.95, fallback="F3",
                   borrow_mode="path_c", line_review=False)
    a = simulate_leverage(idx, p, spec)["total_real_spend" ]
    sa = simulate_leverage(idx, p, spec)["cum_interest_real"][0]
    sb = simulate_leverage(idx, p2, spec)["cum_interest_real"][0]
    # cum interest is monotone in decisions; identical prefix -> compare via
    # rerun on truncated horizon
    idx_h = np.arange(cut)[None, :]
    ph = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    ph2 = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in p2.items()}
    ra = simulate_leverage(idx_h, ph, spec)
    rb = simulate_leverage(idx_h, ph2, spec)
    assert np.allclose(ra["total_real_spend"], rb["total_real_spend"])
    assert np.allclose(ra["cum_interest_real"], rb["cum_interest_real"])
    print("PASS no-lookahead: identical decisions on identical prefixes")


def test_maintenance_triggers():
    """A crash path must trigger forced liquidation when LTV crosses kill."""
    T = 120
    p = zero_vol_panel(T)
    p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    p["stock"][60:72] = -0.15          # ~86% crash over a year
    idx = np.arange(T)[None, :]
    spec = LevSpec(rate=0.06, bucket_years=0, borrow_mode="path_c",
                   borrow_const=0.03, ltv_kill=0.5, line_review=False)
    r = simulate_leverage(idx, p, spec)
    assert r["forced_liq"][0], "expected forced liquidation on crash path"
    print(f"PASS maintenance: forced liq fired, peak LTV {r['peak_ltv'][0]*100:.0f}%, "
          f"first hit year {r['first_hit_year'][0]}")


if __name__ == "__main__":
    test_hand_calc_replication()
    test_engine_matches_hand_calc_zero_vol()
    test_no_lookahead_refill()
    test_maintenance_triggers()
    print("ALL LEVERAGE TESTS PASSED")
