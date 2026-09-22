"""Anti-self-deception tests for the v4 §7 return-stacking layer.
Run: python tests/test_stack.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.build_trend import tsmom_proxy
from engine.core import load_panel, restrict, simulate, Alloc, FixedReal
from engine.stack import stacked_panel, stacked_returns, spending_smoothness


def _synthetic(T, seed=0):
    rng = np.random.default_rng(seed)
    stock = rng.normal(0.006, 0.04, T)
    bond = rng.normal(0.003, 0.015, T)
    bill = np.full(T, 0.002)
    return stock, bond, bill


def test_proxy_no_lookahead():
    """Tampering with returns from month t onward must leave every position
    (= the trend signal x vol scaling) at months <= t unchanged, and every
    proxy return before month t unchanged."""
    T = 240
    stock, bond, bill = _synthetic(T)
    base, pos0 = tsmom_proxy(stock, bond, bill)
    rng = np.random.default_rng(1)
    for t in (60, 120, 200):
        s2, b2 = stock.copy(), bond.copy()
        s2[t:] = rng.normal(0.0, 0.08, T - t)      # wildly different future
        b2[t:] = rng.normal(-0.01, 0.03, T - t)
        alt, pos1 = tsmom_proxy(s2, b2, bill)
        assert np.allclose(pos0[:t + 1], pos1[:t + 1]), \
            f"position at or before month {t} changed when future returns changed"
        assert np.allclose(base[:t], alt[:t]), \
            f"proxy return before month {t} changed when future returns changed"
        # the detector must actually be able to see a change after t
        assert not np.allclose(pos0[t + 13:], pos1[t + 13:]), \
            "tampering the future did not move later positions — test is blind"
    print("PASS proxy no-lookahead: positions through t depend only on data through t-1")


def test_proxy_uses_t_minus_1_only_at_month_t():
    """Stronger form: change ONLY month t's returns; position at t must be
    identical, position at t+1 may differ."""
    T = 120
    stock, bond, bill = _synthetic(T, seed=3)
    _, pos0 = tsmom_proxy(stock, bond, bill)
    t = 80
    s2 = stock.copy()
    s2[t] = +0.5                                  # huge shock in month t
    _, pos1 = tsmom_proxy(s2, bond, bill)
    assert np.allclose(pos0[t], pos1[t]), "position at t read month-t return"
    assert not np.allclose(pos0[t + 1], pos1[t + 1]), "position at t+1 ignored month-t return"
    print("PASS proxy month-t shock: pos[t] unchanged, pos[t+1] reacts")


def test_identity_zero_overlays():
    """With trend = 0, bond == bill (zero bond overlay), spread = 0 and
    fees = 0, RSST, RSSB and the five-five series equal the stock series
    exactly, and the engine gives identical results."""
    panel = restrict(load_panel("us"), "1934-02", "1990-12")
    zero_trend = np.zeros(len(panel["stock"]))
    flat = dict(panel)
    flat["bond"] = panel["bill"].copy()            # bond - bill == 0
    for kind in ("RSST", "RSSB", "FIVEFIVE"):
        r = stacked_returns(kind, flat["stock"], flat["bond"], flat["bill"], zero_trend,
                            spread=0.0, fee_rssb=0.0, fee_rsst=0.0)
        assert np.array_equal(r, panel["stock"]), f"{kind} != stock with zero overlays"
        sp = stacked_panel(flat, zero_trend, kind, spread=0.0, fee_rssb=0.0, fee_rsst=0.0)
        idx = np.arange(360)[None, :] + np.arange(0, 300, 50)[:, None]
        a = simulate(idx, sp, Alloc(1.0), FixedReal(0.04))
        b = simulate(idx, panel, Alloc(1.0), FixedReal(0.04))
        assert np.array_equal(a["annual_real_spend"], b["annual_real_spend"])
        assert np.array_equal(a["final_real_wealth"], b["final_real_wealth"])
    print("PASS identity: zero overlays + zero fee reproduce the stock series exactly")


def test_fee_and_spread_sign():
    """Fees/spread must only ever lower the series, by exactly (fee+spread)/12."""
    T = 50
    stock, bond, bill = _synthetic(T)
    tr = np.zeros(T)
    r0 = stacked_returns("RSST", stock, bond, bill, tr, spread=0.0, fee_rsst=0.0)
    r1 = stacked_returns("RSST", stock, bond, bill, tr, spread=0.01, fee_rsst=0.0099)
    assert np.allclose(r0 - r1, (0.01 + 0.0099) / 12)
    ff = stacked_returns("FIVEFIVE", stock, bond, bill, tr, spread=0.0)
    expect = stock + 0.5 * (bond - bill) - 0.5 * (0.0039 + 0.0099) / 12
    assert np.allclose(ff, expect)
    print("PASS fee/spread arithmetic and five-five exposures")


def test_smoothness_metrics():
    s = np.array([[1.0, 1.0, 0.5, 0.5, 1.0],       # one 50% cut, 2 yrs < 80% of peak
                  [1.0, 1.1, 1.21, 1.331, 1.4641]])  # steady +10%
    m = spending_smoothness(s)
    assert abs(m["spend_dd_max"] - (-0.25)) < 1e-12, m   # median of (-0.5, 0.0)
    assert abs(m["spend_dd_yrs"] - 1.0) < 1e-12, m       # median of (2, 0)
    print("PASS spending-smoothness metrics on a hand-made path")


if __name__ == "__main__":
    test_proxy_no_lookahead()
    test_proxy_uses_t_minus_1_only_at_month_t()
    test_identity_zero_overlays()
    test_fee_and_spread_sign()
    test_smoothness_metrics()
    print("ALL PASS")
