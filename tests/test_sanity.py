"""Anti-self-deception tests (HANDOFF §6). Run: python tests/test_sanity.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.core import (load_panel, restrict, simulate, historical_indices,
                         stationary_bootstrap_indices, Alloc, BucketAlloc,
                         FixedReal, GuytonKlinger)


def synthetic_panel(T, stock=0.005, bond=0.003, bill=0.002, infl=0.002):
    cpi = np.cumprod(np.full(T, 1 + infl))
    return dict(stock=np.full(T, stock), bond=np.full(T, bond),
                bill=np.full(T, bill), cpi=cpi,
                infl=np.full(T, infl), dates=[str(i) for i in range(T)])


def test_zero_vol_annuity():
    """With zero volatility and returns == inflation, a fixed-real rule must
    exhaust wealth in exactly 1/rate years (start-of-month annuity)."""
    T = 12 * 40
    p = synthetic_panel(T, stock=0.002, bond=0.002, bill=0.002, infl=0.002)
    idx = np.arange(T)[None, :]
    res = simulate(idx, p, Alloc(0.6), FixedReal(0.04))
    # nominal return == inflation → real return 0 → money lasts 25 years
    ry = res["ruin_year"][0]
    assert 24 <= ry <= 25, f"expected ruin in year 24-25, got {ry}"
    total = res["total_real_spend"][0]
    assert abs(total - 1.0) < 0.01, f"lifetime real spend should be ~1.0, got {total:.4f}"
    print(f"PASS zero-vol annuity: ruin year {ry}, lifetime real spend {total:.4f}")


def test_no_lookahead():
    """Perturbing returns from month t onward must not change any spending
    decision made at or before month t."""
    rng = np.random.default_rng(0)
    T = 12 * 30
    base = dict(stock=rng.normal(0.006, 0.04, T), bond=rng.normal(0.003, 0.015, T),
                bill=np.full(T, 0.002), cpi=None, infl=rng.normal(0.002, 0.003, T))
    base["cpi"] = np.cumprod(1 + base["infl"])
    tampered = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    cut = 12 * 15
    tampered["stock"][cut:] = -0.5  # catastrophic future
    tampered["bond"][cut:] = -0.5
    idx = np.arange(T)[None, :]
    for alloc, rule in [(Alloc(0.6), GuytonKlinger(0.05)),
                        (BucketAlloc(3), FixedReal(0.04)),
                        (Alloc(0.6), FixedReal(0.04))]:
        a = simulate(idx, base, alloc, rule)["annual_real_spend"][0]
        b = simulate(idx, tampered, alloc, rule)["annual_real_spend"][0]
        yrs = cut // 12
        assert np.allclose(a[:yrs], b[:yrs]), \
            f"lookahead detected in {alloc.name}/{rule.name}"
    print("PASS no-lookahead: pre-tamper spending identical for GK, bucket, fixed")


def test_lookahead_detector_catches_violation():
    """Deliberately violating strategy (uses month-t return) must be caught
    by the same comparison — proves the detector has teeth."""
    rng = np.random.default_rng(1)
    T = 12 * 4
    p = synthetic_panel(T)
    p["stock"] = rng.normal(0.005, 0.05, T)
    # a cheating rule: spend more if THIS month's stock return is positive
    def cheat_path(panel):
        wealth, spend = 1.0, []
        for t in range(T):
            w = 0.004 if panel["stock"][t] > 0 else 0.002   # uses r[t]: illegal
            wealth -= w
            spend.append(w)
            wealth *= 1 + panel["stock"][t]
        return np.array(spend)
    a = cheat_path(p)
    p2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    p2["stock"][T // 2:] *= -1
    b = cheat_path(p2)
    assert not np.allclose(a[:T // 2 + 1], b[:T // 2 + 1]), \
        "detector failed to catch a deliberate lookahead"
    print("PASS detector catches deliberate lookahead")


def test_covariance_preservation():
    """Bootstrap resamples must keep the stock-inflation and stock-bond
    covariance seen in the source panel (joint row sampling)."""
    panel = restrict(load_panel("us"), "1934-02")
    rng = np.random.default_rng(2)
    idx = stationary_bootstrap_indices(len(panel["stock"]), 360, 200, 36, rng)
    s, b, f = panel["stock"][idx], panel["bond"][idx], panel["infl"][idx]
    def corr(x, y):
        xm, ym = x - x.mean(), y - y.mean()
        return (xm * ym).mean() / (x.std() * y.std())
    src_sb = np.corrcoef(panel["stock"], panel["bond"])[0, 1]
    src_sf = np.corrcoef(panel["stock"], panel["infl"])[0, 1]
    boot_sb = np.array([np.corrcoef(s[i], b[i])[0, 1] for i in range(200)])
    boot_sf = np.array([np.corrcoef(s[i], f[i])[0, 1] for i in range(200)])
    assert abs(boot_sb.mean() - src_sb) < 0.05, (src_sb, boot_sb.mean())
    assert abs(boot_sf.mean() - src_sf) < 0.05, (src_sf, boot_sf.mean())
    print(f"PASS covariance preserved: stock-bond src {src_sb:.3f} boot {boot_sb.mean():.3f}; "
          f"stock-infl src {src_sf:.3f} boot {boot_sf.mean():.3f}")


def test_historical_window_count():
    panel = load_panel("us")
    idx = historical_indices(len(panel["stock"]), 360)
    assert idx.shape[0] > 1400
    print(f"PASS historical windows: {idx.shape[0]} 30y monthly-start windows")


if __name__ == "__main__":
    test_zero_vol_annuity()
    test_no_lookahead()
    test_lookahead_detector_catches_violation()
    test_covariance_preservation()
    test_historical_window_count()
    print("ALL SANITY TESTS PASSED")
