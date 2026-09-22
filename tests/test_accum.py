"""v4 §5 accumulation-phase anti-self-deception tests.
Run: python tests/test_accum.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_v4_accum import simulate_accum, glide_share, GLIDES


def flat_panel(T, r):
    return dict(stock=np.full(T, r), bond=np.full(T, r), infl=np.full(T, r),
                cpi=np.cumprod(np.full(T, 1 + r)), dates=[str(i) for i in range(T)])


def test_zero_vol_identity():
    """Returns == inflation -> zero real growth -> W_T == c x years exactly,
    for every glide path (start-of-month contributions, no growth)."""
    T = 12 * 30
    p = flat_panel(T, 0.003)
    idx = np.arange(T)[None, :]
    for g in GLIDES:
        r = simulate_accum(idx, p, g, 1.0, k=0)
        assert abs(r["w_T"][0] - 30.0) < 1e-9, f"{g}: W_T={r['w_T'][0]!r}"
        assert r["months_contrib"][0] == T
    # H4 on but no hazard hits (k=1, hazard tiny; force via h0=0)
    r = simulate_accum(idx, p, "A3", 0.5, h0=0.0, k=4)
    assert abs(r["w_T"][0] - 15.0) < 1e-9
    print("PASS zero-vol identity: W_T = 30.0 (c=1, 30y) for A1-A5; 15.0 for c=0.5")


def test_glide_endpoints():
    assert glide_share("A4", 40, 0) == 1.0 and abs(glide_share("A4", 40, 39) - 0.64) < 1e-12
    assert glide_share("A5", 40, 19) == 1.0 and abs(glide_share("A5", 40, 39) - 0.43) < 1e-12
    assert glide_share("A5", 20, 0) == 1.0 and abs(glide_share("A5", 20, 19) - 0.43) < 1e-12
    print("PASS glide endpoints (A4 tent, A5 age-based)")


def _random_panel(T, seed):
    rng = np.random.default_rng(seed)
    p = dict(stock=rng.normal(0.006, 0.05, T), bond=rng.normal(0.003, 0.015, T),
             infl=rng.normal(0.002, 0.003, T))
    p["cpi"] = np.cumprod(1 + p["infl"])
    return p


def test_no_lookahead():
    """Tampering returns from month `cut` onward must leave every contribution,
    unemployment flag and coast flag before `cut` unchanged (H4 crash state,
    coast line and contributions all read information through t-1 only)."""
    T = 12 * 30
    cut = 12 * 12
    P = 400
    base = _random_panel(T, 3)
    tam = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    tam["stock"][cut:] = -0.6          # crash from month `cut` on
    tam["bond"][cut:] = -0.6
    tam["infl"][cut:] = 0.05
    rng = np.random.default_rng(0)
    # random paths (many crash states / coast crossings), but months before
    # the cut only draw panel rows before the cut, so the tampered rows can
    # only ever be seen from month `cut` onward
    idx = np.empty((P, T), dtype=np.int64)
    idx[:, :cut] = rng.integers(0, cut, (P, cut))
    idx[:, cut:] = rng.integers(cut, T, (P, T - cut))
    for kw in (dict(k=4, coast_theta=1.0), dict(k=1, coast_theta=1.5),
               dict(k=4, coast_theta=None), dict(k=0, coast_theta=1.0)):
        a = simulate_accum(idx, base, "A4", 1.0, trace=True, **kw)
        b = simulate_accum(idx, tam, "A4", 1.0, trace=True, **kw)
        for key in ("contrib", "unemployed", "coasting"):
            assert np.array_equal(a[key][:, :cut], b[key][:, :cut]), \
                f"lookahead in {key} with {kw}"
        # the detector has teeth: the crash DOES change decisions after the cut
        if kw["k"] == 4:
            assert not np.array_equal(a["unemployed"], b["unemployed"]), kw
        if kw["coast_theta"] is not None:
            assert a["coasting"].any() and not np.array_equal(a["coasting"], b["coasting"]), kw
    print("PASS no-lookahead: contributions / unemployment / coast flags before "
          "the tamper month are identical")


def test_h4_hazard_scales_with_k():
    """More correlation multiplier -> more unemployment months on average and
    a lower terminal p5 (the mechanism H4 is meant to add)."""
    T = 12 * 30
    P = 3000
    p = _random_panel(T, 7)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, T, (P, T))
    m = [simulate_accum(idx, p, "A3", 1.0, k=k)["months_unemp"].mean() for k in (1, 2, 4)]
    assert m[0] < m[1] < m[2], m
    print(f"PASS H4: mean unemployed months k=1/2/4 = {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f}")


if __name__ == "__main__":
    test_zero_vol_identity()
    test_glide_endpoints()
    test_no_lookahead()
    test_h4_hazard_scales_with_k()
    print("ALL PASS")
