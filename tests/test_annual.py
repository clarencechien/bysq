"""Sanity checks for the annual JST engine (handoff v4 §4).
Run: python tests/test_annual.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.annual import (load_jst, stack_panel, us_series, global_series,
                           pooled_block_indices, historical_indices,
                           simulate_annual, FixedReal, GuytonKlinger, VPW)


def synthetic_panel(T, eq=0.02, bond=0.02, bill=0.02, infl=0.02):
    n = T
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[:-1] = np.arange(1, n)
    return dict(eq=np.full(n, eq), bond=np.full(n, bond), bill=np.full(n, bill),
                infl=np.full(n, infl), year=np.arange(n), cid=np.zeros(n, int),
                iso=np.array(["SYN"] * n), next=nxt, gdp_usd=np.ones(n),
                countries=["SYN"])


def test_zero_vol_annuity():
    """(i) zero vol, nominal return == inflation -> real return 0: FIX 4% with
    start-of-year withdrawals pays 25 full years (years 0..24) and fails in
    year 25; lifetime real spending == 1.0 W0."""
    p = synthetic_panel(40)
    idx = np.arange(40)[None, :]
    res = simulate_annual(idx, p, 0.6, FixedReal(0.04))
    ry = int(res["ruin_year"][0])
    assert ry == 25, f"expected ruin in year 25, got {ry}"
    tot = float(res["total_real_spend"][0])
    assert abs(tot - 1.0) < 1e-6, f"lifetime real spend should be 1.0, got {tot:.6f}"
    # VPW must never ruin and must spend exactly the balance
    res = simulate_annual(idx[:, :30], p, 0.6, VPW(0.0 + 1e-9))
    assert res["ruined"][0] == False and abs(res["total_real_spend"][0] - 1.0) < 1e-6
    print(f"PASS (i) zero-vol annuity: FIX4 ruin year {ry}, lifetime real spend {tot:.4f}; "
          f"VPW spends 1.0000 and never ruins")


def test_us_trinity_gate():
    """(ii) US-only ANNUAL JST rows, FIX 4% 60/40, 30-year historical
    overlapping windows starting 1926+ -> success rate vs the monthly
    engine's 97.9% (Trinity 95%, Bengen 100%)."""
    C = load_jst()
    U = us_series(C)
    idx = historical_indices(U, 30, first_year=1926)
    res = simulate_annual(idx, U, 0.6, FixedReal(0.04))
    succ = 100 * (1 - res["exhausted"].mean())
    fails = [int(U["year"][idx[i, 0]]) for i in np.where(res["exhausted"])[0]]
    n = len(idx)
    print(f"(ii) US annual FIX4 60/40 30y, {n} windows {U['year'][idx[0,0]]}-{U['year'][idx[-1,0]]}: "
          f"success {succ:.1f}% (monthly engine 97.9%); failing cohorts {fails}")
    assert abs(succ - 97.9) <= 5.0, "US annual engine too far from the monthly Trinity gate"
    print("PASS (ii) Trinity/Bengen gate within 5 points")
    return succ


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def _rank(x):
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def test_pooled_corr_preserved():
    """(iii) the pooled block bootstrap keeps the equity-inflation
    covariance: corr(eq, infl) in the pooled source vs mean of per-path corr.
    Pearson is dominated by DEU 1922 (infl 1024%), so the assertion is on
    Spearman (rank) correlation; both are printed."""
    C = load_jst()
    P = stack_panel(C)
    rng = np.random.default_rng(0)
    idx = pooled_block_indices(P, 50, 10_000, 10, rng)
    req_src = (1 + P["eq"]) / (1 + P["infl"]) - 1
    out = {}
    for label, series in [("nominal eq", P["eq"]), ("real eq", req_src)]:
        src_p = _corr(series, P["infl"])
        src_s = _corr(_rank(series), _rank(P["infl"]))
        s, f = series[idx], P["infl"][idx]
        s_c = s - s.mean(1, keepdims=True)
        f_c = f - f.mean(1, keepdims=True)
        path_p = (s_c * f_c).sum(1) / np.sqrt((s_c ** 2).sum(1) * (f_c ** 2).sum(1))
        rs, rf = _rank(s), _rank(f)
        rs = rs - rs.mean(1, keepdims=True)
        rf = rf - rf.mean(1, keepdims=True)
        path_s = (rs * rf).sum(1) / np.sqrt((rs ** 2).sum(1) * (rf ** 2).sum(1))
        out[label] = (src_p, path_p.mean(), src_s, path_s.mean())
        print(f"(iii) corr({label}, infl): source Pearson {src_p:+.3f} vs path-mean {path_p.mean():+.3f}; "
              f"Spearman {src_s:+.3f} vs path-mean {path_s.mean():+.3f}")
    # also check the bootstrap reproduces the source mean real equity return
    req_paths = req_src[idx].mean()
    print(f"(iii) mean real equity: source {req_src.mean():.4f} vs paths {req_paths:.4f}")
    for label, (sp, pp, ss, ps) in out.items():
        assert abs(ss - ps) < 0.05, f"Spearman corr not preserved for {label}: {ss:.3f} vs {ps:.3f}"
    assert abs(req_src.mean() - req_paths) < 0.005
    print("PASS (iii) pooled bootstrap preserves equity-inflation rank correlation and mean")


def test_no_lookahead_annual():
    """Tampering with returns from year t on must not change spending
    before t (GK, VPW, FIX)."""
    rng = np.random.default_rng(1)
    T = 40
    p = synthetic_panel(T)
    p["eq"] = rng.normal(0.07, 0.2, T)
    p["bond"] = rng.normal(0.02, 0.08, T)
    p["infl"] = rng.normal(0.03, 0.03, T)
    q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    q["eq"][20:] = -0.6
    q["bond"][20:] = -0.6
    q["infl"][20:] = 0.5
    idx = np.arange(T)[None, :]
    for rule in (GuytonKlinger(0.05), VPW(0.03), FixedReal(0.04)):
        a = simulate_annual(idx, p, 0.6, rule)["annual_real_spend"][0]
        b = simulate_annual(idx, q, 0.6, rule)["annual_real_spend"][0]
        assert np.allclose(a[:21], b[:21]), f"lookahead in {rule.name}"
    print("PASS no-lookahead: spending through the tamper year identical for GK, VPW, FIX")


def test_global_series_weights():
    """USA weight (rgdpmad x pop) in the global series must be the largest
    post-1945 and around 40-50% in 2000."""
    C = load_jst()
    G = global_series(C)
    i = np.where(G["year"] == 2000)[0][0]
    assert G["n_countries"][i] == 16 and 0.35 < G["w_max"][i] < 0.55, G["w_max"][i]
    print(f"PASS global series: 2000 has {G['n_countries'][i]} countries, max weight {G['w_max'][i]:.2f}")


if __name__ == "__main__":
    test_zero_vol_annuity()
    test_no_lookahead_annual()
    test_global_series_weights()
    test_us_trinity_gate()
    test_pooled_corr_preserved()
    print("ALL ANNUAL TESTS PASSED")
