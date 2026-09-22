"""Return-stacking synthetic portfolios (v4 §7).

Each stacked portfolio is expressed as ONE synthetic monthly nominal return
series that replaces `panel["stock"]`; the caller then runs the unmodified
engine with Alloc(1.0), so the engine rebalances nothing and every spending
rule sees the stacked fund exactly as it would see a single ETF.

  RSST-like ("100% stock + 100% trend"):
      r = stock_r + haircut * trend_excess - spread/12 - FEE_RSST/12
      trend_excess is already an excess return over cash (AQR TSMOM /
      in-house proxy), so the futures financing at T-bill is implicit;
      `spread` is the extra financing cost above T-bill (per year).
  RSSB-like ("100% stock + 100% bond"):
      r = stock_r + (bond_r - bill_r) - spread/12 - FEE_RSSB/12
  Five-five ("50% RSSB + 50% RSST", monthly rebalanced):
      r = 0.5 * r_rssb + 0.5 * r_rsst
      => exposures 100% stock + 50% bond + 50% trend, fee 0.5*(0.39+0.99)%.

Fees (verified 2026-09-22 on stockanalysis.com; fund pages: RSSB 0.39%,
RSST 0.99%; handoff quoted "about 0.40% / 1.0%").

Everything here is contemporaneous by construction: the overlay return in
month t is an asset return of month t, like any other asset return. The
only decision embedded in the series is the trend SIGNAL, which
data/build_trend.py computes from information through t-1
(tests/test_stack.py checks this).

Idealised upper bound — NOT modelled: tracking error of the real ETFs,
daily (vs monthly here) rebalancing of the leverage, margin/liquidation
risk in a crash, bid-ask and roll costs of the futures, the fact that AQR
TSMOM is a gross hypothetical factor (~12%/yr excess since 1985, far above
what any investable trend fund has delivered net).
"""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "processed")

FEE_RSSB = 0.0039
FEE_RSST = 0.0099
FEE_VT = 0.0007      # control: VT-like global stock ETF
FEE_BND = 0.0003     # control: BND-like bond ETF


def load_trend(col="trend_excess"):
    """dict date -> monthly excess return of the trend overlay.
    col: 'trend_excess' (spliced: proxy pre-1985, AQR after)
         'proxy_scaled' (in-house proxy over the whole period, vol-matched)
         'proxy_raw'    (in-house proxy, unscaled)"""
    out = {}
    with open(os.path.join(PROC, "trend_monthly.csv")) as f:
        for row in csv.DictReader(f):
            out[row["date"]] = float(row[col])
    return out


def align_trend(panel, trend):
    """Trend excess returns aligned to panel['dates']; raises if any month
    is missing (the trend file must cover the sample)."""
    missing = [d for d in panel["dates"] if d not in trend]
    if missing:
        raise KeyError(f"trend series missing {len(missing)} months, e.g. {missing[:3]}")
    return np.array([trend[d] for d in panel["dates"]])


def stacked_returns(kind, stock, bond, bill, trend, spread=0.0, haircut=1.0,
                    fee_rssb=FEE_RSSB, fee_rsst=FEE_RSST):
    """Monthly nominal return of the synthetic stacked fund.
    kind in {'RSST', 'RSSB', 'FIVEFIVE'}; spread and fees are per year."""
    if kind == "RSST":
        return stock + haircut * trend - spread / 12.0 - fee_rsst / 12.0
    if kind == "RSSB":
        return stock + (bond - bill) - spread / 12.0 - fee_rssb / 12.0
    if kind == "FIVEFIVE":
        a = stacked_returns("RSSB", stock, bond, bill, trend, spread, haircut, fee_rssb, fee_rsst)
        b = stacked_returns("RSST", stock, bond, bill, trend, spread, haircut, fee_rssb, fee_rsst)
        return 0.5 * a + 0.5 * b
    raise ValueError(kind)


def stacked_panel(panel, trend, kind, spread=0.0, haircut=1.0,
                  fee_rssb=FEE_RSSB, fee_rsst=FEE_RSST):
    """Copy of `panel` whose 'stock' column is the stacked fund's return.
    Run it with Alloc(1.0). `trend` is a date->excess-return dict or an
    array already aligned to panel['dates']."""
    tr = trend if isinstance(trend, np.ndarray) else align_trend(panel, trend)
    out = dict(panel)
    out["stock"] = stacked_returns(kind, panel["stock"], panel["bond"], panel["bill"],
                                   tr, spread, haircut, fee_rssb, fee_rsst)
    out["stack_kind"] = kind
    return out


def control_panel(panel, fee_stock=FEE_VT, fee_bond=FEE_BND):
    """Controls (100/0, 80/20, 60/40 via Alloc) with ETF-level fees embedded
    so the comparison with the stacked funds is net-of-fee on both sides."""
    out = dict(panel)
    out["stock"] = panel["stock"] - fee_stock / 12.0
    out["bond"] = panel["bond"] - fee_bond / 12.0
    return out


# ------------------------------------------------ historical regime maths --

def window_real_path(panel, first, last, stock_w=1.0, rebalance_months=12):
    """Real wealth index (start = 1.0) of a stock/bond mix over the
    historical months first <= date <= last, rebalanced every
    `rebalance_months` months (12 = the engine's annual rebalance). With
    stock_w = 1.0 this is just the (possibly synthetic) stock series."""
    d = np.array(panel["dates"])
    m = (d >= first) & (d <= last)
    s, b, infl = panel["stock"][m], panel["bond"][m], panel["infl"][m]
    n = len(s)
    ws, wb = stock_w, 1.0 - stock_w
    nom = np.empty(n)
    for t in range(n):
        if t % rebalance_months == 0:
            tot = ws + wb
            ws, wb = tot * stock_w, tot * (1.0 - stock_w)
        ws *= 1.0 + s[t]
        wb *= 1.0 + b[t]
        nom[t] = ws + wb
    price = np.cumprod(1.0 + infl)
    return nom / price


def max_drawdown(path):
    """Max drawdown of an index that starts at 1.0 (peak includes 1.0)."""
    p = np.concatenate([[1.0], path])
    peak = np.maximum.accumulate(p)
    return float((p / peak - 1.0).min())


# ---------------------------------------------- spending-smoothness metrics -

def spending_smoothness(annual_real_spend):
    """Per-path spending-path metrics, medians across paths:
      spend_vol    std of year-over-year % change in real spending
      spend_dd_max max drawdown of real spending from its running peak
      spend_dd_yrs years with spending below 80% of its running peak
    Years after ruin (spend = 0) are excluded from the YoY std (a 100% cut
    would dominate the number and ruin is already scored separately)."""
    s = np.asarray(annual_real_spend, float)
    P, Y = s.shape
    prev, cur = s[:, :-1], s[:, 1:]
    ok = (prev > 0) & (cur > 0)
    chg = np.where(ok, cur / np.maximum(prev, 1e-15) - 1.0, np.nan)
    with np.errstate(invalid="ignore"):
        vol = np.nanstd(chg, axis=1)
    peak = np.maximum.accumulate(s, axis=1)
    rel = s / np.maximum(peak, 1e-15)
    dd_max = (rel - 1.0).min(axis=1)
    yrs = (rel < 0.80).sum(axis=1)
    return dict(spend_vol=float(np.nanmedian(vol)),
                spend_dd_max=float(np.median(dd_max)),
                spend_dd_yrs=float(np.median(yrs)))
