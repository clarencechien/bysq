"""Build data/processed/trend_monthly.csv — the trend-following overlay for
the v4 §7 return-stacking experiment.

  1985-01 .. 2026-05 : AQR "Time Series Momentum Factors, Monthly" —
                       TSMOM (all assets), monthly EXCESS return over cash,
                       used as is (gross of fees/costs; hypothetical).
  1934-02 .. 1984-12 : in-house 2-asset time-series-momentum proxy built
                       from our own S&P composite and 10y bond series
                       (data/processed/us_monthly.csv), scaled so that its
                       volatility over the 1985-01..2026-05 overlap matches
                       AQR TSMOM. Overlap correlation / mean-return gap are
                       printed and recorded in data/PROVENANCE.md.
  2026-06 .. 2026-07 : AQR file ends 2026-05; the two tail months use the
                       scaled proxy (flagged in column `src`).

Proxy definition (all signals use information through t-1 only):
  for a in {stock, bond}:
    ex_a[t]   = r_a[t] - bill[t]                       (monthly excess return)
    sig_a[t]  = sign( prod_{s=t-12..t-1}(1+r_a[s]) / prod(1+bill[s]) - 1 )
    vol_a[t]  = std(ex_a[t-36..t-1]) * sqrt(12)        (>=12 obs, expanding
                                                        up to 36 during warm-up)
    pos_a[t]  = sig_a[t] * TARGET_VOL / vol_a[t], clipped to +-POS_CAP
    ret_a[t]  = pos_a[t] * ex_a[t]
  proxy[t]    = mean_a ret_a[t]
  Months with < 12 months of history (1934-02..1935-01) have pos = 0.

SG Trend Index (spec §7.1) is NOT available from any free programmatic
source and is not used. The AQR "Century of Factor Premia" file contains
only cross-sectional factors (no time-series momentum) and is not used.

python data/build_trend.py  ->  data/processed/trend_monthly.csv
"""
import csv
import datetime
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")
AQR_FILE = os.path.join(RAW, "Time-Series-Momentum-Factors-Monthly.xlsx")
OUT = os.path.join(PROC, "trend_monthly.csv")

TARGET_VOL = 0.10     # per-asset vol target (annual)
LB_SIGNAL = 12        # months
LB_VOL = 36           # months
MIN_VOL_OBS = 12
POS_CAP = 5.0         # |position| cap per asset (notional / NAV)
SPLICE = "1985-01"


def load_us():
    path = os.path.join(PROC, "us_monthly.csv")
    dates, stock, bond, bill = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
            stock.append(float(row["stock_nom"]))
            bond.append(float(row["bond_nom"]))
            bill.append(float(row["bill_nom"]) if row["bill_nom"] else np.nan)
    return dates, np.array(stock), np.array(bond), np.array(bill)


def tsmom_proxy(stock, bond, bill, target_vol=TARGET_VOL, lb_signal=LB_SIGNAL,
                lb_vol=LB_VOL, min_vol_obs=MIN_VOL_OBS, pos_cap=POS_CAP):
    """Unscaled 2-asset TSMOM proxy. Inputs are aligned monthly arrays with
    NO NaNs (caller slices to the bill-available period). Returns (proxy,
    positions) where positions is (T, 2) and positions[t] depends only on
    data through t-1."""
    T = len(stock)
    assets = [stock, bond]
    pos = np.zeros((T, 2))
    for a, r in enumerate(assets):
        ex = r - bill
        for t in range(T):
            if t < lb_signal or t < min_vol_obs:
                continue                       # warm-up: flat
            w_sig = slice(t - lb_signal, t)
            cum_ex = np.prod(1 + r[w_sig]) / np.prod(1 + bill[w_sig]) - 1.0
            sig = 1.0 if cum_ex > 0 else (-1.0 if cum_ex < 0 else 0.0)
            w_vol = slice(max(0, t - lb_vol), t)
            vol = ex[w_vol].std() * np.sqrt(12)
            if vol <= 0:
                continue
            pos[t, a] = np.clip(sig * target_vol / vol, -pos_cap, pos_cap)
    rets = np.stack([pos[:, a] * (assets[a] - bill) for a in range(2)], axis=1)
    return rets.mean(axis=1), pos


def load_aqr():
    import openpyxl
    wb = openpyxl.load_workbook(AQR_FILE, read_only=True, data_only=True)
    ws = wb["TSMOM Factors"]
    rows = list(ws.iter_rows(values_only=True))
    assert rows[17][1] == "TSMOM", rows[17]
    out = {}
    for r in rows[18:]:
        if isinstance(r[0], datetime.datetime) and isinstance(r[1], (int, float)):
            out[r[0].strftime("%Y-%m")] = float(r[1])
    return out


def build(verbose=True):
    dates, stock, bond, bill = load_us()
    ok = ~np.isnan(bill)
    i0 = int(np.argmax(ok))
    assert ok[i0:].all(), "bill series has interior gaps"
    d = dates[i0:]
    proxy_raw, pos = tsmom_proxy(stock[i0:], bond[i0:], bill[i0:])
    aqr = load_aqr()
    aqr_arr = np.array([aqr.get(x, np.nan) for x in d])

    ov = ~np.isnan(aqr_arr)
    k = aqr_arr[ov].std() / proxy_raw[ov].std()
    proxy_scaled = proxy_raw * k
    corr = np.corrcoef(proxy_scaled[ov], aqr_arr[ov])[0, 1]
    mean_gap = 12 * (aqr_arr[ov].mean() - proxy_scaled[ov].mean())
    stats = dict(
        overlap_start=d[int(np.argmax(ov))],
        overlap_end=d[len(d) - 1 - int(np.argmax(ov[::-1]))],
        overlap_months=int(ov.sum()),
        scale_k=float(k),
        corr=float(corr),
        aqr_mean_ann=float(12 * aqr_arr[ov].mean()),
        aqr_vol_ann=float(np.sqrt(12) * aqr_arr[ov].std()),
        proxy_scaled_mean_ann=float(12 * proxy_scaled[ov].mean()),
        proxy_raw_mean_ann=float(12 * proxy_raw[ov].mean()),
        proxy_raw_vol_ann=float(np.sqrt(12) * proxy_raw[ov].std()),
        mean_gap_ann=float(mean_gap),
        proxy_pre1985_mean_ann=float(12 * proxy_scaled[np.array(d) < SPLICE].mean()),
        proxy_pre1985_vol_ann=float(np.sqrt(12) * proxy_scaled[np.array(d) < SPLICE].std()),
        pos_abs_max=float(np.abs(pos).max()),
        pos_abs_p95=float(np.percentile(np.abs(pos[LB_VOL:]), 95)),
        pos_bond_abs_med=float(np.median(np.abs(pos[LB_VOL:, 1]))),
        pos_stock_abs_med=float(np.median(np.abs(pos[LB_VOL:, 0]))),
    )
    # overlap correlation of stock-only / bond-only legs, for the record
    for a, nm in enumerate(("stock", "bond")):
        leg = pos[:, a] * ((stock[i0:] if a == 0 else bond[i0:]) - bill[i0:])
        stats[f"corr_{nm}_leg"] = float(np.corrcoef(leg[ov], aqr_arr[ov])[0, 1])

    spliced = np.where(ov, aqr_arr, proxy_scaled)
    src = np.where(ov, "aqr", np.where(np.array(d) < SPLICE, "proxy", "proxy_tail"))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "trend_excess", "src", "aqr_tsmom", "proxy_scaled",
                    "proxy_raw", "pos_stock", "pos_bond"])
        for i, x in enumerate(d):
            w.writerow([x, f"{spliced[i]:.8f}", src[i],
                        "" if np.isnan(aqr_arr[i]) else f"{aqr_arr[i]:.8f}",
                        f"{proxy_scaled[i]:.8f}", f"{proxy_raw[i]:.8f}",
                        f"{pos[i, 0]:.4f}", f"{pos[i, 1]:.4f}"])
    if verbose:
        print(f"wrote {OUT}: {len(d)} rows {d[0]}..{d[-1]}")
        for kk, v in stats.items():
            print(f"  {kk:>24s}: {v:.4f}" if isinstance(v, float) else f"  {kk:>24s}: {v}")
        if stats["corr"] < 0.4:
            print("  WARNING: overlap correlation < 0.4 — pre-1985 stitch is weak; "
                  "discount 1934-1984 stacking results.")
    return stats


if __name__ == "__main__":
    build()
