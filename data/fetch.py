"""Data layer: download raw series and build processed monthly panels.

Sources (all real, official or academic; no generated data):
  - Shiller monthly dataset (S&P composite price, dividends, US CPI, GS10 yield), 1871-01+
    http://www.econ.yale.edu/~shiller/data/ie_data.xls
  - FRED TB3MS: 3-month T-bill secondary market rate, 1934-01+
  - FRED EXTAUS: New Taiwan Dollars per USD, monthly average, 1983-10+
  - DGBAS (行政院主計總處) 消費者物價基本分類指數 總指數, monthly, 1971-01+
    nstatdb.dgbas.gov.tw funid=A030101015 (base year 2021=100)

Run: python data/fetch.py            # uses cached raw files if present
     python data/fetch.py --refresh  # force re-download
"""
import json
import os
import subprocess
import sys
import math
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
PROC = os.path.join(HERE, "processed")
CA = os.environ.get("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

SOURCES = {
    "shiller.xls": "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "tb3ms.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS",
    "extaus.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXTAUS",
    "dgbas_cpi.json": ("https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx?"
                       "sys=220&ymf=6001&ymt=11512&kind=21&type=1&funid=A030101015"
                       "&cycle=1&outmode=8&compmode=0&outkind=1&fldspc=0,1,&codspc0=0,1,&rdm=R123456"),
}


def download(refresh=False):
    os.makedirs(RAW, exist_ok=True)
    for fname, url in SOURCES.items():
        path = os.path.join(RAW, fname)
        if os.path.exists(path) and not refresh:
            print(f"cached  {fname}")
            continue
        cmd = ["curl", "-sSL", "--max-time", "120", "-o", path, url]
        if os.path.exists(CA):
            cmd = ["curl", "--cacert", CA, "-sSL", "--max-time", "120", "-o", path, url]
        subprocess.run(cmd, check=True)
        print(f"fetched {fname}")


def ym_add(ym, k):
    y, m = ym
    m += k
    return (y + (m - 1) // 12, (m - 1) % 12 + 1)


def parse_shiller():
    import xlrd
    wb = xlrd.open_workbook(os.path.join(RAW, "shiller.xls"))
    sh = wb.sheet_by_name("Data")
    rows = {}
    for r in range(8, sh.nrows):
        d = sh.cell_value(r, 0)
        if not isinstance(d, float):
            continue
        y = int(d)
        m = int(round((d - y) * 100))
        if not (1 <= m <= 12):
            continue
        p = sh.cell_value(r, 1)
        div = sh.cell_value(r, 2)   # annualized dividend rate
        cpi = sh.cell_value(r, 4)
        gs10 = sh.cell_value(r, 6)  # percent
        rows[(y, m)] = dict(
            p=float(p) if p != "" else None,
            d=float(div) if div != "" else None,
            cpi=float(cpi) if cpi != "" else None,
            gs10=float(gs10) if gs10 != "" else None,
        )
    return rows


def parse_fred(fname):
    out = {}
    with open(os.path.join(RAW, fname)) as f:
        for row in csv.reader(f):
            if not row or row[0] == "observation_date":
                continue
            try:
                y, m, _ = row[0].split("-")
                out[(int(y), int(m))] = float(row[1])
            except ValueError:
                continue
    return out


def parse_dgbas():
    with open(os.path.join(RAW, "dgbas_cpi.json"), encoding="utf-8") as f:
        j = json.load(f)
    vals = j["orgdata"][0] if isinstance(j["orgdata"][0], list) else j["orgdata"]
    # The DB clips the requested range to actual availability and reports it
    # back: ymf='7001' means ROC year 70 month 01 = 1981-01.
    ymf = str(j["ymf"])
    start = (int(ymf[:-2]) + 1911, int(ymf[-2:]))
    out = {}
    ym = start
    for v in vals:
        if v is not None and v != "":
            out[ym] = float(v)
        ym = ym_add(ym, 1)
    return out


def bond_tr_monthly(y_prev_pct, y_now_pct, maturity=10):
    """Total return of a constant-maturity par bond over one month.

    Standard replication approach: at t-1 buy a par bond with annual coupon
    equal to the prevailing GS10 yield; one month later reprice at the new
    yield (maturity held at 10y — the error from 1 month of aging is
    negligible) and collect one month of coupon accrual.
    """
    c = y_prev_pct / 100.0
    y = y_now_pct / 100.0
    if y <= 0:
        price = 1.0 + c * maturity - 0  # degenerate, never hit in sample
    else:
        ann = (1 - (1 + y) ** -maturity) / y
        price = c * ann + (1 + y) ** -maturity
    return price - 1.0 + c / 12.0


def build():
    os.makedirs(PROC, exist_ok=True)
    shiller = parse_shiller()
    tbill = parse_fred("tb3ms.csv")
    fx = parse_fred("extaus.csv")
    twcpi = parse_dgbas()

    # --- US panel: monthly nominal USD returns + US CPI level ---
    months = sorted(shiller.keys())
    us_rows = []
    for i in range(1, len(months)):
        ym, prev = months[i], months[i - 1]
        s0, s1 = shiller[prev], shiller[ym]
        if None in (s0["p"], s1["p"], s0["gs10"], s1["gs10"], s1["cpi"]):
            continue
        d = s1["d"] if s1["d"] is not None else s0["d"]
        if d is None:
            continue
        stock = (s1["p"] + d / 12.0) / s0["p"] - 1.0
        bond = bond_tr_monthly(s0["gs10"], s1["gs10"])
        bill_rate = tbill.get(prev)  # decision uses previous month's rate
        bill = (1 + bill_rate / 100.0) ** (1 / 12.0) - 1.0 if bill_rate is not None else None
        us_rows.append((ym, stock, bond, bill, s1["cpi"]))

    with open(os.path.join(PROC, "us_monthly.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "stock_nom", "bond_nom", "bill_nom", "us_cpi"])
        for (y, m), st, bd, bl, cpi in us_rows:
            w.writerow([f"{y}-{m:02d}", f"{st:.8f}", f"{bd:.8f}",
                        "" if bl is None else f"{bl:.8f}", f"{cpi:.5f}"])
    print(f"us_monthly.csv: {len(us_rows)} rows "
          f"{us_rows[0][0]} .. {us_rows[-1][0]}")

    # --- Taiwan panel: same rows converted to TWD, deflated by Taiwan CPI ---
    tw_rows = []
    for i, ((y, m), st, bd, bl, _) in enumerate(us_rows):
        prev = months[months.index((y, m)) - 1] if (y, m) in months else None
        if (y, m) not in fx or prev not in fx or (y, m) not in twcpi:
            continue
        if bl is None:
            continue
        fx_ret = fx[(y, m)] / fx[prev] - 1.0  # TWD per USD: + = TWD depreciates
        tw_rows.append(((y, m),
                        (1 + st) * (1 + fx_ret) - 1,
                        (1 + bd) * (1 + fx_ret) - 1,
                        (1 + bl) * (1 + fx_ret) - 1,
                        twcpi[(y, m)],
                        fx[(y, m)]))

    with open(os.path.join(PROC, "tw_monthly.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "stock_twd_nom", "bond_twd_nom", "usdbill_twd_nom",
                    "tw_cpi", "usdtwd"])
        for (y, m), st, bd, bl, cpi, r in tw_rows:
            w.writerow([f"{y}-{m:02d}", f"{st:.8f}", f"{bd:.8f}", f"{bl:.8f}",
                        f"{cpi:.4f}", f"{r:.4f}"])
    print(f"tw_monthly.csv: {len(tw_rows)} rows "
          f"{tw_rows[0][0]} .. {tw_rows[-1][0]}")

    # provenance sanity: overlap check US CPI vs TW CPI inflation correlation
    return us_rows, tw_rows


if __name__ == "__main__":
    download(refresh="--refresh" in sys.argv)
    build()
