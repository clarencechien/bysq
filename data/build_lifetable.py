"""Parse 內政部 簡易生命表 (abridged life table, ODS) into a single-age qx
table 0..110 for 全體/男性/女性, extrapolating the open 85+ group with a
Gompertz fit on ages 70-84 (standard actuarial closure; flagged in
PROVENANCE). Output: data/processed/tw_lifetable_<roc_year>.csv
Run: python data/build_lifetable.py"""
import csv, os, sys, zipfile
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}'
X = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'


def read_rows(path):
    root = ET.fromstring(zipfile.ZipFile(path).read("content.xml"))
    rows = []
    for tbl in root.iter(T + 'table'):
        for row in tbl.iter(T + 'table-row'):
            if int(row.get(T + 'number-rows-repeated', '1')) > 50:
                continue
            cells = []
            for c in row.iter(T + 'table-cell'):
                crep = int(c.get(T + 'number-columns-repeated', '1'))
                txt = "".join(t.text or "" for t in c.iter(X + 'p'))
                cells.extend([txt] * min(crep, 8))
            rows.append(cells)
    return rows


def parse(path):
    rows = read_rows(path)
    out = {}
    cur = None
    for r in rows:
        if not r:
            continue
        if r[0] in ("全體", "男性", "女性"):
            cur = {"全體": "all", "男性": "m", "女性": "f"}[r[0]]
            out.setdefault(cur, {})
            continue
        if cur is None:
            continue
        a = r[0].strip()
        if a.endswith("M"):
            continue
        if a.endswith("+"):
            a = a[:-1]
        if not a.isdigit():
            continue
        try:
            qx, lx, ex = float(r[1]), float(r[2]), float(r[6])
        except (ValueError, IndexError):
            continue
        out[cur][int(a)] = (qx, lx, ex)
    return out


def close_table(tab, max_age=110):
    """Gompertz closure of the open 85+ group: qx(85+k) = q84 * exp(b*(k+1)),
    with the slope b solved so that the closed table reproduces the PUBLISHED
    remaining life expectancy at 85 (e85 of the 85+ row). Ages 0-84 are the
    published single-age qx verbatim."""
    q84 = tab[84][0]
    e85_pub = tab[85][2]

    def e85(b):
        q = np.minimum(q84 * np.exp(b * np.arange(1, max_age - 85 + 1)), 1.0)
        q[-1] = 1.0
        surv = np.cumprod(1 - q)
        return surv.sum() + 0.5

    lo, hi = 0.01, 0.30
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if e85(mid) > e85_pub:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    qx = np.zeros(max_age + 1)
    for age in range(85):
        qx[age] = tab[age][0]
    for k, age in enumerate(range(85, max_age + 1)):
        qx[age] = min(q84 * np.exp(b * (k + 1)), 1.0)
    qx[max_age] = 1.0
    return qx


def build(path, roc_year):
    tabs = parse(path)
    q = {k: close_table(v) for k, v in tabs.items()}
    outp = os.path.join(HERE, "processed", f"tw_lifetable_{roc_year}.csv")
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "qx_all", "qx_m", "qx_f"])
        for age in range(111):
            w.writerow([age, f"{q['all'][age]:.6f}", f"{q['m'][age]:.6f}", f"{q['f'][age]:.6f}"])
    # verification: recompute e_x from closed table vs published
    for k in ("all", "m", "f"):
        for age in (0, 40, 50, 60, 65):
            surv = np.cumprod(1 - q[k][age:])
            e = surv.sum() + 0.5   # sum_{t>=1} l_{x+t}/l_x + 0.5
            print(f"{roc_year} {k} e{age}: closed-table {e:.2f} vs published {tabs[k][age][2]:.2f}")
    print("wrote", outp)


if __name__ == "__main__":
    build(os.path.join(HERE, "raw", "moi_lifetable_114.ods"), 114)
    build(os.path.join(HERE, "raw", "moi_lifetable_113.ods"), 113)
