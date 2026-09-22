"""Annual-frequency engine for the JST (Jordà-Schularick-Taylor Macrohistory
R6) 16-country panel — handoff v4 §4 ("H2 rescue").

Deliberately SEPARATE from engine/core.py (the monthly engine): same design
constraints (one shared time axis per sampled country-year; withdrawal at
the START of the year; every decision reads information through year t-1
only), but annual steps and a pooled-across-countries block bootstrap.

Views
-----
Everything is the "domestic investor" view (Anarkulova/Cederburg/O'Doherty
2022): local-currency nominal returns deflated by local CPI of the SAME
country-year. No USD view is built.

Samples
-------
pooled   : stationary block bootstrap POOLED across countries. Each block
           picks a uniformly random usable country-year as start (so each
           country is weighted by its number of usable years), then runs
           forward inside that country's contiguous segment; block length
           is geometric(mean_block). If the block runs off the end of a
           contiguous segment a fresh start is re-drawn (no 2020->1870 join,
           no cross-country join inside a block).
global   : one real-GDP-weighted "world of domestic investors" series.
           Weights = rgdpmad x pop (Maddison real GDP per capita in 1990
           international $ times population, both from JST R6), renormalised
           over the countries usable in that year. The JST `gdp` column is
           nominal local currency in country-specific UNITS (USA in billions,
           BEL in millions of francs, ...), so gdp / xrusd is NOT comparable
           across countries (USA would get a 0.4% weight in 1922) and is not
           used. Then the same stationary bootstrap the monthly engine uses
           (engine.core.stationary_bootstrap_indices, modular wrap).
us       : the USA rows of JST, same single-series bootstrap (control that
           separates 'annual vs monthly' from 'US survivorship').

Usable country-year = eq_tr, bond_tr, bill_rate, cpi(t) and cpi(t-1) all
present (inflation computable). Hyperinflation years that survive that
filter (DEU 1922/1924, JPN 1945, ...) are KEPT; run_v4_jst.py reports the
max inflation / equity return of each sample so the reader can judge.
"""
import csv
import os
import numpy as np

from engine.core import (FixedReal, GuytonKlinger, VPW,
                         stationary_bootstrap_indices,
                         spend_fail_abs, ce_spending)

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(os.path.dirname(HERE), "data", "processed")
JST_CSV = os.path.join(PROC, "jst_annual.csv")
JST_RAW = os.path.join(os.path.dirname(HERE), "data", "raw", "JSTdatasetR6.xlsx")
JST_W = os.path.join(PROC, "jst_rgdp_weights.csv")


def load_jst_weights(path=JST_W, raw=JST_RAW):
    """{(iso, year): rgdpmad * pop} — real GDP in 1990 international $
    (thousands), the only size measure in JST with common units across
    countries. Built from the raw xlsx once and cached in data/processed."""
    if not os.path.exists(path):
        import openpyxl
        wb = openpyxl.load_workbook(raw, read_only=True)
        ws = wb[wb.sheetnames[0]]
        it = ws.iter_rows(values_only=True)
        hdr = list(next(it))
        iy, ii, ir, ip = (hdr.index(c) for c in ("year", "iso", "rgdpmad", "pop"))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "iso", "rgdpmad", "pop", "rgdp"])
            for r in it:
                g = (r[ir] * r[ip]) if (r[ir] is not None and r[ip] is not None) else ""
                w.writerow([r[iy], r[ii], "" if r[ir] is None else r[ir],
                            "" if r[ip] is None else r[ip], g])
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["iso"], int(r["year"]))] = _f(r["rgdp"])
    return out


# ---------------------------------------------------------------- data ------

def load_jst(path=JST_CSV):
    """dict iso -> dict of aligned arrays over the USABLE years of that
    country: year, eq, bond, bill (nominal, local, decimal), infl
    (cpi_t/cpi_{t-1}-1), gdp_usd (nominal GDP / xrusd; NaN if gdp missing),
    crisis. Countries with no usable year are dropped (CAN, IRL in R6)."""
    raw = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            raw.setdefault(r["iso"], []).append(r)
    wts = load_jst_weights()
    out = {}
    for iso, rows in raw.items():
        rows.sort(key=lambda r: int(r["year"]))
        cpi_by_year = {int(r["year"]): _f(r["cpi"]) for r in rows}
        rec = []
        for r in rows:
            y = int(r["year"])
            eq, bd, bl, cpi = (_f(r["eq_tr"]), _f(r["bond_tr"]),
                               _f(r["bill_rate"]), _f(r["cpi"]))
            cpi_prev = cpi_by_year.get(y - 1, np.nan)
            if any(np.isnan(v) for v in (eq, bd, bl, cpi, cpi_prev)):
                continue
            gdp, xr = _f(r["gdp"]), _f(r["xrusd"])
            gdp_usd = gdp / xr if (not np.isnan(gdp) and not np.isnan(xr) and xr > 0) else np.nan
            rec.append((y, eq, bd, bl, cpi / cpi_prev - 1.0, gdp_usd,
                        _f(r["crisisJST"]), wts.get((iso, y), np.nan)))
        if not rec:
            continue
        a = np.array(rec, dtype=float)
        out[iso] = dict(year=a[:, 0].astype(int), eq=a[:, 1], bond=a[:, 2],
                        bill=a[:, 3], infl=a[:, 4], gdp_usd=a[:, 5],
                        crisis=a[:, 6], rgdp=a[:, 7])
    return out


def _f(v):
    try:
        return float(v) if v not in ("", None) else np.nan
    except ValueError:
        return np.nan


def stack_panel(countries, include=None, exclude=()):
    """Stack per-country arrays into one row-indexed panel.
    `next[i]` = i+1 if row i+1 is the same country and the following
    calendar year (contiguous), else -1: the block bootstrap walks it."""
    isos = [c for c in sorted(countries) if (include is None or c in include)
            and c not in exclude]
    cols = {k: [] for k in ("eq", "bond", "bill", "infl", "year", "gdp_usd", "rgdp")}
    cid, iso_arr = [], []
    for i, iso in enumerate(isos):
        c = countries[iso]
        for k in cols:
            cols[k].append(c[k])
        cid.append(np.full(len(c["year"]), i))
        iso_arr += [iso] * len(c["year"])
    panel = {k: np.concatenate(v) for k, v in cols.items()}
    panel["cid"] = np.concatenate(cid)
    panel["iso"] = np.array(iso_arr)
    n = len(panel["eq"])
    nxt = np.full(n, -1, dtype=np.int64)
    same = (panel["cid"][1:] == panel["cid"][:-1]) & \
           (panel["year"][1:] == panel["year"][:-1] + 1)
    nxt[:-1][same] = np.arange(1, n)[same]
    panel["next"] = nxt
    panel["countries"] = isos
    return panel


def filter_countries(countries, keep):
    """Drop country-years where keep(country_dict, i) is False (e.g. the
    hyperinflation sensitivity: lambda c, i: c['infl'][i] <= 0.5). Contiguous
    segments are recomputed by stack_panel, so dropped years split blocks."""
    out = {}
    for iso, c in countries.items():
        m = np.array([bool(keep(c, i)) for i in range(len(c["year"]))])
        if m.any():
            out[iso] = {k: v[m] for k, v in c.items()}
    return out


def global_series(countries, weights="rgdp"):
    """Real-GDP-weighted 'world of domestic investors' series, one row per
    year (weights='rgdp' = rgdpmad x pop; 'equal' -> 1/n; 'gdp_usd' = the
    broken gdp/xrusd, kept only for the record).

    For each calendar year the LOCAL-CURRENCY REAL returns of every usable
    country are averaged with weights w_c / sum(w) over the countries usable
    in that year. Weighted inflation is averaged the same way, and nominal
    returns are rebuilt as (1+real)(1+infl)-1 so that simulate_annual, which
    deflates by `infl`, recovers the weighted real return exactly.
    This is NOT a USD index and NOT a cap-weighted index (JST has no market
    caps). A missing weight is filled from the country's nearest available
    year (documented caveat)."""
    years = sorted(set(int(y) for c in countries.values() for y in c["year"]))
    rows = []
    # fill missing gdp within country by nearest available year
    gfill = {}
    for iso, c in countries.items():
        g = (c["rgdp"] if weights == "rgdp" else c["gdp_usd"]).copy()
        if weights == "equal":
            g = np.ones_like(g)
        if np.isnan(g).any():
            ok = ~np.isnan(g)
            g = np.interp(c["year"], c["year"][ok], g[ok])
        gfill[iso] = dict(zip(c["year"], g))
    lookup = {iso: {int(y): i for i, y in enumerate(c["year"])}
              for iso, c in countries.items()}
    for y in years:
        req, rb, rbl, inf, w = [], [], [], [], []
        for iso, c in countries.items():
            i = lookup[iso].get(y)
            if i is None:
                continue
            d = 1.0 + c["infl"][i]
            req.append((1 + c["eq"][i]) / d - 1)
            rb.append((1 + c["bond"][i]) / d - 1)
            rbl.append((1 + c["bill"][i]) / d - 1)
            inf.append(c["infl"][i])
            w.append(gfill[iso][y])
        w = np.array(w, float)
        w = w / w.sum()
        rows.append((y, np.dot(w, req), np.dot(w, rb), np.dot(w, rbl),
                     np.dot(w, inf), len(w), w.max()))
    a = np.array(rows, float)
    infl = a[:, 4]
    panel = dict(year=a[:, 0].astype(int),
                 eq=(1 + a[:, 1]) * (1 + infl) - 1,
                 bond=(1 + a[:, 2]) * (1 + infl) - 1,
                 bill=(1 + a[:, 3]) * (1 + infl) - 1,
                 infl=infl, gdp_usd=np.full(len(a), np.nan), rgdp=np.full(len(a), np.nan),
                 cid=np.zeros(len(a), int), iso=np.array(["GLOBAL"] * len(a)),
                 n_countries=a[:, 5].astype(int), w_max=a[:, 6],
                 countries=["GLOBAL"])
    n = len(a)
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[:-1] = np.arange(1, n)
    panel["next"] = nxt
    return panel


def us_series(countries):
    return stack_panel(countries, include=["USA"])


# ----------------------------------------------------------- sampling ------

def pooled_block_indices(panel, t_out, n_paths, mean_block, rng):
    """Stationary block bootstrap pooled across countries (see module doc).
    panel: from stack_panel(). Returns (n_paths, t_out) row indices."""
    n = len(panel["eq"])
    nxt = panel["next"]
    p = 1.0 / mean_block
    jump = rng.random((n_paths, t_out)) < p
    starts = rng.integers(0, n, (n_paths, t_out))
    idx = np.empty((n_paths, t_out), dtype=np.int64)
    cur = starts[:, 0].copy()
    idx[:, 0] = cur
    for t in range(1, t_out):
        cont = nxt[cur]
        cur = np.where(jump[:, t] | (cont < 0), starts[:, t], cont)
        idx[:, t] = cur
    return idx


def single_series_indices(panel, t_out, n_paths, mean_block, rng):
    """Same stationary bootstrap as the monthly engine (modular wrap) over a
    single contiguous series (global or US)."""
    return stationary_bootstrap_indices(len(panel["eq"]), t_out, n_paths,
                                        mean_block, rng)


def historical_indices(panel, t_out, first_year=None):
    """All overlapping windows of t_out consecutive years that stay inside
    one contiguous country segment (annual starts)."""
    n = len(panel["eq"])
    rows = []
    for s in range(n):
        if first_year is not None and panel["year"][s] < first_year:
            continue
        w, cur, ok = [s], s, True
        for _ in range(t_out - 1):
            cur = panel["next"][cur]
            if cur < 0:
                ok = False
                break
            w.append(cur)
        if ok:
            rows.append(w)
    return np.array(rows, dtype=np.int64).reshape(-1, t_out)


# --------------------------------------------------------- strategies ------

class FloorVPW:
    """S1 (handoff §1.1): external real lifetime income `floor_income` per
    year (units of W0, e.g. 0.02) + the whole portfolio on VPW. Spending
    metric = income + portfolio draw. Run with stock_w=1.0 for 'S1'."""
    def __init__(self, floor_income=0.02, expected_real=0.03):
        self.floor_income = floor_income
        self.vpw = VPW(expected_real)
        self.rate = None
        self.name = f"S1-fl{floor_income*100:g}+VPW{expected_real*100:g}"


class LadderVPW:
    """S2-lite: at t=0 buy a real ladder paying `floor` per year (units of
    W0) at the start of each of the first `n_years` years, priced at
    `real_rate`: cost = sum_{k=1..n} floor (1+real_rate)^-k (the spec's
    formula; an annuity-immediate price for what is paid as an annuity-due,
    i.e. ~1% under-priced). The ladder sits outside the portfolio and pays
    deterministically; the rest (W0 - cost) runs VPW (stock_w=1.0 for S2)."""
    def __init__(self, n_years=15, floor=0.02, real_rate=0.01, expected_real=0.03):
        self.n_years, self.floor, self.real_rate = n_years, floor, real_rate
        self.cost = sum(floor * (1 + real_rate) ** -k for k in range(1, n_years + 1))
        self.vpw = VPW(expected_real)
        self.rate = None
        self.name = f"S2L{n_years}-fl{floor*100:g}+VPW{expected_real*100:g}"


# --------------------------------------------------------- simulator -------

def _vpw_factor(rule, y, years):
    n_left = rule.years_left(y, years)
    r = rule.expected_real
    return r / (1 - (1 + r) ** -n_left) if n_left > 1 else 1.0


def simulate_annual(idx, panel, stock_w, rule, w0=1.0):
    """Annual steps, vectorised across paths.

    idx: (P, years) row indices into the stacked panel.
    Order inside year y: (1) spending decision from balances after returns
    through y-1 and the price level through y-1; (2) rebalance to stock_w;
    (3) start-of-year withdrawal; (4) apply year-y returns.
    Real spending = nominal draw / price level at the start of the year,
    where the price level compounds the sampled country-year's inflation.
    Returns the same metric dict as engine.core.simulate (annual_real_spend
    is (P, years))."""
    P, years = idx.shape
    eq = panel["eq"][idx]
    bd = panel["bond"][idx]
    infl = panel["infl"][idx]
    price = np.ones((P, years))
    price[:, 1:] = np.cumprod(1.0 + infl[:, :-1], axis=1)

    external = np.zeros((P, years))            # S1 income / S2 ladder payouts
    w0 = np.broadcast_to(np.asarray(w0, float), (P,)).copy()
    inner = rule
    if isinstance(rule, FloorVPW):
        external[:] = rule.floor_income
        inner = rule.vpw
    elif isinstance(rule, LadderVPW):
        external[:, :min(rule.n_years, years)] = rule.floor
        w0 = w0 - rule.cost
        inner = rule.vpw

    bal_s = w0 * stock_w
    bal_b = w0 * (1.0 - stock_w)

    if isinstance(inner, VPW):
        spend_nom = w0 * _vpw_factor(inner, 0, years)
    else:
        spend_nom = np.full(P, inner.rate * 1.0)     # fraction of REFERENCE wealth 1.0
    initial_rate = spend_nom / np.maximum(w0, 1e-12)
    alive = np.ones(P, bool)
    ruin_year = np.full(P, -1)
    spend_real = np.zeros((P, years))
    prev_port_neg = np.zeros(P, bool)

    for y in range(years):
        wealth = bal_s + bal_b
        if y > 0:
            yr_infl = infl[:, y - 1]
            if isinstance(inner, FixedReal):
                spend_nom = spend_nom * (1.0 + yr_infl)
            elif isinstance(inner, GuytonKlinger):
                r = inner
                cur_wr = np.where(wealth > 0, spend_nom / np.maximum(wealth, 1e-12), np.inf)
                adj = np.minimum(yr_infl, r.cap)
                skip = prev_port_neg & (cur_wr > initial_rate)
                spend_nom = spend_nom * (1.0 + np.where(skip, 0.0, np.maximum(adj, 0.0)))
                cur_wr = np.where(wealth > 0, spend_nom / np.maximum(wealth, 1e-12), np.inf)
                if y < years - r.cpr_off_years:
                    hit = cur_wr > r.upper * initial_rate
                    spend_nom = np.where(hit, spend_nom * (1 - r.cut), spend_nom)
                low = (spend_nom / np.maximum(wealth, 1e-12)) < r.lower * initial_rate
                spend_nom = np.where(low & (wealth > 0), spend_nom * (1 + r.raise_), spend_nom)
                if r.floor_ratio > 0:
                    spend_nom = np.maximum(spend_nom, r.floor_ratio * r.rate * price[:, y])
            elif isinstance(inner, VPW):
                spend_nom = wealth * _vpw_factor(inner, y, years)
            else:
                raise TypeError(f"rule {type(inner).__name__} not supported by the annual engine")
        # rebalance (GK's portfolio-management rule reduces to this annually)
        wealth = bal_s + bal_b
        bal_s = wealth * stock_w
        bal_b = wealth * (1.0 - stock_w)
        # start-of-year withdrawal, proportional
        want = np.where(alive, spend_nom, 0.0)
        got = np.minimum(want, wealth)
        frac = np.where(wealth > 0, got / np.maximum(wealth, 1e-300), 0.0)
        bal_s = bal_s * (1.0 - frac)
        bal_b = bal_b * (1.0 - frac)
        newly = alive & (got < want - 1e-12)
        ruin_year = np.where(newly & (ruin_year < 0), y, ruin_year)
        alive = alive & ~newly
        spend_real[:, y] = got / price[:, y] + external[:, y]
        # apply year-y returns
        port_ret = stock_w * eq[:, y] + (1 - stock_w) * bd[:, y]
        bal_s = np.maximum(bal_s * (1.0 + eq[:, y]), 0.0)
        bal_b = np.maximum(bal_b * (1.0 + bd[:, y]), 0.0)
        prev_port_neg = port_ret < 0

    final_real_wealth = (bal_s + bal_b) / (price[:, -1] * (1 + infl[:, -1]))
    return dict(
        total_real_spend=spend_real.sum(axis=1),
        ruined=(ruin_year >= 0) & (ruin_year < years - 1),
        planned_depletion=(ruin_year == years - 1),
        exhausted=(ruin_year >= 0),
        ruin_year=ruin_year,
        final_real_wealth=final_real_wealth,
        annual_real_spend=spend_real,
    )


# ------------------------------------------------------------ scoring ------

FLOORS = (0.020, 0.025)


def score(res, floors=FLOORS):
    """Mirror of run_v3.score on the absolute ruler."""
    s = res["annual_real_spend"]
    row = dict(ruin_pct=round(100 * res["ruined"].mean(), 2),
               spend_p5=round(float(np.percentile(res["total_real_spend"], 5)), 3),
               spend_med=round(float(np.median(res["total_real_spend"])), 3),
               endw_med=round(float(np.median(res["final_real_wealth"])), 2))
    for fl in floors:
        row[f"sfabs{int(round(fl*1000))}"] = round(100 * spend_fail_abs(s, fl).mean(), 2)
    for g in (2, 4, 8):
        row[f"ce_g{g}"] = round(ce_spending(s, g), 4)
    return row


def real_stats(panel, rows=None):
    """Summary of a (sub)panel: real equity/bond/bill returns, inflation."""
    r = np.arange(len(panel["eq"])) if rows is None else rows
    d = 1.0 + panel["infl"][r]
    req = (1 + panel["eq"][r]) / d - 1
    rb = (1 + panel["bond"][r]) / d - 1
    rbl = (1 + panel["bill"][r]) / d - 1
    inf = panel["infl"][r]
    geo = lambda x: float(np.exp(np.mean(np.log1p(np.maximum(x, -0.999999)))) - 1)
    return dict(n_country_years=int(len(r)),
                n_countries=int(len(np.unique(panel["cid"][r]))),
                req_mean=round(float(req.mean()), 4), req_med=round(float(np.median(req)), 4),
                req_geo=round(geo(req), 4),
                rb_mean=round(float(rb.mean()), 4), rb_med=round(float(np.median(rb)), 4),
                rbl_mean=round(float(rbl.mean()), 4), rbl_med=round(float(np.median(rbl)), 4),
                infl_p50=round(float(np.median(inf)), 4),
                infl_p95=round(float(np.percentile(inf, 95)), 4),
                infl_max=round(float(inf.max()), 4),
                eq_nom_max=round(float(panel["eq"][r].max()), 4),
                req_min=round(float(req.min()), 4))
