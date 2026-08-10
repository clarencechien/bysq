"""租 vs 買：對靠投資組合過活的家庭，房子是資產還是負債？

## 設計：為什麼不能拿「月租」比「房貸月付金」

月付金裡的本金是**儲蓄**不是支出；而現金買房沒有月付金，卻有機會成本。
正確的比較是**不可回收成本**：
    租 → 租金，100% 不可回收
    買 → 資金機會成本 + 持有成本 + 攤銷交易成本 − 實質增值

買划算的條件（毛租金收益率 y = 年租金 ÷ 房價）：**y > (r − g) + c**
    r = 資金機會成本、g = 實質增值率、c = 持有成本率
本專案的 r 取「可持續提領率」，因為整份研究都用「這筆資產能支撐什麼生活」衡量資產。

## 公平的模擬比較：固定「非住房生活水準」

錯誤做法是讓兩邊花一樣的總額——那等於讓買方把房租那筆錢拿去多消費。
正確做法：兩邊的**非住房消費目標相同**，住房那一欄租方填租金、買方填持有成本(+房貸利息)，
比較「能不能撐住」與「非住房消費的終身總額」。

## 第二條 bar：地板效應（本專案獨有）

租金是 100% 砍不掉、與通膨連動、沒有終點的負債；v3 已證明砍不掉的支出佔比 ≥70%
時動態規則失效。買房消滅最大的地板項目，但要用投資組合去換。兩效果相反，
交叉點用模擬找。

python run_rent_vs_buy.py -> results/rent_vs_buy.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_PATHS = 10_000
START_AGE, HORIZON_Y = 50, 55
T = HORIZON_Y * 12
STOCK_W = 0.8
CARRY = 0.009          # 持有成本率/年：稅+管理費+維護+攤銷交易成本
ESSENTIALS = 25_000.0  # 住房以外砍不掉的支出


def breakeven_table():
    out = []
    for r in (0.030, 0.035, 0.039):
        for g in (-0.01, 0.0, 0.01, 0.02):
            y = (r - g) + CARRY
            out.append(dict(r=round(r * 100, 1), g=round(g * 100, 1),
                            c=round(CARRY * 100, 1), breakeven_yield=round(y * 100, 2),
                            price_to_annual_rent=round(1 / y, 1)))
    return out


def implied(price, monthly_rent, r=0.035, c=CARRY):
    y = monthly_rent * 12 / price
    return dict(price_wan=round(price / 1e4), rent_monthly=round(monthly_rent),
                gross_yield=round(y * 100, 2),
                breakeven_g=round(((r + c) - y) * 100, 2),
                annual_edge_to_rent_wan=round(((r + c) * price - monthly_rent * 12) / 1e4))


def simulate(idx, panel, mode, wealth0, house_price, monthly_rent, nonhousing,
             death_h, death_w, ltv=0.0, mort_mode="float", mort_nom=0.022,
             mort_spread=0.015, g_real=0.0, rent_drift=0.0, buy_at_month=None):
    """mode: 'rent' | 'own'。兩邊的非住房消費目標相同 (nonhousing)。

    ltv>0 = 房貸（只付息；本金屬儲蓄，由遺產結清）。
    mort_mode='float' → 利率 = 抽樣短率 + 加碼（台灣是機動利率，這是實況）
    mort_mode='fixed' → 固定名目利率（台灣買不到，當對照組）
    rent_drift    → 租金在 CPI 之上的年漂移（例如 0.01 = 每年實質 +1%）
    buy_at_month  → mode='rent' 時：該月被迫用現金買房（房價同步套用
                    rent_drift 的實質漂移——租金會漲的世界裡房價也在漲）；
                    買不起的路徑繼續租，另計比例
    """
    P, Tn = idx.shape
    stock_r, bond_r, infl_m = panel["stock"][idx], panel["bond"][idx], panel["infl"][idx]
    bill = panel["bill"][idx]
    price = np.ones((P, Tn))
    lvl = np.ones(P)
    for t in range(1, Tn):
        lvl = lvl * (1.0 + infl_m[:, t - 1])
        price[:, t] = lvl

    if mode == "own":
        equity = house_price * (1 - ltv)
        debt = house_price * ltv
        w0 = wealth0 - equity
        carry_real = CARRY * house_price / 12.0
    else:
        equity = debt = 0.0
        w0 = wealth0
        carry_real = 0.0
    if w0 <= 0:
        return None

    bal = np.stack([np.full(P, w0 * STOCK_W), np.full(P, w0 * (1 - STOCK_W))], axis=1)
    ruined = np.zeros(P, bool)
    floor_breach = np.zeros(P, bool)
    lifetime_nh = np.zeros(P)      # 非住房消費（實質）
    housing_paid = np.zeros(P)
    draw_nom = np.zeros(P)
    floor_ref = np.zeros(P)
    owns = np.zeros(P, bool)       # buy_at_month 之後轉為屋主的路徑
    bought_price_real = np.zeros(P)

    for t in range(Tn):
        y = t // 12
        both = (t < death_h) & (t < death_w)
        one = ((t < death_h) | (t < death_w)) & ~both
        alive = both | one
        nh_t = np.where(both, nonhousing, np.where(one, nonhousing * 0.8, 0.0))
        drift = (1 + rent_drift) ** (t / 12.0)

        # 被迫購屋事件（現金買，價格 = 房價 × 同步實質漂移）
        if mode == "rent" and buy_at_month is not None and t == buy_at_month:
            cost_real = house_price * drift
            wealth = bal.sum(axis=1)
            can = alive & (wealth / price[:, t] >= cost_real * 1.02)
            frac_pay = np.where(can, cost_real * price[:, t] /
                                np.maximum(wealth, 1e-9), 0.0)
            bal *= (1.0 - frac_pay)[:, None]
            owns = can
            bought_price_real = np.where(can, cost_real, 0.0)

        # 住房那一欄（名目）
        if mode == "rent":
            housing = np.where(owns,
                               CARRY * bought_price_real / 12.0 * price[:, t],
                               monthly_rent * drift * price[:, t])
        else:
            if mort_mode == "float":
                m_m = (1 + bill[:, t]) * (1 + mort_spread) ** (1 / 12) - 1
            else:
                m_m = (1 + mort_nom) ** (1 / 12) - 1
            housing = carry_real * price[:, t] + debt * price[:, t] / price[:, 0] * m_m

        floor_nom = (ESSENTIALS * np.where(one, 0.9, 1.0)) * price[:, t] + housing
        floor_ref = floor_nom / price[:, t]

        wealth = bal.sum(axis=1)
        if t % 12 == 0:
            n_left = max((100 - START_AGE) - y, 1)
            f = 0.03 / (1 - 1.03 ** -n_left) if n_left > 1 else 1.0
            d = (wealth / price[:, t]) * f / 12.0                  # VPW，實質
            d = np.maximum(d, floor_nom / price[:, t])             # 補到地板
            d = np.minimum(d, 2.0 * nh_t + housing / price[:, t])  # 上限：非住房 2 倍
            draw_nom = d * price[:, t]

            bal[:, 0] = wealth * STOCK_W
            bal[:, 1] = wealth * (1 - STOCK_W)

        want = np.where(alive & ~ruined, draw_nom, 0.0)
        wealth = bal.sum(axis=1)
        take = np.minimum(want, wealth)
        frac = np.where(wealth > 0, take / np.maximum(wealth, 1e-9), 0.0)
        bal *= (1.0 - frac)[:, None]
        ruined |= alive & ~ruined & (take < want - 1e-6)

        spend_real = np.where(alive, take / price[:, t], 0.0)
        h_real = np.where(alive, housing / price[:, t], 0.0)
        nh_real = np.maximum(spend_real - h_real, 0.0)
        lifetime_nh += nh_real
        housing_paid += h_real
        floor_breach |= alive & (spend_real < floor_ref - 1)

        bal[:, 0] *= (1.0 + stock_r[:, t])
        bal[:, 1] *= (1.0 + bond_r[:, t])

    if mode == "own":
        house_end = house_price * (1 + g_real) ** HORIZON_Y - debt
    else:
        house_end = bought_price_real          # 被迫購屋者期末持有的房（g=0）
    net_end = bal.sum(axis=1) / price[:, -1] + house_end
    h0 = (monthly_rent if mode == "rent"
          else CARRY * house_price / 12 + debt * ((1 + mort_nom) ** (1 / 12) - 1))
    return dict(mode=mode, ltv=ltv,
                could_not_buy_pct=round(100 * float(
                    ((~owns) & (np.maximum(death_h, death_w) > buy_at_month)).mean()), 2)
                if (mode == "rent" and buy_at_month is not None) else None,
                ruin_pct=round(100 * float(ruined.mean()), 2),
                floor_breach_pct=round(100 * float(floor_breach.mean()), 2),
                nh_lifetime_med=round(float(np.median(lifetime_nh)) / 1e4),
                nh_lifetime_p5=round(float(np.percentile(lifetime_nh, 5)) / 1e4),
                housing_lifetime_med=round(float(np.median(housing_paid)) / 1e4),
                net_end_med=round(float(np.median(net_end)) / 1e4),
                start_wr=round(100 * (nonhousing + h0) * 12 / max(wealth0 - equity, 1), 2),
                floor_share=round(100 * (ESSENTIALS + h0) / (nonhousing + h0), 1))


def main():
    from run_fire_case import death_months_from50
    us = restrict(load_panel("us"), "1934-02")
    rng = np.random.default_rng(7)
    dh = death_months_from50(N_PATHS, 18.2, 0.105, rng)
    dw = death_months_from50(N_PATHS, 21.9, 0.110, rng)
    idx = stationary_bootstrap_indices(len(us["stock"]), T, N_PATHS, 36,
                                       np.random.default_rng(0))

    out = {"breakeven": breakeven_table(),
           "cases": [implied(6000e4, 60_000), implied(4000e4, 60_000),
                     implied(3000e4, 60_000), implied(2000e4, 40_000),
                     implied(1200e4, 40_000), implied(1000e4, 40_000)]}
    print("== 封閉解：要漲多少才划算（現金買、r=3.5%、c=0.9%）==")
    for c in out["cases"]:
        print(f"  {c['price_wan']:>5}萬房／月租{c['rent_monthly']:>6}：收益率 {c['gross_yield']:>4.2f}%"
              f" → 需實質年漲 {c['breakeven_g']:+.2f}%"
              f"（否則每年租方多出 {c['annual_edge_to_rent_wan']:>4}萬可花）")

    # --- 收益率掃描：找模擬上的交叉點（現金買）---
    print("\n== 模擬：收益率掃描（資產5000萬、非住房11萬/月、租金4萬，現金買）==")
    W, RENT, NH = 5000e4, 40_000.0, 110_000.0
    base = simulate(idx, us, "rent", W, 0, RENT, NH, dh, dw)
    print(f"  租屋基準： 破產{base['ruin_pct']:5.2f}%  非住房終身{base['nh_lifetime_med']:>5}萬"
          f"  p5 {base['nh_lifetime_p5']:>5}萬  地板占比{base['floor_share']:.1f}%"
          f"  起始提領率{base['start_wr']:.2f}%")
    scan = [{"mode": "rent", **base}]
    for yld in (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06):
        hp = RENT * 12 / yld
        r = simulate(idx, us, "own", W, hp, RENT, NH, dh, dw)
        if r is None:
            print(f"  收益率{yld*100:4.1f}% → 房價{round(hp/1e4):>5}萬：買不起")
            continue
        r["yield"] = round(yld * 100, 1); r["house_wan"] = round(hp / 1e4)
        scan.append(r)
        d = r["nh_lifetime_med"] - base["nh_lifetime_med"]
        print(f"  收益率{yld*100:4.1f}% 房價{r['house_wan']:>5}萬：破產{r['ruin_pct']:5.2f}%"
              f"  非住房終身{r['nh_lifetime_med']:>5}萬  p5 {r['nh_lifetime_p5']:>5}萬"
              f"  地板{r['floor_share']:4.1f}%  起始提領率{r['start_wr']:5.2f}%"
              f"  → 相對租屋 {d:+5}萬")
    out["yield_scan"] = scan

    # --- 房貸：機動 vs 固定 ---
    print("\n== 模擬：貸款買（房價3000萬=收益率1.6%、頭期2成，其餘投資）==")
    HP = 3000e4
    mort = []
    for ltv, mode_, nomr, label in [
            (0.0, "float", 0.022, "全現金"),
            (0.8, "float", 0.022, "貸8成・機動(短率+1.5%)"),
            (0.8, "fixed", 0.022, "貸8成・固定2.2%（台灣買不到）"),
            (0.8, "fixed", 0.035, "貸8成・固定3.5%")]:
        r = simulate(idx, us, "own", W, HP, RENT, NH, dh, dw,
                     ltv=ltv, mort_mode=mode_, mort_nom=nomr)
        if r is None:
            print(f"  {label:28s} 買不起"); continue
        r["label"] = label
        mort.append(r)
        d = r["nh_lifetime_med"] - base["nh_lifetime_med"]
        print(f"  {label:28s} 破產{r['ruin_pct']:5.2f}%  非住房終身{r['nh_lifetime_med']:>5}萬"
              f"  p5 {r['nh_lifetime_p5']:>5}萬  期末淨資產{r['net_end_med']:>6}萬"
              f"  → 相對租屋 {d:+5}萬")
    out["mortgage"] = mort
    out["rent_base"] = base

    with open(os.path.join(RESULTS, "rent_vs_buy.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote results/rent_vs_buy.json")


if __name__ == "__main__":
    main()
