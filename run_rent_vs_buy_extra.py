"""租 vs 買：補強信度與效度的加跑。

  A. 租金在 CPI 之上漂移 +1%/+2% 的世界 → 交叉點（bar）移動多少
  B. 「75 歲被迫買房」情境（房價與租金同步漂移）→ 「遲早要買」論的公平版
  C. 台灣引擎（1983–2023 黃金樣本）對照 → 樣本相依性
  D. 3 個 seed × 關鍵格 → 抽樣離散度
python run_rent_vs_buy_extra.py -> results/rent_vs_buy_extra.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.core import load_panel, restrict, stationary_bootstrap_indices
from run_rent_vs_buy import simulate, N_PATHS, T
from run_fire_case import death_months_from50

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
W, RENT, NH = 5000e4, 40_000.0, 110_000.0

us = restrict(load_panel("us"), "1934-02")
tw = load_panel("tw")
tw["bill"] = tw.pop("usdbill")
rng = np.random.default_rng(7)
dh = death_months_from50(N_PATHS, 18.2, 0.105, rng)
dw = death_months_from50(N_PATHS, 21.9, 0.110, rng)
IDX = {s: stationary_bootstrap_indices(len(us["stock"]), T, N_PATHS, 36,
                                       np.random.default_rng(s)) for s in (0, 1, 2)}
IDX_TW = stationary_bootstrap_indices(len(tw["stock"]), T, N_PATHS, 36,
                                      np.random.default_rng(0))
out = {}

# ---- A. 租金漂移下的交叉點 --------------------------------------------------
print("== A. 租金每年比 CPI 多漲 0/1/2% 時，租屋基準 vs 各收益率的買 ==")
out["drift"] = []
for drift in (0.0, 0.01, 0.02):
    base = simulate(IDX[0], us, "rent", W, 0, RENT, NH, dh, dw, rent_drift=drift)
    row = dict(drift=drift, rent=base, buys=[])
    print(f"  漂移 CPI+{drift*100:.0f}%：租屋 破產{base['ruin_pct']:5.2f}% "
          f"非住房終身{base['nh_lifetime_med']:>5}萬 p5 {base['nh_lifetime_p5']:>5}萬")
    for yld in (0.020, 0.030, 0.040, 0.050, 0.060):
        hp = RENT * 12 / yld
        r = simulate(IDX[0], us, "own", W, hp, RENT, NH, dh, dw)
        r["yield"] = round(yld * 100, 1)
        r["house_wan"] = round(hp / 1e4)
        r["edge_vs_rent"] = r["nh_lifetime_med"] - base["nh_lifetime_med"]
        row["buys"].append(r)
        print(f"      買 收益率{yld*100:4.1f}%（{r['house_wan']:>5}萬）："
              f"破產{r['ruin_pct']:5.2f}% 終身{r['nh_lifetime_med']:>5}萬 "
              f"→ 相對租 {r['edge_vs_rent']:+5}萬")
    out["drift"].append(row)

# ---- B. 75 歲被迫買 ---------------------------------------------------------
print("\n== B. 75 歲被迫現金買房（房價與租金同步漂移）vs 現在買 vs 一直租 ==")
out["forced75"] = []
for drift in (0.0, 0.01, 0.02):
    for hp in (1500e4, 3000e4):
        f75 = simulate(IDX[0], us, "rent", W, hp, RENT, NH, dh, dw,
                       rent_drift=drift, buy_at_month=(75 - 50) * 12)
        now = simulate(IDX[0], us, "own", W, hp, RENT, NH, dh, dw)
        f75.update(drift=drift, house_wan=round(hp / 1e4))
        out["forced75"].append(dict(scenario=f75, buy_now=now))
        print(f"  漂移+{drift*100:.0f}% 房{round(hp/1e4):>4}萬 | 75歲買：破產{f75['ruin_pct']:5.2f}% "
              f"買不起{f75['could_not_buy_pct']:5.1f}% 終身{f75['nh_lifetime_med']:>5}萬 "
              f"| 現在買：破產{now['ruin_pct']:5.2f}% 終身{now['nh_lifetime_med']:>5}萬")

# ---- C. 台灣引擎對照 --------------------------------------------------------
print("\n== C. 台灣引擎（黃金樣本，偏樂觀上界）==")
out["tw"] = []
base_tw = simulate(IDX_TW, tw, "rent", W, 0, RENT, NH, dh, dw)
out["tw"].append(dict(mode="rent", **{k: v for k, v in base_tw.items() if k != "mode"}))
print(f"  租屋：破產{base_tw['ruin_pct']:5.2f}% 終身{base_tw['nh_lifetime_med']:>5}萬")
for yld in (0.025, 0.035, 0.050):
    hp = RENT * 12 / yld
    r = simulate(IDX_TW, tw, "own", W, hp, RENT, NH, dh, dw)
    r["yield"] = round(yld * 100, 1)
    r["edge_vs_rent"] = r["nh_lifetime_med"] - base_tw["nh_lifetime_med"]
    out["tw"].append(r)
    print(f"  買 收益率{yld*100:4.1f}%：破產{r['ruin_pct']:5.2f}% "
          f"終身{r['nh_lifetime_med']:>5}萬 → 相對租 {r['edge_vs_rent']:+5}萬")

# ---- D. 多 seed 離散度 ------------------------------------------------------
print("\n== D. 3 seeds × 關鍵格（美國引擎）==")
out["seeds"] = []
for s in (0, 1, 2):
    b = simulate(IDX[s], us, "rent", W, 0, RENT, NH, dh, dw)
    o5 = simulate(IDX[s], us, "own", W, RENT * 12 / 0.05, RENT, NH, dh, dw)
    o2 = simulate(IDX[s], us, "own", W, RENT * 12 / 0.02, RENT, NH, dh, dw)
    out["seeds"].append(dict(seed=s, rent=b, own_y5=o5, own_y2=o2))
    print(f"  seed {s}: 租 終身{b['nh_lifetime_med']:>5} | 買@5% {o5['nh_lifetime_med']:>5} "
          f"(差{o5['nh_lifetime_med']-b['nh_lifetime_med']:+4}) | 買@2% {o2['nh_lifetime_med']:>5} "
          f"(差{o2['nh_lifetime_med']-b['nh_lifetime_med']:+5})")

with open(os.path.join(RESULTS, "rent_vs_buy_extra.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nwrote results/rent_vs_buy_extra.json")
