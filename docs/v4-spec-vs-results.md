# v4 規格 vs 結果對照（`docs/handoff-v4.md` 逐節）

| 手冊節 | 要求 | 做了什麼 | 輸出 | 狀態 |
|---|---|---|---|---|
| §0.1 | `ruined` = 耗盡且距終點 >1 年；`planned_depletion` 另開；重出 v3 VPW/RMD 列 | `engine/core.py` `simulate()` 回傳 `ruined / planned_depletion / exhausted`；`run_v4_v3rerun.py` | `results/v4_v3rerun.csv`；REPORT-v4 v4.0.1 | ✅ 排序不變 |
| §0.2 | 勞保終身（CPI 門檻調整）、勞退定期給付至平均餘命、每 3 年重算；重跑受影響專題 | `engine/household.py`（勞保 5% 階梯、勞退 `LR_TERM/LR_RATE` 名目定期）；`run_v4_lr.py` 量化舊模型偏誤；專題 A/B 原本把勞退當一次領併入資產，不受影響 | `results/v4_lr.csv`；v4.0.2 | ✅ |
| §1.1 S1–S4 | 勞保地板（足額/七折/歸零）+ 100% 股 VPW；階梯 N∈{10,15,20}；遞延年金 80/85 起、附加費用 {10,20,30}%；疊加 | `engine/household.py` + `run_v4_floor.py`：pension_eff ∈ {0, .7, 1, 1.75, 2.5}% W0、階梯實質/名目、年金名目/實質（假設性）、S4 | `results/v4_floor.csv`、`results/v4_floor_frontier.png`；v4.1 | ✅ 年金用精算價 × (1+load)，標假設性 |
| §1.2 | 對照 VPW 60/40、80/20、GK4/GK5；floor_ratio ∈ {0.35, 0.5, 0.7, 1.0} | 同上，65 與 50 歲、US/TW 兩引擎、死亡率 | 同上 | ✅ |
| §1.3 | floor_ratio × 策略主表；地板成本 vs 保障前緣圖 | v4.1 主表；`results/v4_floor_frontier.png` | | ✅ |
| §2.1 | 內政部生命表抽樣、單身/聯合、P(耗盡且存活)、P(跌破且存活)、活到 p90 條件；退休年齡 {40,50,60,65} | `engine/mortality.py`（114 年簡易表、Gompertz 封閉）+ `run_v4_mortality.py` 事後層 | `results/v4_mortality.csv`；v4.2 | ✅ |
| §2.2 | longevity_shift ∈ {0,+3,+6}；翻轉標不穩健 | 同上；60/40 vs 100/0 在 40 歲 VPW 翻轉 → 不穩健 | | ✅ |
| §2.3 | 65 歲 vs 40 歲答案拆開 | v4.2 讀法 1 | | ✅ 手冊「對 FIRE 者低估」不成立 |
| §3.1 | V-cap {5/2.5, 5/5, 10/5}、V-yale {0.5,0.7,0.8}、V-floor；spend_vol / dd_max / dd_yrs；GK 存廢判定 | `VPW(smooth=, floor_abs=)`；`run_v4_smooth.py` | `results/v4_smooth.csv`；v4.3.1 | ✅ V-yale 0.7、W7 通過 → GK 失去理由 |
| §3.2 | 台灣版 RMD：餘命 + buffer {0,5,10}，r=5% 與 1/CAPE | `VPW(life_table=, cape_rate=)`（CAPE 從 Shiller 檔） | v4.3.2 | ✅ buffer ≥ +10 |
| §3.3 | 斷崖測試：到 100、90→110、100→120…；對照每年更新、年金 | `VPW(horizon=, extensions=)`；v4.3.2 | | ✅ 每次延長砍四成；死亡率尺下讀 |
| §3.4 | W6 funded ratio、W7 風險護欄（無前視） | `FundedRatio`、`RiskGuardrail`（Milevsky-Robinson 封閉式，固定假設參數） | v4.3.1 | ✅ W7 > GK；W6 末段加支迴圈待修 |
| §4 | JST 18 國年頻引擎；池化 block bootstrap；市值加權全球；美國單國對照；核心子集 | `engine/annual.py`、`run_v4_jst.py`、`tests/test_annual.py`；16 國可用；**市值無 → 實質 GDP 加權**；另加 ex-US、ex-惡性通膨 | `results/v4_jst*.csv`、`docs/v4-notes-jst.md`；v4.4 | ✅ DMS 無免費來源未交叉驗證 |
| §5 | A1–A5 × {20,30,40}、H4 k∈{1,2,4}、Coast FIRE、整條分布銜接、聯合表 | `run_v4_accum.py`、`tests/test_accum.py`；`simulate(w0=)` | `results/v4_accum*.csv`、`docs/v4-notes-accum.md`；v4.5 | ✅ |
| §6 | 長照 60–120 萬 × {3,5,8} 年、起始由存活模型抽、崩盤中條件情境；桶存廢 | `simulate(extra_spend=)`、`run_v4_ltc.py`；三個財富層 | `results/v4_ltc.csv`；v4.6 | ✅ 桶退場；換房/醫療一次性大額未另跑（長照 8 年 × 120 萬已涵蓋量級） |
| §7 | RSSB+RSST 合成：股 + (GS10−Tbill) + 趨勢（SG Trend / AQR TSMOM）+ 融資 + 費用；四個問題 | `data/build_trend.py`、`engine/stack.py`、`run_v4_stack.py`、`tests/test_stack.py` | `results/v4_stack*.csv`、`docs/v4-notes-stack.md`；v4.7 | ✅ SG Trend 無來源；1934–84 拼接弱（r=0.37） |
| §8 Path A | 台股質借序列接入 + 地緣衝擊情境 | **未做**：序列在信貸專案、不在本 repo | — | ❌ 阻塞於資料 |
| §9.1 | 行為棄守 (X, Y, q, m) | `HH(behavior=)`；`run_v4_realism.py` | `results/v4_behavior.csv`；v4.9.1 | ✅ 支出波動觸發棄守未建模 |
| §9.2 | 稅與費用 | `simulate(fee=)` 均勻拖累 {0.3, 0.7, 1.2}% | `results/v4_fees.csv`；v4.9.2 | ✅ 高周轉的額外交易成本、遺產稅、最低稅負制未逐項建模 |
| §9.3 | 支出微笑 | `HH(smile=)` | `results/v4_smile.csv`；v4.9.3 | ✅ |
| §9.4 | Shiller/FRED 補到最新；含與不含 2024+ 兩版 | `data/fetch.py` 重建到 2026-07；`restrict(last_date=)` | `results/v4_sample.csv`；v4.9.4 | ✅ 009826 只記錄不進引擎 |
| §10.1 | 逐策略偏誤敏感度（報酬與利率各自插值） | `run_v4_bias.py`（`path_c_var` 模式） | `results/v4_bias.csv`；v4.10.1 | ✅ |
| §10.2 | 死亡率回套 v1–v3 結論 | v4.2 末段清單 | | ✅ |
| §12 | 六個決策問題 | REPORT-v4 v4.12 | | ✅ |
| 規格外 | FIRE 範例（無勞保勞退、1000/3000/5000 萬、40/50/65 歲、三配置、四種世界） | `run_v4_fire50.py`、`run_v4_fire50_intl.py` | `results/v4_fire50.csv`、`results/v4_fire50_intl.csv`；REPORT-v4 v4.11；`docs/bysq_report_v4.html` | ✅ |
| 規格外 | 台灣人持有 VT + BND 含匯率（美元計價全球股、台幣實質匯率疊加、遺產） | `engine/annual.py::global_usd_series`、`simulate_annual(fx_overlay=, wealth_path)`、`run_v4_twvt.py` | `results/v4_twvt.csv`；`docs/bysq_report_twvt.html` | ✅ 匯率每年獨立抽樣，失去危機期同步貶值效果 |

未做／部分：§8 Path A（資料不在 repo）；§4 DMS 交叉驗證（付費資料）；§6 換房等一次性非長照大額（量級被長照情境涵蓋）；§9.2 逐項稅制。
