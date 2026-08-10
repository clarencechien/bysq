# bysq — 台灣視角提領策略研究（60/40 vs GK 護欄 vs BYSQ 現金桶）

以真實歷史資料回答：「60/40 + 4%」、「Guyton-Klinger 護欄」、「100% 股票 + 現金桶（BYSQ）」
在 30/50 年提領期的真實取捨。規格見兩份 HANDOFF；本版為誠實縮小範圍的實作，
所有縮減之處在 `data/PROVENANCE.md` 與 `REPORT.md` 明列。

## 重現全部數字

```bash
pip install numpy xlrd
python data/fetch.py          # 下載 Shiller / FRED / 主計總處原始資料
python tests/test_sanity.py   # 防自欺測試（無前視、零波動退化、共變異數保留）
python run_experiments.py     # 主矩陣 + 提領率掃描 → results/*.csv
```

## 結構

- `data/fetch.py` — 資料下載與拼接；`data/PROVENANCE.md` — 來源與已知缺口
- `engine/core.py` — 綁定同一時間軸的 stationary block bootstrap、
  月頻期初提領模擬、完整四規則 GK、無前視現金桶
- `tests/test_sanity.py` — HANDOFF §6 防自欺協定
- `run_experiments.py` — 策略矩陣（US 1934+、TW 1983+ 兩引擎）
- `results/` — 輸出表；`REPORT.md` — 結論與失效條件
