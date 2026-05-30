# 模型變數說明（SP500-only + AutoML → `y_next`）

> 資料集：`quant/model/model_dataset.parquet`（目前約 6k 列，依資料更新而變動；以檔案內容為準）  
> Universe：**S&P 500 only**（`quant/data/sp500_constituents.csv`，並做 `.`→`-` ticker 正規化）  
> 模型：`FLAML AutoML`（目前最佳為 `xgb_limitdepth`；以 `quant/model/model_meta.json` 為準）  
> 特徵清單：`quant/model/model_meta.json` → `feature_cols`（42 個；`close` / `adjusted_close` 僅供推導、不進模型）  
> 統計表（可 CSV 匯入）：`quant/model/feature_stats.csv`  
> 產生腳本：`python scripts/feature_stats.py`

---

## 0. Walk-forward 與 TTM 暖機

| 設定 | 值 |
|------|-----|
| `FEATURE_WARMUP_QUARTERS` | **4**（2020Q2–2021Q1：TTM/YoY 幾乎全 NaN，不進 pool search / 訓練） |
| Pool search | Q5–Q16（2021Q2–2024Q1） |
| CV OOF 預測季 | Q13–Q16（2023Q2–2024Q1），每 fold 訓練 = Q5 起 expanding |
| Final OOS | Q17–Q20（2024Q2–2025Q1） |

---

## 1. 預測目標 `y_next`

| 項目 | 說明 |
|------|------|
| **定義** | 下一季是否為超額 Top30：`y_next ∈ {0, 1}` |
| **超額報酬** | `excess_return = 季報酬 − 當季 SPY 報酬`（市值加權 S&P 500） |
| **當季 `y_t`** | `1` 若 `excess_return` 橫截面排名 ≤ **30**，否則 `0` |
| **組合** | 模型 Top30；權重公式由 CV OOF 從 `WEIGHT_SCHEME_POOL` 選出 |
| **月度對照** | `labels_monthly` + `holdings_by_month.json`（selected vs actual_top30） |
| **程式** | `build_labels.py`、`monthly_portfolio.py` |

**注意**：`return` / `excess_return` 為信號季；`y_next` / `is_top30_next` 為下一季。

---

## 2. Point-in-Time（PIT）對齊

每一列 = **一檔股票 × 一個季度末 `as_of_date`**。

1. 股價特徵：該季 **最後一個交易日** 的 `close` / `adjusted_close` / `volume` / `shares_outstanding`。
2. 基本面：在該 `as_of_date` 當天，取 **`publish_date ≤ as_of_date`** 中 **最新一筆** 已公布季報（避免 look-ahead）。
3. `sector`：由 SimFin `IndustryId` 對照 `load_industries()` 取得（約 99.7% 非 Unknown）。

程式：`quant/pipeline/build_features.py`

---

## 3. 訓練時 NA / inf 處理（不補值版）

實作：`quant/pipeline/train_model.py` → `prepare_feature_matrix()`

| 步驟 | 說明 |
|------|------|
| 1 | `inf` / `-inf` → `NaN` |
| 2 | **不做補值**（保留 `NaN` 讓 tree model 用「missing branch」處理） |
| 3 | 數值 **clip 至 [-1e6, 1e6]**（避免極端值/溢位） |

**Loss**：**Weighted BCE** — FLAML `task=classification`、`metric=log_loss`；訓練時 `sample_weight`：`w_neg=1`、`w_pos=n_neg/n_pos`（每個 CV fold / 最終模型在該 fold 訓練集上計算）。推論分數 = `predict_proba` 的 **P(y_next=1)**。

AutoML 設定（見 `model_meta.json`）：`skip_transform=True`；estimator 含 `xgb_limitdepth`, `xgboost`, `lgbm`（可處理 NaN）。

**未進模型的欄位**（僅存於 parquet）：`symbol`, `company_name`, `sector`, `industry`, `quarter`, 日期欄、`operating_cash_flow`, `capital_expenditure`, `free_cash_flow`, `y_t`, `y_next`, `return`, `sharpe`, `quarter_end`、`close`、`adjusted_close`（後兩者僅供推導 `market_cap` / 動量，見 §4.B）。

---

## 4. 模型特徵：公式與缺失率（42 個）

缺失率來自 **進模型前** 的原始 `model_dataset`。`inf%` 為該欄 `inf` 占比。實際數字以 `quant/model/feature_stats.csv` 為準（會隨資料更新）。

### A. 原始基本面（單季，PIT 財報）

| 變數 | 計算方式 | 缺失% | inf% |
|------|----------|------:|-----:|
| `revenue` | SimFin 損益表 `Revenue` | (see `feature_stats.csv`) | 0 |
| `gross_profit` | `Gross Profit` | (see `feature_stats.csv`) | 0 |
| `operating_income` | `Operating Income (Loss)` | 0.01 | 0 |
| `net_income` | `Net Income` | 0.00 | 0 |
| `eps_diluted` | `Net Income (Common) / Shares (Diluted)` | 0.80 | 0 |
| `total_assets` | 資產負債表 | 0.00 | 0 |
| `total_liabilities` | 資產負債表 | 0.02 | 0 |
| `total_equity` | 資產負債表 | 0.01 | 0 |
| `cash_and_equivalents` | 現金及約當 | (see `feature_stats.csv`) | 0 |
| `total_debt` | `Short Term Debt + Long Term Debt` | 0.00 | 0 |

### B. 市場 / 價格（季末）

| 變數 | 計算方式 | 缺失% | inf% |
|------|----------|------:|-----:|
| `shares_outstanding` | 優先股價檔 `Shares Outstanding`，否則財報稀釋/基本股數 | (see `feature_stats.csv`) | 0 |
| `volume` | 季末成交量 | 0.00 | 0 |
| `market_cap` | `adjusted_close × shares_outstanding` | 0.18 | 0 |
| `enterprise_value` | `market_cap + total_debt - cash_and_equivalents` | (see `feature_stats.csv`) | 0 |

> **註**：`close`（未還原季末收盤）與 `adjusted_close`（還原季末收盤）仍會計算並存於 parquet，用來推導 `market_cap`（`adjusted_close × shares`）與動量 `return_3m/6m/12m`（用 `close`），但**不作為模型特徵**。消融實驗（`scripts/exp_price_features.py`）顯示移除這兩個原始價格 level 後 CV OOF 年化超額 Sharpe 不降反升（1.69→1.79），且彼此高度冗餘。

### C. 成長率（YoY）

需至少 4 季歷史；`pct_change(4)` 按 symbol 排序後計算。

| 變數 | 計算方式 | 缺失% | inf% |
|------|----------|------:|-----:|
| `revenue_growth_yoy` | `revenue_ttm` 與 4 季前比 | (see `feature_stats.csv`) | (see `feature_stats.csv`) |
| `eps_growth_yoy` | `eps_diluted` 與 4 季前比 | (see `feature_stats.csv`) | (see `feature_stats.csv`) |

> `revenue_ttm` = 連續 4 季 `revenue` rolling sum（`min_periods=4`）。其他 `*_ttm` 同理。

### D. 獲利能力（TTM 比率）

分母為 0 時結果為 NaN（`_safe_div`）。

| 變數 | 公式 | 缺失% | inf% |
|------|------|------:|-----:|
| `gross_margin` | `gross_profit_ttm / revenue_ttm` | (see `feature_stats.csv`) | 0 |
| `operating_margin` | `operating_income_ttm / revenue_ttm` | (see `feature_stats.csv`) | 0 |
| `net_margin` | `net_income_ttm / revenue_ttm` | (see `feature_stats.csv`) | 0 |
| `roe` | `net_income_ttm / total_equity` | (see `feature_stats.csv`) | 0 |
| `roa` | `net_income_ttm / total_assets` | (see `feature_stats.csv`) | 0 |

### E. 估值 / 殖利率

| 變數 | 公式 | 缺失% | inf% |
|------|------|------:|-----:|
| `pe_ratio` | `market_cap / net_income_ttm` | (see `feature_stats.csv`) | 0 |
| `pb_ratio` | `market_cap / total_equity` | (see `feature_stats.csv`) | 0 |
| `ev_to_ebit` | `enterprise_value / operating_income_ttm` | (see `feature_stats.csv`) | 0 |
| `ebit_to_tev` | `operating_income_ttm / enterprise_value` | (see `feature_stats.csv`) | 0 |
| `fcf_yield` | `free_cash_flow_ttm / market_cap`（見下方 FCF proxy） | (see `feature_stats.csv`) | 0 |
| `earnings_yield` | `net_income_ttm / market_cap` | (see `feature_stats.csv`) | 0 |

**FCF proxy（SimFin 無顯式 CapEx 欄）**：

- `capital_expenditure = max(0, -Change in Fixed Assets & Intangibles)`
- `free_cash_flow = operating_cash_flow + Change in Fixed Assets & Intangibles`
- 再對 OCF / CapEx / FCF 做 4 季 TTM → `*_ttm`

### F. 槓桿

| 變數 | 公式 | 缺失% | inf% |
|------|------|------:|-----:|
| `debt_to_equity` | `total_debt / total_equity` | (see `feature_stats.csv`) | 0 |
| `debt_to_assets` | `total_debt / total_assets` | 0.00 | 0 |

### G. 動量 / 流動性

以 **未還原 `close`** 按季序列計算 `pct_change`（季為單位）。

| 變數 | 公式 | 缺失% | inf% |
|------|------|------:|-----:|
| `return_3m` | 相對 1 季前季末 close | (see `feature_stats.csv`) | 0 |
| `return_6m` | 相對 2 季前 | (see `feature_stats.csv`) | (see `feature_stats.csv`) |
| `return_12m` | 相對 4 季前 | (see `feature_stats.csv`) | (see `feature_stats.csv`) |
| `volume_turnover` | `volume / shares_outstanding` | (see `feature_stats.csv`) | 0 |

### H. 同業分位（`quarter × sector`）

`rank(pct=True)`，0–1，越高越好。

| 變數 | 對排名欄位 | 缺失% | inf% |
|------|------------|------:|-----:|
| `sector_rank_revenue_growth` | `revenue_growth_yoy` | (see `feature_stats.csv`) | 0 |
| `sector_rank_eps_growth` | `eps_growth_yoy` | (see `feature_stats.csv`) | 0 |
| `sector_rank_roe` | `roe` | (see `feature_stats.csv`) | 0 |
| `sector_rank_operating_margin` | `operating_margin` | (see `feature_stats.csv`) | 0 |
| `sector_rank_ebit_to_tev` | `ebit_to_tev` | (see `feature_stats.csv`) | 0 |
| `sector_rank_fcf_yield` | `fcf_yield` | (see `feature_stats.csv`) | 0 |
| `sector_rank_return_12m` | `return_12m` | (see `feature_stats.csv`) | 0 |
| `sector_rank_low_debt` | `1 - rank(debt_to_assets)` | 0.00 | 0 |

---

## 5. 缺失率摘要（由高到低；以 `feature_stats.csv` 為準）

### Top missing（由 `feature_stats.csv` 產生）

<!-- AUTO:TOP_MISSING_START -->
| feature | miss_pct | inf_pct |
|---|---:|---:|
| `revenue_growth_yoy` | 38.12% | 0.0000% |
| `sector_rank_revenue_growth` | 38.12% | 0.0000% |
| `gross_margin` | 23.14% | 0.0000% |
| `eps_growth_yoy` | 22.02% | 0.0167% |
| `sector_rank_eps_growth` | 22.02% | 0.0000% |
| `return_12m` | 21.97% | 0.0000% |
| `sector_rank_return_12m` | 21.97% | 0.0000% |
| `fcf_yield` | 19.38% | 0.0000% |
| `sector_rank_fcf_yield` | 19.38% | 0.0000% |
| `ebit_to_tev` | 17.06% | 0.0000% |
| `ev_to_ebit` | 17.06% | 0.0000% |
| `sector_rank_ebit_to_tev` | 17.06% | 0.0000% |
| `earnings_yield` | 16.96% | 0.0000% |
| `net_margin` | 16.96% | 0.0000% |
| `operating_margin` | 16.96% | 0.0000% |
<!-- AUTO:TOP_MISSING_END -->

### 完整來源（唯一事實來源）

以下內容不手動維護，請以 `quant/model/feature_stats.csv` 為準：

- 快速重算：`python scripts/feature_stats.py`
- 讀取欄位：
  - `miss_pct`：缺失率（%）
  - `inf_pct`：inf 比例（%）
  - `p01/p25/p50/p75/p99/min/max`：分位數與極值

（手寫摘要容易跟最新資料/feature 變動不同步，所以改成以 CSV 作為唯一事實來源。）

**缺值主因**：① 需 4 季歷史（YoY、TTM、12m 動量）；② 財報欄位本身缺失（如 `gross_profit`）；③ FCF proxy / TTM 鏈條斷裂。

---

## 6. 資料流

```text
SimFin raw (quant/data_raw/)
  → build_curated（parquet）
  → build_features（PIT + TTM + 比率 + sector rank；**SP500-only universe**）
  → build_labels（y_t, y_next；**SP500-only universe 內排名**）
  → build_model_dataset（inner join，drop y_next 為 NaN）
  → train_model（prepare_feature_matrix（不補值）→ FLAML AutoML）
```

---

## 7. 相關檔案

| 檔案 | 用途 |
|------|------|
| `quant/pipeline/build_features.py` | 特徵公式 |
| `quant/pipeline/build_labels.py` | 標籤公式 |
| `quant/pipeline/train_model.py` | NA/inf 處理與訓練 |
| `quant/pipeline/fetch_simfin.py` | 下載與 EPS/Debt/FCF proxy |
| `quant/model/model_meta.json` | 特徵列表、train/val quarters、AutoML 最佳模型與設定 |
| `quant/model/feature_stats.csv` | 缺失率與分位數 |
| `scripts/verify_y_next.py` | 驗證 `y_next=1` 與下一季排名 |
| `scripts/eval_model.py` | OOS 評估（Top-k 命中率 + per-quarter Spearman） |
| `quant/model/EVAL_REPORT.md` | OOS 評估報告（表格） |

---

<!-- AUTO:LAST_UPDATED_START -->
*最後更新：2026-05-31（SP500-only universe；label 在 SP500 內排名；AutoML + NaN 不補值；OOS 評估報告已產生）*
<!-- AUTO:LAST_UPDATED_END -->
