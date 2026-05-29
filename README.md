# Interview MVP (Clean Rebuild)

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Build artifacts (raw → curated → features/labels → model → backtest)
```bash
python scripts/build_all.py --force
```

### 3) Run backend
```bash
python -m uvicorn backend.main:app --reload --port 8001
```

### 4) Run frontend
```bash
cd frontend
npm install
npm run dev
```

## 1 分鐘面試 Demo

開兩個終端機：

### 終端 A：一鍵建置 + 後端
```bash
python scripts/build_all.py --force
python -m uvicorn backend.main:app --reload --port 8001
```

### 終端 B：前端
```bash
cd frontend
npm run dev
```

打開前端後，你會看到：
- 最近 10 個季度回測曲線點（strategy vs benchmark）
- 可選季度的 Top holdings

Chat API（離線版）可測：
```bash
curl -X POST http://localhost:8001/api/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"模型用什麼?\"}]}"
```

## Data contract (MVP)

Raw files live under `quant/data_raw/` (or can be symlinked there).

Minimum required:
- `prices_daily.csv`: `Date,symbol,close,volume` (daily)
- `constituents.csv`: `symbol,company_name,sector,industry`
- Quarterly fundamentals (income + balance):
  - `symbol,report_date,publish_date,revenue,gross_profit,operating_income,net_income,eps_diluted,shares_basic,shares_diluted`
  - `symbol,report_date,publish_date,total_assets,total_liabilities,total_equity,cash_and_equivalents,total_debt`

## Reproducibility
- Pipeline outputs are written under `quant/*/` folders (no database).
- Each step writes a `*_meta.json` with row counts + source hashes.
- **模型變數、計算公式、NA 處理與缺失率**：見 [`quant/model/MODEL_VARIABLES.md`](quant/model/MODEL_VARIABLES.md)

## 真實資料（SimFin）下載
1. 到 SimFin 註冊取得免費 API key（SimFin 帳號頁面可找到 API key）。
2. 在專案根目錄建立 `.env`：
```
SIMFIN_API_KEY=你的key
```
3. 重新跑：
```bash
python scripts/build_all.py --force
```
若沒有設定 `SIMFIN_API_KEY`，會自動改用可重現的 sample data。

### Cashflow 欄位（SimFin 無顯式 CapEx/FCF 時的 proxy）

部分 SimFin `us-cashflow-quarterly` 匯出**沒有** `Capital Expenditures`、`Free Cash Flow` 欄位。此時下載腳本會用同一季的 **`Change in Fixed Assets & Intangibles`**（`change_in_fixed_assets`）做近似：

- **`capital_expenditure`** = `max(0, -change_in_fixed_assets)`（把「購置固定資產等」常見的負向現金流轉成正值，表示估算的資本支出金額）
- **`free_cash_flow`** = `operating_cash_flow + change_in_fixed_assets`（與「營運現金流減去資本支出」在單一投資科目假設下一致）

若之後資料集出現官方 `Capital Expenditures` / `Free Cash Flow`，可再改為優先使用官方欄位。

# 投資組合分析 Dashboard

S&P 500 飆股預測、季度再平衡回測、Fama-French 因子分析與 AI 問答 Dashboard。

## 功能

- **數據 Agent**：依規格書建構基本面特徵 mart（TTM、成長、估值、產業排名），point-in-time 對齊
- **飆股預測**：FLAML AutoML（19 特徵）預測下一季飆股分數 `y_next`
- **回測**：月度 Top 30 逆波動加權；基準為 **SPY**（S&P 500 市值加權指數 ETF，與 labels 相同之月/季初末 close 定義）
- **因子分析**：FF5 + Momentum OLS 回歸
- **Chatbot**：LLM 問答區間表現、持股、方法論

## 快速開始

```bash
pip install -r requirements.txt
cd frontend && npm install

# 一鍵建置：數據 Agent → 驗證 → AutoML → 回測
python scripts/build_artifacts.py --force --time-budget 120

# 後端
python -m uvicorn backend.main:app --reload --port 8001

# 前端
cd frontend; npm run dev
```

開啟 http://localhost:5173

## 數據 Agent 流程

對應 [fundamental_stock_prediction_data_agent_spec.md](fundamental_stock_prediction_data_agent_spec.md)：

```text
universe → prices → fundamentals → factors
→ feature_mart (PIT: publish_date <= as_of_date)
→ labels → validation → modeling_dataset
→ FLAML AutoML → backtest
```

配置：[quant/config.yaml](quant/config.yaml)

## 專案結構

```
quant/
  config.yaml
  data_agent/     # Task 1-8 agents
  pipelines/      # 編排腳本
  pipeline/       # 相容層
  modeling/       # FLAML train/predict
  backtest/
  cache/          # Parquet + model 產物
backend/
frontend/
tests/
scripts/build_artifacts.py
```

## API

| Endpoint | 說明 |
|----------|------|
| GET /api/backtest/curve | 策略 vs 基準曲線 |
| GET /api/backtest/metrics | 區間指標 |
| GET /api/holdings/{quarter} | Top 10 持股 |
| GET /api/factor-analysis | 因子回歸 |
| GET /api/model/features | AutoML 特徵與 importance |
| POST /api/chat | AI 問答 |

## 標籤與特徵

**y_t** = 1 若當季超額報酬在 S&P500 內排名前 30；組合 Top10 逆波動加權（見 `build_labels.py`、`monthly_portfolio.py`）

**x_t**（19 個 AutoML 變數）：成長、獲利、估值、槓桿、產業相對排名 — 見 `quant/config.py` 的 `MODEL_FEATURE_COLS`

## 驗證

```bash
python tests/test_no_lookahead.py
```

產出 `quant/cache/data_validation_report.json`
