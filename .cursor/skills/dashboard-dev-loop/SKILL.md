---
name: dashboard-dev-loop
description: Rebuild quant artifacts, restart the FastAPI backend on port 8001, and verify endpoints on Windows/PowerShell for this S&P 500 quant dashboard. Use when the user asks to rerun the pipeline, restart the backend, regenerate backtest data, or verify that APIs/frontend still work after changes.
disable-model-invocation: true
---

# Dashboard 開發回圈（重建 → 重啟 → 驗證）

本專案在 Windows + PowerShell 上開發。後端 FastAPI 跑在 **port 8001**，前端 Vite/React。
以下是每次改完程式碼後反覆執行的標準流程。

## 重要環境慣例

- 一律先設 `PYTHONPATH`：`$env:PYTHONPATH="d:\interview2"`
- **驗證 API 不要用 `curl`**（在 PowerShell 會卡住）；改用 `python` + `urllib.request`。
- 後端非 `--reload` 模式時，改完程式碼一定要重啟才會生效。
- 任何生成/更新後，依專案規則更新 `history.log`（見最後一節）。

## 1. 重建資料產物（依改動範圍擇一）

只改回測/權重/組合邏輯（不重訓模型）→ 只重跑回測：

```bash
$env:PYTHONPATH="d:\interview2"; python -c "from quant.pipeline.backtest import run_backtest; run_backtest(force=True)"
```

改到特徵/標籤/模型/超參選擇 → 全量重建（含 train + backtest + validate）：

```bash
$env:PYTHONPATH="d:\interview2"; python scripts/build_all.py --force
```

其他常用單步：

```bash
python scripts/factor_analysis.py   # 重算因子分析 (quant/factor/factor_analysis.json)
python scripts/feature_stats.py     # 重算 feature_stats.csv
python scripts/eval_model.py        # OOS 評估報告
```

## 2. 重啟後端（先殺掉佔用 8001 的舊行程）

```bash
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Start-Sleep -Seconds 2; $env:PYTHONPATH="d:\interview2"; python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

用背景方式啟動（block_until_ms=0），再等 `Uvicorn running` 字樣確認啟動成功。
舊行程被殺時出現的 "Restart backend" 錯誤通知是預期清理，可忽略。

## 3. 驗證端點（用 python，不要用 curl）

```bash
python -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://127.0.0.1:8001/api/model/summary',timeout=10).read()); print(list(d)[:8])"
```

注意：在 `python -c` 字串內**避免使用反引號**（PowerShell 會當逃脫字元而報錯）。

## 4. 前端型別檢查 / 建置

```bash
cd frontend; npx tsc --noEmit -p tsconfig.app.json   # 只型別檢查
cd frontend; npm run build                            # 完整 build (tsc + vite)
```

## 5. 更新 history.log（專案規則，務必執行）

- 新增一個版本段落：`[vX.Y] 一句話摘要 - YYYY-MM-DD`
- 用 1–3 行 bullet 紀錄差異（哪些檔案、改了什麼、影響哪個分頁/端點）
