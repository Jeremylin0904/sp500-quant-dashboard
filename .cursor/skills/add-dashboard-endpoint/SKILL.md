---
name: add-dashboard-endpoint
description: Add a new backend data endpoint (FastAPI) and wire it into the React dashboard for this S&P 500 quant project. Use when the user asks to expose new model/backtest/factor data via an API and show it on a new tab, panel, chart, or table in the frontend.
disable-model-invocation: true
---

# 新增 dashboard 資料端點（後端 → 前端串接）

從「JSON/CSV 產物」到「前端分頁/面板」的固定模式。每個新資料來源都照這條鏈做。

## 架構分層

- 產物：`quant/backtest/*.json`、`quant/model/*`、`quant/factor/*`
- 載入：`backend/services/data_service.py`（`@lru_cache` loader + `clear_cache()`）
- 路由：`backend/routers/<area>.py`（`APIRouter(prefix="/api/<area>")`）
- 註冊：`backend/main.py`（`app.include_router(...)`）
- 前端：`frontend/src/App.tsx`（`fetchJson`）+ 視圖元件（`frontend/src/components/*.tsx`）

## 後端步驟

1. 在 `data_service.py` 加 loader（小檔直接讀；需容忍 NaN/Infinity 用 `_load_json_lenient`）：

```python
@lru_cache(maxsize=1)
def get_my_thing() -> dict:
    path = BACKTEST_DIR / "my_thing.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
```

2. **務必**把新 loader 加進 `clear_cache()`（否則重建後 API 仍回舊資料）。
3. 若要彙整多來源，仿照 `build_model_summary()` / `build_model_variables()` 包一層 `@lru_cache` builder。
4. 在對應 router 加 endpoint；新 area 要在 `main.py` `include_router`。

```python
@router.get("/my-thing")
def my_thing():
    return get_my_thing()
```

## 前端步驟

1. 在 `App.tsx` 定義型別 + `useState`，在載入區用 `fetchJson<T>("/api/...")` 取資料。
2. 大型分頁拆成獨立元件（見 `MethodologyView.tsx`、`VariablesView.tsx`、`ScatterChart.tsx`）。
3. 新分頁：擴充 `type Tab`、`navItems`、`tabTitle` 對照表，並用 `{tab === "x" && <XView .../>}` 渲染。
4. 樣式加到 `frontend/src/App.css`（沿用 CSS 變數：`--accent`、`--panel`、`--muted`…，淺色主題）。

## 驗證

1. 重啟後端並用 python urllib 打新端點（見 `dashboard-dev-loop` skill）。
2. `cd frontend; npx tsc --noEmit -p tsconfig.app.json` 要過。
3. 更新 `history.log`（版本段落 + bullet 差異）。

## 常見坑

- 忘了加 `clear_cache()` → 改資料後 API 不更新。
- 在 PowerShell 用 `curl` 驗證會卡住，改用 python urllib。
- `python -c` 字串內不要用反引號。
