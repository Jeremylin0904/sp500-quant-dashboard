---
name: update-model-variables-doc
description: Regenerates quant/model/MODEL_VARIABLES.md from the latest model artifacts (model_meta.json + feature_stats.csv), ensuring the embedded Top missing table and last-updated date stay in sync. Use when the user asks to update MODEL_VARIABLES.md, feature missingness tables, or wants a repeatable documentation refresh.
disable-model-invocation: true
---

# Update MODEL_VARIABLES.md（可重複執行的文件更新 skill）

## 目標

把 `quant/model/MODEL_VARIABLES.md` 的「Top missing 表格」與「最後更新日期」**自動化更新**，避免手動維護造成 drift。

## 產物與資料來源（Source of Truth）

- `quant/model/feature_stats.csv`：缺失率/inf% 的唯一事實來源（由 `scripts/feature_stats.py` 生成）
- `quant/model/model_meta.json`：模型實際使用的 `feature_cols`
- `quant/model/MODEL_VARIABLES.md`：文件本體（含 AUTO markers）

## 執行流程（照做就好）

1. 先重算特徵統計（會覆寫 `feature_stats.csv`）：

```bash
python scripts/feature_stats.py
```

2. 更新文件（會覆寫 `MODEL_VARIABLES.md` 的 AUTO 區塊）：

```bash
python scripts/update_model_variables_doc.py
```

3.（可選）如果你也想同步更新 OOS 評估報告：

```bash
python scripts/eval_model.py
```

4. 依專案規則，任何「生成/更新程式碼或文件」後都要更新 `history.log`：
   - 新增一個版本段落（例如 `[vX.Y] ... - YYYY-MM-DD`）
   - 用 1–3 行 bullet 紀錄本次差異（哪些檔案被更新、更新內容是什麼）

## AUTO 區塊標記（不要手改區塊內容）

`MODEL_VARIABLES.md` 會用以下 markers 讓腳本精準替換內容：

- `<!-- AUTO:TOP_MISSING_START -->` ... `<!-- AUTO:TOP_MISSING_END -->`
- `<!-- AUTO:LAST_UPDATED_START -->` ... `<!-- AUTO:LAST_UPDATED_END -->`

## 驗證（快速 smoke check）

- `MODEL_VARIABLES.md` 的 Top missing 表格數字應與 `feature_stats.csv` 一致
- 表格列出的 feature 必須是 `model_meta.json` 的 `feature_cols` 子集（避免把未入模欄位寫進文件）

