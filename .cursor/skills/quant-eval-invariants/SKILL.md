---
name: quant-eval-invariants
description: Non-negotiable evaluation and backtesting rules for this S&P 500 quant ML project (CV-OOF-only selection, TTM warmup, frozen out-of-sample walk-forward, point-in-time alignment). Use when changing model selection, hyperparameter/weight-scheme search, walk-forward folds, labels, or anything that could leak future information.
disable-model-invocation: true
---

# 量化評估不可違反原則（避免資料洩漏 / 過擬合選型）

改動模型選型、超參搜尋、walk-forward、標籤、回測時，先確認下列規則都成立。

## 1. 選型只看 CV OOF，永遠不看 Final OOS

- 模型 + 權重方案的選擇，**只能**用 CV OOF（樣本內 out-of-fold）結果，指標為
  `ann_sharpe_excess`（年化超額 Sharpe）。
- **Final OOS 嚴禁參與任何選型**，只在凍結後做最終評估。
- 任何「用 OOS 表現挑模型/權重」的程式或結論都是錯的，必須改回 CV OOF。

## 2. TTM 特徵暖機：剔除前 4 季

- `FEATURE_WARMUP_QUARTERS = 4`（`quant/config.py`）。
- YoY 成長率與 TTM 比率需 ≥4 季歷史，前 4 季（2020Q2–2021Q1）特徵幾乎全 NaN，
  **不進 pool search / 訓練**。

## 3. Walk-forward 結構（expanding window）

- Pool search：AutoML 120s 於暖機後、OOS 前的季別 → 取 top-K 模型設定。
- CV OOF：8-fold expanding，前 4 折（預測 2023Q2–2024Q1）用來選型。
- Final OOS：凍結 (模型+config+權重) 後，對最後 4 季（2024Q2–2025Q1）expanding 重訓+預測。
- 每折訓練集 = 「預測季之前所有可用季」，不可洩漏未來。

## 4. Point-in-Time（PIT）對齊，不可前視

- 基本面只取 `publish_date ≤ as_of_date` 的最新一筆季報。
- 回測/評估的歷史季別分數**必須**用 `walk_forward_scores.parquet`（逐折產生），
  **絕不可**用對全期 fit 的 `model.pkl` 分數（`model.pkl` 僅供部署）。

## 5. 標籤定義

- `y_t = 1` 若該季 `excess_return`（季報酬 − 當季 SPY）橫截面排名 ≤ `TOP_N_LABEL`（30）。
- `y_next` = 下一季的 `y_t`，是模型預測目標；不要把信號季與實現季搞混。

## 改完之後

- 重跑 `python scripts/build_all.py --force` 並確認 `model_meta.json` 的
  `cv_oof_quarters` / `final_oos_quarters` / `feature_warmup_excluded` 仍正確。
- 更新 `history.log`。
