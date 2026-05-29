# Expanding Walk-Forward 評估（Top-30 超額報酬標籤）
- **Hyperparam pool**：Q1–Q16 AutoML 120s → top-5 configs
- **CV OOF**：每個 config 固定 hyperparam 跑 8-fold expanding OOF，取 **mean fold log_loss** 最佳
- **Final OOS**：凍結 hyperparam（xgboost，pool rank #3），Q17–Q20 expanding retrain
- **全部預測季**：2023Q2, 2023Q3, 2023Q4, 2024Q1, 2024Q2, 2024Q3, 2024Q4, 2025Q1
- **CV OOF 季**：2023Q2, 2023Q3, 2023Q4, 2024Q1
- **Final OOS 季**：2024Q2, 2024Q3, 2024Q4, 2025Q1
- **Final OOS 列數**：1317
- **Final OOS mean Spearman**：0.1750

## Hyperparam pool — 8-fold OOF 平均 log_loss

| rank | estimator | mean fold log_loss | pooled OOF AUC |
|---:|---|---:|---:|
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 1 | xgb_limitdepth | 0.4249 | 0.714 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 2 | lgbm | 0.4926 | 0.690 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 5 | lgbm | 0.5066 | 0.682 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 3 | xgboost **← selected** | 0.5366 | 0.678 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |
| 4 | xgboost | 0.5427 | 0.673 |

## Final OOS — Confusion matrix（Top30 / 季）

### Pred：每季分數 **Top 30**

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| **Actual 0** | 1141 | 97 |
| **Actual 1** | 56 | 23 |

- Accuracy 88.38% · Precision 19.17% · Recall 29.11% · F1 23.12% · Specificity 92.16%

## 每季 Spearman / Top-k（Final OOS）

| quarter | n | n_pos | spearman | P@10 | R@10 |
|---|---:|---:|---:|---:|---:|
| 2024Q2 | 324 | 20 | 0.1810 | 20.00% | 10.00% |
| 2024Q3 | 329 | 20 | 0.2629 | 40.00% | 20.00% |
| 2024Q4 | 331 | 13 | -0.0728 | 0.00% | 0.00% |
| 2025Q1 | 333 | 26 | 0.3289 | 60.00% | 23.08% |
