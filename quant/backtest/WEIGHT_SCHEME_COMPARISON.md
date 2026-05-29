# 權重方案比較（Top30 · walk-forward 分數選股）
同一套 `walk_forward_scores`；僅改權重公式。Sharpe 以**月頻超額報酬**年化。
**選型依據：僅看 CV OOF（Q9–Q16）**；Final OOS 僅供事後監控，不可回頭用來挑方案。
## CV OOF（Q9–Q16）— 選型主表 · 按超額 Sharpe 排序
| rank | scheme | 累積超額 | 年化超額 | 策略 Sharpe | 超額 Sharpe | MDD |
|---:|---|---:|---:|---:|---:|---:|
| 1 | rank 線性 decay | 113.16% | 54.02% | 1.71 | 1.72 | — |
| 2 | 等權 Top30 | 70.78% | 36.03% | 1.57 | 1.60 | — |
| 3 | score/σ normalize | 48.93% | 25.88% | 1.46 | 1.52 | — |
| 4 | 1/σ normalize（baseline） ← baseline | 46.32% | 24.62% | 1.44 | 1.48 | — |
| 5 | softmax(score/(T·σ)) T=0.10 | 9.87% | 5.66% | 0.89 | 0.33 | — |
| 6 | softmax(score/(T·σ)) T=0.20 | 6.77% | 3.91% | 0.84 | 0.26 | — |
| 7 | softmax(score/(T·σ)) T=0.05 | — | — | — | — | — |

### CV OOF · 按累積超額排序（Sharpe 之外參考）
| rank | scheme | 累積超額 | 超額 Sharpe |
|---:|---|---:|---:|
| 1 | rank 線性 decay | 113.16% | 1.72 |
| 2 | 等權 Top30 | 70.78% | 1.60 |
| 3 | score/σ normalize | 48.93% | 1.52 |
| 4 | 1/σ normalize（baseline） | 46.32% | 1.48 |
| 5 | softmax(score/(T·σ)) T=0.10 | 9.87% | 0.33 |
| 6 | softmax(score/(T·σ)) T=0.20 | 6.77% | 0.26 |
| 7 | softmax(score/(T·σ)) T=0.05 | — | — |

## Final OOS（Q17–Q20）— 僅監控 · 不作選型
| scheme | 累積超額 | 超額 Sharpe | OOF rank（Sharpe） |
|---|---:|---:|---:|
| 等權 Top30 | 29.97% | 1.53 | 2 |
| 1/σ normalize（baseline） | 26.72% | 1.45 | 4 |
| rank 線性 decay | 34.18% | 1.51 | 1 |
| score/σ normalize | 26.90% | 1.46 | 3 |
| softmax(score/(T·σ)) T=0.05 | — | — | 7 |
| softmax(score/(T·σ)) T=0.10 | -15.01% | -0.35 | 5 |
| softmax(score/(T·σ)) T=0.20 | -14.02% | -0.34 | 6 |

## 結論（僅依 CV OOF）
- **推薦方案（超額 Sharpe）**：rank 線性 decay （OOF 超額 Sharpe 1.72，vs baseline 1.48，+0.24）
- **累積超額最高（OOF）**：rank 線性 decay （113.16%），但 Sharpe 排 #1 — 波動較高
- **score/σ**：OOF Sharpe 1.52，略優 baseline，改善有限
- **inv_vol baseline**：OOF 超額 Sharpe 垫底（1.48）；引入分數權重整體優於純 inv-vol
- Final OOS 表僅供事後對照；**不可**用 OOS 結果回頭改選方案。
- `回歸超額報酬` 需另訓練 regression 目標，未含在此表。
