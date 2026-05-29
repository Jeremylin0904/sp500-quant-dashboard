from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.data_service import (
    build_model_summary,
    get_backtest_monthly,
    get_holdings_by_month,
    get_performance_report,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("")
def chat(req: ChatRequest):
    last = req.messages[-1].content if req.messages else ""
    summary = build_model_summary()
    perf = get_performance_report()
    monthly = get_backtest_monthly()
    holdings = get_holdings_by_month()

    oos = perf.get("out_of_sample") or {}
    oos_eval = summary.get("oos_eval") or {}
    loss = summary.get("loss") or {}
    port = summary.get("portfolio") or {}

    text = (
        "離線 MVP chatbot。可問：模型、回測、持股、方法論。"
        f" 目前：{loss.get('name', 'weighted_bce')} · 基準 {port.get('benchmark_symbol', 'SPY')} · "
        f"Top{port.get('top_n_holdings', 30)} 月度調倉。"
    )

    if "模型" in last or "feature" in last.lower() or "loss" in last.lower():
        text = (
            f"模型：{summary.get('best_estimator')}（{summary.get('task')}）。"
            f" Loss：{loss.get('name')}（{loss.get('description', '')}）。"
            f" OOS Spearman={oos_eval.get('spearman_mean')}, "
            f"Top30 P/R={oos_eval.get('top30_precision')}/{oos_eval.get('top30_recall')}, "
            f"OOF AUC={summary.get('cv', {}).get('oof_roc_auc')}。"
        )
    elif "回測" in last or "curve" in last.lower() or "績效" in last:
        n_oos = len(monthly.get("monthly_out_of_sample") or [])
        text = (
            f"月度回測 vs {port.get('benchmark_symbol')}："
            f"OOS {n_oos} 個月，累積超額（幾何）{oos.get('total_excess_geometric', 0):.1%}，"
            f"最大回撤 {oos.get('max_drawdown', 0):.1%}。"
        )
    elif "持股" in last or "hold" in last.lower():
        ms = sorted(holdings.keys())
        text = f"持股月份（最近 5）：{', '.join(ms[-5:]) if ms else '（無）'}。"
    elif "方法" in last or "方法論" in last:
        text = (
            "流程：季末 PIT 特徵 → 預測下季超額 Top30（y_next）→ weighted BCE 分類 → "
            "每月末用最近完成信號季選 Top30、逆波動加權 → 持有下一日曆月；"
            f"基準為 SPY 市值加權。"
        )

    return {"message": text}
