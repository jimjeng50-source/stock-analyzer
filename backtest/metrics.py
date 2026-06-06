"""回測績效指標計算。"""

import numpy as np
import pandas as pd


def calc_metrics(bt_result: dict, initial_capital: float) -> dict:
    """
    輸入 run_backtest 回傳的 result dict，計算完整績效指標。
    """
    equity_df = bt_result.get("equity_df", pd.DataFrame())
    trades    = bt_result.get("trades", [])
    final     = bt_result.get("final_capital", initial_capital)

    metrics = {
        "total_return":     0.0,
        "annual_return":    0.0,
        "max_drawdown":     0.0,
        "sharpe_ratio":     0.0,
        "win_rate":         0.0,
        "profit_factor":    0.0,
        "avg_holding_days": 0.0,
        "trade_count":      0,
        "bh_return":        0.0,
        "excess_return":    0.0,
        "avg_win":          0.0,
        "avg_loss":         0.0,
    }

    if equity_df.empty:
        return metrics

    # ── 報酬率 ────────────────────────────────────────────
    total_ret = (final - initial_capital) / initial_capital * 100
    metrics["total_return"] = round(total_ret, 2)

    # 年化報酬（CAGR）
    n_days = len(equity_df)
    if n_days > 1:
        ann_ret = ((1 + total_ret / 100) ** (365 / n_days) - 1) * 100
        metrics["annual_return"] = round(ann_ret, 2)

    # ── 最大回撤 ──────────────────────────────────────────
    eq = equity_df["strategy"].astype(float)
    peak = eq.expanding().max()
    dd   = (eq - peak) / peak * 100
    metrics["max_drawdown"] = round(float(dd.min()), 2)

    # ── Sharpe Ratio（年化，無風險利率 2%） ───────────────
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() * 252 - 0.02) / (daily_ret.std() * np.sqrt(252))
        metrics["sharpe_ratio"] = round(float(sharpe), 2)

    # ── 交易統計 ──────────────────────────────────────────
    sell_trades = [t for t in trades if "pnl" in t and t["pnl"] is not None]
    metrics["trade_count"] = len([t for t in trades if t["action"] == "買進"])

    if sell_trades:
        pnls = [t["pnl"] for t in sell_trades]
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p <= 0]

        metrics["win_rate"] = round(len(wins) / len(pnls) * 100, 1)
        metrics["avg_win"]  = round(np.mean(wins), 0) if wins else 0
        metrics["avg_loss"] = round(np.mean(loss), 0) if loss else 0

        total_profit = sum(wins)
        total_loss   = abs(sum(loss))
        metrics["profit_factor"] = round(total_profit / (total_loss + 1e-9), 2)

        hold_days = [t["holding_days"] for t in sell_trades if t["holding_days"]]
        metrics["avg_holding_days"] = round(np.mean(hold_days), 1) if hold_days else 0

    # ── Buy & Hold 比較 ───────────────────────────────────
    if "buyhold" in equity_df.columns:
        bh_final = float(equity_df["buyhold"].iloc[-1])
        bh_ret   = (bh_final - initial_capital) / initial_capital * 100
        metrics["bh_return"]     = round(bh_ret, 2)
        metrics["excess_return"] = round(total_ret - bh_ret, 2)

    return metrics
