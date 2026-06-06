"""回測結果視覺化：Plotly 深色主題圖表。"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _dark(fig: go.Figure, height: int = 400, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def equity_curve_chart(bt_result: dict) -> go.Figure:
    """淨值曲線圖（策略 vs Buy & Hold），含買賣標記與最大回撤區間。"""
    equity_df = bt_result.get("equity_df", pd.DataFrame())
    trades     = bt_result.get("trades", [])

    if equity_df.empty:
        return go.Figure()

    dates = equity_df["date"]
    eq    = equity_df["strategy"].astype(float)

    fig = go.Figure()

    # 策略淨值
    fig.add_trace(go.Scatter(x=dates, y=eq, name="評分策略",
                             line=dict(color="#00d4ff", width=2)))

    # Buy & Hold
    if "buyhold" in equity_df.columns:
        fig.add_trace(go.Scatter(x=dates, y=equity_df["buyhold"].astype(float),
                                 name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dot")))

    # 最大回撤陰影
    peak = eq.expanding().max()
    dd   = (eq - peak) / peak * 100
    dd_min_idx = dd.idxmin()
    if not pd.isna(dd_min_idx):
        peak_idx = eq[:dd_min_idx + 1].idxmax()
        fig.add_vrect(
            x0=dates.iloc[peak_idx], x1=dates.iloc[dd_min_idx],
            fillcolor="rgba(255,75,75,0.15)", line_width=0,
            annotation_text=f"最大回撤 {dd.min():.1f}%",
            annotation_position="top left",
        )

    # 買進標記
    buys  = [t for t in trades if t["action"] == "買進"]
    sells = [t for t in trades if "賣出" in t["action"] and t.get("pnl") is not None]

    if buys:
        buy_dates = [t["date"] for t in buys]
        buy_eq    = [float(equity_df[equity_df["date"] == pd.to_datetime(d)]["strategy"].values[0])
                     if pd.to_datetime(d) in equity_df["date"].values else None for d in buy_dates]
        valid = [(d, v) for d, v in zip(buy_dates, buy_eq) if v is not None]
        if valid:
            vd, vv = zip(*valid)
            fig.add_trace(go.Scatter(x=list(vd), y=list(vv), mode="markers",
                                     marker=dict(symbol="triangle-up", size=10, color="#6bcb77"),
                                     name="買進"))

    if sells:
        sell_dates = [t["date"] for t in sells]
        sell_eq    = [float(equity_df[equity_df["date"] == pd.to_datetime(d)]["strategy"].values[0])
                      if pd.to_datetime(d) in equity_df["date"].values else None for d in sell_dates]
        valid = [(d, v) for d, v in zip(sell_dates, sell_eq) if v is not None]
        if valid:
            vd, vv = zip(*valid)
            fig.add_trace(go.Scatter(x=list(vd), y=list(vv), mode="markers",
                                     marker=dict(symbol="triangle-down", size=10, color="#ff6b6b"),
                                     name="賣出"))

    return _dark(fig, height=450, title="策略淨值曲線 vs Buy & Hold")


def monthly_return_heatmap(equity_df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """月份報酬率熱圖。"""
    if equity_df.empty:
        return go.Figure()

    eq = equity_df.set_index("date")["strategy"].astype(float)
    eq.index = pd.to_datetime(eq.index)
    monthly = eq.resample("ME").last().pct_change() * 100
    monthly = monthly.dropna()

    if monthly.empty:
        return go.Figure()

    monthly_df = monthly.reset_index()
    monthly_df.columns = ["date", "ret"]
    monthly_df["year"]  = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.month

    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]

    z    = pivot.values
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(pivot.columns), y=[str(y) for y in pivot.index],
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="報酬率%"),
    ))
    return _dark(fig, height=max(200, len(pivot) * 40 + 80), title="月份報酬率熱圖")


def score_distribution_hist(trades: list) -> go.Figure:
    """買進/賣出時的評分分佈直方圖。"""
    buy_scores  = [t["score"] for t in trades if t["action"] == "買進" and t.get("score")]
    sell_scores = [t["score"] for t in trades if "賣出" in t.get("action","") and t.get("score")]

    fig = go.Figure()
    if buy_scores:
        fig.add_trace(go.Histogram(x=buy_scores, name="買進時評分",
                                   marker_color="#6bcb77", opacity=0.75, nbinsx=20))
    if sell_scores:
        fig.add_trace(go.Histogram(x=sell_scores, name="賣出時評分",
                                   marker_color="#ff6b6b", opacity=0.75, nbinsx=20))

    fig.add_vline(x=65, line_dash="dash", line_color="#ffd93d", annotation_text="買進門檻")
    fig.add_vline(x=45, line_dash="dash", line_color="#ff9f43", annotation_text="賣出門檻")
    fig.update_layout(barmode="overlay")
    return _dark(fig, height=320, title="交易時評分分佈")


def pnl_waterfall(trades: list) -> go.Figure:
    """每筆交易損益瀑布圖。"""
    sell_trades = [t for t in trades if "賣出" in t.get("action","") and t.get("pnl") is not None]
    if not sell_trades:
        return go.Figure()

    pnls   = [t["pnl"] for t in sell_trades]
    labels = [t["date"] for t in sell_trades]
    colors = ["#6bcb77" if p > 0 else "#ff6b6b" for p in pnls]

    fig = go.Figure(go.Bar(x=labels, y=pnls, marker_color=colors,
                           text=[f"{p:,.0f}" for p in pnls], textposition="outside"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    return _dark(fig, height=320, title="每筆交易損益（元）")


def build_charts(bt_result: dict, initial_capital: float) -> dict:
    """產生所有回測圖表，回傳 dict of go.Figure。"""
    equity_df = bt_result.get("equity_df", pd.DataFrame())
    trades    = bt_result.get("trades", [])
    return {
        "equity":    equity_curve_chart(bt_result),
        "monthly":   monthly_return_heatmap(equity_df, initial_capital),
        "score_dist": score_distribution_hist(trades),
        "pnl":       pnl_waterfall(trades),
    }
