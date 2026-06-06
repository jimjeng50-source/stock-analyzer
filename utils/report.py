import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _radar_fig(category_scores: dict) -> go.Figure:
    labels_map = {
        "chips": "籌碼面", "fundamental": "基本面",
        "technical": "技術面", "momentum": "動能面", "risk": "風險面",
    }
    cats = list(labels_map.keys())
    labels = [labels_map[c] for c in cats]
    values = [category_scores.get(c, 0) for c in cats]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill="toself", line=dict(color="#00d4ff"),
        fillcolor="rgba(0,212,255,0.2)",
    ))
    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(range=[0, 100])),
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
    )
    return fig


def _bar_fig(category_scores: dict) -> go.Figure:
    labels_map = {
        "chips": "籌碼面", "fundamental": "基本面",
        "technical": "技術面", "momentum": "動能面", "risk": "風險面",
    }
    cats = list(labels_map.keys())
    labels = [labels_map[c] for c in cats]
    values = [category_scores.get(c, 0) for c in cats]
    colors = ["#ff6b6b" if v < 40 else "#ffd93d" if v < 65 else "#6bcb77" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", xaxis=dict(range=[0, 110]),
        margin=dict(l=10, r=40, t=10, b=10), height=250,
    )
    return fig


def save_html_report(
    stock_id: str,
    score_result: dict,
    price_df: pd.DataFrame,
    institutional_df: pd.DataFrame,
    revenue_df: pd.DataFrame,
    ai_advice: str = "",
    output_dir: str = "output",
) -> str:
    """產生 HTML 報告並儲存至 output_dir，回傳儲存路徑。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{stock_id}_{timestamp}.html")

    total = score_result.get("total_score", 0)
    rec = score_result.get("recommendation", "")
    cat = score_result.get("category_scores", {})
    raw = score_result.get("raw_factors", {})

    # 雷達圖與長條圖
    radar_html = pio.to_html(_radar_fig(cat), full_html=False, include_plotlyjs="cdn")
    bar_html = pio.to_html(_bar_fig(cat), full_html=False, include_plotlyjs=False)

    # K 線圖
    kline_html = ""
    if not price_df.empty and all(c in price_df.columns for c in ["open", "high", "low", "close"]):
        fig = go.Figure(go.Candlestick(
            x=price_df["date"], open=price_df["open"],
            high=price_df["high"], low=price_df["low"], close=price_df["close"],
            name="K線",
        ))
        fig.update_layout(template="plotly_dark", title="股價走勢", height=350, xaxis_rangeslider_visible=False)
        kline_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # 原始因子表格
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4g}</td></tr>"
        for k, v in sorted(raw.items())
    )
    factors_table = f"<table border='1' cellpadding='6' style='border-collapse:collapse;width:100%'><thead><tr><th>因子</th><th>數值</th></tr></thead><tbody>{rows}</tbody></table>"

    ai_section = ""
    if ai_advice:
        ai_section = f"<h2>AI 投資建議</h2><pre style='white-space:pre-wrap;background:#1e1e2e;padding:16px;border-radius:8px'>{ai_advice}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>台股分析報告 - {stock_id}</title>
<style>
  body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;padding:24px}}
  h1{{color:#58a6ff}} h2{{color:#79c0ff;margin-top:32px}}
  .score-box{{display:inline-block;font-size:3rem;font-weight:bold;color:#ffd93d;margin-right:16px}}
  .rec-box{{display:inline-block;font-size:1.4rem;padding:8px 16px;background:#161b22;border-radius:8px}}
  table td,table th{{text-align:left;padding:6px 12px}}
  table tr:nth-child(even){{background:#161b22}}
</style>
</head>
<body>
<h1>📊 台股評分報告 — {stock_id}</h1>
<p>產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div><span class="score-box">{total:.1f}</span><span class="rec-box">{rec}</span></div>
<h2>各面向評分</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap">
  <div style="flex:1;min-width:300px">{radar_html}</div>
  <div style="flex:1;min-width:300px">{bar_html}</div>
</div>
{f'<h2>股價走勢</h2>{kline_html}' if kline_html else ''}
<h2>原始因子數值</h2>
{factors_table}
{ai_section}
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    return filename
