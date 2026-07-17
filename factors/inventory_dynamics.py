"""
factors/inventory_dynamics.py
存貨動態與庫存重估信號

目的：捕捉「庫存重估獲利」型公司（記憶體模組、面板、鋼鐵、塑化等原物料/
商品持有者）的獲利拐點 —— 這類公司的爆發性獲利來自「低價庫存 × 產品報價
上漲」，是資產負債表（存貨）與毛利率的交互作用，單純看營收趨勢看不到。

以威剛（3260）為例：某季 EPS 暴增，主因是手上低成本 DRAM 庫存遇上報價
暴漲，毛利率急速擴張。財務上可觀察到的前兆：
  1. 存貨部位偏高／建立中（存貨/營收比上升、存貨週轉天數變化）
  2. 毛利率「加速」向上（不只上升，而是季增幅本身在放大）

反向（下行風險）：存貨堆積（週轉天數上升）+ 毛利率下滑 =
手上抱著高價庫存遇到報價下跌 → 提前示警。

全部由 FinMind 財報資料計算，資料不足時回傳中性信號（不影響主流程）。
"""

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 庫存重估信號對 Forward EPS 的最大調整幅度（±20%）
MAX_REVALUATION_IMPACT = 0.20


def _quarterly_series(df: pd.DataFrame, keywords: list) -> Optional[pd.Series]:
    """從 FinMind long-format 財報表，依關鍵字取某科目的季度序列。"""
    if df is None or df.empty or "type" not in df.columns or "value" not in df.columns:
        return None
    for kw in keywords:
        mask = df["type"].str.contains(kw, case=False, na=False)
        if mask.any():
            s = df[mask].groupby("date")["value"].first().sort_index()
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not s.empty:
                return s
    return None


def compute_inventory_dynamics(fetcher, stock_id: str, n_quarters: int = 8) -> dict:
    """
    計算存貨動態與庫存重估信號。

    Returns:
        {
            "dsi_latest": float,          # 最新存貨週轉天數
            "dsi_trend": float,           # 週轉天數近期變化（天，正=變慢/堆積）
            "inv_to_rev_latest": float,   # 存貨/季營收
            "inv_to_rev_trend": float,    # 存貨/營收比變化
            "gm_latest": float,           # 最新毛利率（%）
            "gm_momentum": float,         # 毛利率相對近期基準（百分點）
            "gm_acceleration": float,     # 毛利率季增幅的加速度（百分點）
            "revaluation_score": float,   # -1~+1，正=重估順風、負=庫存風險
            "signal_label": str,
            "reasons": list[str],
            "error": Optional[str],
        }
    """
    result = {
        "dsi_latest": None, "dsi_trend": None,
        "inv_to_rev_latest": None, "inv_to_rev_trend": None,
        "gm_latest": None, "gm_momentum": None, "gm_acceleration": None,
        "revaluation_score": 0.0, "signal_label": "中性", "reasons": [],
        "error": None,
    }

    try:
        from utils.tz import now_tw
        start = (now_tw() - timedelta(days=(n_quarters + 3) * 100)).strftime("%Y-%m-%d")

        bs_df = fetcher._fm_request("TaiwanStockBalanceSheet", stock_id, start)
        is_df = fetcher._fm_request("TaiwanStockFinancialStatements", stock_id, start)

        inv = _quarterly_series(bs_df, ["存貨", "Inventories", "Inventory"])
        rev = _quarterly_series(is_df, ["OperatingRevenue", "Revenue", "營業收入"])
        cogs = _quarterly_series(is_df, ["CostOfGoodsSold", "銷貨成本", "CostOfRevenue", "營業成本"])
        gp = _quarterly_series(is_df, ["GrossProfit", "毛利", "gross_profit"])

        if inv is None or rev is None:
            result["error"] = "存貨或營收資料不足"
            return result

        # 對齊季度
        idx = inv.index
        if cogs is not None:
            idx = idx.intersection(cogs.index)
        idx = idx.intersection(rev.index)
        if len(idx) < 3:
            result["error"] = "可對齊季度不足（需 ≥3 季）"
            return result

        inv = inv[idx].tail(n_quarters)
        rev_q = rev[idx].tail(n_quarters)

        # ── 存貨週轉天數 (DSI) ─────────────────────────────────────────────
        if cogs is not None:
            cogs_q = cogs[idx].tail(n_quarters)
            dsi = (inv / (cogs_q.abs() / 90.0)).replace([np.inf, -np.inf], np.nan).dropna()
            if len(dsi) >= 3:
                result["dsi_latest"] = round(float(dsi.iloc[-1]), 1)
                result["dsi_trend"] = round(float(dsi.iloc[-1] - dsi.iloc[-4:-1].mean()), 1)

        # ── 存貨/營收比 ────────────────────────────────────────────────────
        inv_to_rev = (inv / (rev_q.abs() + 1e-9)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(inv_to_rev) >= 3:
            result["inv_to_rev_latest"] = round(float(inv_to_rev.iloc[-1]), 3)
            result["inv_to_rev_trend"] = round(
                float(inv_to_rev.iloc[-1] - inv_to_rev.iloc[-4:-1].mean()), 3
            )

        # ── 毛利率動能與加速度 ─────────────────────────────────────────────
        gm = None
        if gp is not None:
            gp_q = gp[idx].tail(n_quarters)
            gm = (gp_q / (rev_q.abs() + 1e-9) * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if gm is not None and len(gm) >= 4:
            result["gm_latest"] = round(float(gm.iloc[-1]), 1)
            gm_momentum = float(gm.iloc[-1] - gm.iloc[-4:-1].mean())
            # 加速度：最近一季季增幅 − 前一季季增幅
            d_last = float(gm.iloc[-1] - gm.iloc[-2])
            d_prev = float(gm.iloc[-2] - gm.iloc[-3])
            gm_accel = d_last - d_prev
            result["gm_momentum"] = round(gm_momentum, 2)
            result["gm_acceleration"] = round(gm_accel, 2)
        else:
            gm_momentum, gm_accel = 0.0, 0.0

        # ── 庫存重估信號 ───────────────────────────────────────────────────
        score, reasons = _revaluation_score(
            gm_momentum, gm_accel,
            result["dsi_trend"], result["inv_to_rev_latest"],
            result["inv_to_rev_trend"], inv_to_rev,
        )
        result["revaluation_score"] = round(score, 3)
        result["reasons"] = reasons
        result["signal_label"] = (
            "庫存重估順風" if score >= 0.3 else
            "庫存去化風險" if score <= -0.3 else "中性"
        )

    except Exception as e:
        logger.debug("存貨動態計算失敗 %s：%s", stock_id, e)
        result["error"] = str(e)

    return result


def _revaluation_score(gm_momentum, gm_accel, dsi_trend, inv_to_rev, inv_trend, inv_series):
    """
    組合庫存重估信號（-1~+1）。核心是毛利率拐點，存貨部位放大方向。
    """
    reasons = []
    score = 0.0

    # 1) 毛利率動能（相對近期基準，±5pp → ±0.6）
    gm_part = float(np.clip(gm_momentum / 5.0, -0.6, 0.6))
    score += gm_part
    if gm_momentum >= 2:
        reasons.append(f"毛利率較近期基準 +{gm_momentum:.1f}pp，明顯擴張")
    elif gm_momentum <= -2:
        reasons.append(f"毛利率較近期基準 {gm_momentum:.1f}pp，明顯收縮")

    # 2) 毛利率加速度（拐點訊號，±4pp → ±0.4）
    accel_part = float(np.clip(gm_accel / 4.0, -0.4, 0.4))
    score += accel_part
    if gm_accel >= 1.5:
        reasons.append(f"毛利率季增幅加速（加速度 +{gm_accel:.1f}pp）→ 報價/庫存重估效應")
    elif gm_accel <= -1.5:
        reasons.append(f"毛利率季增幅轉弱（加速度 {gm_accel:.1f}pp）")

    # 3) 存貨部位放大方向：高存貨在順風時提供更多重估動能、逆風時風險更大
    inv_elevated = False
    if inv_series is not None and len(inv_series) >= 4:
        inv_elevated = inv_to_rev is not None and inv_to_rev > float(inv_series.iloc[-4:-1].mean())

    if score > 0 and inv_elevated:
        score *= 1.2
        reasons.append("存貨部位偏高，若報價續揚重估獲利空間更大")
    elif score < 0 and (dsi_trend is not None and dsi_trend > 5):
        score *= 1.2
        reasons.append(f"存貨週轉天數上升 {dsi_trend:.0f} 天，高價庫存去化壓力")

    return float(np.clip(score, -1.0, 1.0)), reasons


def apply_product_price_scenario(
    forward_eps: float, ttm_eps: float, inv_to_rev: Optional[float],
    gm_latest: Optional[float], product_price_chg_pct: float,
) -> dict:
    """
    產品報價情境調整（使用者手動輸入對主力產品報價的看法時）。

    對庫存重估型公司，產品報價變動幾乎直接流入毛利。粗估：
      毛利增量 ≈ 報價變動% × 營收 × 傳導係數
    以存貨/營收比作為「原物料/商品成分」的代理，估算傳導強度。

    Returns:
        {"scenario_eps": float, "eps_delta_pct": float, "note": str}
    """
    if not product_price_chg_pct or ttm_eps is None or ttm_eps == 0:
        return {"scenario_eps": forward_eps, "eps_delta_pct": 0.0, "note": ""}

    # 傳導係數：存貨/營收比越高，商品報價傳導越強（上限 0.6）
    passthrough = 0.3
    if inv_to_rev is not None:
        passthrough = float(min(0.6, max(0.15, inv_to_rev)))

    # 報價變動 → 毛利率變動（百分點）；再換算成 EPS 槓桿
    # 假設營業利益率槓桿 ≈ 毛利變動的 70% 落到稅前
    gm_delta_pp = product_price_chg_pct * passthrough
    # EPS 對毛利率的敏感度：以 TTM EPS 為基礎，毛利率每變動 1pp 約影響
    # 稅後淨利率 ~0.8pp；用 EPS/淨利率概念粗估
    eps_leverage = gm_delta_pp * 0.8 / 100.0  # 轉小數
    scenario_eps = forward_eps * (1 + eps_leverage * 3)  # 放大係數（庫存重估非線性）

    delta_pct = (scenario_eps / forward_eps - 1) * 100 if forward_eps else 0.0
    note = (f"產品報價 {product_price_chg_pct:+.0f}% 情境（傳導係數 {passthrough:.2f}）："
            f"Forward EPS 調整 {delta_pct:+.0f}%")
    return {"scenario_eps": round(scenario_eps, 2),
            "eps_delta_pct": round(delta_pct, 1), "note": note}
