import numpy as np
import pandas as pd


def _pct_change(a: float, b: float) -> float:
    """計算百分比變化，除數為 0 時回傳 0。"""
    if b == 0 or np.isnan(b):
        return 0.0
    return (a - b) / abs(b) * 100


def compute_fundamental(
    revenue_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    current_price: float,
) -> dict:
    """
    計算基本面因子（原始數值，未標準化）。

    回傳 dict 包含：
        rev_yoy, rev_mom, rev_3m_trend, rev_12m_high,
        eps_latest, eps_qoq, eps_yoy,
        gross_margin, gpm_trend, pe_ratio
    """
    result = {
        "rev_yoy": 0.0, "rev_mom": 0.0, "rev_3m_trend": 0,
        "rev_12m_high": 0, "eps_latest": 0.0, "eps_qoq": 0.0,
        "eps_yoy": 0.0, "gross_margin": 0.0, "gpm_trend": 0.0,
        "pe_ratio": 0.0,
    }

    # ── 月營收 ──────────────────────────────────────────
    if not revenue_df.empty and "revenue" in revenue_df.columns:
        rev = revenue_df.copy().sort_values("date").dropna(subset=["revenue"])
        rev["revenue"] = pd.to_numeric(rev["revenue"], errors="coerce")
        rev = rev.dropna(subset=["revenue"])

        if len(rev) >= 2:
            latest_rev = float(rev["revenue"].iloc[-1])
            prev_month_rev = float(rev["revenue"].iloc[-2])
            result["rev_mom"] = round(_pct_change(latest_rev, prev_month_rev), 2)

        if len(rev) >= 13:
            yoy_base = float(rev["revenue"].iloc[-13])
            latest_rev = float(rev["revenue"].iloc[-1])
            result["rev_yoy"] = round(_pct_change(latest_rev, yoy_base), 2)

        # 近 3 個月趨勢（每月均較前月成長 = 1，否則 -1）
        if len(rev) >= 4:
            last4 = rev["revenue"].values[-4:]
            if last4[1] > last4[0] and last4[2] > last4[1] and last4[3] > last4[2]:
                result["rev_3m_trend"] = 1
            elif last4[1] < last4[0] and last4[2] < last4[1] and last4[3] < last4[2]:
                result["rev_3m_trend"] = -1

        # 是否創近 12 個月新高
        if len(rev) >= 12:
            hist_max = rev["revenue"].values[-12:-1].max()
            if float(rev["revenue"].iloc[-1]) > hist_max:
                result["rev_12m_high"] = 1

    # ── 季財報 ──────────────────────────────────────────
    if not financial_df.empty and "type" in financial_df.columns and "value" in financial_df.columns:
        fs = financial_df.copy().sort_values("date")
        fs["value"] = pd.to_numeric(fs["value"], errors="coerce")

        # 嘗試多種 EPS 欄位名稱（FinMind 可能為英文或中文）
        eps_keywords = ["EPS", "每股盈餘", "eps"]
        eps_df = pd.DataFrame()
        for kw in eps_keywords:
            mask = fs["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                eps_df = fs[mask].copy()
                break

        if not eps_df.empty:
            # 以日期去重後排序（同期可能有多筆，取第一筆）
            eps_q = (
                eps_df.groupby("date")["value"]
                .first()
                .sort_index()
                .dropna()
            )
            if len(eps_q) >= 1:
                result["eps_latest"] = round(float(eps_q.iloc[-1]), 2)
            if len(eps_q) >= 2:
                result["eps_qoq"] = round(float(eps_q.iloc[-1]) - float(eps_q.iloc[-2]), 2)
            if len(eps_q) >= 5:
                result["eps_yoy"] = round(float(eps_q.iloc[-1]) - float(eps_q.iloc[-5]), 2)

            # PE ratio（本益比）= 股價 / (最近四季 EPS 加總)
            if len(eps_q) >= 4 and current_price > 0:
                ttm_eps = float(eps_q.tail(4).sum())
                if ttm_eps > 0:
                    result["pe_ratio"] = round(current_price / ttm_eps, 1)
                elif ttm_eps <= 0:
                    result["pe_ratio"] = -1.0  # 虧損

        # 毛利率
        gp_keywords = ["GrossProfit", "毛利", "gross_profit"]
        rev_keywords = ["OperatingRevenue", "Revenue", "營業收入", "revenue"]

        gp_df = pd.DataFrame()
        for kw in gp_keywords:
            mask = fs["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                gp_df = fs[mask].copy()
                break

        rev_fs_df = pd.DataFrame()
        for kw in rev_keywords:
            mask = fs["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                rev_fs_df = fs[mask].copy()
                break

        if not gp_df.empty and not rev_fs_df.empty:
            gp_q = gp_df.groupby("date")["value"].first().sort_index().dropna()
            rv_q = rev_fs_df.groupby("date")["value"].first().sort_index().dropna()
            common_dates = gp_q.index.intersection(rv_q.index)
            if len(common_dates) >= 1:
                gm_series = (gp_q[common_dates] / (rv_q[common_dates] + 1e-9) * 100).replace([np.inf, -np.inf], np.nan).dropna()
                if len(gm_series) >= 1:
                    result["gross_margin"] = round(float(gm_series.iloc[-1]), 2)
                if len(gm_series) >= 2:
                    result["gpm_trend"] = round(float(gm_series.iloc[-1]) - float(gm_series.iloc[-2]), 2)

        # 嘗試直接從財報取毛利率欄位
        if result["gross_margin"] == 0.0:
            gm_keywords = ["GrossMargin", "毛利率"]
            for kw in gm_keywords:
                mask = fs["type"].str.contains(kw, case=False, na=False)
                if mask.any():
                    gm_df = fs[mask].copy()
                    gm_q = gm_df.groupby("date")["value"].first().sort_index().dropna()
                    if len(gm_q) >= 1:
                        result["gross_margin"] = round(float(gm_q.iloc[-1]), 2)
                    if len(gm_q) >= 2:
                        result["gpm_trend"] = round(float(gm_q.iloc[-1]) - float(gm_q.iloc[-2]), 2)
                    break

    return result
