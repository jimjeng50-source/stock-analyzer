import numpy as np
import pandas as pd
from typing import Optional


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


def compute_financial_quality_metrics(stock_id: str) -> dict:
    """
    計算財務品質四大指標：DSI、DSO、FCF Yield、Capex Intensity。

    資料來源：
        FinMind TaiwanStockCashFlowsStatement（現金流）
        TaiwanStockBalanceSheet（資產負債表）
        TaiwanStockFinancialStatements（損益表）

    Returns:
        {
            "dsi": float,             # 存貨周轉天數
            "dso": float,             # 應收帳款天數
            "fcf_yield": float,       # 自由現金流殖利率（%）
            "capex_intensity": float, # Capex/Revenue（%）
            "quality_score": float,   # 0-100
            "quality_label": str,     # "優質"|"良好"|"普通"|"偏弱"
            "highlights": list[str],
            "concerns": list[str],
        }
    """
    result = {
        "dsi": None, "dso": None,
        "fcf_yield": None, "capex_intensity": None,
        "quality_score": 50.0,
        "quality_label": "普通",
        "highlights": [],
        "concerns": [],
    }

    try:
        from data.fetcher import DataFetcher
        from utils.tz import now_tw
        from datetime import timedelta

        fetcher = DataFetcher()
        end = now_tw().strftime("%Y-%m-%d")
        start = (now_tw() - timedelta(days=730)).strftime("%Y-%m-%d")

        # ── 現金流量表 ─────────────────────────────────────────────────────
        cf_df = fetcher._fm_request("TaiwanStockCashFlowsStatement", stock_id, start)
        op_cf, capex = 0.0, 0.0
        if not cf_df.empty and "type" in cf_df.columns and "value" in cf_df.columns:
            cf_df["value"] = pd.to_numeric(cf_df["value"], errors="coerce")
            # 營業現金流
            for kw in ["營業活動之淨現金流入", "OperatingActivities", "CashFromOperations"]:
                m = cf_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    op_cf = float(cf_df[m]["value"].dropna().tail(4).sum())
                    break
            # 資本支出（Capex 通常為負值）
            for kw in ["取得不動產", "資本支出", "CapitalExpenditure", "PurchaseOfProperty"]:
                m = cf_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    capex_raw = float(cf_df[m]["value"].dropna().tail(4).sum())
                    capex = abs(capex_raw)
                    break

        # ── 資產負債表 ─────────────────────────────────────────────────────
        bs_df = fetcher._fm_request("TaiwanStockBalanceSheet", stock_id, start)
        inventory, accounts_receivable = 0.0, 0.0
        if not bs_df.empty and "type" in bs_df.columns and "value" in bs_df.columns:
            bs_df["value"] = pd.to_numeric(bs_df["value"], errors="coerce")
            for kw in ["存貨", "Inventories", "Inventory"]:
                m = bs_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    inventory = float(bs_df[m]["value"].dropna().iloc[-1])
                    break
            for kw in ["應收帳款", "AccountsReceivable", "TradeReceivables"]:
                m = bs_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    accounts_receivable = float(bs_df[m]["value"].dropna().iloc[-1])
                    break

        # ── 損益表：取年化營收和銷貨成本 ─────────────────────────────────
        is_df = fetcher._fm_request("TaiwanStockFinancialStatements", stock_id, start)
        revenue_ttm, cogs_ttm = 0.0, 0.0
        if not is_df.empty and "type" in is_df.columns and "value" in is_df.columns:
            is_df["value"] = pd.to_numeric(is_df["value"], errors="coerce")
            for kw in ["OperatingRevenue", "Revenue", "營業收入"]:
                m = is_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    rev_q = is_df[m].groupby("date")["value"].first().sort_index().dropna()
                    revenue_ttm = float(rev_q.tail(4).sum())
                    break
            for kw in ["CostOfGoodsSold", "銷貨成本", "CostOfRevenue"]:
                m = is_df["type"].str.contains(kw, case=False, na=False)
                if m.any():
                    cogs_q = is_df[m].groupby("date")["value"].first().sort_index().dropna()
                    cogs_ttm = float(cogs_q.tail(4).sum())
                    break

        # ── 計算指標 ──────────────────────────────────────────────────────
        if cogs_ttm > 0 and inventory > 0:
            dsi = round(inventory / (cogs_ttm / 365), 1)
            result["dsi"] = dsi

        if revenue_ttm > 0 and accounts_receivable > 0:
            dso = round(accounts_receivable / (revenue_ttm / 365), 1)
            result["dso"] = dso

        fcf = op_cf - capex
        market_price = fetcher.get_market_price(stock_id)
        if market_price > 0:
            # 估算市值：使用 yfinance 或預設股本估算
            # 先以 FCF / 假設市值（取季末資本額替代）估算
            pass  # market cap 估算複雜，先以 revenue 比例估算
        if revenue_ttm > 0 and fcf != 0.0:
            result["fcf_yield"] = round(fcf / revenue_ttm * 100, 2)

        if revenue_ttm > 0 and capex > 0:
            result["capex_intensity"] = round(capex / revenue_ttm * 100, 2)

        # ── 評分（各 25 分，合計 100 分）────────────────────────────────
        scores = []

        # DSI 分數：越低越好（參考行業中位數 60 天）
        dsi_score = 25.0
        if result["dsi"] is not None:
            dsi = result["dsi"]
            if dsi <= 30:
                dsi_score = 25.0
            elif dsi <= 60:
                dsi_score = 20.0
            elif dsi <= 90:
                dsi_score = 12.0
            else:
                dsi_score = 5.0
        scores.append(dsi_score)

        # DSO 分數
        dso_score = 25.0
        if result["dso"] is not None:
            dso = result["dso"]
            if dso <= 30:
                dso_score = 25.0
            elif dso <= 60:
                dso_score = 20.0
            elif dso <= 90:
                dso_score = 12.0
            else:
                dso_score = 5.0
        scores.append(dso_score)

        # FCF Yield 分數
        fcf_score = 12.5
        if result["fcf_yield"] is not None:
            fy = result["fcf_yield"]
            if fy >= 5:
                fcf_score = 25.0
            elif fy >= 0:
                fcf_score = max(0, fy / 5 * 25)
            else:
                fcf_score = 0.0
        scores.append(fcf_score)

        # Capex Intensity 分數
        capex_score = 10.0
        if result["capex_intensity"] is not None:
            ci = result["capex_intensity"]
            if ci <= 5:
                capex_score = 10.0   # 低投入，成熟期
            elif ci <= 15:
                capex_score = 25.0  # 積極擴張，成長期
            else:
                capex_score = 15.0  # 重資本，需觀察
        scores.append(capex_score)

        quality_score = round(sum(scores), 1)
        result["quality_score"] = quality_score

        if quality_score >= 80:
            result["quality_label"] = "優質"
        elif quality_score >= 60:
            result["quality_label"] = "良好"
        elif quality_score >= 40:
            result["quality_label"] = "普通"
        else:
            result["quality_label"] = "偏弱"

        # ── 亮點與隱憂 ────────────────────────────────────────────────────
        highlights, concerns = [], []
        if result["dsi"] and result["dsi"] <= 30:
            highlights.append(f"存貨周轉天數 {result['dsi']:.0f} 天，庫存管理優異")
        elif result["dsi"] and result["dsi"] > 90:
            concerns.append(f"存貨周轉天數 {result['dsi']:.0f} 天偏高，需注意庫存積壓")

        if result["dso"] and result["dso"] <= 30:
            highlights.append(f"應收帳款天數 {result['dso']:.0f} 天，收款效率佳")
        elif result["dso"] and result["dso"] > 90:
            concerns.append(f"應收帳款天數 {result['dso']:.0f} 天偏高，資金占用大")

        if result["fcf_yield"] and result["fcf_yield"] >= 5:
            highlights.append(f"自由現金流佔營收 {result['fcf_yield']:.1f}%，現金創造力強")
        elif result["fcf_yield"] and result["fcf_yield"] < 0:
            concerns.append(f"自由現金流為負（{result['fcf_yield']:.1f}%），燒錢階段需留意")

        if result["capex_intensity"] and 5 <= result["capex_intensity"] <= 15:
            highlights.append(f"Capex 強度 {result['capex_intensity']:.1f}%，積極擴產期")
        elif result["capex_intensity"] and result["capex_intensity"] > 20:
            concerns.append(f"Capex 強度 {result['capex_intensity']:.1f}% 偏高，回報待觀察")

        result["highlights"] = highlights[:2]
        result["concerns"] = concerns[:2]

    except Exception:
        result["quality_score"] = 50.0
        result["quality_label"] = "普通"

    return result
