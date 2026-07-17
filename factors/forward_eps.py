"""
factors/forward_eps.py
Forward EPS 推估模組

依賴:
    data/fetcher.py 中的 DataFetcher.get_quarterly_eps()、
    get_monthly_revenue()、get_quarterly_gross_margin()、
    get_historical_pe()、get_market_price()
"""

import numpy as np
import pandas as pd
from typing import Optional

from data.fetcher import DataFetcher


class ForwardEPSCalculator:
    """
    Forward EPS 推估器

    計算邏輯：
    1. 取近 8 季 EPS 實績，計算 TTM EPS
    2. 取近 6 個月月營收 YoY 平均作為成長率代理
    3. 計算毛利率趨勢調整係數（連續擴張 → 上修，收縮 → 下修）
    3.5 存貨動態與庫存重估信號：對記憶體模組/面板/原物料等庫存重估型
        公司，偵測「低價庫存 × 報價上漲」的獲利拐點（存貨部位 + 毛利
        加速度），修正 Forward EPS —— 這是單純營收趨勢看不到的爆發來源
    4. Forward EPS = TTM EPS × (1 + 營收成長 + 毛利趨勢 + 庫存重估)
    5. 用歷史 P/E 分位數計算三情境目標價
    6. 計算 PEG Ratio；選填產品報價情境調整
    """

    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def calculate(self, stock_id: str, product_price_chg_pct: float = None) -> dict:
        """
        計算完整 Forward EPS 數據包。

        Args:
            product_price_chg_pct: 主力產品報價變動看法（%），選填。
                提供時對庫存重估型公司加做報價情境調整。

        Returns:
            dict — 含 ttm_eps、forward_eps_1y、target_price、
            inventory_dynamics（存貨動態與庫存重估信號）、
            price_scenario（報價情境，若提供）等欄位。
        """
        result = {
            "stock_id": stock_id,
            "ttm_eps": None,
            "eps_growth_rate": None,
            "forward_eps_1y": None,
            "gm_adjustment": None,
            "revaluation_adjustment": None,
            "inventory_dynamics": None,
            "price_scenario": None,
            "target_price": {"bull": None, "base": None, "bear": None},
            "current_price": None,
            "upside_pct": None,
            "peg_ratio": None,
            "pe_band": {"p25": None, "median": None, "p75": None, "current": None},
            "confidence": "low",
            "confidence_reason": "",
            "error": None,
        }

        try:
            # Step 1: TTM EPS
            eps_data = self.fetcher.get_quarterly_eps(stock_id, n_quarters=8)
            if eps_data is None or len(eps_data) < 4:
                result["error"] = "EPS 資料不足（需至少 4 季）"
                return result

            ttm_eps = float(eps_data.tail(4)["eps"].sum())
            result["ttm_eps"] = round(ttm_eps, 2)

            # Step 2: 營收 YoY 成長率（取近 6 個月平均）
            confidence_flags = []
            rev_data = self.fetcher.get_monthly_revenue(stock_id, months=6)
            if rev_data is None or len(rev_data) < 3:
                growth_proxy = 0.0
                confidence_flags.append("rev_insufficient")
            else:
                yoy_series = rev_data["revenue_yoy"].dropna()
                if yoy_series.empty:
                    growth_proxy = 0.0
                    confidence_flags.append("rev_yoy_insufficient")
                else:
                    growth_proxy = float(yoy_series.mean()) / 100.0  # 轉成小數

            # Step 3: 毛利率趨勢調整
            gm_data = self.fetcher.get_quarterly_gross_margin(stock_id, n_quarters=6)
            if gm_data is None or len(gm_data) < 3:
                gm_adjustment = 0.0
                confidence_flags.append("gm_insufficient")
            else:
                gm_series = gm_data["gross_margin"].values.astype(float)
                gm_mean = float(gm_series.mean())
                gm_std = float(gm_series.std())
                if gm_std > 0:
                    gm_adjustment = float((gm_series[-1] - gm_mean) / gm_std) * 0.1
                    gm_adjustment = max(-0.05, min(0.05, gm_adjustment))
                else:
                    gm_adjustment = 0.0

            result["gm_adjustment"] = round(gm_adjustment, 4)

            # Step 3.5: 存貨動態與庫存重估信號（威剛型：低價庫存 × 報價上漲）
            reval_adjustment = 0.0
            try:
                from factors.inventory_dynamics import (
                    compute_inventory_dynamics, MAX_REVALUATION_IMPACT,
                    apply_product_price_scenario,
                )
                inv_dyn = compute_inventory_dynamics(self.fetcher, stock_id)
                result["inventory_dynamics"] = inv_dyn
                if not inv_dyn.get("error"):
                    reval_adjustment = inv_dyn["revaluation_score"] * MAX_REVALUATION_IMPACT
                    result["revaluation_adjustment"] = round(reval_adjustment, 4)
            except Exception as e:
                inv_dyn = None

            # Step 4: Forward EPS（營收成長 + 毛利趨勢 + 庫存重估）
            adjusted_growth = growth_proxy + gm_adjustment + reval_adjustment
            forward_eps = ttm_eps * (1 + adjusted_growth)
            result["eps_growth_rate"] = round(adjusted_growth, 4)
            result["forward_eps_1y"] = round(forward_eps, 2)

            # Step 4.5: 產品報價情境（選填，使用者有報價看法時）
            if product_price_chg_pct and inv_dyn and not inv_dyn.get("error"):
                try:
                    result["price_scenario"] = apply_product_price_scenario(
                        forward_eps, ttm_eps,
                        inv_dyn.get("inv_to_rev_latest"),
                        inv_dyn.get("gm_latest"),
                        product_price_chg_pct,
                    )
                except Exception:
                    pass

            # Step 5: 歷史 P/E 分位數（取近 3 年日頻資料）
            pe_data = self.fetcher.get_historical_pe(stock_id, years=3)
            current_price = self.fetcher.get_market_price(stock_id)
            result["current_price"] = current_price

            if pe_data is not None and len(pe_data) > 60 and ttm_eps > 0:
                pe_series = pe_data["pe_ratio"].dropna()
                pe_series = pe_series[pe_series > 0]
                if len(pe_series) >= 20:
                    p25 = float(pe_series.quantile(0.25))
                    median = float(pe_series.median())
                    p75 = float(pe_series.quantile(0.75))
                    current_pe = current_price / ttm_eps if ttm_eps > 0 else None

                    result["pe_band"] = {
                        "p25": round(p25, 1),
                        "median": round(median, 1),
                        "p75": round(p75, 1),
                        "current": round(current_pe, 1) if current_pe else None,
                    }
                    result["target_price"] = {
                        "bull": round(p75 * forward_eps, 1),
                        "base": round(median * forward_eps, 1),
                        "bear": round(p25 * forward_eps, 1),
                    }
                    if current_price > 0:
                        result["upside_pct"] = round(
                            (result["target_price"]["base"] - current_price)
                            / current_price * 100,
                            1,
                        )

                    # PEG Ratio：僅在成長率為正時計算
                    if adjusted_growth > 0 and current_pe:
                        result["peg_ratio"] = round(
                            current_pe / (adjusted_growth * 100), 2
                        )
                else:
                    confidence_flags.append("pe_data_insufficient")
            else:
                confidence_flags.append("pe_data_insufficient")

            # Step 6: 信心度評估
            if not confidence_flags:
                result["confidence"] = "high"
                result["confidence_reason"] = "各項資料齊全，推估可信度高"
            elif len(confidence_flags) == 1:
                result["confidence"] = "medium"
                result["confidence_reason"] = f"部分資料缺漏：{', '.join(confidence_flags)}"
            else:
                result["confidence"] = "low"
                result["confidence_reason"] = (
                    f"多項資料缺漏：{', '.join(confidence_flags)}，請謹慎參考"
                )

        except Exception as e:
            result["error"] = f"計算異常：{str(e)}"
            result["confidence"] = "low"

        return result
