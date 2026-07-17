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

    def calculate(self, stock_id: str, product_price_chg_pct: float = None,
                  consensus_eps: float = None, guidance_eps: float = None) -> dict:
        """
        計算完整 Forward EPS 數據包（多算法）。

        Args:
            product_price_chg_pct: 主力產品報價變動看法（%），選填。
            consensus_eps: 市場共識 EPS（手動輸入外資/投顧預估），選填。
            guidance_eps: 公司財測 EPS（手動輸入法說會指引），選填。

        Returns:
            dict — 含 ttm_eps、forward_eps_1y、target_price、
            inventory_dynamics、price_scenario，以及 methods（多種算法比較）。
        """
        result = {
            "stock_id": stock_id,
            "ttm_eps": None,
            "eps_growth_rate": None,
            "forward_eps_1y": None,
            "base_eps_1y": None,
            "gm_adjustment": None,
            "revaluation_adjustment": None,
            "inventory_dynamics": None,
            "price_scenario": None,
            "methods": {},
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
            inv_dyn = None
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

            # Step 4: 多算法 Forward EPS
            # (a) 定量模型-基礎：營收成長 + 毛利趨勢（原本的算法）
            base_growth = growth_proxy + gm_adjustment
            base_eps = ttm_eps * (1 + base_growth)
            # (b) 定量模型-含庫存重估：再加庫存重估修正（威剛型）
            inv_growth = base_growth + reval_adjustment
            inventory_eps = ttm_eps * (1 + inv_growth)

            # 主要輸出：以含庫存重估者為準（供評分/目標價使用）
            adjusted_growth = inv_growth
            forward_eps = inventory_eps
            result["eps_growth_rate"] = round(adjusted_growth, 4)
            result["forward_eps_1y"] = round(forward_eps, 2)
            result["base_eps_1y"] = round(base_eps, 2)

            # 資料是否足以支撐定量推估（營收或毛利至少一項有效）
            _rev_missing = ("rev_insufficient" in confidence_flags
                            or "rev_yoy_insufficient" in confidence_flags)
            _gm_missing = "gm_insufficient" in confidence_flags
            _quant_ok = not (_rev_missing and _gm_missing)

            # (c) 情境敏感度：以毛利率波動度張開樂觀/中性/悲觀
            gm_vol = 0.05
            if gm_data is not None and len(gm_data) >= 3:
                _gm_arr = gm_data["gross_margin"].values.astype(float)
                _mean = float(_gm_arr.mean())
                if _mean != 0:
                    gm_vol = float(min(0.15, max(0.03, _gm_arr.std() / abs(_mean))))
            scenario = {
                "bull": round(ttm_eps * (1 + base_growth + gm_vol), 2),
                "base": round(base_eps, 2),
                "bear": round(ttm_eps * (1 + base_growth - gm_vol), 2),
            }

            # 組多算法比較表
            result["methods"] = {
                "quant_base": {
                    "label": "定量模型-基礎", "eps": round(base_eps, 2),
                    "growth_pct": round(base_growth * 100, 1),
                    "available": _quant_ok,
                    "source": "營收成長率 + 毛利趨勢（TTM 外推）",
                    "note": "" if _quant_ok else "營收/毛利資料不足，僅等於 TTM",
                },
                "quant_inventory": {
                    "label": "定量模型-含庫存重估", "eps": round(inventory_eps, 2),
                    "growth_pct": round(inv_growth * 100, 1),
                    "available": _quant_ok and inv_dyn is not None and not inv_dyn.get("error"),
                    "source": "基礎 + 存貨/毛利加速度庫存重估信號",
                    "note": (inv_dyn.get("signal_label", "") if (inv_dyn and not inv_dyn.get("error"))
                             else "存貨資料不足，等同基礎模型"),
                },
                "scenario": {
                    "label": "情境敏感度（樂觀/中性/悲觀）", "eps": scenario["base"],
                    "bull": scenario["bull"], "base": scenario["base"], "bear": scenario["bear"],
                    "available": _quant_ok,
                    "source": "以毛利率波動度張開三情境",
                    "note": "",
                },
                "consensus": {
                    "label": "市場共識（外資/投顧平均）",
                    "eps": round(float(consensus_eps), 2) if consensus_eps else None,
                    "available": bool(consensus_eps),
                    "source": "IBES/Bloomberg 分析師預估平均",
                    "note": "無免費資料源，如有券商報告數字可手動輸入" if not consensus_eps else "手動輸入",
                },
                "guidance": {
                    "label": "公司財測指引",
                    "eps": round(float(guidance_eps), 2) if guidance_eps else None,
                    "available": bool(guidance_eps),
                    "source": "公司法說會官方財務預測",
                    "note": "台股多數不提供正式 EPS 財測，如有可手動輸入" if not guidance_eps else "手動輸入",
                },
            }

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
