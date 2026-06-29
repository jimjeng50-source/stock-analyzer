"""
factors/supply_chain.py
產業鏈 Lead-Lag 景氣信號分析模組

定義台灣主要產業鏈的上中下游對應關係，計算各層級景氣信號，
應用 Lead-Lag 效應推估資金流向。
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from data.fetcher import DataFetcher
from utils.tz import now_tw

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 產業鏈定義
# ─────────────────────────────────────────────────────────────────────────────

SUPPLY_CHAIN_MAP = {
    "semiconductor": {
        "name": "半導體產業鏈",
        "lead_lag_months": 2,
        "tiers": {
            "upstream": {
                "description": "矽晶圓、材料、化學品",
                "stocks": ["5483", "6481", "4904"],
                "weight": 0.5,
            },
            "midstream": {
                "description": "晶圓代工、IC設計、設備",
                "stocks": ["2330", "2303", "2454", "3231"],
                "weight": 0.3,
            },
            "downstream": {
                "description": "封裝測試、PCB、被動元件",
                "stocks": ["2311", "3711", "2408", "2327"],
                "weight": 0.2,
            },
        },
    },
    "ai_server": {
        "name": "AI 伺服器供應鏈",
        "lead_lag_months": 1,
        "tiers": {
            "upstream": {
                "description": "記憶體、散熱材料、電源",
                "stocks": ["3037", "2408", "6274"],
                "weight": 0.5,
            },
            "midstream": {
                "description": "伺服器 ODM、網通",
                "stocks": ["2317", "2382", "4904", "3704"],
                "weight": 0.3,
            },
            "downstream": {
                "description": "系統整合、雲端服務",
                "stocks": ["2301", "2356"],
                "weight": 0.2,
            },
        },
    },
    "ev_components": {
        "name": "電動車零組件",
        "lead_lag_months": 3,
        "tiers": {
            "upstream": {
                "description": "電芯材料、銅箔基板",
                "stocks": ["6409", "4205"],
                "weight": 0.5,
            },
            "midstream": {
                "description": "電池模組、馬達控制",
                "stocks": ["1513", "3105"],
                "weight": 0.3,
            },
            "downstream": {
                "description": "車用電子、充電樁",
                "stocks": ["2231", "6215"],
                "weight": 0.2,
            },
        },
    },
}


def get_stock_chain(stock_id: str) -> Optional[tuple]:
    """
    查詢某股票屬於哪條產業鏈的哪個層級。

    Returns:
        (chain_key, tier_name) 或 None（不在任何產業鏈中）
    """
    for chain_key, chain_info in SUPPLY_CHAIN_MAP.items():
        for tier_name, tier_info in chain_info["tiers"].items():
            if stock_id in tier_info["stocks"]:
                return (chain_key, tier_name)
    return None


def _normalize(x: float, low: float, high: float) -> float:
    """線性正規化至 [-1, 1]。"""
    if high == low:
        return 0.0
    return float(np.clip((x - low) / (high - low) * 2 - 1, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SupplyChainAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SupplyChainAnalyzer:
    """
    產業鏈景氣信號分析器

    主要功能：
    1. 計算各層級景氣信號分數
    2. 應用 Lead-Lag 效應推估目標個股的資金流向
    3. 輸出人類可讀的資金流向解說文字
    """

    def __init__(self, fetcher: DataFetcher):
        self.fetcher = fetcher

    def analyze_chain(self, chain_name: str) -> dict:
        """
        分析指定產業鏈的景氣狀態。

        Args:
            chain_name: SUPPLY_CHAIN_MAP 的 key（如 "semiconductor"）

        Returns:
            {
                "chain_name": str,
                "overall_signal": float,
                "signal_label": str,
                "tier_signals": {...},
                "lead_lag_months": int,
                "capital_flow_direction": str,
                "top_stocks_to_watch": list,
                "timestamp": str,
            }
        """
        chain_info = SUPPLY_CHAIN_MAP.get(chain_name)
        if not chain_info:
            return {
                "chain_name": chain_name,
                "overall_signal": 0.0,
                "signal_label": "中性盤整",
                "tier_signals": {},
                "lead_lag_months": 0,
                "capital_flow_direction": "查無此產業鏈資訊",
                "top_stocks_to_watch": [],
                "timestamp": now_tw().strftime("%Y-%m-%d %H:%M"),
            }

        tier_signals = {}
        for tier_name, tier_info in chain_info["tiers"].items():
            try:
                signal = self._calculate_tier_signal(
                    tier_info["stocks"], tier_info["weight"]
                )
            except Exception:
                signal = 0.0
            tier_signals[tier_name] = round(signal, 3)

        # 加權整體信號（upstream 最有領先性故權重已含在 SUPPLY_CHAIN_MAP 中）
        overall = 0.0
        for tier_name, tier_info in chain_info["tiers"].items():
            overall += tier_signals.get(tier_name, 0.0) * tier_info["weight"]
        overall = round(float(np.clip(overall, -1.0, 1.0)), 3)

        signal_label = self._interpret_signal(overall)
        capital_flow_text = self._generate_capital_flow_text(
            chain_name, tier_signals, chain_info["lead_lag_months"]
        )

        # 建議關注個股：下游信號滯後於上游信號時，下游有補漲機會
        top_stocks = []
        upstream_sig = tier_signals.get("upstream", 0.0)
        downstream_sig = tier_signals.get("downstream", 0.0)
        if upstream_sig > 0.3 and downstream_sig < upstream_sig - 0.2:
            downstream_stocks = chain_info["tiers"]["downstream"]["stocks"]
            top_stocks = downstream_stocks[:3]

        return {
            "chain_name": chain_info["name"],
            "overall_signal": overall,
            "signal_label": signal_label,
            "tier_signals": tier_signals,
            "lead_lag_months": chain_info["lead_lag_months"],
            "capital_flow_direction": capital_flow_text,
            "top_stocks_to_watch": top_stocks,
            "timestamp": now_tw().strftime("%Y-%m-%d %H:%M"),
        }

    def analyze_for_stock(self, stock_id: str) -> dict:
        """
        為單一個股分析其所在產業鏈的信號。

        Returns:
            {
                "stock_id": str,
                "chain_name": str,
                "tier": str,
                "chain_signal": float,
                "upstream_signal": float,
                "lead_lag_impact": str,
                "expected_impact_in": str,
                "chain_score": float,
                "interpretation": str,
            }
        """
        chain_result = get_stock_chain(stock_id)
        if chain_result is None:
            return {
                "stock_id": stock_id,
                "chain_name": "不在追蹤產業鏈中",
                "tier": "—",
                "chain_signal": 0.0,
                "upstream_signal": 0.0,
                "lead_lag_impact": "中性",
                "expected_impact_in": "—",
                "chain_score": 50.0,
                "interpretation": "此個股不在系統追蹤的產業鏈中，無法提供 Lead-Lag 分析。",
            }

        chain_key, tier_name = chain_result
        chain_info = SUPPLY_CHAIN_MAP[chain_key]
        chain_analysis = self.analyze_chain(chain_key)

        chain_signal = chain_analysis["overall_signal"]
        upstream_signal = chain_analysis["tier_signals"].get("upstream", 0.0)
        lead_lag_months = chain_info["lead_lag_months"]

        # 判斷 Lead-Lag 對此股的影響
        if tier_name == "upstream":
            # 上游股本身帶動後續
            lead_lag_impact = "受益" if chain_signal > 0.2 else ("中性" if chain_signal > -0.2 else "承壓")
            expected_impact = "當前即刻反映"
        elif tier_name == "midstream":
            lead_lag_impact = "受益" if upstream_signal > 0.3 else ("中性" if upstream_signal > -0.2 else "承壓")
            expected_impact = f"約 {max(1, lead_lag_months // 2)} 個月後"
        else:  # downstream
            lead_lag_impact = "受益" if upstream_signal > 0.3 else ("中性" if upstream_signal > -0.2 else "承壓")
            expected_impact = f"約 {lead_lag_months} 個月後"

        # chain_score：0-100 供 scorer 使用
        chain_score = round((chain_signal + 1.0) / 2.0 * 100, 1)

        tier_desc_map = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}
        tier_desc = tier_desc_map.get(tier_name, tier_name)
        chain_name_zh = chain_info["name"]

        interpretation = (
            f"{stock_id} 位於{chain_name_zh}{tier_desc}（{chain_info['tiers'][tier_name]['description']}）。"
            f"當前產業鏈整體信號為 {chain_analysis['signal_label']}（{chain_signal:+.2f}），"
            f"上游信號 {upstream_signal:+.2f}。"
            f"Lead-Lag 分析顯示此股預計{expected_impact}受到{lead_lag_impact}效應影響。"
        )

        return {
            "stock_id": stock_id,
            "chain_name": chain_name_zh,
            "tier": tier_desc,
            "chain_signal": chain_signal,
            "upstream_signal": upstream_signal,
            "lead_lag_impact": lead_lag_impact,
            "expected_impact_in": expected_impact,
            "chain_score": chain_score,
            "interpretation": interpretation,
        }

    def _calculate_tier_signal(self, stocks: list, weight: float) -> float:
        """
        計算單一層級信號。

        信號 = 0.4 × normalize(rev_yoy, -30, 30)
              + 0.4 × normalize(fi_net_20d, -1e8, 1e8)
              + 0.2 × normalize(price_momentum_20d, -20, 20)

        Returns: 層級信號值 -1.0 到 +1.0
        """
        rev_yoy_list, fi_net_list, mom_list = [], [], []

        for stock_id in stocks[:4]:  # 限制查詢數量
            try:
                # 月營收 YoY
                rev_df = self.fetcher.get_monthly_revenue(stock_id, months=3)
                if rev_df is not None and not rev_df.empty and "revenue_yoy" in rev_df.columns:
                    yoy = rev_df["revenue_yoy"].dropna()
                    if not yoy.empty:
                        rev_yoy_list.append(float(yoy.iloc[-1]))
                time.sleep(0.3)

                # 外資 20 日淨買賣超
                fi_series = self.fetcher.get_institutional_net(stock_id, days=25)
                if not fi_series.empty:
                    fi_net_list.append(float(fi_series.tail(20).sum()))
                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"產業鏈信號取得失敗 {stock_id}: {e}")
                continue

        # 合併計算信號
        rev_signal = _normalize(float(np.mean(rev_yoy_list)), -30, 30) if rev_yoy_list else 0.0
        fi_signal = _normalize(float(np.mean(fi_net_list)), -1e7, 1e7) if fi_net_list else 0.0
        # 價格動能簡化：用外資信號代理（避免額外 API 請求）
        mom_signal = fi_signal * 0.5

        return float(np.clip(
            0.4 * rev_signal + 0.4 * fi_signal + 0.2 * mom_signal,
            -1.0, 1.0
        ))

    def _interpret_signal(self, signal: float) -> str:
        """將數值信號轉為人類可讀標籤。"""
        if signal > 0.6:
            return "強勢擴張"
        elif signal > 0.2:
            return "緩步回溫"
        elif signal > -0.2:
            return "中性盤整"
        elif signal > -0.6:
            return "景氣降溫"
        return "明顯收縮"

    def _generate_capital_flow_text(
        self, chain_key: str, tier_signals: dict, lead_lag_months: int
    ) -> str:
        """生成資金流向推論的自然語言文字。"""
        chain_info = SUPPLY_CHAIN_MAP.get(chain_key, {})
        chain_name = chain_info.get("name", chain_key)
        tiers = chain_info.get("tiers", {})

        upstream_sig = tier_signals.get("upstream", 0.0)
        midstream_sig = tier_signals.get("midstream", 0.0)
        downstream_sig = tier_signals.get("downstream", 0.0)

        upstream_label = self._interpret_signal(upstream_sig)
        midstream_label = self._interpret_signal(midstream_sig)
        downstream_label = self._interpret_signal(downstream_sig)

        upstream_desc = tiers.get("upstream", {}).get("description", "上游")
        midstream_desc = tiers.get("midstream", {}).get("description", "中游")

        text = (
            f"{chain_name}上游（{upstream_desc}）信號為{upstream_label}（{upstream_sig:+.2f}），"
        )

        if upstream_sig > 0.3 and midstream_sig < upstream_sig - 0.2:
            text += (
                f"依過去 Lead-Lag 規律，資金可能在約 {lead_lag_months} 個月後開始佈局"
                f"中游（{midstream_desc}）族群。"
                f"當前中游信號仍為{midstream_label}（{midstream_sig:+.2f}），"
                f"顯示市場尚未充分 price in 上游強勢，存在提前卡位機會。"
            )
        elif upstream_sig < -0.3:
            text += (
                f"上游景氣走弱，預計 {lead_lag_months} 個月後可能對中下游造成壓力，"
                f"建議適度降低產業鏈整體持倉。"
            )
        else:
            text += (
                f"中游信號為{midstream_label}（{midstream_sig:+.2f}），"
                f"下游信號為{downstream_label}（{downstream_sig:+.2f}），"
                f"整體產業鏈處於{self._interpret_signal((upstream_sig + midstream_sig + downstream_sig) / 3)}狀態。"
            )

        return text
