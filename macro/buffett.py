"""
巴菲特指標（Buffett Indicator）：台股總市值 / 台灣 GDP（%）。

比率越高 → 股市相對 GDP 越貴：
  < 80%  : 顯著低估
  80-100%: 小幅低估
  100-120%: 合理
  120-150%: 偏高估
  > 150% : 嚴重高估

資料來源：
  總市值 → FinMind TaiwanStockPER（含 market_cap 欄位）；
            失敗時以 yfinance TAIEX 點位比例估算
  GDP    → World Bank API NY.GDP.MKTP.CN（台幣現價）；
            失敗時使用內建靜態備援值
"""

import requests
import numpy as np
import pandas as pd
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

from utils.tz import now_tw

# ── 靜態台灣 GDP 備援（兆台幣，來源：行政院主計總處）──────────────
_TAIWAN_GDP_TRILLION = {
    2019: 18.05,
    2020: 19.76,
    2021: 22.68,
    2022: 22.65,
    2023: 21.96,
    2024: 23.80,
    2025: 24.50,
}

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"

# TAIEX 1 點 ≈ 0.0029 兆台幣市值（2023-2024 歷史中位估算）
_TAIEX_CAP_FACTOR = 0.0029


def _fetch_twse_market_cap_trillion(token: str = "") -> float:
    """取得台股最新總市值（兆台幣）。"""
    # ── 方法 1：FinMind TaiwanStockPER ───────────────────────────
    if token:
        try:
            end = now_tw().strftime("%Y-%m-%d")
            start = (now_tw() - timedelta(days=14)).strftime("%Y-%m-%d")
            resp = requests.get(_FINMIND_API, params={
                "dataset": "TaiwanStockPER",
                "start_date": start,
                "end_date": end,
                "token": token,
            }, timeout=15)
            body = resp.json()
            if body.get("status") == 200 and body.get("data"):
                df = pd.DataFrame(body["data"])
                for col in ["market_cap", "MarketCap", "capitalization"]:
                    if col in df.columns:
                        cap = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not cap.empty:
                            # FinMind 單位為百萬元或億元，依實際欄位大小推算
                            v = float(cap.iloc[-1])
                            if v > 1e9:          # 單位：元
                                return v / 1e12
                            elif v > 1e6:        # 單位：百萬元
                                return v / 1e6
                            elif v > 1000:       # 單位：億元
                                return v / 1e4
                            else:                # 單位：兆元
                                return v
        except Exception:
            pass

    # ── 方法 2：yfinance TAIEX 比例估算 ────────────────────────
    try:
        import yfinance as yf
        raw = yf.download("^TWII", period="5d", progress=False, auto_adjust=True)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            taiex = float(raw[col].dropna().iloc[-1])
            return round(taiex * _TAIEX_CAP_FACTOR, 2)
    except Exception:
        pass

    return 55.0  # 最終備援近似值


def _get_taiwan_gdp_trillion() -> float:
    """取得台灣最新年度 GDP（兆台幣）。"""
    # ── World Bank API（NY.GDP.MKTP.CN = 本國貨幣 GDP）────────────
    try:
        url = "https://api.worldbank.org/v2/country/TW/indicator/NY.GDP.MKTP.CN"
        resp = requests.get(url, params={"format": "json", "mrv": 3}, timeout=10)
        data = resp.json()
        if isinstance(data, list) and len(data) > 1 and data[1]:
            for item in data[1]:
                v = item.get("value")
                if v is not None:
                    gdp_twd = float(v)  # 元
                    return round(gdp_twd / 1e12, 2)
    except Exception:
        pass

    # 靜態備援
    year = min(now_tw().year, max(_TAIWAN_GDP_TRILLION.keys()))
    return _TAIWAN_GDP_TRILLION.get(year, 22.0)


def _buffett_signal(ratio: float) -> tuple[str, str]:
    if ratio < 80:
        return "🟢", "台股大幅低估，長線布局機會"
    elif ratio < 100:
        return "🟡", "台股小幅低估，中性偏多"
    elif ratio < 120:
        return "🟡", "台股合理估值，中性觀望"
    elif ratio < 150:
        return "🟠", "台股偏高估值，控制持倉"
    return "🔴", "台股嚴重高估，風險極高"


def _buffett_score(ratio: float) -> float:
    """比率越低越好，映射為 0~1 分（供 macro_scorer 使用）。"""
    # ratio=60% → 1.0；ratio=180% → 0.0
    return float(np.clip(1 - (ratio - 60) / 120, 0.0, 1.0))


def compute_buffett(token: str = "") -> dict:
    """
    計算台灣巴菲特指標。

    Returns
    -------
    {
        "ratio":        125.3,      # 市值/GDP (%)
        "market_cap":   65.2,       # 台股總市值（兆台幣，估算）
        "gdp":          22.7,       # 台灣 GDP（兆台幣）
        "score":        0.45,       # 0~1 分（越高越便宜）
        "signal":       "🟠 台股偏高估值，控制持倉",
        "color":        "🟠",
        "interpretation": "台股偏高估值，控制持倉",
        "historical_context": "...",
    }
    """
    market_cap = _fetch_twse_market_cap_trillion(token)
    gdp = _get_taiwan_gdp_trillion()
    if gdp <= 0:
        gdp = 22.0

    ratio = round(market_cap / gdp * 100, 1)
    score = _buffett_score(ratio)
    color, interp = _buffett_signal(ratio)

    # 歷史脈絡說明
    if ratio < 80:
        hist = "歷史上此水位（<80%）出現於 2008 年金融海嘯後，為長線難得買點"
    elif ratio < 100:
        hist = "類似 2012～2013 年整體股市低迷期，多數優質股仍具吸引力"
    elif ratio < 120:
        hist = "台股歷史合理中樞區間，市場估值未過度偏離基本面"
    elif ratio < 150:
        hist = "類似 2020～2021 年資金行情末段，需留意流動性風險"
    else:
        hist = "超越歷史高點，比較接近 2000 年科技泡沫水位，極度謹慎"

    return {
        "ratio": ratio,
        "market_cap": round(market_cap, 2),
        "gdp": round(gdp, 2),
        "score": round(score, 4),
        "signal": f"{color} {interp}",
        "color": color,
        "interpretation": interp,
        "historical_context": hist,
    }
