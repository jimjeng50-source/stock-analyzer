"""
data/yf_fundamentals.py
yfinance 基本面備援（免費、免 token）

FinMind 免費配額用盡或資料缺漏時，用 yfinance 的 .info 補足：
    TTM EPS、Forward EPS（分析師共識）、本益比、營收成長、毛利率。

yfinance 對台股大型股覆蓋良好，小型股可能缺欄位（回傳 None，呼叫端需容錯）。
結果快取於 process 記憶體，避免同一次執行重複打 Yahoo。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

_CACHE = {}


def get_yf_fundamentals(stock_id: str) -> dict:
    """
    Returns（缺欄位為 None）:
        {
          "trailing_eps": float,
          "forward_eps": float,       # 分析師共識預估 EPS
          "trailing_pe": float,
          "forward_pe": float,
          "revenue_growth": float,    # 小數，如 0.153 = 15.3%
          "earnings_growth": float,   # 小數
          "gross_margins": float,     # 小數，如 0.53 = 53%
          "source_suffix": str,       # ".TW" or ".TWO"
        }
    全部失敗回傳空 dict。
    """
    if not _HAS_YF or not stock_id:
        return {}
    if stock_id in _CACHE:
        return _CACHE[stock_id]

    result = {}
    for suffix in (".TW", ".TWO"):
        try:
            info = yf.Ticker(f"{stock_id}{suffix}").info
            # 至少要有 EPS 或 PE 才視為有效命中
            if info and (info.get("trailingEps") is not None
                         or info.get("forwardEps") is not None
                         or info.get("trailingPE") is not None):
                result = {
                    "trailing_eps": _num(info.get("trailingEps")),
                    "forward_eps": _num(info.get("forwardEps")),
                    "trailing_pe": _num(info.get("trailingPE")),
                    "forward_pe": _num(info.get("forwardPE")),
                    "revenue_growth": _num(info.get("revenueGrowth")),
                    "earnings_growth": _num(info.get("earningsGrowth")),
                    "gross_margins": _num(info.get("grossMargins")),
                    "source_suffix": suffix,
                }
                break
        except Exception as e:
            logger.debug("yfinance 基本面抓取失敗 %s%s：%s", stock_id, suffix, e)

    _CACHE[stock_id] = result
    return result


def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        # yfinance 有時回傳極端佔位值
        if f != f or abs(f) > 1e12:
            return None
        return f
    except (TypeError, ValueError):
        return None
