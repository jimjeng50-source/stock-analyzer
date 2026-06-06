"""台幣匯率因子：USDTWD 下跌 = 台幣升值 = 外資流入正訊號。"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


def _get_usdtwd(days: int = 90) -> pd.Series:
    if not _HAS_YF:
        return pd.Series(dtype=float)
    end = datetime.today()
    start = (end - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    for ticker in ["TWD=X", "USDTWD=X"]:
        try:
            raw = yf.download(ticker, start=start, end=end.strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                return raw[col].dropna()
        except Exception as e:
            print(f"[警告] 匯率資料取得失敗 ({ticker}): {e}")
    return pd.Series(dtype=float)


def compute_fx() -> dict:
    """
    回傳台幣匯率因子 dict。
    所有數值已取反：正值 = 台幣升值（有利股市），負值 = 台幣貶值。
    """
    result = {"twd_5d_chg": 0.0, "twd_20d_chg": 0.0,
               "twd_trend": 0.0, "twd_vs_ma20": 0.0}

    close = _get_usdtwd()
    if len(close) < 6:
        return result

    c_now = float(close.iloc[-1])

    if len(close) >= 6:
        result["twd_5d_chg"] = round(-(c_now / float(close.iloc[-6]) - 1) * 100, 3)

    if len(close) >= 21:
        result["twd_20d_chg"] = round(-(c_now / float(close.iloc[-21]) - 1) * 100, 3)

    if len(close) >= 10:
        arr = close.tail(10).values.astype(float)
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        result["twd_trend"] = round(-slope, 4)  # 負斜率 = 台幣升值趨勢

    if len(close) >= 20:
        ma20 = float(close.tail(20).mean())
        result["twd_vs_ma20"] = round(-(c_now / ma20 - 1) * 100, 3)

    return result


def get_fx_series(days: int = 90) -> pd.Series:
    """回傳 USDTWD 原始收盤價 Series（供圖表使用）。"""
    return _get_usdtwd(days)
