"""VIX 恐慌指數因子：VIX 越低代表市場越平靜，得分越高。"""

import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


def _get_vix(days: int = 90) -> pd.Series:
    if not _HAS_YF:
        return pd.Series(dtype=float)
    end = datetime.today()
    start = (end - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("^VIX", start=start, end=end.strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            return raw[col].dropna()
    except Exception as e:
        print(f"[警告] VIX 資料取得失敗: {e}")
    return pd.Series(dtype=float)


def vix_to_score(vix: float) -> float:
    """
    VIX → 0~1 分數。
    < 15 平靜 → 1.0；≥ 30 極度恐慌 → 0.0。
    """
    if vix < 15:
        return 1.00
    elif vix < 20:
        return 0.75
    elif vix < 25:
        return 0.50
    elif vix < 30:
        return 0.25
    return 0.00


def vix_to_label(vix: float) -> str:
    if vix < 15:
        return "市場平靜"
    elif vix < 20:
        return "輕微波動"
    elif vix < 25:
        return "中度緊張"
    elif vix < 30:
        return "明顯恐慌"
    return "極度恐慌"


def compute_vix() -> dict:
    result = {"vix_level": 20.0, "vix_5d_chg": 0.0, "vix_signal": 0.5}

    close = _get_vix()
    if close.empty:
        return result

    vix_now = float(close.iloc[-1])
    result["vix_level"] = round(vix_now, 2)
    result["vix_signal"] = vix_to_score(vix_now)

    if len(close) >= 6:
        result["vix_5d_chg"] = round(float(close.iloc[-1]) - float(close.iloc[-6]), 2)

    return result


def get_vix_series(days: int = 90) -> pd.Series:
    """回傳 VIX 原始收盤 Series（供圖表使用）。"""
    return _get_vix(days)
