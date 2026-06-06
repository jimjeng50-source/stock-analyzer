"""外資期貨未平倉因子：外資台指期淨多單為市場方向最強先行指標。"""

import pandas as pd
import requests
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from config import FINMIND_TOKEN

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


def _fm_get(dataset: str, data_id: str = "", days: int = 90) -> pd.DataFrame:
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        params = {
            "dataset": dataset,
            "start_date": start,
            "end_date": end.strftime("%Y-%m-%d"),
            "token": FINMIND_TOKEN,
        }
        if data_id:
            params["data_id"] = data_id
        resp = requests.get(_FINMIND_API, params=params, timeout=20)
        body = resp.json()
        if body.get("status") == 200 and body.get("data"):
            df = pd.DataFrame(body["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"[警告] FinMind {dataset} 失敗：{e}")
    return pd.DataFrame()


def _futures_score(net: float) -> float:
    """
    外資台指期淨多單（口）→ 0~1 分數（Sigmoid，scale=80,000口）。
    外資期貨部位歷史範圍約 -150,000 ~ +150,000 口。
    net=0 → 0.50（中性）
    net=+80,000 → 0.73（偏多）
    net=-80,000 → 0.27（偏空）
    net=-66,772 → 約 0.30（明顯偏空但非極端）
    """
    import numpy as np
    return round(float(1 / (1 + np.exp(-net / 80_000))), 4)


def compute_futures() -> dict:
    result = {
        "fi_future_net": 0,
        "fi_future_5d_chg": 0,
        "fi_future_trend": 0.0,
        "fi_option_pc_ratio": 1.0,
        "futures_score": 0.5,
    }

    if not FINMIND_TOKEN:
        return result

    # 台指期外資未平倉，明確指定 data_id="TX"（台指期）
    df = _fm_get("TaiwanFuturesInstitutionalInvestors", data_id="TX", days=60)

    if df.empty:
        return result

    # FinMind 實際欄位名稱（由 debug 確認）：
    # institutional_investors（法人名稱）、
    # long_open_interest_balance_volume（多方未平倉口數）、
    # short_open_interest_balance_volume（空方未平倉口數）

    inv_col = "institutional_investors"
    if inv_col not in df.columns:
        # 舊版欄位名稱備援
        inv_col = next((c for c in df.columns if "investor" in c.lower() or c == "name"), None)

    if inv_col is None:
        return result

    fi_df = df[df[inv_col].str.contains("外資", na=False)].copy()

    if fi_df.empty:
        return result

    # 計算淨未平倉口數 = 多方 - 空方
    long_col  = "long_open_interest_balance_volume"
    short_col = "short_open_interest_balance_volume"

    # 備援：找含 long/short 的欄位
    if long_col not in fi_df.columns:
        long_col  = next((c for c in fi_df.columns if "long" in c.lower() and "volume" in c.lower()), None)
    if short_col not in fi_df.columns:
        short_col = next((c for c in fi_df.columns if "short" in c.lower() and "volume" in c.lower()), None)

    if not long_col or not short_col:
        return result

    fi_df["net"] = (pd.to_numeric(fi_df[long_col],  errors="coerce").fillna(0) -
                    pd.to_numeric(fi_df[short_col], errors="coerce").fillna(0))
    net_col = "net"

    net_series = fi_df.groupby("date")[net_col].sum().sort_index().dropna()

    if net_series.empty:
        return result

    net_now = int(net_series.iloc[-1])
    result["fi_future_net"] = net_now
    result["futures_score"] = _futures_score(net_now)

    if len(net_series) >= 6:
        result["fi_future_5d_chg"] = int(net_series.iloc[-1] - net_series.iloc[-6])

    if len(net_series) >= 10:
        import numpy as np
        arr = net_series.tail(10).values.astype(float)
        result["fi_future_trend"] = round(float(np.polyfit(range(len(arr)), arr, 1)[0]), 0)

    return result
