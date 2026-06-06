"""全市場外資資金流向：抓取 FinMind 整體市場三大法人彙計。"""

import pandas as pd
import requests
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from config import FINMIND_TOKEN

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"

# 以大型權值股代理全市場外資買賣超（避免需要全市場匯總API）
_PROXY_STOCKS = ["2330", "2454", "2317", "2382", "2308", "3711", "2303", "2881", "2882", "2412"]


def _fm_stock(stock_id: str, days: int = 30) -> pd.DataFrame:
    end = datetime.today()
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(_FINMIND_API, params={
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": start,
            "end_date": end.strftime("%Y-%m-%d"),
            "token": FINMIND_TOKEN,
        }, timeout=15)
        body = resp.json()
        if body.get("status") == 200 and body.get("data"):
            df = pd.DataFrame(body["data"])
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _consecutive_days(series: pd.Series) -> int:
    arr = series.dropna().values
    if not len(arr):
        return 0
    sign = 1 if arr[-1] > 0 else -1
    count = 0
    for v in reversed(arr):
        if v * sign > 0:
            count += 1
        else:
            break
    return count * sign


def compute_fund_flow() -> dict:
    result = {
        "fi_total_5d": 0,
        "fi_total_20d": 0,
        "fi_consecutive_days": 0,
        "it_total_5d": 0,
        "flow_score": 0.5,
    }

    if not FINMIND_TOKEN:
        return result

    # 以代理股票加總模擬全市場外資流向
    fi_agg = {}
    it_agg = {}

    for sid in _PROXY_STOCKS[:5]:   # 限制 5 支避免 API 過多請求
        df = _fm_stock(sid, days=30)
        if df.empty or "name" not in df.columns:
            continue
        for col in ["buy", "sell"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "net" not in df.columns and "buy" in df.columns:
            df["net"] = df["buy"] - df["sell"]

        # FinMind 實際回傳英文名稱（Foreign_Investor / Investment_Trust 等）
        inv_col = "name" if "name" in df.columns else "institutional_investors"
        # 外資：Foreign_Investor（排除 Foreign_Dealer_Self）
        fi_mask = df[inv_col].isin(["Foreign_Investor", "外資及陸資", "外資"])
        # 投信：Investment_Trust
        it_mask = df[inv_col].isin(["Investment_Trust", "投信"])
        fi = df[fi_mask].groupby("date")["net"].sum()
        it = df[it_mask].groupby("date")["net"].sum()

        for d, v in fi.items():
            fi_agg[d] = fi_agg.get(d, 0) + v
        for d, v in it.items():
            it_agg[d] = it_agg.get(d, 0) + v

    if not fi_agg:
        return result

    fi_series = pd.Series(fi_agg).sort_index()
    it_series = pd.Series(it_agg).sort_index()

    result["fi_total_5d"]         = int(fi_series.tail(5).sum())
    result["fi_total_20d"]        = int(fi_series.tail(20).sum())
    result["fi_consecutive_days"] = _consecutive_days(fi_series)
    result["it_total_5d"]         = int(it_series.tail(5).sum()) if not it_series.empty else 0

    fi5  = result["fi_total_5d"]
    cons = result["fi_consecutive_days"]
    s5   = float(1 / (1 + np.exp(-fi5 / 10_000)))
    sc   = float(1 / (1 + np.exp(-cons / 3)))
    result["flow_score"] = round(s5 * 0.6 + sc * 0.4, 4)

    return result
