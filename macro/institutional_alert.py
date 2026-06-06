"""
三大法人五日多空指標與異常警示。

針對外資、投信、自營商各自分析：
  - 近5日 / 20日累計淨買賣超
  - 連續買超 / 賣超天數
  - Z 分數（相對近20日均值的偏離）
  - 趨勢逆轉偵測
  - 三大法人同步偵測

以代理股票（台積電、聯發科、鴻海等）加總模擬全市場流向，
並在出現以下異常時發出警示：
  1. 單日淨買賣超超過 Z > ±2.5σ
  2. 連續賣超 / 買超 ≥ 5 天
  3. 趨勢逆轉（前5日方向 ↔ 後5日方向）
  4. 外資 + 投信同步大買超 / 大賣超
"""

import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from config import FINMIND_TOKEN

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"

# 代理股票（按市值加權排序）
_PROXY_STOCKS = ["2330", "2454", "2317", "2382", "2308", "3711", "2303", "2881", "2882", "2412"]

_FI_NAMES     = {"Foreign_Investor", "外資及陸資", "外資"}
_IT_NAMES     = {"Investment_Trust", "投信"}
_DEALER_NAMES = {"Dealer_self", "Dealer_Hedging", "自營商", "自營商(自行買賣)"}

# 連續買賣超警示門檻（天）
_CONSECUTIVE_ALERT_DAYS = 5


def _fm_stock(stock_id: str, days: int = 45) -> pd.DataFrame:
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


def _aggregate_all(days: int = 45) -> dict[str, pd.Series]:
    """彙整代理股票的外資、投信、自營商每日淨買賣超。"""
    fi_agg, it_agg, dl_agg = {}, {}, {}

    for sid in _PROXY_STOCKS[:6]:
        df = _fm_stock(sid, days=days)
        if df.empty or "name" not in df.columns:
            continue
        for col in ["buy", "sell"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "net" not in df.columns and "buy" in df.columns:
            df["net"] = df["buy"] - df["sell"]
        if "net" not in df.columns:
            continue

        for date, grp in df.groupby("date"):
            fi_agg[date] = fi_agg.get(date, 0) + float(grp[grp["name"].isin(_FI_NAMES)]["net"].sum())
            it_agg[date] = it_agg.get(date, 0) + float(grp[grp["name"].isin(_IT_NAMES)]["net"].sum())
            dl_agg[date] = dl_agg.get(date, 0) + float(grp[grp["name"].isin(_DEALER_NAMES)]["net"].sum())

    return {
        "fi":     pd.Series(fi_agg).sort_index()     if fi_agg else pd.Series(dtype=float),
        "it":     pd.Series(it_agg).sort_index()     if it_agg else pd.Series(dtype=float),
        "dealer": pd.Series(dl_agg).sort_index()     if dl_agg else pd.Series(dtype=float),
    }


def _consecutive_days(series: pd.Series) -> int:
    arr = series.dropna().values
    if not len(arr):
        return 0
    sign = 1 if arr[-1] > 0 else (-1 if arr[-1] < 0 else 0)
    if sign == 0:
        return 0
    count = 0
    for v in reversed(arr):
        if (v > 0 and sign == 1) or (v < 0 and sign == -1):
            count += 1
        else:
            break
    return count * sign


def _detect_anomalies(series: pd.Series, name: str) -> list[dict]:
    """偵測異常，回傳警示字典列表。"""
    alerts = []
    if series.empty or len(series) < 6:
        return alerts

    vals = series.dropna()

    # ── Z 分數：相對近20日的偏離 ──────────────────────────
    if len(vals) >= 10:
        recent = vals.tail(20)
        mean = float(recent.mean())
        std = float(recent.std())
        last_val = float(vals.iloc[-1])
        if std > 0:
            z = (last_val - mean) / std
            if z < -2.5:
                alerts.append({
                    "level": "🔴",
                    "type": "異常大量賣超",
                    "name": name,
                    "msg": f"{name} 昨日賣超偏離均值 {z:.1f}σ（20日），異常大量出場",
                })
            elif z > 2.5:
                alerts.append({
                    "level": "🟢",
                    "type": "異常大量買超",
                    "name": name,
                    "msg": f"{name} 昨日買超偏離均值 +{z:.1f}σ（20日），異常大量進場",
                })

    # ── 連續買賣超 ────────────────────────────────────────
    cons = _consecutive_days(vals)
    if cons <= -_CONSECUTIVE_ALERT_DAYS:
        alerts.append({
            "level": "🔴",
            "type": "持續賣超警示",
            "name": name,
            "msg": f"{name} 已連續賣超 {abs(cons)} 天，中期趨勢偏空",
        })
    elif cons >= _CONSECUTIVE_ALERT_DAYS:
        alerts.append({
            "level": "🟢",
            "type": "持續買超訊號",
            "name": name,
            "msg": f"{name} 已連續買超 {cons} 天，中期趨勢偏多",
        })

    # ── 趨勢逆轉偵測 ─────────────────────────────────────
    if len(vals) >= 10:
        pre5  = float(vals.iloc[-10:-5].sum())
        last5 = float(vals.tail(5).sum())
        if pre5 > 0 and last5 < 0:
            alerts.append({
                "level": "🔴",
                "type": "趨勢逆轉（由多轉空）",
                "name": name,
                "msg": f"{name} 近5日轉為賣超（前5日+{pre5:,.0f} → 後5日{last5:,.0f}），趨勢逆轉",
            })
        elif pre5 < 0 and last5 > 0:
            alerts.append({
                "level": "🟢",
                "type": "趨勢逆轉（由空轉多）",
                "name": name,
                "msg": f"{name} 近5日轉為買超（前5日{pre5:,.0f} → 後5日+{last5:,.0f}），趨勢逆轉",
            })

    return alerts


def _build_stats(series: pd.Series, label: str) -> dict:
    if series.empty:
        return {
            "5d_net": 0, "20d_net": 0, "consecutive": 0,
            "direction": "中性", "z_score": 0.0, "label": label,
        }
    net5  = int(series.tail(5).sum())
    net20 = int(series.tail(20).sum())
    cons  = _consecutive_days(series)

    z = 0.0
    if len(series) >= 10:
        recent = series.tail(20)
        std = float(recent.std())
        if std > 0:
            z = round((float(series.iloc[-1]) - float(recent.mean())) / std, 2)

    direction = "多" if net5 > 0 else ("空" if net5 < 0 else "中性")
    return {
        "5d_net": net5, "20d_net": net20,
        "consecutive": cons, "direction": direction,
        "z_score": z, "label": label,
    }


def compute_institutional_signals() -> dict:
    """
    計算三大法人五日多空指標與異常警示。

    Returns
    -------
    {
        "fi":     {"5d_net", "20d_net", "consecutive", "direction", "z_score", "label"},
        "it":     {...},
        "dealer": {...},
        "alerts": [{"level", "type", "name", "msg"}, ...],
        "combined_signal": "多頭" | "空頭" | "分歧",
        "series": {"fi": pd.Series, "it": pd.Series, "dealer": pd.Series},
        "available": bool,
    }
    """
    if not FINMIND_TOKEN:
        return _empty_result()

    data = _aggregate_all(days=45)
    fi_s, it_s, dl_s = data["fi"], data["it"], data["dealer"]

    fi_stats = _build_stats(fi_s,    "外資")
    it_stats = _build_stats(it_s,    "投信")
    dl_stats = _build_stats(dl_s,    "自營商")

    # ── 三大法人多空計票 ──────────────────────────────────
    bull = sum(1 for s in [fi_stats, it_stats, dl_stats] if s["direction"] == "多")
    bear = sum(1 for s in [fi_stats, it_stats, dl_stats] if s["direction"] == "空")
    combined = "多頭" if bull >= 2 else ("空頭" if bear >= 2 else "分歧")

    # ── 各法人異常偵測 ────────────────────────────────────
    alerts = []
    for s, name in [(fi_s, "外資"), (it_s, "投信"), (dl_s, "自營商")]:
        alerts.extend(_detect_anomalies(s, name))

    # ── 外資 + 投信同步偵測（最強訊號）────────────────────
    if not fi_s.empty and not it_s.empty and len(fi_s) >= 5 and len(it_s) >= 5:
        fi5 = float(fi_s.tail(5).sum())
        it5 = float(it_s.tail(5).sum())
        if fi5 < 0 and it5 < 0:
            alerts.append({
                "level": "🔴",
                "type": "外資投信同步賣超",
                "name": "綜合",
                "msg": f"外資+投信近5日均賣超（外資 {fi5:,.0f} / 投信 {it5:,.0f}），雙重利空警示",
            })
        elif fi5 > 0 and it5 > 0:
            alerts.append({
                "level": "🟢",
                "type": "外資投信同步買超",
                "name": "綜合",
                "msg": f"外資+投信近5日均買超（外資 +{fi5:,.0f} / 投信 +{it5:,.0f}），雙重利多",
            })

    return {
        "fi": fi_stats,
        "it": it_stats,
        "dealer": dl_stats,
        "alerts": alerts,
        "combined_signal": combined,
        "series": data,
        "available": True,
    }


def _empty_result() -> dict:
    empty_stats = {"5d_net": 0, "20d_net": 0, "consecutive": 0, "direction": "中性", "z_score": 0.0}
    return {
        "fi": {**empty_stats, "label": "外資"},
        "it": {**empty_stats, "label": "投信"},
        "dealer": {**empty_stats, "label": "自營商"},
        "alerts": [],
        "combined_signal": "中性",
        "series": {k: pd.Series(dtype=float) for k in ["fi", "it", "dealer"]},
        "available": False,
    }


def per_stock_alerts(institutional_df: pd.DataFrame, stock_id: str = "") -> list[dict]:
    """
    針對單一股票的三大法人資料產生警示。
    直接使用 FinMindFetcher.get_institutional() 的回傳值。
    """
    if institutional_df.empty or "name" not in institutional_df.columns:
        return []

    alerts = []
    for names, label in [(_FI_NAMES, "外資"), (_IT_NAMES, "投信"), (_DEALER_NAMES, "自營商")]:
        sub = institutional_df[institutional_df["name"].isin(names)]
        if sub.empty:
            continue
        series = sub.groupby("date")["net"].sum().sort_index()
        alerts.extend(_detect_anomalies(series, f"{stock_id} {label}"))
    return alerts
