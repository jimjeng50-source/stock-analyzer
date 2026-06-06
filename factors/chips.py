import numpy as np
import pandas as pd


def _safe_last(series: pd.Series, default=0.0):
    val = series.dropna()
    return float(val.iloc[-1]) if not val.empty else default


def _linear_slope(series: pd.Series) -> float:
    """最小二乘線性迴歸斜率，資料不足時回傳 0。"""
    arr = series.dropna().values
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    coeffs = np.polyfit(x, arr, 1)
    return float(coeffs[0])


def _consecutive_days(series: pd.Series) -> int:
    """
    計算最近連續買超天數（正值）或連續賣超天數（負值）。
    series 為每日淨買賣超（正=買超，負=賣超）。
    """
    arr = series.dropna().values
    if len(arr) == 0:
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


def compute_chips(institutional_df: pd.DataFrame, margin_df: pd.DataFrame) -> dict:
    """
    計算籌碼面因子，所有因子為原始數值（未標準化）。

    回傳 dict 包含：
        fi_5d_net, fi_20d_net, fi_consecutive, fi_trend,
        it_5d_net, it_20d_net, it_consecutive,
        dealer_5d_net, margin_chg_5d, short_chg_5d
    """
    result = {
        "fi_5d_net": 0.0, "fi_20d_net": 0.0, "fi_consecutive": 0,
        "fi_trend": 0.0, "it_5d_net": 0.0, "it_20d_net": 0.0,
        "it_consecutive": 0, "dealer_5d_net": 0.0,
        "margin_chg_5d": 0.0, "short_chg_5d": 0.0,
    }

    # ── 三大法人 ──────────────────────────────────────────
    # FinMind name 欄位可能為中文或英文，兩者都支援
    _FI_NAMES     = {"外資", "外資及陸資", "Foreign_Investor"}
    _IT_NAMES     = {"投信", "Investment_Trust"}
    _DEALER_NAMES = {"自營商", "自營商(自行買賣)", "Dealer_self", "Dealer_Hedging"}

    if not institutional_df.empty and "name" in institutional_df.columns and "net" in institutional_df.columns:
        # 外資
        fi_mask = institutional_df["name"].isin(_FI_NAMES)
        fi_df = institutional_df[fi_mask].copy()
        if not fi_df.empty:
            fi_net = fi_df.groupby("date")["net"].sum().sort_index()
            result["fi_5d_net"] = float(fi_net.tail(5).sum())
            result["fi_20d_net"] = float(fi_net.tail(20).sum())
            result["fi_consecutive"] = _consecutive_days(fi_net)
            result["fi_trend"] = _linear_slope(fi_net.tail(10))

        # 投信
        it_mask = institutional_df["name"].isin(_IT_NAMES)
        it_df = institutional_df[it_mask].copy()
        if not it_df.empty:
            it_net = it_df.groupby("date")["net"].sum().sort_index()
            result["it_5d_net"] = float(it_net.tail(5).sum())
            result["it_20d_net"] = float(it_net.tail(20).sum())
            result["it_consecutive"] = _consecutive_days(it_net)

        # 自營商
        dealer_mask = institutional_df["name"].isin(_DEALER_NAMES)
        dealer_df = institutional_df[dealer_mask].copy()
        if not dealer_df.empty:
            dealer_net = dealer_df.groupby("date")["net"].sum().sort_index()
            result["dealer_5d_net"] = float(dealer_net.tail(5).sum())

    # ── 融資融券 ──────────────────────────────────────────
    if not margin_df.empty:
        # 融資餘額欄位（FinMind 可能為 MarginPurchaseTodayBalance 或 margin_purchase_today_balance）
        margin_col = next(
            (c for c in margin_df.columns if "marginpurchasetodaybalance" in c.lower().replace("_", "")),
            None
        )
        short_col = next(
            (c for c in margin_df.columns if "shortsaletodaybalance" in c.lower().replace("_", "")),
            None
        )

        if margin_col:
            margin_bal = pd.to_numeric(margin_df[margin_col], errors="coerce").dropna()
            if len(margin_bal) >= 5:
                chg = (margin_bal.iloc[-1] - margin_bal.iloc[-5]) / (margin_bal.iloc[-5] + 1e-9) * 100
                result["margin_chg_5d"] = float(chg)

        if short_col:
            short_bal = pd.to_numeric(margin_df[short_col], errors="coerce").dropna()
            if len(short_bal) >= 5:
                chg = (short_bal.iloc[-1] - short_bal.iloc[-5]) / (short_bal.iloc[-5] + 1e-9) * 100
                result["short_chg_5d"] = float(chg)

    return result
