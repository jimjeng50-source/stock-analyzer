import numpy as np
import pandas as pd


def compute_momentum(price_df: pd.DataFrame) -> dict:
    """
    計算動能面因子（原始數值，未標準化）。

    回傳 dict 包含：
        ret_5d, ret_1m, ret_3m, vol_20d, high_52w_pct, momentum_accel
    """
    result = {
        "ret_5d": 0.0, "ret_1m": 0.0, "ret_3m": 0.0,
        "vol_20d": 30.0, "high_52w_pct": -10.0, "momentum_accel": 0.0,
    }

    if price_df.empty or "close" not in price_df.columns:
        return result

    df = price_df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").dropna()

    if len(close) < 5:
        return result

    c_now = float(close.iloc[-1])

    # ── 報酬率 ──────────────────────────────────────────
    if len(close) >= 6:
        result["ret_5d"] = round((c_now / float(close.iloc[-6]) - 1) * 100, 2)

    if len(close) >= 21:
        result["ret_1m"] = round((c_now / float(close.iloc[-21]) - 1) * 100, 2)

    if len(close) >= 61:
        result["ret_3m"] = round((c_now / float(close.iloc[-61]) - 1) * 100, 2)

    # ── 波動度（20 日年化） ──────────────────────────────
    if len(close) >= 20:
        log_ret = np.log(close / close.shift(1)).dropna()
        recent_vol = log_ret.tail(20)
        vol_daily = float(recent_vol.std())
        result["vol_20d"] = round(vol_daily * np.sqrt(252) * 100, 2)

    # ── 距 52 週高點百分比（負值） ──────────────────────
    lookback = min(len(close), 252)
    high_52w = float(close.tail(lookback).max())
    if high_52w > 0:
        result["high_52w_pct"] = round((c_now / high_52w - 1) * 100, 2)

    # ── 動能加速度（5 日報酬 - 20 日報酬） ──────────────
    result["momentum_accel"] = round(result["ret_5d"] - result["ret_1m"], 2)

    return result
