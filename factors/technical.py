import numpy as np
import pandas as pd

try:
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    print("[警告] ta 套件未安裝，技術面因子將設為中性值")


def _get_last(series: pd.Series, default=0.0):
    val = series.dropna()
    return float(val.iloc[-1]) if not val.empty else default


def compute_technical(price_df: pd.DataFrame) -> dict:
    """
    計算技術面因子（原始數值，未標準化）。

    回傳 dict 包含：
        above_ma5, above_ma20, above_ma60, ma_alignment, ma20_deviation,
        rsi_14, rsi_signal, macd_histogram, macd_cross,
        bb_position, vol_ratio, vol_trend
    """
    neutral = {
        "above_ma5": 0, "above_ma20": 0, "above_ma60": 0,
        "ma_alignment": 1, "ma20_deviation": 0.0,
        "rsi_14": 50.0, "rsi_signal": 0,
        "macd_histogram": 0.0, "macd_cross": 0,
        "bb_position": 0.5, "vol_ratio": 1.0, "vol_trend": 0,
    }

    if price_df.empty or "close" not in price_df.columns or not _HAS_TA:
        return neutral

    df = price_df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None

    if len(close) < 20:
        return neutral

    result = neutral.copy()

    try:
        # ── 均線 ──────────────────────────────────────────
        ma5 = SMAIndicator(close=close, window=5).sma_indicator()
        ma10 = SMAIndicator(close=close, window=10).sma_indicator()
        ma20 = SMAIndicator(close=close, window=20).sma_indicator()
        ma60 = SMAIndicator(close=close, window=60).sma_indicator()

        c = _get_last(close)
        m5 = _get_last(ma5)
        m10 = _get_last(ma10)
        m20 = _get_last(ma20)
        m60 = _get_last(ma60)

        result["above_ma5"] = 1 if c > m5 else -1
        result["above_ma20"] = 1 if c > m20 else -1
        result["above_ma60"] = 1 if c > m60 else -1

        # 多頭排列計數（MA5>MA10>MA20>MA60 各算 1 分，共 0~3）
        alignment = 0
        if m5 > m10:
            alignment += 1
        if m10 > m20:
            alignment += 1
        if m20 > m60:
            alignment += 1
        result["ma_alignment"] = alignment

        if m20 > 0:
            result["ma20_deviation"] = round((c - m20) / m20 * 100, 2)

        # ── RSI ──────────────────────────────────────────
        rsi = RSIIndicator(close=close, window=14).rsi()
        rsi_val = _get_last(rsi, default=50.0)
        result["rsi_14"] = round(rsi_val, 2)

        if rsi_val < 30:
            result["rsi_signal"] = 1   # 超賣，潛在買點
        elif rsi_val > 70:
            result["rsi_signal"] = -1  # 超買，潛在賣點
        else:
            result["rsi_signal"] = 0

        # ── MACD ──────────────────────────────────────────
        macd_obj = MACD(close=close)
        hist = macd_obj.macd_diff()
        macd_line = macd_obj.macd()
        sig_line = macd_obj.macd_signal()

        result["macd_histogram"] = round(_get_last(hist), 4)

        # 黃金/死亡交叉：判斷最後兩根柱狀圖符號變化
        hist_clean = hist.dropna()
        if len(hist_clean) >= 2:
            prev, curr = float(hist_clean.iloc[-2]), float(hist_clean.iloc[-1])
            if prev < 0 and curr >= 0:
                result["macd_cross"] = 1   # 黃金交叉
            elif prev > 0 and curr <= 0:
                result["macd_cross"] = -1  # 死亡交叉

        # ── 布林通道 ──────────────────────────────────────
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_pos = bb.bollinger_pband().clip(0, 1)
        result["bb_position"] = round(_get_last(bb_pos, default=0.5), 4)

        # ── 成交量 ──────────────────────────────────────────
        if volume is not None and len(volume.dropna()) >= 20:
            vol_ma20 = SMAIndicator(close=volume, window=20).sma_indicator()
            vol_ma5 = SMAIndicator(close=volume, window=5).sma_indicator()

            vm20 = _get_last(vol_ma20, default=1.0)
            v_today = _get_last(volume, default=0.0)
            vm5 = _get_last(vol_ma5, default=1.0)

            result["vol_ratio"] = round(v_today / (vm20 + 1e-9), 2)
            result["vol_trend"] = 1 if vm5 > vm20 else -1

    except Exception as e:
        print(f"[警告] 技術指標計算失敗：{e}")
        return neutral

    return result
