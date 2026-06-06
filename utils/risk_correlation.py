"""
風險相關係數分析。

流程：
  1. 找出歷史中跌幅 ≥ drop_threshold 的所有交易日
  2. 對各前置技術/法人指標計算 Pearson 相關係數
  3. 輸出相關係數最高的因子，列為風險指標

用法：
    from utils.risk_correlation import compute_risk_correlations
    result = compute_risk_correlations(price_df, institutional_df, drop_threshold=-5)
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from scipy.stats import pearsonr, spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """安全版 Pearson r（返回 r, p_value）。"""
    try:
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        if len(x) < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0, 1.0
        if _HAS_SCIPY:
            r, p = pearsonr(x, y)
            return (float(r) if not np.isnan(r) else 0.0), float(p)
        r = float(np.corrcoef(x, y)[0, 1])
        return (r if not np.isnan(r) else 0.0), 1.0
    except Exception:
        return 0.0, 1.0


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _factor_label(key: str) -> str:
    _MAP = {
        "rsi_14":        "RSI(14)",
        "c_vs_ma5":      "偏離MA5 (%)",
        "c_vs_ma20":     "偏離MA20 (%)",
        "c_vs_ma60":     "偏離MA60 (%)",
        "vol_10d":       "10日年化波動率",
        "vol_20d":       "20日年化波動率",
        "vol_ratio":     "量比（當日/20日均量）",
        "ret_5d_pre":    "前5日報酬 (%)",
        "ret_20d_pre":   "前20日報酬 (%)",
        "ret_lag1":      "前1日報酬 (%)",
        "ret_lag2":      "前2日報酬 (%)",
        "bb_pos":        "布林通道位置 (0~1)",
        "macd_hist":     "MACD 柱狀值",
        "fi_net":        "外資當日買賣超（張）",
        "fi_net_5d":     "外資5日累計買賣超",
        "it_net":        "投信當日買賣超（張）",
        "it_net_5d":     "投信5日累計買賣超",
        "ma5_slope":     "MA5 斜率（5日變化）",
        "high_dist":     "距近10日高點 (%)",
        "prev_drop":     "前日是否大跌（≥3%）",
    }
    return _MAP.get(key, key)


def _interpret(factor: str, corr: float) -> str:
    label = _factor_label(factor)
    strength = "強" if abs(corr) > 0.45 else ("中" if abs(corr) > 0.25 else "弱")
    sign_txt = "正相關" if corr > 0 else "負相關"
    if factor in ("vol_10d", "vol_20d"):
        return f"{label}：波動率高時跌幅更劇烈（{strength}）r={corr:.3f}"
    if factor in ("rsi_14",):
        if corr < 0:
            return f"{label}：RSI 越低時越容易大跌（{strength}，超賣可能持續）r={corr:.3f}"
        return f"{label}：RSI 越高（超買）時越容易大跌（{strength}）r={corr:.3f}"
    if factor in ("fi_net", "fi_net_5d"):
        return f"{label}：外資賣超越多當日跌幅越大（{strength}）r={corr:.3f}"
    if "c_vs_ma" in factor:
        return f"{label}：價格偏離均線越多則下跌風險越大（{strength}）r={corr:.3f}"
    return f"{label}：{sign_txt}（{strength}）r={corr:.3f}"


# ── 主要函式 ─────────────────────────────────────────────────────────────────

def compute_risk_correlations(
    price_df: pd.DataFrame,
    institutional_df: pd.DataFrame | None = None,
    drop_threshold: float = -5.0,
    lookback_days: int = 365,
) -> dict:
    """
    分析歷史跌幅 ≥ |drop_threshold| 日的統計相關性。

    Parameters
    ----------
    price_df        : DataFrame，須含 [date, close]；建議含 volume
    institutional_df: DataFrame，須含 [date, name, net]（可為 None）
    drop_threshold  : 跌幅閾值（負數），預設 -5.0 代表跌幅 ≥ 5%
    lookback_days   : 回溯天數，預設 365 天

    Returns
    -------
    {
        "drop_count": int,
        "avg_drop":   float,
        "max_drop":   float,
        "drop_events": list[dict],        # 每次大跌日期+跌幅
        "correlations": list[dict],       # 所有相關係數
        "top_risk_factors": list[dict],   # 前 5 個最高負相關（跌跌相關）
        "risk_score": float,              # 0~100，數值越高當前風險越高
        "risk_level": str,
        "message": str,
    }
    """
    if price_df is None or price_df.empty or "close" not in price_df.columns:
        return _empty_result("無價格資料")

    df = price_df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["close"]).tail(lookback_days + 60).reset_index(drop=True)
    df["ret"] = df["close"].pct_change() * 100

    # ── 識別大跌事件 ──────────────────────────────────────────────
    drop_mask = df["ret"] <= drop_threshold
    drop_df   = df[drop_mask][["date", "ret"]].rename(columns={"ret": "pct_change"})
    drop_count = len(drop_df)

    if drop_count < 5:
        return {
            **_empty_result(),
            "drop_count": drop_count,
            "drop_events": drop_df.to_dict("records"),
            "message": (
                f"歷史跌幅≥{abs(drop_threshold):.0f}% 事件僅 {drop_count} 次，"
                "樣本不足（建議延長回溯天數或降低閾值）"
            ),
        }

    # ── 衍生因子特徵 ──────────────────────────────────────────────
    df = _build_features(df, institutional_df)

    feature_cols = [c for c in df.columns if c not in {"date", "ret", "close", "open", "high", "low", "volume"}]

    # ── 計算相關係數 ──────────────────────────────────────────────
    y = df["ret"].values
    correlations = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].values
        r, p = _safe_pearson(x, y)
        if abs(r) > 0.05:
            correlations.append({
                "factor":         col,
                "factor_label":   _factor_label(col),
                "correlation":    round(r, 4),
                "p_value":        round(p, 4),
                "abs_corr":       abs(r),
                "interpretation": _interpret(col, r),
            })

    correlations.sort(key=lambda x: x["abs_corr"], reverse=True)

    # 負相關因子 = 該因子偏高時，跌幅往往偏大（風險因子）
    top_risk = [c for c in correlations if c["correlation"] < -0.05][:5]

    # ── 即時風險分數（以當前因子值估算）─────────────────────────────
    risk_score, risk_level = _current_risk_score(df, top_risk)

    avg_drop = round(float(drop_df["pct_change"].mean()), 2)
    max_drop = round(float(drop_df["pct_change"].min()), 2)

    return {
        "drop_count":       drop_count,
        "avg_drop":         avg_drop,
        "max_drop":         max_drop,
        "drop_events":      drop_df.tail(30).to_dict("records"),
        "correlations":     correlations[:12],
        "top_risk_factors": top_risk,
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "message":          "",
    }


def _build_features(df: pd.DataFrame, inst_df: pd.DataFrame | None) -> pd.DataFrame:
    """衍生技術指標特徵，回傳擴充後的 DataFrame。"""
    try:
        c = df["close"].astype(float)
        v = (df["volume"].astype(float) if "volume" in df.columns
             else pd.Series(np.nan, index=df.index))

        ma5  = c.rolling(5,  min_periods=1).mean()
        ma20 = c.rolling(20, min_periods=1).mean()
        ma60 = c.rolling(60, min_periods=1).mean()

        df["c_vs_ma5"]  = (c - ma5)  / (ma5  + 1e-9) * 100
        df["c_vs_ma20"] = (c - ma20) / (ma20 + 1e-9) * 100
        df["c_vs_ma60"] = (c - ma60) / (ma60 + 1e-9) * 100

        df["rsi_14"] = _rsi(c, 14)

        log_ret = np.log(c / c.shift(1))
        df["vol_10d"] = log_ret.rolling(10).std() * np.sqrt(252) * 100
        df["vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

        if not v.isna().all():
            mv = v.rolling(20, min_periods=1).mean()
            df["vol_ratio"] = v / (mv + 1e-9)

        df["ret_5d_pre"]  = c.pct_change(5).shift(1)  * 100
        df["ret_20d_pre"] = c.pct_change(20).shift(1) * 100
        df["ret_lag1"]    = df["ret"].shift(1)
        df["ret_lag2"]    = df["ret"].shift(2)

        # 布林通道位置
        std20 = c.rolling(20).std()
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        band = (bb_upper - bb_lower).replace(0, np.nan)
        df["bb_pos"] = (c - bb_lower) / band

        # MACD 柱狀值
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd - signal

        # 距近10日高點偏離
        roll_high = c.rolling(10).max()
        df["high_dist"] = (c - roll_high) / (roll_high + 1e-9) * 100

        # 前一日是否大跌（≥3%）
        df["prev_drop"] = (df["ret"].shift(1) <= -3).astype(float)

        # MA5 斜率
        df["ma5_slope"] = ma5.diff(5)

        # 法人買賣超特徵
        if inst_df is not None and not inst_df.empty and "net" in inst_df.columns:
            _add_institutional_features(df, inst_df)

    except Exception:
        pass

    return df


def _add_institutional_features(df: pd.DataFrame, inst_df: pd.DataFrame) -> None:
    """將法人買賣超指標併入 df（in-place）。"""
    inst = inst_df.copy()
    inst["date"] = pd.to_datetime(inst["date"])

    _FI = {"外資", "外資及陸資", "Foreign_Investor"}
    _IT = {"投信", "Investment_Trust"}

    fi = inst[inst["name"].isin(_FI)].groupby("date")["net"].sum()
    it = inst[inst["name"].isin(_IT)].groupby("date")["net"].sum()

    df["fi_net"] = df["date"].map(fi.to_dict())
    df["it_net"] = df["date"].map(it.to_dict())
    df["fi_net_5d"] = df["fi_net"].rolling(5).sum()
    df["it_net_5d"] = df["it_net"].rolling(5).sum()


def _current_risk_score(df: pd.DataFrame, top_risk: list[dict]) -> tuple[float, str]:
    """
    以最新一行的因子值與歷史相關係數估算當前風險分數（0~100）。
    分數越高代表當前環境與歷史大跌時較接近。
    """
    if df.empty or not top_risk:
        return 50.0, "資料不足"

    latest = df.dropna(subset=["close"]).iloc[-1]
    scores = []

    for item in top_risk:
        col = item["factor"]
        r   = item["correlation"]   # 負值
        if col not in df.columns:
            continue
        all_vals = df[col].dropna()
        if all_vals.empty or len(all_vals) < 10:
            continue
        cur_val = float(latest.get(col, np.nan))
        if np.isnan(cur_val):
            continue

        # 負相關：該因子偏低 → 跌幅偏大
        # 以百分位換算：當前值在歷史分布的百分位越低，風險越高
        pct = float((all_vals < cur_val).mean())  # 當前值高於多少比例的歷史值
        # 負相關因子：pct 越低（當前值偏低）→ 風險越高
        risk_component = (1 - pct) * abs(r) * 100
        scores.append(risk_component)

    if not scores:
        return 50.0, "無足夠相關因子"

    risk_score = round(float(np.mean(scores)), 1)
    risk_score = max(0.0, min(100.0, risk_score))

    if risk_score >= 70:
        level = "🔴 高風險"
    elif risk_score >= 50:
        level = "🟠 中高風險"
    elif risk_score >= 30:
        level = "🟡 中低風險"
    else:
        level = "🟢 低風險"

    return risk_score, level


def _empty_result(message: str = "") -> dict:
    return {
        "drop_count": 0, "avg_drop": 0.0, "max_drop": 0.0,
        "drop_events": [], "correlations": [], "top_risk_factors": [],
        "risk_score": 50.0, "risk_level": "資料不足", "message": message,
    }
