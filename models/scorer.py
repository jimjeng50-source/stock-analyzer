import numpy as np
from config import FACTOR_WEIGHTS, SCORE_THRESHOLDS


def _sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Sigmoid 歸一化，以 center 為中心，scale 控制靈敏度。回傳 (0, 1)。"""
    return float(1 / (1 + np.exp(-(x - center) / (scale + 1e-9))))


def _inv_sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """反向 Sigmoid（越小越好）。"""
    return 1.0 - _sigmoid(x, center, scale)


def _binary(x: float) -> float:
    """將 {-1, 0, 1} 線性映射至 {0, 0.5, 1}。"""
    return float(np.clip((x + 1) / 2, 0, 1))


def _linear(x: float, lo: float, hi: float) -> float:
    """線性截斷歸一化至 [0, 1]。"""
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def _pe_norm(pe: float) -> float:
    """本益比歸一化：越低越好，虧損（PE<=0）給低分。"""
    if pe <= 0:
        return 0.25
    if pe < 10:
        return 0.95
    if pe < 15:
        return 0.85
    if pe < 25:
        return 0.65
    if pe < 40:
        return 0.45
    if pe < 60:
        return 0.30
    return 0.15


# ── 各因子歸一化函式映射表 ───────────────────────────────
_NORMALIZERS = {
    # 籌碼面
    "fi_5d_net":      lambda x: _sigmoid(x, 0, 5_000),
    "fi_20d_net":     lambda x: _sigmoid(x, 0, 20_000),
    "fi_consecutive": lambda x: _sigmoid(x, 0, 3),
    "fi_trend":       lambda x: _sigmoid(x, 0, 500),
    "it_5d_net":      lambda x: _sigmoid(x, 0, 3_000),
    "it_20d_net":     lambda x: _sigmoid(x, 0, 10_000),
    "it_consecutive": lambda x: _sigmoid(x, 0, 3),
    "dealer_5d_net":  lambda x: _sigmoid(x, 0, 2_000),
    "margin_chg_5d":  lambda x: _sigmoid(x, 0, 5),    # 融資增加偏負面
    "short_chg_5d":   lambda x: _inv_sigmoid(x, 0, 5), # 融券增加偏負面

    # 技術面
    "above_ma5":      _binary,
    "above_ma20":     _binary,
    "above_ma60":     _binary,
    "ma_alignment":   lambda x: _linear(x, 0, 3),
    "ma20_deviation": lambda x: _sigmoid(x, 0, 5),
    "rsi_14":         lambda x: _sigmoid(x, 50, 15),
    "rsi_signal":     _binary,
    "macd_histogram": lambda x: _sigmoid(x, 0, 0.3),
    "macd_cross":     _binary,
    "bb_position":    lambda x: float(np.clip(x, 0, 1)),
    "vol_ratio":      lambda x: _sigmoid(x, 1, 0.5),
    "vol_trend":      _binary,

    # 基本面
    "rev_yoy":        lambda x: _sigmoid(x, 0, 15),
    "rev_mom":        lambda x: _sigmoid(x, 0, 8),
    "rev_3m_trend":   _binary,
    "rev_12m_high":   lambda x: float(np.clip(x, 0, 1)),
    "eps_latest":     lambda x: _sigmoid(x, 1, 2),
    "eps_qoq":        lambda x: _sigmoid(x, 0, 1),
    "eps_yoy":        lambda x: _sigmoid(x, 0, 1),
    "gross_margin":   lambda x: _sigmoid(x, 30, 15),
    "gpm_trend":      lambda x: _sigmoid(x, 0, 2),
    "pe_ratio":       _pe_norm,

    # 動能面
    "ret_5d":         lambda x: _sigmoid(x, 0, 3),
    "ret_1m":         lambda x: _sigmoid(x, 0, 10),
    "ret_3m":         lambda x: _sigmoid(x, 0, 20),
    "high_52w_pct":   lambda x: _sigmoid(x, -10, 10),  # 接近高點得分高
    "momentum_accel": lambda x: _sigmoid(x, 0, 3),

    # 風險面
    "vol_20d": lambda x: _inv_sigmoid(x, 30, 15),  # 波動越低風險分越高
}


# 因子分類
_CATEGORY_FACTORS = {
    "chips": [
        "fi_5d_net", "fi_20d_net", "fi_consecutive", "fi_trend",
        "it_5d_net", "it_20d_net", "it_consecutive",
        "dealer_5d_net", "margin_chg_5d", "short_chg_5d",
    ],
    "technical": [
        "above_ma5", "above_ma20", "above_ma60", "ma_alignment",
        "ma20_deviation", "rsi_14", "rsi_signal",
        "macd_histogram", "macd_cross", "bb_position",
        "vol_ratio", "vol_trend",
    ],
    "fundamental": [
        "rev_yoy", "rev_mom", "rev_3m_trend", "rev_12m_high",
        "eps_latest", "eps_qoq", "eps_yoy",
        "gross_margin", "gpm_trend", "pe_ratio",
    ],
    "momentum": [
        "ret_5d", "ret_1m", "ret_3m",
        "high_52w_pct", "momentum_accel",
    ],
    "risk": [
        "vol_20d",
    ],
}


def _recommendation(score: float) -> str:
    if score >= SCORE_THRESHOLDS["strong_buy"]:
        return "⭐⭐ 強力買進"
    if score >= SCORE_THRESHOLDS["buy"]:
        return "⭐ 買進"
    if score >= SCORE_THRESHOLDS["hold"]:
        return "◆ 持有觀望"
    if score >= SCORE_THRESHOLDS["sell"]:
        return "▼ 減碼"
    return "✕ 賣出"


class Scorer:
    """
    加權因子評分模型。

    使用方式：
        scorer = Scorer(weights)
        result = scorer.score(chips, technical, fundamental, momentum)
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or FACTOR_WEIGHTS
        # 確保權重總和為 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(
        self,
        chips: dict,
        technical: dict,
        fundamental: dict,
        momentum: dict,
    ) -> dict:
        """
        計算綜合評分。

        回傳：
            {
                "total_score": float,         # 0~100
                "category_scores": dict,      # 各類別 0~100
                "recommendation": str,
                "raw_factors": dict,          # 所有原始因子值
            }
        """
        # 合併所有因子
        raw = {}
        raw.update(chips)
        raw.update(technical)
        raw.update(fundamental)
        raw.update(momentum)

        # 標準化後逐類別平均
        cat_scores = {}
        for cat, factors in _CATEGORY_FACTORS.items():
            scores = []
            for f in factors:
                val = raw.get(f, None)
                if val is None:
                    continue
                norm_fn = _NORMALIZERS.get(f)
                if norm_fn is None:
                    continue
                try:
                    s = norm_fn(float(val))
                    s = float(np.clip(s, 0, 1))
                    scores.append(s)
                except Exception:
                    pass
            cat_scores[cat] = round(float(np.mean(scores)) * 100, 1) if scores else 50.0

        # 加權總分
        total = sum(cat_scores.get(cat, 50.0) * self.weights.get(cat, 0) for cat in _CATEGORY_FACTORS)
        total = round(total, 1)

        return {
            "total_score": total,
            "category_scores": cat_scores,
            "recommendation": _recommendation(total),
            "raw_factors": raw,
        }
