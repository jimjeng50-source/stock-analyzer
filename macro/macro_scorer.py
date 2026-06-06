"""
總體資金面評分：整合匯率、VIX、期貨、資金流向，
輸出 macro_score (0~1) 與個股評分乘數 (0.7~1.0)。
"""

import numpy as np


_WEIGHTS = {
    "fx_score":      0.25,
    "vix_score":     0.25,
    "futures_score": 0.30,
    "flow_score":    0.20,
}


def _fx_to_score(fx: dict) -> float:
    """匯率因子 → 0~1 分數。"""
    chg5 = fx.get("twd_5d_chg", 0)
    chg20 = fx.get("twd_20d_chg", 0)
    trend = fx.get("twd_trend", 0)
    vs_ma = fx.get("twd_vs_ma20", 0)

    def sig(x, s=1.0):
        return float(1 / (1 + np.exp(-x / s)))

    s = sig(chg5, 0.5) * 0.4 + sig(chg20, 1.0) * 0.3 + sig(trend * 100, 0.5) * 0.15 + sig(vs_ma, 1.0) * 0.15
    return round(s, 4)


def _macro_signal(score: float) -> str:
    if score >= 0.75:
        return "🟢 資金強勢流入，做多環境佳"
    elif score >= 0.55:
        return "🟡 資金小幅流入，中性偏多"
    elif score >= 0.40:
        return "🟠 資金中性，保守操作"
    return "🔴 資金流出警示，降低持倉"


def calc_macro_score() -> dict:
    """
    計算總體資金面評分。

    回傳：
    {
        "macro_score":  0.72,        # 0~1
        "components": {
            "fx_score":      0.80,
            "vix_score":     0.75,
            "futures_score": 0.65,
            "flow_score":    0.68,
        },
        "signal":     "資金積極流入",
        "multiplier": 0.916,         # 0.7~1.0，用於個股評分
        "raw": { ... }               # 所有子模組原始數值
    }
    """
    from macro.fx import compute_fx
    from macro.vix import compute_vix
    from macro.futures import compute_futures
    from macro.fund_flow import compute_fund_flow

    raw_fx = compute_fx()
    raw_vix = compute_vix()
    raw_futures = compute_futures()
    raw_flow = compute_fund_flow()

    components = {
        "fx_score":      _fx_to_score(raw_fx),
        "vix_score":     raw_vix.get("vix_signal", 0.5),
        "futures_score": raw_futures.get("futures_score", 0.5),
        "flow_score":    raw_flow.get("flow_score", 0.5),
    }

    macro_score = round(
        sum(components[k] * _WEIGHTS[k] for k in _WEIGHTS), 4
    )

    # 乘數：macro_score=0 → 0.7，macro_score=1 → 1.0
    multiplier = round(0.7 + 0.3 * macro_score, 4)

    return {
        "macro_score": macro_score,
        "components": components,
        "signal": _macro_signal(macro_score),
        "multiplier": multiplier,
        "raw": {**raw_fx, **raw_vix, **raw_futures, **raw_flow},
    }
