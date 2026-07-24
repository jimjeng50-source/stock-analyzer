"""
screener/weight_tuner.py
週度推薦邏輯調參：依近期已評估推薦的「各因子分 vs 實際 60 日報酬」相關性，
微調因子權重，朝提升勝率（目標 70%）的方向調整。

設計原則（避免過擬合/劇烈擺動）：
- 樣本不足（< MIN_SAMPLES）直接跳過，維持現行權重。
- 用相關性做「乘法微調」，每個權重每週最多變動 MAX_STEP。
- 夾限至 [W_FLOOR, W_CAP] 後正規化為總和 1。
- 只有實質變動（>0.005）才視為有調整。
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_WIN_RATE = 0.70

# 因子 → daily_recommendations 內對應的分數欄位
FACTOR_COLS = {
    "fundamental": "fundamental_score",
    "chips": "chips_score",
    "risk": "risk_score",
    "technical": "technical_score",
    "momentum": "momentum_score",
}

MIN_SAMPLES = 12
LEARNING_RATE = 0.5
MAX_STEP = 0.05     # 每週每個權重最多變動 0.05
W_FLOOR = 0.05
W_CAP = 0.60


def _safe_corr(scores: np.ndarray, returns: np.ndarray) -> float:
    """相關係數，資料不足或零變異回傳 0。"""
    mask = ~np.isnan(scores) & ~np.isnan(returns)
    if mask.sum() < MIN_SAMPLES:
        return 0.0
    s, r = scores[mask], returns[mask]
    if np.std(s) == 0 or np.std(r) == 0:
        return 0.0
    c = np.corrcoef(s, r)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def compute_tuned_weights(df, current_weights: dict, target_win_rate: float = TARGET_WIN_RATE):
    """
    回傳 (new_weights, report)。

    report: {n, win_rate, correlations, changes, tuned(bool), reason, target}
    """
    report = {
        "n": 0, "win_rate": None, "correlations": {}, "changes": {},
        "tuned": False, "reason": "", "target": target_win_rate,
    }
    if df is None or df.empty or "return_60d_pct" not in getattr(df, "columns", []):
        report["reason"] = "無已評估的推薦資料（尚未有 60 日報酬）"
        return dict(current_weights), report

    d = df.copy()
    d["return_60d_pct"] = pd.to_numeric(d["return_60d_pct"], errors="coerce")
    d = d.dropna(subset=["return_60d_pct"])
    report["n"] = int(len(d))
    if len(d) < MIN_SAMPLES:
        report["reason"] = f"樣本不足（{len(d)}<{MIN_SAMPLES}），維持現行權重"
        return dict(current_weights), report

    returns = d["return_60d_pct"].values.astype(float)
    report["win_rate"] = round(float((returns > 0).mean()), 3)

    corrs = {}
    for factor, col in FACTOR_COLS.items():
        if col in d.columns:
            scores = pd.to_numeric(d[col], errors="coerce").values.astype(float)
            corrs[factor] = _safe_corr(scores, returns)
        else:
            corrs[factor] = 0.0
    report["correlations"] = {k: round(v, 3) for k, v in corrs.items()}

    # 乘法微調：分數與報酬正相關的因子加權，負相關的減權
    new = {}
    for f, w in current_weights.items():
        delta = w * LEARNING_RATE * corrs.get(f, 0.0)
        delta = max(-MAX_STEP, min(MAX_STEP, delta))
        new[f] = w + delta
    # 夾限 + 正規化
    new = {f: min(W_CAP, max(W_FLOOR, v)) for f, v in new.items()}
    total = sum(new.values()) or 1.0
    new = {f: v / total for f, v in new.items()}

    report["changes"] = {
        f: (round(current_weights[f], 3), round(new[f], 3)) for f in current_weights
    }
    report["tuned"] = any(
        abs(new[f] - current_weights[f]) > 0.005 for f in current_weights
    )
    report["reason"] = (
        "已依近期績效微調權重" if report["tuned"]
        else "近期績效無明確方向，權重幾乎不變"
    )
    return new, report


def format_tune_message(report: dict) -> str:
    """把調參結果整理成 Telegram 訊息。"""
    zh = {"fundamental": "基本面", "chips": "籌碼", "risk": "風險",
          "technical": "技術", "momentum": "動能"}
    lines = ["🔧 每週推薦邏輯回顧"]
    wr = report.get("win_rate")
    tgt = report.get("target", TARGET_WIN_RATE)
    if wr is not None:
        gap = "✅ 已達標" if wr >= tgt else f"距目標 {(tgt - wr) * 100:.0f} 個百分點"
        lines.append(f"近期 60 日勝率：{wr * 100:.0f}%（目標 {tgt * 100:.0f}%，{gap}）")
    lines.append(f"評估樣本：{report.get('n', 0)} 筆")
    lines.append("")
    lines.append(report.get("reason", ""))

    changes = report.get("changes", {})
    if report.get("tuned") and changes:
        lines.append("")
        lines.append("權重調整（舊→新）：")
        for f, (old, new) in changes.items():
            arrow = "↑" if new > old else ("↓" if new < old else "→")
            lines.append(f"  {zh.get(f, f)} {old:.0%} {arrow} {new:.0%}")
    lines.append("")
    lines.append("＊調整依據近期實際報酬與各因子分的相關性，"
                 "每週小幅修正以提升長期勝率。僅供研究參考，不構成投資建議。")
    return "\n".join(lines)
