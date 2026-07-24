"""
tests/test_weight_tuner.py
週度調參 compute_tuned_weights / format_tune_message 單元測試
"""

import pandas as pd

from screener.weight_tuner import (
    compute_tuned_weights,
    format_tune_message,
    MIN_SAMPLES,
    W_FLOOR,
    W_CAP,
)

_CURRENT = {"chips": 0.20, "fundamental": 0.45, "technical": 0.10,
            "momentum": 0.10, "risk": 0.15}


def _df_fundamental_drives(n=20):
    """基本面分與 60 日報酬正相關；其餘因子固定（零變異→相關 0）。"""
    fund = [50 + 2 * i for i in range(n)]
    ret = [(f - 70) * 0.5 for f in fund]        # 與 fundamental 完全線性相關
    return pd.DataFrame({
        "fundamental_score": fund,
        "chips_score": [60] * n,
        "risk_score": [60] * n,
        "technical_score": [60] * n,
        "momentum_score": [60] * n,
        "return_60d_pct": ret,
    })


class TestComputeTunedWeights:
    def test_insufficient_samples_unchanged(self):
        df = _df_fundamental_drives(n=MIN_SAMPLES - 1)
        new, rep = compute_tuned_weights(df, _CURRENT)
        assert new == _CURRENT
        assert rep["tuned"] is False
        assert "樣本不足" in rep["reason"]

    def test_empty_df_unchanged(self):
        new, rep = compute_tuned_weights(pd.DataFrame(), _CURRENT)
        assert new == _CURRENT
        assert rep["tuned"] is False

    def test_positive_corr_raises_weight(self):
        df = _df_fundamental_drives(n=20)
        new, rep = compute_tuned_weights(df, _CURRENT)
        assert rep["tuned"] is True
        assert new["fundamental"] > _CURRENT["fundamental"]     # 正相關→加權
        assert rep["correlations"]["fundamental"] > 0.9

    def test_weights_normalized_and_clamped(self):
        df = _df_fundamental_drives(n=20)
        new, _ = compute_tuned_weights(df, _CURRENT)
        assert abs(sum(new.values()) - 1.0) < 1e-9
        for v in new.values():
            assert W_FLOOR - 1e-9 <= v <= W_CAP + 1e-9

    def test_negative_corr_lowers_weight(self):
        df = _df_fundamental_drives(n=20)
        df["return_60d_pct"] = -df["return_60d_pct"]           # 反轉→負相關
        new, rep = compute_tuned_weights(df, _CURRENT)
        assert rep["correlations"]["fundamental"] < -0.9
        assert new["fundamental"] < _CURRENT["fundamental"]

    def test_win_rate_reported(self):
        df = _df_fundamental_drives(n=20)
        _, rep = compute_tuned_weights(df, _CURRENT)
        assert rep["win_rate"] is not None
        assert 0.0 <= rep["win_rate"] <= 1.0
        assert rep["n"] == 20

    def test_step_capped(self):
        """單週單一權重變動不超過約 MAX_STEP（正規化前）。"""
        df = _df_fundamental_drives(n=20)
        new, _ = compute_tuned_weights(df, _CURRENT)
        # 正規化後仍應接近，不會暴衝
        assert new["fundamental"] - _CURRENT["fundamental"] < 0.06


class TestFormatMessage:
    def test_message_has_winrate_and_target(self):
        df = _df_fundamental_drives(n=20)
        _, rep = compute_tuned_weights(df, _CURRENT)
        msg = format_tune_message(rep)
        assert "每週推薦邏輯回顧" in msg
        assert "勝率" in msg
        assert "70%" in msg           # 目標
        assert "不構成投資建議" in msg

    def test_message_insufficient_samples(self):
        df = _df_fundamental_drives(n=3)
        _, rep = compute_tuned_weights(df, _CURRENT)
        msg = format_tune_message(rep)
        assert "樣本不足" in msg
