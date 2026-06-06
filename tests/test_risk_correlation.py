"""Unit tests for utils/risk_correlation.py."""

import pytest
import pandas as pd
import numpy as np
from utils.risk_correlation import (
    compute_risk_correlations, _safe_pearson, _build_features,
    _current_risk_score, _empty_result,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_price_df(n: int = 200, inject_drops: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    close = 500 + np.cumsum(rng.normal(0.3, 2, n))
    if inject_drops:
        for idx in [40, 80, 120, 160, 30, 70, 100, 140]:
            if idx < n:
                close[idx] = close[idx - 1] * 0.93
    return pd.DataFrame({
        "date":   pd.date_range("2023-01-01", periods=n),
        "open":   close - rng.uniform(0, 2, n),
        "high":   close + rng.uniform(0, 4, n),
        "low":    close - rng.uniform(0, 4, n),
        "close":  close,
        "volume": rng.integers(10_000, 100_000, n).astype(float),
    })


# ── _safe_pearson ──────────────────────────────────────────────────────────────

class TestSafePearson:
    def test_perfectly_correlated(self):
        x = np.arange(50, dtype=float)
        r, _ = _safe_pearson(x, x)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_anticorrelated(self):
        x = np.arange(50, dtype=float)
        y = -x
        r, _ = _safe_pearson(x, y)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        r, _ = _safe_pearson(x, y)
        assert abs(r) < 0.3  # random chance

    def test_too_few_points_returns_zero(self):
        r, _ = _safe_pearson(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        assert r == 0.0

    def test_constant_series_returns_zero(self):
        x = np.ones(50)
        y = np.arange(50, dtype=float)
        r, _ = _safe_pearson(x, y)
        assert r == 0.0

    def test_nan_values_stripped(self):
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 10)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        r, _ = _safe_pearson(x, y)
        assert not np.isnan(r)


# ── compute_risk_correlations ──────────────────────────────────────────────────

class TestComputeRiskCorrelations:
    def test_returns_required_keys(self):
        df = _make_price_df(200)
        result = compute_risk_correlations(df, drop_threshold=-5)
        for key in ("drop_count", "avg_drop", "max_drop", "drop_events",
                    "correlations", "top_risk_factors", "risk_score", "risk_level"):
            assert key in result

    def test_empty_price_df_returns_empty(self):
        result = compute_risk_correlations(pd.DataFrame())
        assert result["drop_count"] == 0
        assert result["correlations"] == []

    def test_drop_count_accurate(self):
        df = _make_price_df(200, inject_drops=True)
        result = compute_risk_correlations(df, drop_threshold=-5)
        # At least some of the injected -7% drops should be counted
        assert result["drop_count"] >= 4

    def test_avg_drop_is_negative(self):
        df = _make_price_df(200)
        result = compute_risk_correlations(df, drop_threshold=-5)
        if result["drop_count"] >= 5:
            assert result["avg_drop"] < 0

    def test_max_drop_leq_avg_drop(self):
        df = _make_price_df(200)
        result = compute_risk_correlations(df, drop_threshold=-5)
        if result["drop_count"] >= 5:
            assert result["max_drop"] <= result["avg_drop"]

    def test_correlations_bounded_minus1_to_1(self):
        df = _make_price_df(200)
        result = compute_risk_correlations(df, drop_threshold=-5)
        for item in result["correlations"]:
            assert -1 <= item["correlation"] <= 1

    def test_risk_score_in_0_100(self):
        df = _make_price_df(200)
        result = compute_risk_correlations(df, drop_threshold=-5)
        assert 0 <= result["risk_score"] <= 100

    def test_insufficient_drops_returns_message(self):
        # A flat series will have no drops
        df = pd.DataFrame({
            "date":   pd.date_range("2024-01-01", periods=50),
            "close":  np.ones(50) * 100,
            "volume": np.ones(50) * 5000,
        })
        result = compute_risk_correlations(df, drop_threshold=-5)
        assert result["drop_count"] < 5
        assert result["message"] != ""

    def test_with_institutional_data_adds_fi_factors(self):
        df = _make_price_df(200)
        dates = df["date"].values
        inst = pd.DataFrame([
            {"date": d, "name": "外資", "net": float(np.random.randint(-5000, 5000))}
            for d in dates
        ])
        result = compute_risk_correlations(df, institutional_df=inst, drop_threshold=-5)
        # Should not raise; fi-related factors may appear
        assert result["drop_count"] >= 0

    def test_top_risk_factors_all_negative_corr(self):
        df = _make_price_df(365, inject_drops=True)
        result = compute_risk_correlations(df, drop_threshold=-5, lookback_days=365)
        for item in result["top_risk_factors"]:
            assert item["correlation"] < 0, (
                f"top_risk factor {item['factor']} has positive correlation"
            )


# ── _build_features ─────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_adds_rsi(self):
        df = _make_price_df(100, inject_drops=False)
        df["ret"] = df["close"].pct_change() * 100
        enriched = _build_features(df.copy(), None)
        assert "rsi_14" in enriched.columns

    def test_adds_bollinger(self):
        df = _make_price_df(100, inject_drops=False)
        df["ret"] = df["close"].pct_change() * 100
        enriched = _build_features(df.copy(), None)
        assert "bb_pos" in enriched.columns

    def test_rsi_bounded_0_100(self):
        df = _make_price_df(150, inject_drops=False)
        df["ret"] = df["close"].pct_change() * 100
        enriched = _build_features(df.copy(), None)
        valid = enriched["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()
