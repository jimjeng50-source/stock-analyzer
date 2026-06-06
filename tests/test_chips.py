"""Unit tests for factors/chips.py — _consecutive_days, _linear_slope, compute_chips."""

import pytest
import pandas as pd
import numpy as np
from factors.chips import _consecutive_days, _linear_slope, compute_chips


# ── _consecutive_days ──────────────────────────────────────────────────────────

class TestConsecutiveDays:
    def test_all_positive_returns_positive_count(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert _consecutive_days(s) == 5

    def test_all_negative_returns_negative_count(self):
        s = pd.Series([-1, -2, -3])
        assert _consecutive_days(s) == -3

    def test_last_zero_returns_zero(self):
        s = pd.Series([1, 2, 3, 0])
        assert _consecutive_days(s) == 0

    def test_mixed_counts_from_tail(self):
        s = pd.Series([5, -2, 3, 4, 6])
        assert _consecutive_days(s) == 3

    def test_empty_series_returns_zero(self):
        assert _consecutive_days(pd.Series(dtype=float)) == 0

    def test_single_positive(self):
        assert _consecutive_days(pd.Series([7.0])) == 1

    def test_single_negative(self):
        assert _consecutive_days(pd.Series([-3.0])) == -1

    def test_sign_flip_resets(self):
        s = pd.Series([10, 10, -1, 5, 5, 5])
        assert _consecutive_days(s) == 3

    def test_nan_ignored(self):
        s = pd.Series([np.nan, 1.0, 2.0, 3.0])
        assert _consecutive_days(s) == 3


# ── _linear_slope ──────────────────────────────────────────────────────────────

class TestLinearSlope:
    def test_increasing_returns_positive(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _linear_slope(s) > 0

    def test_decreasing_returns_negative(self):
        s = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _linear_slope(s) < 0

    def test_flat_returns_near_zero(self):
        s = pd.Series([3.0, 3.0, 3.0, 3.0])
        assert abs(_linear_slope(s)) < 1e-9

    def test_single_value_returns_zero(self):
        assert _linear_slope(pd.Series([42.0])) == 0.0

    def test_empty_returns_zero(self):
        assert _linear_slope(pd.Series(dtype=float)) == 0.0

    def test_nan_stripped(self):
        s = pd.Series([np.nan, 1.0, 2.0, 3.0])
        assert _linear_slope(s) > 0


# ── compute_chips ──────────────────────────────────────────────────────────────

@pytest.fixture
def inst_df_buying():
    dates = pd.date_range("2024-01-01", periods=30)
    rows = []
    for d in dates:
        rows += [
            {"date": d, "name": "外資", "buy": 6000, "sell": 1000, "net": 5000},
            {"date": d, "name": "投信", "buy": 2000, "sell": 500,  "net": 1500},
            {"date": d, "name": "自營商", "buy": 300, "sell": 400,  "net": -100},
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def inst_df_selling():
    dates = pd.date_range("2024-01-01", periods=30)
    rows = []
    for d in dates:
        rows += [
            {"date": d, "name": "Foreign_Investor", "buy": 500, "sell": 5000, "net": -4500},
            {"date": d, "name": "Investment_Trust", "buy": 200, "sell": 1000, "net": -800},
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def margin_df():
    dates = pd.date_range("2024-01-01", periods=30)
    return pd.DataFrame({
        "date": dates,
        "MarginPurchaseTodayBalance": np.linspace(150_000, 200_000, 30),
        "ShortSaleTodayBalance":      np.linspace(20_000,  15_000,  30),
    })


class TestComputeChips:
    def test_returns_all_expected_keys(self, inst_df_buying, margin_df):
        result = compute_chips(inst_df_buying, margin_df)
        for key in ("fi_5d_net", "fi_20d_net", "fi_consecutive", "fi_trend",
                    "it_5d_net", "it_20d_net", "it_consecutive",
                    "dealer_5d_net", "margin_chg_5d", "short_chg_5d"):
            assert key in result, f"Missing key: {key}"

    def test_empty_dfs_return_zeros(self):
        result = compute_chips(pd.DataFrame(), pd.DataFrame())
        for k, v in result.items():
            assert v == 0 or v == 0.0

    def test_bullish_fi_nets_positive(self, inst_df_buying, margin_df):
        result = compute_chips(inst_df_buying, margin_df)
        assert result["fi_5d_net"] > 0
        assert result["fi_20d_net"] > 0
        assert result["fi_consecutive"] > 0

    def test_bearish_fi_nets_negative(self, inst_df_selling, margin_df):
        result = compute_chips(inst_df_selling, margin_df)
        assert result["fi_5d_net"] < 0

    def test_english_name_variants_recognized(self):
        dates = pd.date_range("2024-01-01", periods=10)
        rows = [{"date": d, "name": "Foreign_Investor", "buy": 3000, "sell": 1000, "net": 2000}
                for d in dates]
        df = pd.DataFrame(rows)
        result = compute_chips(df, pd.DataFrame())
        assert result["fi_5d_net"] > 0

    def test_margin_increase_detected(self):
        dates = pd.date_range("2024-01-01", periods=10)
        balances = np.linspace(100_000, 150_000, 10)
        margin = pd.DataFrame({
            "date": dates,
            "MarginPurchaseTodayBalance": balances,
            "ShortSaleTodayBalance": np.ones(10) * 20_000,
        })
        result = compute_chips(pd.DataFrame(), margin)
        assert result["margin_chg_5d"] > 0   # margins increased → positive value

    def test_short_decrease_detected(self):
        dates = pd.date_range("2024-01-01", periods=10)
        shorts = np.linspace(30_000, 10_000, 10)  # short sales declining
        margin = pd.DataFrame({
            "date": dates,
            "MarginPurchaseTodayBalance": np.ones(10) * 100_000,
            "ShortSaleTodayBalance": shorts,
        })
        result = compute_chips(pd.DataFrame(), margin)
        assert result["short_chg_5d"] < 0  # short decreased → negative change
