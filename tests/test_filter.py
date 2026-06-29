"""
tests/test_filter.py
Tests for screener/filter.py — QuickFilter
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from screener.filter import FilterConfig, QuickFilter


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "stock_id": "2330",
            "stock_name": "台積電",
            "market": "TWSE",
            "industry": "半導體",
            "market_cap_b": 20000,
            "avg_volume_k": 50000,
            "last_price": 850.0,
        },
        {
            "stock_id": "0001",
            "stock_name": "仙股",
            "market": "TWSE",
            "industry": "其他",
            "market_cap_b": 0.1,
            "avg_volume_k": 5,
            "last_price": 3.0,
        },
        {
            "stock_id": "2317",
            "stock_name": "鴻海",
            "market": "TWSE",
            "industry": "電子",
            "market_cap_b": 3000,
            "avg_volume_k": 60000,
            "last_price": 180.0,
        },
    ])


# Config that avoids any API calls (no EPS/revenue fetching).
_TEST_CONFIG = FilterConfig(
    min_price=10.0,
    max_price=5000.0,
    min_market_cap_b=1.0,
    min_avg_volume_k=100,
    require_positive_eps_ttm=False,
    min_revenue_yoy=-100.0,
)


def _make_filter():
    """Return a QuickFilter with test config and time.sleep mocked out."""
    return QuickFilter(fetcher=None, config=_TEST_CONFIG)


class TestQuickFilter:
    """Tests for QuickFilter.run()."""

    def test_filter_removes_low_price(self, sample_df):
        """Stock '0001' with price 3.0 is below min_price=10.0 and must be removed."""
        qf = _make_filter()
        with patch("screener.filter.time.sleep"):
            passed_df, _ = qf.run(sample_df)
        assert "0001" not in passed_df["stock_id"].tolist()

    def test_filter_keeps_valid_stocks(self, sample_df):
        """'2330' and '2317' satisfy all criteria and must remain."""
        qf = _make_filter()
        with patch("screener.filter.time.sleep"):
            passed_df, _ = qf.run(sample_df)
        ids = passed_df["stock_id"].tolist()
        assert "2330" in ids
        assert "2317" in ids

    def test_report_has_required_keys(self, sample_df):
        """Filter report must contain total_input, steps, total_passed, removed_ids."""
        qf = _make_filter()
        with patch("screener.filter.time.sleep"):
            _, report = qf.run(sample_df)
        for key in ("total_input", "steps", "total_passed", "removed_ids"):
            assert key in report, f"Missing key: {key}"

    def test_total_input_matches(self, sample_df):
        """report['total_input'] must equal the number of rows in input DataFrame."""
        qf = _make_filter()
        with patch("screener.filter.time.sleep"):
            _, report = qf.run(sample_df)
        assert report["total_input"] == 3

    def test_removed_ids_contains_junk_stock(self, sample_df):
        """report['removed_ids'] must include '0001'."""
        qf = _make_filter()
        with patch("screener.filter.time.sleep"):
            _, report = qf.run(sample_df)
        assert "0001" in report["removed_ids"]

    def test_exclude_list_works(self, sample_df):
        """Adding '2330' to exclude_stock_ids must remove it from results."""
        config = FilterConfig(
            min_price=10.0,
            max_price=5000.0,
            min_market_cap_b=1.0,
            min_avg_volume_k=100,
            require_positive_eps_ttm=False,
            min_revenue_yoy=-100.0,
            exclude_stock_ids=["2330"],
        )
        qf = QuickFilter(fetcher=None, config=config)
        with patch("screener.filter.time.sleep"):
            passed_df, _ = qf.run(sample_df)
        assert "2330" not in passed_df["stock_id"].tolist()

    def test_empty_df_returns_empty(self):
        """run() on an empty DataFrame returns an empty DataFrame without error."""
        qf = _make_filter()
        empty = pd.DataFrame(
            columns=["stock_id", "stock_name", "market", "industry",
                     "market_cap_b", "avg_volume_k", "last_price"]
        )
        with patch("screener.filter.time.sleep"):
            passed_df, report = qf.run(empty)
        assert passed_df.empty
        assert report["total_input"] == 0
        assert report["total_passed"] == 0
