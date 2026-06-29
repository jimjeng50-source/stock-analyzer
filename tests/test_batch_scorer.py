"""
tests/test_batch_scorer.py
BatchScorer 批次評分引擎單元測試
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from screener.batch_scorer import BatchScorer


_MOCK_SCORE_RESULT = {
    "total_score": 75.0,
    "recommendation": "買進",
    "category_scores": {
        "chips": 80, "fundamental": 70, "technical": 75,
        "momentum": 78, "risk": 65,
    },
}

_PRICE_DF = pd.DataFrame({"close": [100.0]})

_REQUIRED_COLS = [
    "stock_id", "total_score", "recommendation",
    "chips_score", "fundamental_score", "technical_score",
    "momentum_score", "risk_score", "current_price", "error", "scored_at",
]


def _setup_mocks(MockFetcher, MockScorer):
    MockFetcher.return_value.get_price.return_value = _PRICE_DF
    MockFetcher.return_value.get_institutional.return_value = pd.DataFrame()
    MockFetcher.return_value.get_margin_trading.return_value = pd.DataFrame()
    MockFetcher.return_value.get_monthly_revenue.return_value = pd.DataFrame()
    MockFetcher.return_value.get_financial_statements.return_value = pd.DataFrame()
    MockScorer.return_value.score.return_value = _MOCK_SCORE_RESULT


_PATCHES = [
    patch("screener.batch_scorer.FinMindFetcher"),
    patch("screener.batch_scorer.compute_chips", return_value={}),
    patch("screener.batch_scorer.compute_technical", return_value={}),
    patch("screener.batch_scorer.compute_fundamental", return_value={}),
    patch("screener.batch_scorer.compute_momentum", return_value={}),
    patch("screener.batch_scorer.Scorer"),
    patch("screener.batch_scorer.time.sleep"),
]


class TestBatchScorer:

    def test_score_universe_returns_dataframe(self):
        """score_universe(['2330','2317']) 應回傳 DataFrame。"""
        with patch("screener.batch_scorer.FinMindFetcher") as MockFetcher, \
             patch("screener.batch_scorer.compute_chips", return_value={}), \
             patch("screener.batch_scorer.compute_technical", return_value={}), \
             patch("screener.batch_scorer.compute_fundamental", return_value={}), \
             patch("screener.batch_scorer.compute_momentum", return_value={}), \
             patch("screener.batch_scorer.Scorer") as MockScorer, \
             patch("screener.batch_scorer.time.sleep"):
            _setup_mocks(MockFetcher, MockScorer)
            bs = BatchScorer(max_workers=1)
            result = bs.score_universe(["2330", "2317"], show_progress=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_result_sorted_by_score_desc(self):
        """結果應依 total_score 由高到低排序。"""
        with patch("screener.batch_scorer.FinMindFetcher") as MockFetcher, \
             patch("screener.batch_scorer.compute_chips", return_value={}), \
             patch("screener.batch_scorer.compute_technical", return_value={}), \
             patch("screener.batch_scorer.compute_fundamental", return_value={}), \
             patch("screener.batch_scorer.compute_momentum", return_value={}), \
             patch("screener.batch_scorer.Scorer") as MockScorer, \
             patch("screener.batch_scorer.time.sleep"):
            _setup_mocks(MockFetcher, MockScorer)
            bs = BatchScorer(max_workers=1)
            result = bs.score_universe(["2330", "2317"], show_progress=False)
        scores = result["total_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_required_columns_present(self):
        """結果 DataFrame 應包含所有必要欄位。"""
        with patch("screener.batch_scorer.FinMindFetcher") as MockFetcher, \
             patch("screener.batch_scorer.compute_chips", return_value={}), \
             patch("screener.batch_scorer.compute_technical", return_value={}), \
             patch("screener.batch_scorer.compute_fundamental", return_value={}), \
             patch("screener.batch_scorer.compute_momentum", return_value={}), \
             patch("screener.batch_scorer.Scorer") as MockScorer, \
             patch("screener.batch_scorer.time.sleep"):
            _setup_mocks(MockFetcher, MockScorer)
            bs = BatchScorer(max_workers=1)
            result = bs.score_universe(["2330"], show_progress=False)
        for col in _REQUIRED_COLS:
            assert col in result.columns, f"缺少欄位：{col}"

    def test_failed_stock_excluded_from_result(self):
        """評分失敗的個股應放入 failed_df，不出現在主結果中。"""
        def fetcher_side_effect(stock_id):
            mock = MagicMock()
            if stock_id == "FAIL":
                mock.get_price.side_effect = RuntimeError("模擬失敗")
            else:
                mock.get_price.return_value = _PRICE_DF
                mock.get_institutional.return_value = pd.DataFrame()
                mock.get_margin_trading.return_value = pd.DataFrame()
                mock.get_monthly_revenue.return_value = pd.DataFrame()
                mock.get_financial_statements.return_value = pd.DataFrame()
            return mock

        with patch("screener.batch_scorer.FinMindFetcher", side_effect=fetcher_side_effect), \
             patch("screener.batch_scorer.compute_chips", return_value={}), \
             patch("screener.batch_scorer.compute_technical", return_value={}), \
             patch("screener.batch_scorer.compute_fundamental", return_value={}), \
             patch("screener.batch_scorer.compute_momentum", return_value={}), \
             patch("screener.batch_scorer.Scorer") as MockScorer, \
             patch("screener.batch_scorer.time.sleep"):
            MockScorer.return_value.score.return_value = _MOCK_SCORE_RESULT
            bs = BatchScorer(max_workers=1)
            result = bs.score_universe(["2330", "FAIL"], show_progress=False)

        assert "FAIL" not in result["stock_id"].tolist()
        assert not bs.failed_df.empty
        assert "FAIL" in bs.failed_df["stock_id"].tolist()

    def test_empty_input_returns_empty(self):
        """空輸入清單應回傳空 DataFrame。"""
        bs = BatchScorer(max_workers=1)
        result = bs.score_universe([], show_progress=False)
        assert result.empty

    def test_single_stock(self):
        """單支股票評分應回傳 1 行 DataFrame，total_score 正確。"""
        with patch("screener.batch_scorer.FinMindFetcher") as MockFetcher, \
             patch("screener.batch_scorer.compute_chips", return_value={}), \
             patch("screener.batch_scorer.compute_technical", return_value={}), \
             patch("screener.batch_scorer.compute_fundamental", return_value={}), \
             patch("screener.batch_scorer.compute_momentum", return_value={}), \
             patch("screener.batch_scorer.Scorer") as MockScorer, \
             patch("screener.batch_scorer.time.sleep"):
            _setup_mocks(MockFetcher, MockScorer)
            bs = BatchScorer(max_workers=1)
            result = bs.score_universe(["2330"], show_progress=False)
        assert len(result) == 1
        assert result.iloc[0]["stock_id"] == "2330"
        assert result.iloc[0]["total_score"] == pytest.approx(75.0)
