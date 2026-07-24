"""
tests/test_free_fallback.py
共用免費資料備援層 augment_factors_with_free_sources 單元測試（全 mock）
"""

import pandas as pd
from unittest.mock import patch

from data.free_fallback import augment_factors_with_free_sources


_EMPTY_FUND = {
    "rev_yoy": 0.0, "gross_margin": 0.0, "pe_ratio": 0.0, "eps_latest": 0.0,
}
_EMPTY_CHIPS = {
    "fi_5d_net": 0.0, "it_5d_net": 0.0, "dealer_5d_net": 0.0, "margin_chg_5d": 0.0,
}
_YF = {
    "revenue_growth": 0.10, "gross_margins": 0.40,
    "trailing_pe": 20.0, "trailing_eps": 5.0,
}
_NONEMPTY = pd.DataFrame({"x": [1]})


def _t86_df():
    return pd.DataFrame([
        {"date": "2026-07-24", "name": "外資", "net": 500_000},
        {"date": "2026-07-24", "name": "投信", "net": 100_000},
    ])


class TestFundamentalTrigger:
    def test_augments_when_both_missing(self):
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value=_YF), \
             patch("data.twse_chips.get_t86_institutional", return_value=pd.DataFrame()):
            _, fund = augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=_NONEMPTY,
                revenue_df=pd.DataFrame(), financial_df=pd.DataFrame())
        assert fund["rev_yoy"] == 10.0
        assert fund["pe_ratio"] == 20.0

    def test_augments_when_only_revenue_missing(self):
        """Gap G：只缺月營收（財報有）也要補（OR 條件）。"""
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value=_YF), \
             patch("data.twse_chips.get_t86_institutional", return_value=pd.DataFrame()):
            _, fund = augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=_NONEMPTY,
                revenue_df=pd.DataFrame(), financial_df=_NONEMPTY)
        assert fund["rev_yoy"] == 10.0

    def test_augments_when_only_financial_missing(self):
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value=_YF), \
             patch("data.twse_chips.get_t86_institutional", return_value=pd.DataFrame()):
            _, fund = augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=_NONEMPTY,
                revenue_df=_NONEMPTY, financial_df=pd.DataFrame())
        assert fund["gross_margin"] == 40.0

    def test_no_yf_call_when_both_present(self):
        """FinMind 財報齊全時，不應打 yfinance。"""
        with patch("data.yf_fundamentals.get_yf_fundamentals") as mock_yf, \
             patch("data.twse_chips.get_t86_institutional", return_value=pd.DataFrame()):
            augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=_NONEMPTY,
                revenue_df=_NONEMPTY, financial_df=_NONEMPTY)
        mock_yf.assert_not_called()


class TestChipsTrigger:
    def test_augments_chips_when_institutional_missing(self):
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value={}), \
             patch("data.twse_chips.get_t86_institutional", return_value=_t86_df()):
            chips, _ = augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=pd.DataFrame(),
                revenue_df=_NONEMPTY, financial_df=_NONEMPTY)
        assert chips["fi_5d_net"] == 500_000
        assert chips["it_5d_net"] == 100_000

    def test_allow_t86_false_skips_t86(self):
        """歷史回溯 allow_t86=False → 不打 T86。"""
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value={}), \
             patch("data.twse_chips.get_t86_institutional") as mock_t86:
            augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=pd.DataFrame(),
                revenue_df=_NONEMPTY, financial_df=_NONEMPTY, allow_t86=False)
        mock_t86.assert_not_called()

    def test_no_t86_call_when_institutional_present(self):
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value={}), \
             patch("data.twse_chips.get_t86_institutional") as mock_t86:
            augment_factors_with_free_sources(
                "2330", chips=dict(_EMPTY_CHIPS), fundamental=dict(_EMPTY_FUND),
                current_price=100.0, institutional_df=_NONEMPTY,
                revenue_df=_NONEMPTY, financial_df=_NONEMPTY)
        mock_t86.assert_not_called()
