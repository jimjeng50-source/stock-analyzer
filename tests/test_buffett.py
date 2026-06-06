"""Unit tests for macro/buffett.py."""

import pytest
from unittest.mock import patch, MagicMock
from macro.buffett import (
    _buffett_signal, _buffett_score, compute_buffett,
    _fetch_twse_market_cap_trillion, _get_taiwan_gdp_trillion,
)


class TestBuffettSignal:
    def test_below_80_green(self):
        color, _ = _buffett_signal(70)
        assert color == "🟢"

    def test_80_to_100_yellow(self):
        color, _ = _buffett_signal(90)
        assert color == "🟡"

    def test_100_to_120_yellow(self):
        color, _ = _buffett_signal(110)
        assert color == "🟡"

    def test_120_to_150_orange(self):
        color, _ = _buffett_signal(135)
        assert color == "🟠"

    def test_above_150_red(self):
        color, _ = _buffett_signal(160)
        assert color == "🔴"

    def test_returns_tuple_of_strings(self):
        color, interp = _buffett_signal(100)
        assert isinstance(color, str)
        assert isinstance(interp, str) and len(interp) > 0


class TestBuffettScore:
    def test_low_ratio_high_score(self):
        assert _buffett_score(60) == pytest.approx(1.0)

    def test_high_ratio_low_score(self):
        assert _buffett_score(180) == pytest.approx(0.0)

    def test_bounded_0_to_1(self):
        for r in [0, 50, 100, 150, 200, 300]:
            s = _buffett_score(float(r))
            assert 0 <= s <= 1

    def test_monotonically_decreasing(self):
        ratios = [80, 100, 120, 140, 160]
        scores = [_buffett_score(r) for r in ratios]
        assert scores == sorted(scores, reverse=True)


class TestComputeBuffett:
    def test_returns_required_keys(self):
        with patch("macro.buffett._fetch_twse_market_cap_trillion", return_value=55.0), \
             patch("macro.buffett._get_taiwan_gdp_trillion", return_value=22.0):
            result = compute_buffett()
        for key in ("ratio", "market_cap", "gdp", "score", "signal", "color", "interpretation"):
            assert key in result

    def test_ratio_calculated_correctly(self):
        with patch("macro.buffett._fetch_twse_market_cap_trillion", return_value=44.0), \
             patch("macro.buffett._get_taiwan_gdp_trillion", return_value=22.0):
            result = compute_buffett()
        assert result["ratio"] == pytest.approx(200.0, abs=0.1)

    def test_score_in_0_1_range(self):
        with patch("macro.buffett._fetch_twse_market_cap_trillion", return_value=50.0), \
             patch("macro.buffett._get_taiwan_gdp_trillion", return_value=22.0):
            result = compute_buffett()
        assert 0 <= result["score"] <= 1

    def test_zero_gdp_fallback(self):
        with patch("macro.buffett._fetch_twse_market_cap_trillion", return_value=50.0), \
             patch("macro.buffett._get_taiwan_gdp_trillion", return_value=0.0):
            result = compute_buffett()
        # Should not raise and ratio should be finite
        assert result["ratio"] > 0
        assert not (result["ratio"] != result["ratio"])  # not NaN


class TestFetchTwseMarketCap:
    def test_fallback_returns_positive_float(self):
        with patch("macro.buffett.requests.get", side_effect=Exception("timeout")):
            with patch("yfinance.download") as mock_yf:
                mock_df = MagicMock()
                mock_df.empty = True
                mock_yf.return_value = mock_df
                cap = _fetch_twse_market_cap_trillion(token="")
        assert isinstance(cap, float) and cap > 0

    def test_no_token_skips_finmind(self):
        with patch("macro.buffett.requests.get") as mock_req:
            with patch("yfinance.download") as mock_yf:
                mock_df = MagicMock()
                mock_df.empty = True
                mock_yf.return_value = mock_df
                _fetch_twse_market_cap_trillion(token="")
        # requests.get should not have been called for FinMind
        mock_req.assert_not_called()


class TestGetTaiwanGdp:
    def test_returns_positive_float(self):
        with patch("macro.buffett.requests.get", side_effect=Exception("offline")):
            gdp = _get_taiwan_gdp_trillion()
        assert isinstance(gdp, float) and gdp > 0

    def test_world_bank_success_path(self):
        fake_data = [
            {},
            [{"value": 22_000_000_000_000.0, "date": "2023"}]
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_data
        with patch("macro.buffett.requests.get", return_value=mock_resp):
            gdp = _get_taiwan_gdp_trillion()
        assert gdp == pytest.approx(22.0, abs=0.5)
