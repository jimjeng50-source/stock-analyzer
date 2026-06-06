"""Unit tests for models/scorer.py — normalization functions and Scorer.score()."""

import pytest
import numpy as np
from models.scorer import (
    _sigmoid, _inv_sigmoid, _binary, _linear, _pe_norm,
    _NORMALIZERS, _recommendation, Scorer,
)


# ── Normalization primitives ───────────────────────────────────────────────────

class TestSigmoid:
    def test_center_returns_half(self):
        assert _sigmoid(0, 0, 1) == pytest.approx(0.5, abs=1e-6)

    def test_large_positive_near_one(self):
        assert _sigmoid(1000, 0, 1) > 0.999

    def test_large_negative_near_zero(self):
        assert _sigmoid(-1000, 0, 1) < 0.001

    def test_center_offset(self):
        assert _sigmoid(10, 10, 1) == pytest.approx(0.5, abs=1e-6)

    def test_output_bounded(self):
        # sigmoid can saturate to 0.0 or 1.0 at floating-point extremes
        for x in np.linspace(-100, 100, 50):
            v = _sigmoid(float(x))
            assert 0 <= v <= 1


class TestInvSigmoid:
    def test_inverts_sigmoid_at_center(self):
        assert _inv_sigmoid(0, 0, 1) == pytest.approx(0.5, abs=1e-6)

    def test_large_negative_near_one(self):
        assert _inv_sigmoid(-1000, 0, 1) > 0.999

    def test_sum_to_one_with_sigmoid(self):
        for x in [-5, 0, 5]:
            assert _sigmoid(x) + _inv_sigmoid(x) == pytest.approx(1.0, abs=1e-9)


class TestBinary:
    def test_minus_one_maps_to_zero(self):
        assert _binary(-1) == pytest.approx(0.0)

    def test_zero_maps_to_half(self):
        assert _binary(0) == pytest.approx(0.5)

    def test_plus_one_maps_to_one(self):
        assert _binary(1) == pytest.approx(1.0)

    def test_clips_above_one(self):
        assert _binary(2) == pytest.approx(1.0)

    def test_clips_below_zero(self):
        assert _binary(-2) == pytest.approx(0.0)


class TestLinear:
    def test_below_lo_returns_zero(self):
        assert _linear(-5, 0, 100) == 0.0

    def test_above_hi_returns_one(self):
        assert _linear(150, 0, 100) == 1.0

    def test_midpoint(self):
        assert _linear(50, 0, 100) == pytest.approx(0.5)

    def test_degenerate_range_returns_half(self):
        assert _linear(5, 5, 5) == 0.5


class TestPeNorm:
    def test_negative_pe_returns_025(self):
        assert _pe_norm(-1) == 0.25
        assert _pe_norm(0) == 0.25

    def test_low_pe_high_score(self):
        assert _pe_norm(8) == pytest.approx(0.95)

    def test_very_high_pe_low_score(self):
        assert _pe_norm(100) == pytest.approx(0.15)

    def test_monotonically_decreasing(self):
        pes = [10, 15, 25, 40, 60, 80]
        scores = [_pe_norm(p) for p in pes]
        assert scores == sorted(scores, reverse=True)


# ── Normalizers dict completeness ─────────────────────────────────────────────

class TestNormalizers:
    _EXPECTED_FACTORS = [
        "fi_5d_net", "fi_20d_net", "fi_consecutive", "fi_trend",
        "it_5d_net", "it_20d_net", "it_consecutive", "dealer_5d_net",
        "margin_chg_5d", "short_chg_5d",
        "above_ma5", "above_ma20", "above_ma60", "ma_alignment",
        "rsi_14", "macd_histogram", "macd_cross", "bb_position",
        "vol_ratio", "vol_trend",
        "rev_yoy", "rev_mom", "eps_latest", "pe_ratio", "gross_margin",
        "ret_5d", "ret_1m", "ret_3m", "vol_20d",
    ]

    def test_all_expected_factors_present(self):
        for factor in self._EXPECTED_FACTORS:
            assert factor in _NORMALIZERS, f"Missing normalizer for {factor}"

    def test_all_normalizers_return_0_to_1(self):
        test_values = [0, 1, -1, 100, -100, 0.5, -0.5]
        for factor, fn in _NORMALIZERS.items():
            for v in test_values:
                result = fn(float(v))
                assert 0 <= result <= 1, (
                    f"Normalizer {factor}({v}) = {result} outside [0,1]"
                )


# ── Recommendation thresholds ──────────────────────────────────────────────────

class TestRecommendation:
    def test_strong_buy(self):
        assert "強力買進" in _recommendation(85)

    def test_buy(self):
        assert "買進" in _recommendation(70)

    def test_hold(self):
        assert "持有" in _recommendation(55)

    def test_sell_down(self):
        assert "減碼" in _recommendation(38)

    def test_sell(self):
        assert "賣出" in _recommendation(20)

    @pytest.mark.parametrize("score", [0, 29, 30, 44, 45, 64, 65, 79, 80, 100])
    def test_boundary_values_return_string(self, score):
        r = _recommendation(float(score))
        assert isinstance(r, str) and len(r) > 0


# ── Scorer integration ─────────────────────────────────────────────────────────

class TestScorer:
    def _sample_factors(self):
        chips = {
            "fi_5d_net": 3000, "fi_20d_net": 10000, "fi_consecutive": 3,
            "fi_trend": 200, "it_5d_net": 500, "it_20d_net": 2000,
            "it_consecutive": 2, "dealer_5d_net": -100,
            "margin_chg_5d": -2, "short_chg_5d": -1,
        }
        technical = {
            "above_ma5": 1, "above_ma20": 1, "above_ma60": 1,
            "ma_alignment": 3, "rsi_14": 58, "macd_histogram": 0.4,
            "macd_cross": 0, "bb_position": 0.6, "vol_ratio": 1.5,
            "vol_trend": 1, "ma20_deviation": 2, "rsi_signal": 0,
        }
        fundamental = {
            "rev_yoy": 15, "rev_mom": 5, "rev_3m_trend": 1, "rev_12m_high": 1,
            "eps_latest": 4, "eps_qoq": 0.5, "eps_yoy": 1,
            "gross_margin": 35, "gpm_trend": 2, "pe_ratio": 18,
        }
        momentum = {
            "ret_5d": 2, "ret_1m": 8, "ret_3m": 15,
            "high_52w_pct": -8, "momentum_accel": 1,
        }
        return chips, technical, fundamental, momentum

    def test_returns_required_keys(self):
        scorer = Scorer()
        chips, tech, fund, mom = self._sample_factors()
        result = scorer.score(chips, tech, fund, mom)
        for key in ("total_score", "category_scores", "recommendation", "raw_factors"):
            assert key in result

    def test_score_in_range(self):
        scorer = Scorer()
        chips, tech, fund, mom = self._sample_factors()
        result = scorer.score(chips, tech, fund, mom)
        assert 0 <= result["total_score"] <= 100

    def test_category_scores_present(self):
        scorer = Scorer()
        chips, tech, fund, mom = self._sample_factors()
        result = scorer.score(chips, tech, fund, mom)
        for cat in ("chips", "technical", "fundamental", "momentum", "risk"):
            assert cat in result["category_scores"]

    def test_weights_auto_normalize(self):
        weights = {"chips": 3, "fundamental": 2, "technical": 2, "momentum": 1, "risk": 1}
        scorer = Scorer(weights)
        total = sum(scorer.weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_missing_factors_dont_crash(self):
        scorer = Scorer()
        result = scorer.score({}, {}, {}, {})
        assert 0 <= result["total_score"] <= 100

    def test_bullish_factors_score_higher_than_bearish(self):
        scorer = Scorer()
        chips_bull = {"fi_5d_net": 50000, "fi_consecutive": 10}
        chips_bear = {"fi_5d_net": -50000, "fi_consecutive": -10}
        r_bull = scorer.score(chips_bull, {}, {}, {})
        r_bear = scorer.score(chips_bear, {}, {}, {})
        assert r_bull["category_scores"]["chips"] > r_bear["category_scores"]["chips"]

    def test_raw_factors_merged(self):
        scorer = Scorer()
        chips = {"fi_5d_net": 100}
        tech  = {"rsi_14": 55}
        result = scorer.score(chips, tech, {}, {})
        assert "fi_5d_net" in result["raw_factors"]
        assert "rsi_14" in result["raw_factors"]
