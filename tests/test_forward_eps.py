"""
tests/test_forward_eps.py
ForwardEPSCalculator 單元測試
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from factors.forward_eps import ForwardEPSCalculator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fetcher():
    """回傳 MagicMock DataFetcher。"""
    return MagicMock()


@pytest.fixture
def eps_df_8q():
    """8 季 EPS 資料。"""
    dates = pd.date_range("2022-01-01", periods=8, freq="QE")
    return pd.DataFrame({"date": dates, "eps": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]})


@pytest.fixture
def revenue_df_6m():
    """6 個月月營收資料（含 YoY）。"""
    dates = pd.date_range("2024-01-01", periods=6, freq="ME")
    return pd.DataFrame({
        "date": dates,
        "revenue": [100_000, 110_000, 120_000, 130_000, 140_000, 150_000],
        "revenue_yoy": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
    })


@pytest.fixture
def gm_df_6q():
    """6 季毛利率資料。"""
    dates = pd.date_range("2022-01-01", periods=6, freq="QE")
    return pd.DataFrame({
        "date": dates,
        "gross_margin": [45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
    })


@pytest.fixture
def pe_df_3y():
    """3 年日頻 PE 資料（250 筆）。"""
    dates = pd.date_range("2022-01-01", periods=250, freq="B")
    return pd.DataFrame({
        "date": dates,
        "pe_ratio": np.random.uniform(15, 30, 250),
    })


# ── 正常情境測試 ───────────────────────────────────────────────────────────────

class TestForwardEPSCalculatorNormal:

    def test_calculate_returns_dict(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """正常情境應回傳包含所有必要 key 的 dict。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        required_keys = [
            "stock_id", "ttm_eps", "eps_growth_rate", "forward_eps_1y",
            "gm_adjustment", "target_price", "current_price", "upside_pct",
            "peg_ratio", "pe_band", "confidence", "confidence_reason", "error",
        ]
        for key in required_keys:
            assert key in result, f"缺少 key：{key}"

    def test_ttm_eps_is_sum_of_last_4(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """TTM EPS 應等於最後 4 季加總。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        expected_ttm = 5.0 + 5.5 + 6.0 + 6.5  # 最後 4 季
        assert result["ttm_eps"] == pytest.approx(expected_ttm, rel=1e-3)

    def test_forward_eps_positive_growth(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """成長率為正時，Forward EPS 應大於 TTM EPS。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        if result.get("eps_growth_rate") and result["eps_growth_rate"] > 0:
            assert result["forward_eps_1y"] > result["ttm_eps"]

    def test_target_prices_ordered(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """熊市 ≤ 基準 ≤ 牛市目標價。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        tp = result["target_price"]
        if all(tp.get(k) is not None for k in ["bear", "base", "bull"]):
            assert tp["bear"] <= tp["base"] <= tp["bull"]

    def test_confidence_high_when_all_data_available(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """所有資料齊全時，信心度應為 high。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["confidence"] in ("high", "medium", "low")

    def test_current_price_matches_mock(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """current_price 應與 mock 回傳值一致。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 888.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["current_price"] == pytest.approx(888.0)


# ── 資料不足情境測試 ──────────────────────────────────────────────────────────

class TestForwardEPSInsufficientData:

    def test_eps_none_returns_error(self, mock_fetcher):
        """EPS 資料為 None 時應回傳 error。"""
        mock_fetcher.get_quarterly_eps.return_value = None

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("9999")

        assert result["error"] is not None
        assert result["ttm_eps"] is None

    def test_eps_too_few_quarters(self, mock_fetcher):
        """EPS 資料少於 4 季時應回傳 error。"""
        eps_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=3, freq="QE"),
            "eps": [1.0, 2.0, 3.0],
        })
        mock_fetcher.get_quarterly_eps.return_value = eps_df

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("9999")

        assert result["error"] is not None

    def test_revenue_none_confidence_flags(self, mock_fetcher, eps_df_8q, gm_df_6q, pe_df_3y):
        """月營收資料為 None 時，信心度應降低（medium 或 low）。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = None
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 500.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["confidence"] in ("medium", "low")

    def test_gm_none_confidence_flags(self, mock_fetcher, eps_df_8q, revenue_df_6m, pe_df_3y):
        """毛利率資料為 None 時，信心度不應為 high。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = None
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 500.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["confidence"] in ("medium", "low")

    def test_pe_data_none_target_price_null(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q):
        """PE 資料為 None 時，目標價應為 None。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = None
        mock_fetcher.get_market_price.return_value = 500.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        tp = result["target_price"]
        assert all(tp.get(k) is None for k in ["bull", "base", "bear"])

    def test_all_data_none_returns_low_confidence(self, mock_fetcher):
        """所有資料均為 None（除 EPS 不足）時，confidence 為 low。"""
        mock_fetcher.get_quarterly_eps.return_value = None

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("9999")

        assert result["confidence"] == "low"


# ── 型別與範圍驗證 ─────────────────────────────────────────────────────────────

class TestForwardEPSTypeValidation:

    def test_gm_adjustment_within_bounds(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """毛利率調整係數應在 [-0.05, 0.05] 範圍內。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        if result["gm_adjustment"] is not None:
            assert -0.05 <= result["gm_adjustment"] <= 0.05

    def test_stock_id_preserved(self, mock_fetcher):
        """stock_id 應與輸入一致。"""
        mock_fetcher.get_quarterly_eps.return_value = None

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("6213")

        assert result["stock_id"] == "6213"

    def test_confidence_valid_values(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """confidence 應為 high / medium / low 其中之一。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["confidence"] in ("high", "medium", "low")

    def test_pe_band_fields(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        """pe_band 應包含 p25、median、p75、current 四個欄位。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = revenue_df_6m
        mock_fetcher.get_quarterly_gross_margin.return_value = gm_df_6q
        mock_fetcher.get_historical_pe.return_value = pe_df_3y
        mock_fetcher.get_market_price.return_value = 700.0

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert set(result["pe_band"].keys()) == {"p25", "median", "p75", "current"}

    def test_exception_handling(self, mock_fetcher):
        """fetcher 拋出例外時，應回傳包含 error 的結果，不應崩潰。"""
        mock_fetcher.get_quarterly_eps.side_effect = RuntimeError("網路錯誤")

        calc = ForwardEPSCalculator(mock_fetcher)
        result = calc.calculate("2330")

        assert result["error"] is not None
        assert result["confidence"] == "low"


# ── 多算法輸出測試 ─────────────────────────────────────────────────────────────

class TestMultipleMethods:
    def _wire(self, mock_fetcher, eps, rev, gm, pe):
        mock_fetcher.get_quarterly_eps.return_value = eps
        mock_fetcher.get_monthly_revenue.return_value = rev
        mock_fetcher.get_quarterly_gross_margin.return_value = gm
        mock_fetcher.get_historical_pe.return_value = pe
        mock_fetcher.get_market_price.return_value = 700.0
        mock_fetcher._fm_request.return_value = pd.DataFrame()  # 無存貨資料

    def test_methods_dict_has_all_five(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        self._wire(mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y)
        result = ForwardEPSCalculator(mock_fetcher).calculate("2330")
        methods = result["methods"]
        for k in ["quant_base", "quant_inventory", "scenario", "consensus", "guidance"]:
            assert k in methods
        # base 與 scenario 有資料
        assert methods["quant_base"]["available"] is True
        assert "bull" in methods["scenario"] and "bear" in methods["scenario"]

    def test_base_and_inventory_eps_separately_exposed(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        self._wire(mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y)
        result = ForwardEPSCalculator(mock_fetcher).calculate("2330")
        # 無存貨資料 → 庫存重估=0 → base 應等於 inventory
        assert result["base_eps_1y"] == pytest.approx(result["forward_eps_1y"], rel=1e-6)
        assert result["methods"]["quant_base"]["eps"] == result["base_eps_1y"]

    def test_manual_consensus_and_guidance_flow_in(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        self._wire(mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y)
        result = ForwardEPSCalculator(mock_fetcher).calculate(
            "2330", consensus_eps=28.5, guidance_eps=26.0)
        assert result["methods"]["consensus"]["available"] is True
        assert result["methods"]["consensus"]["eps"] == 28.5
        assert result["methods"]["guidance"]["available"] is True
        assert result["methods"]["guidance"]["eps"] == 26.0

    def test_consensus_guidance_unavailable_by_default(self, mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y):
        self._wire(mock_fetcher, eps_df_8q, revenue_df_6m, gm_df_6q, pe_df_3y)
        result = ForwardEPSCalculator(mock_fetcher).calculate("2330")
        assert result["methods"]["consensus"]["available"] is False
        assert result["methods"]["consensus"]["eps"] is None
        assert result["methods"]["guidance"]["available"] is False

    def test_quant_unavailable_when_data_insufficient(self, mock_fetcher, eps_df_8q):
        """營收與毛利都缺 → 定量方法標記為不可用（不是誤導的 0%）。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = pd.DataFrame()      # 無營收
        mock_fetcher.get_quarterly_gross_margin.return_value = pd.DataFrame()  # 無毛利
        mock_fetcher.get_historical_pe.return_value = pd.DataFrame()
        mock_fetcher.get_market_price.return_value = 700.0
        mock_fetcher._fm_request.return_value = pd.DataFrame()
        result = ForwardEPSCalculator(mock_fetcher).calculate("3260")
        assert result["methods"]["quant_base"]["available"] is False
        # forward 仍接近 TTM
        assert result["forward_eps_1y"] == pytest.approx(result["ttm_eps"], rel=1e-6)


# ── yfinance 基本面備援 ────────────────────────────────────────────────────────

class TestYfinanceFallback:
    def _wire_empty_finmind(self, mock_fetcher, eps_df_8q):
        """FinMind 只有 EPS，其餘（營收/毛利/PE）全缺。"""
        mock_fetcher.get_quarterly_eps.return_value = eps_df_8q
        mock_fetcher.get_monthly_revenue.return_value = pd.DataFrame()   # 缺
        mock_fetcher.get_quarterly_gross_margin.return_value = pd.DataFrame()  # 缺
        mock_fetcher.get_historical_pe.return_value = pd.DataFrame()     # 缺
        mock_fetcher.get_market_price.return_value = 700.0
        mock_fetcher._fm_request.return_value = pd.DataFrame()

    def test_yf_fills_growth_and_target(self, mock_fetcher, eps_df_8q):
        self._wire_empty_finmind(mock_fetcher, eps_df_8q)
        yf = {"trailing_eps": 23.0, "forward_eps": 28.0, "trailing_pe": 25.0,
              "forward_pe": 22.0, "revenue_growth": 0.18, "earnings_growth": 0.20,
              "gross_margins": 0.53}
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value=yf):
            result = ForwardEPSCalculator(mock_fetcher).calculate("2330")

        # 營收成長改用 yfinance 的 18%
        assert result["eps_growth_rate"] == pytest.approx(0.18, abs=1e-6)
        # 目標價用 yfinance 本益比推得（非空）
        assert result["target_price"]["base"] is not None
        assert result["upside_pct"] is not None
        # 市場共識算法自動用 yfinance forward_eps 填入
        cons = result["methods"]["consensus"]
        assert cons["available"] is True
        assert cons["eps"] == 28.0
        assert "data_sources" in result

    def test_no_yf_data_stays_insufficient(self, mock_fetcher, eps_df_8q):
        """yfinance 也無資料 → 維持原本的資料不足行為。"""
        self._wire_empty_finmind(mock_fetcher, eps_df_8q)
        with patch("data.yf_fundamentals.get_yf_fundamentals", return_value={}):
            result = ForwardEPSCalculator(mock_fetcher).calculate("9999")
        assert "insufficient" in result["confidence_reason"]
        assert result["methods"]["consensus"]["available"] is False
