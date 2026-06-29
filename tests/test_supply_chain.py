"""
tests/test_supply_chain.py
SupplyChainAnalyzer 與 get_stock_chain 單元測試
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from factors.supply_chain import (
    SUPPLY_CHAIN_MAP,
    SupplyChainAnalyzer,
    get_stock_chain,
    _normalize,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fetcher():
    f = MagicMock()
    # 預設：月營收回傳帶有 YoY 的 DataFrame
    rev_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="ME"),
        "revenue": [100_000, 110_000, 120_000],
        "revenue_yoy": [10.0, 15.0, 20.0],
    })
    f.get_monthly_revenue.return_value = rev_df

    # 外資淨買賣超
    fi_series = pd.Series(
        [100_000, -50_000, 200_000, 150_000, 300_000],
        index=pd.date_range("2024-04-01", periods=5, freq="B"),
    )
    f.get_institutional_net.return_value = fi_series
    return f


@pytest.fixture
def empty_fetcher():
    f = MagicMock()
    f.get_monthly_revenue.return_value = None
    f.get_institutional_net.return_value = pd.Series(dtype=float)
    return f


# ── SUPPLY_CHAIN_MAP 結構驗證 ─────────────────────────────────────────────────

class TestSupplyChainMap:

    def test_three_chains_defined(self):
        """應定義 3 條產業鏈。"""
        assert len(SUPPLY_CHAIN_MAP) == 3

    def test_chain_keys(self):
        """應包含 semiconductor、ai_server、ev_components。"""
        assert "semiconductor" in SUPPLY_CHAIN_MAP
        assert "ai_server" in SUPPLY_CHAIN_MAP
        assert "ev_components" in SUPPLY_CHAIN_MAP

    def test_each_chain_has_three_tiers(self):
        """每條鏈應有 upstream、midstream、downstream 三層。"""
        for chain_key, chain_info in SUPPLY_CHAIN_MAP.items():
            tiers = chain_info.get("tiers", {})
            assert "upstream" in tiers, f"{chain_key} 缺少 upstream"
            assert "midstream" in tiers, f"{chain_key} 缺少 midstream"
            assert "downstream" in tiers, f"{chain_key} 缺少 downstream"

    def test_tier_weights_sum_to_one(self):
        """各層權重總和應為 1.0。"""
        for chain_key, chain_info in SUPPLY_CHAIN_MAP.items():
            total_weight = sum(
                tier["weight"] for tier in chain_info["tiers"].values()
            )
            assert abs(total_weight - 1.0) < 1e-6, f"{chain_key} 權重總和 = {total_weight}"

    def test_lead_lag_months_positive(self):
        """Lead-Lag 月數應為正整數。"""
        for chain_key, chain_info in SUPPLY_CHAIN_MAP.items():
            assert chain_info["lead_lag_months"] > 0


# ── get_stock_chain 測試 ──────────────────────────────────────────────────────

class TestGetStockChain:

    def test_tsmc_in_semiconductor_midstream(self):
        """台積電（2330）應在半導體產業鏈的中游。"""
        result = get_stock_chain("2330")
        assert result is not None
        chain_key, tier = result
        assert chain_key == "semiconductor"
        assert tier == "midstream"

    def test_stock_not_in_any_chain_returns_none(self):
        """不在任何產業鏈的股票應回傳 None。"""
        result = get_stock_chain("9999")
        assert result is None

    def test_returns_tuple(self):
        """回傳值應為 tuple。"""
        result = get_stock_chain("2317")
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_upstream_stock(self):
        """上游股票應被正確識別。"""
        # 5483 (環球晶) 在半導體上游
        result = get_stock_chain("5483")
        if result is not None:
            chain_key, tier = result
            assert tier == "upstream"


# ── _normalize 測試 ───────────────────────────────────────────────────────────

class TestNormalize:

    def test_middle_value_zero(self):
        """中間值應正規化為 0。"""
        assert _normalize(50, 0, 100) == pytest.approx(0.0)

    def test_low_value_minus_one(self):
        """最低值應正規化為 -1。"""
        assert _normalize(0, 0, 100) == pytest.approx(-1.0)

    def test_high_value_plus_one(self):
        """最高值應正規化為 +1。"""
        assert _normalize(100, 0, 100) == pytest.approx(1.0)

    def test_clamps_below_minus_one(self):
        """超出範圍的值應截斷至 -1。"""
        assert _normalize(-100, 0, 100) == pytest.approx(-1.0)

    def test_clamps_above_plus_one(self):
        """超出範圍的值應截斷至 +1。"""
        assert _normalize(200, 0, 100) == pytest.approx(1.0)

    def test_equal_low_high_returns_zero(self):
        """low == high 時不應崩潰，回傳 0。"""
        assert _normalize(50, 50, 50) == pytest.approx(0.0)


# ── SupplyChainAnalyzer 測試 ──────────────────────────────────────────────────

class TestSupplyChainAnalyzer:

    def test_analyze_chain_returns_dict(self, mock_fetcher):
        """analyze_chain 應回傳包含必要 key 的 dict。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_chain("semiconductor")

        required = [
            "chain_name", "overall_signal", "signal_label",
            "tier_signals", "lead_lag_months", "capital_flow_direction",
            "top_stocks_to_watch", "timestamp",
        ]
        for key in required:
            assert key in result, f"缺少 key：{key}"

    def test_overall_signal_in_range(self, mock_fetcher):
        """overall_signal 應在 [-1.0, 1.0] 範圍內。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_chain("semiconductor")
        assert -1.0 <= result["overall_signal"] <= 1.0

    def test_signal_label_valid(self, mock_fetcher):
        """signal_label 應為預期的字串之一。"""
        valid_labels = {"強勢擴張", "緩步回溫", "中性盤整", "景氣降溫", "明顯收縮"}
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_chain("ai_server")
        assert result["signal_label"] in valid_labels

    def test_all_three_chains_analyzable(self, mock_fetcher):
        """三條產業鏈均應可分析。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        for chain_key in SUPPLY_CHAIN_MAP.keys():
            result = analyzer.analyze_chain(chain_key)
            assert result["overall_signal"] is not None

    def test_invalid_chain_returns_neutral(self, mock_fetcher):
        """無效產業鏈名稱應回傳 0 信號。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_chain("nonexistent_chain")
        assert result["overall_signal"] == 0.0

    def test_analyze_for_stock_tsmc(self, mock_fetcher):
        """analyze_for_stock(2330) 應回傳有效結果。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_for_stock("2330")

        required = [
            "stock_id", "chain_name", "tier", "chain_signal",
            "upstream_signal", "lead_lag_impact", "expected_impact_in",
            "chain_score", "interpretation",
        ]
        for key in required:
            assert key in result, f"缺少 key：{key}"

    def test_chain_score_in_range(self, mock_fetcher):
        """chain_score 應在 0-100 範圍內。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_for_stock("2330")
        assert 0 <= result["chain_score"] <= 100

    def test_unknown_stock_chain_score_50(self, mock_fetcher):
        """不在任何產業鏈的股票，chain_score 應為 50（中性）。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        result = analyzer.analyze_for_stock("9999")
        assert result["chain_score"] == pytest.approx(50.0)

    def test_empty_data_doesnt_crash(self, empty_fetcher):
        """資料為空時，分析器不應崩潰。"""
        analyzer = SupplyChainAnalyzer(empty_fetcher)
        result = analyzer.analyze_chain("ev_components")
        assert "overall_signal" in result
        assert -1.0 <= result["overall_signal"] <= 1.0

    def test_interpret_signal(self, mock_fetcher):
        """_interpret_signal 應回傳對應標籤。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        assert analyzer._interpret_signal(0.8) == "強勢擴張"
        assert analyzer._interpret_signal(0.4) == "緩步回溫"
        assert analyzer._interpret_signal(0.0) == "中性盤整"
        assert analyzer._interpret_signal(-0.4) == "景氣降溫"
        assert analyzer._interpret_signal(-0.8) == "明顯收縮"

    def test_capital_flow_text_contains_chain_name(self, mock_fetcher):
        """_generate_capital_flow_text 輸出應包含產業鏈名稱。"""
        analyzer = SupplyChainAnalyzer(mock_fetcher)
        tier_signals = {"upstream": 0.5, "midstream": 0.1, "downstream": 0.0}
        text = analyzer._generate_capital_flow_text("semiconductor", tier_signals, 2)
        assert "半導體" in text
