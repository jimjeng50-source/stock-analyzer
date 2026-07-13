"""tests/test_pool_analyzer.py — 候選池全工具分析"""
from unittest.mock import patch

import pandas as pd

from screener.pool_analyzer import analyze_pool


def _scored():
    return pd.DataFrame([
        {"stock_id": "2330", "total_score": 82.0, "recommendation": "買進",
         "chips_score": 80, "fundamental_score": 85, "technical_score": 78,
         "momentum_score": 80, "risk_score": 70, "current_price": 850.0},
        {"stock_id": "2317", "total_score": 68.0, "recommendation": "持有",
         "chips_score": 65, "fundamental_score": 70, "technical_score": 66,
         "momentum_score": 64, "risk_score": 62, "current_price": 180.0},
    ])


class TestAnalyzePool:
    def test_merges_scoring_and_deep(self):
        deep = {
            "2330": {"forward_eps": 45.0, "eps_growth_rate": 18.0,
                     "target_price_base": 1000.0, "upside_pct": 17.6,
                     "chain_name": "半導體", "chain_signal": 0.6},
            "2317": {"forward_eps": 12.0, "eps_growth_rate": 5.0},
        }
        with patch("screener.batch_scorer.BatchScorer") as MockBS, \
             patch("screener.pool_analyzer._deep_analysis", return_value=deep):
            MockBS.return_value.score_universe.return_value = _scored()
            df = analyze_pool(["2330", "2317"], name_map={"2330": "台積電", "2317": "鴻海"})

        assert list(df["stock_id"]) == ["2330", "2317"]     # 依綜合分排序
        row = df[df["stock_id"] == "2330"].iloc[0]
        assert row["stock_name"] == "台積電"
        assert row["forward_eps"] == 45.0
        assert row["target_price"] == 1000.0
        assert row["chain_name"] == "半導體"

    def test_failed_stocks_appended(self):
        """BatchScorer 只回成功者，失敗個股要補回並標記。"""
        with patch("screener.batch_scorer.BatchScorer") as MockBS, \
             patch("screener.pool_analyzer._deep_analysis", return_value={}):
            MockBS.return_value.score_universe.return_value = _scored()
            df = analyze_pool(["2330", "2317", "9999"],
                              name_map={"9999": "壞股"})
        bad = df[df["stock_id"] == "9999"].iloc[0]
        assert bad["error"] != ""
        assert pd.isna(bad["total_score"]) or bad["total_score"] is None

    def test_empty_input(self):
        assert analyze_pool([]).empty

    def test_all_failed_returns_skeleton(self):
        with patch("screener.batch_scorer.BatchScorer") as MockBS:
            MockBS.return_value.score_universe.return_value = pd.DataFrame()
            df = analyze_pool(["2330"], name_map={"2330": "台積電"})
        assert len(df) == 1
        assert df.iloc[0]["error"] != ""
