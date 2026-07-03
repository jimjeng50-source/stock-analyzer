"""
tests/test_recommender.py
DailyRecommender 主動推薦主控器單元測試
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from screener.recommender import DailyRecommender


@pytest.fixture
def mock_universe_df():
    return pd.DataFrame([
        {"stock_id": "2330", "stock_name": "台積電", "market": "TWSE",
         "industry": "半導體", "market_cap_b": 20000, "avg_volume_k": 50000, "last_price": 850.0},
        {"stock_id": "2317", "stock_name": "鴻海", "market": "TWSE",
         "industry": "電子", "market_cap_b": 3000, "avg_volume_k": 60000, "last_price": 180.0},
    ])


@pytest.fixture
def mock_scored_df():
    return pd.DataFrame([
        {"stock_id": "2330", "stock_name": "台積電", "industry": "半導體",
         "total_score": 82.5, "recommendation": "強力買進",
         "chips_score": 88, "fundamental_score": 80, "technical_score": 78,
         "momentum_score": 82, "risk_score": 70, "current_price": 850.0, "error": ""},
        {"stock_id": "2317", "stock_name": "鴻海", "industry": "電子",
         "total_score": 71.0, "recommendation": "買進",
         "chips_score": 72, "fundamental_score": 68, "technical_score": 75,
         "momentum_score": 70, "risk_score": 65, "current_price": 180.0, "error": ""},
    ])


def _run_with_mocks(mock_universe_df, mock_scored_df, dry_run=True, universe_empty=False):
    """Helper: run DailyRecommender.run() with all external dependencies mocked."""
    univ = mock_universe_df if not universe_empty else pd.DataFrame()
    with patch("screener.recommender.UniverseManager") as MockUMgr, \
         patch("screener.recommender.QuickFilter") as MockFilter, \
         patch("screener.recommender.BatchScorer") as MockBScorer, \
         patch("screener.recommender.RecommendationDB") as MockDB, \
         patch.object(DailyRecommender, "_run_deep_analysis", return_value={}):

        MockUMgr.return_value.get_universe.return_value = univ
        MockUMgr.return_value.merge_with_custom.return_value = univ
        MockFilter.return_value.run.return_value = (univ, {"steps": [], "removed_ids": []})
        mock_bs = MockBScorer.return_value
        mock_bs.score_universe.return_value = mock_scored_df if not universe_empty else pd.DataFrame()
        mock_bs.failed_df = pd.DataFrame()

        rec = DailyRecommender()
        result = rec.run(dry_run=dry_run)
        save_called = MockDB.return_value.save_recommendations.called

    return result, save_called


class TestDailyRecommender:

    def test_run_dry_run_returns_dict(self, mock_universe_df, mock_scored_df):
        """run(dry_run=True) 應回傳包含必要 key 的 dict。"""
        result, _ = _run_with_mocks(mock_universe_df, mock_scored_df)
        for key in ("date", "recommendations", "scan_summary", "message", "error"):
            assert key in result, f"缺少 key：{key}"

    def test_run_dry_run_no_db_write(self, mock_universe_df, mock_scored_df):
        """dry_run=True 時不應寫入資料庫。"""
        _, save_called = _run_with_mocks(mock_universe_df, mock_scored_df, dry_run=True)
        assert not save_called

    def test_recommendations_have_required_keys(self, mock_universe_df, mock_scored_df):
        """每筆推薦應包含所有必要欄位。"""
        result, _ = _run_with_mocks(mock_universe_df, mock_scored_df)
        required = ["rank", "stock_id", "stock_name", "total_score", "recommendation",
                    "current_price", "key_reasons", "risk_warning", "industry", "score_breakdown"]
        for rec in result["recommendations"]:
            for key in required:
                assert key in rec, f"推薦缺少 key：{key}"

    def test_message_contains_disclaimer(self, mock_universe_df, mock_scored_df):
        """推播訊息應包含免責聲明。"""
        result, _ = _run_with_mocks(mock_universe_df, mock_scored_df)
        assert "不構成任何投資建議" in result["message"]

    def test_recommendations_sorted_by_rank(self, mock_universe_df, mock_scored_df):
        """推薦清單應按 rank 升序排列（1, 2, 3...）。"""
        result, _ = _run_with_mocks(mock_universe_df, mock_scored_df)
        ranks = [r["rank"] for r in result["recommendations"]]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_fallback_reasons_returns_3(self):
        """_fallback_reasons 應回傳恰好 3 條理由。"""
        rec = DailyRecommender.__new__(DailyRecommender)
        score_data = {
            "chips_score": 80, "fundamental_score": 70, "technical_score": 60,
            "momentum_score": 75, "risk_score": 50,
        }
        reasons = rec._fallback_reasons(score_data)
        assert isinstance(reasons, list)
        assert len(reasons) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in reasons)

    def test_error_returned_when_universe_empty(self, mock_universe_df, mock_scored_df):
        """候選股票池為空時，result['error'] 不應為 None。"""
        result, _ = _run_with_mocks(mock_universe_df, mock_scored_df, universe_empty=True)
        assert result["error"] is not None

    # ── 無達標推薦時的觀察名單 fallback ────────────────────────────────────────

    def test_watch_list_when_no_stock_reaches_threshold(self, mock_universe_df):
        """全部低於推薦門檻（70）但高於快篩門檻（60）→ 0 推薦 + 觀察名單。"""
        low_scored = pd.DataFrame([
            {"stock_id": "2330", "stock_name": "台積電", "industry": "半導體",
             "total_score": 66.0, "recommendation": "持有",
             "chips_score": 60, "fundamental_score": 65, "technical_score": 70,
             "momentum_score": 68, "risk_score": 60, "current_price": 850.0, "error": ""},
            {"stock_id": "2317", "stock_name": "鴻海", "industry": "電子",
             "total_score": 62.0, "recommendation": "持有",
             "chips_score": 58, "fundamental_score": 60, "technical_score": 66,
             "momentum_score": 64, "risk_score": 58, "current_price": 180.0, "error": ""},
        ])
        result, save_called = _run_with_mocks(mock_universe_df, low_scored, dry_run=False)

        assert result["error"] is None
        assert result["recommendations"] == []
        assert len(result["watch_list"]) == 2
        assert result["watch_list"][0]["stock_id"] == "2330"   # 最高分在前
        assert save_called is False                             # 觀察名單不寫入 DB
        assert "無個股達推薦門檻" in result["message"]
        assert "觀察名單" in result["message"]
        assert "66" in result["message"]                        # 顯示最高分
