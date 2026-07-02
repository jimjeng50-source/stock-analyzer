"""
tests/test_watchlist_sync.py
月營收追蹤清單同步（近 60 天每日推薦個股）+ 推薦 Forward EPS 欄位
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alerts.revenue_calendar import RevenueCalendar
from screener.recommendation_db import RecommendationDB


@pytest.fixture
def calendar(tmp_path):
    fetcher = MagicMock()
    return RevenueCalendar(fetcher, db_path=str(tmp_path / "revenue.db"))


def _rec_df(rows):
    return pd.DataFrame(rows)


class TestSyncFromRecommendations:
    def test_adds_recommended_stocks_as_auto(self, calendar):
        recs = _rec_df([
            {"stock_id": "2330", "stock_name": "台積電"},
            {"stock_id": "2317", "stock_name": "鴻海"},
        ])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=recs):
            stats = calendar.sync_from_recommendations(n_days=60)

        assert stats["added"] == 2
        wl = {w["stock_id"]: w for w in calendar.get_watchlist()}
        assert wl["2330"]["source"] == "auto"
        assert wl["2330"]["stock_name"] == "台積電"

    def test_removes_auto_stocks_outside_window(self, calendar):
        """超出 60 天窗口的 auto 股被移除，手動股保留。"""
        # 先同步進兩支 auto
        recs = _rec_df([
            {"stock_id": "2330", "stock_name": "台積電"},
            {"stock_id": "2317", "stock_name": "鴻海"},
        ])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=recs):
            calendar.sync_from_recommendations()
        # 手動加一支
        calendar.add_to_watchlist("2454", "聯發科")

        # 第二次同步：只剩 2330 在窗口內
        recs2 = _rec_df([{"stock_id": "2330", "stock_name": "台積電"}])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=recs2):
            stats = calendar.sync_from_recommendations()

        ids = {w["stock_id"] for w in calendar.get_watchlist()}
        assert ids == {"2330", "2454"}      # 2317 移除、手動 2454 保留
        assert stats["removed"] == 1

    def test_manual_stock_not_downgraded(self, calendar):
        """已手動加入的股票再被推薦，來源仍是 manual。"""
        calendar.add_to_watchlist("2330", "台積電")
        recs = _rec_df([{"stock_id": "2330", "stock_name": "台積電"}])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=recs):
            stats = calendar.sync_from_recommendations()

        wl = calendar.get_watchlist()
        assert stats["added"] == 0
        assert wl[0]["source"] == "manual"

    def test_empty_recommendations_no_error(self, calendar):
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=pd.DataFrame()):
            stats = calendar.sync_from_recommendations()
        assert stats["added"] == 0
        assert stats["total"] == 0


class TestForwardEPSInRecommendations:
    def test_forward_eps_roundtrip(self, tmp_path):
        db = RecommendationDB(db_path=str(tmp_path / "feps.db"))
        db.save_recommendations(date.today(), [
            {"rank": 1, "stock_id": "2330", "stock_name": "台積電",
             "total_score": 85.0, "current_price": 800.0, "key_reasons": [],
             "forward_eps": 42.5, "eps_growth_rate": 18.3},
        ])
        row = db.get_recommendations(date.today())[0]
        assert row["forward_eps"] == 42.5
        assert row["eps_growth_pct"] == 18.3

    def test_forward_eps_nullable(self, tmp_path):
        """深度分析失敗時 forward_eps 為 None，不影響儲存。"""
        db = RecommendationDB(db_path=str(tmp_path / "feps2.db"))
        db.save_recommendations(date.today(), [
            {"rank": 1, "stock_id": "2317", "stock_name": "鴻海",
             "total_score": 75.0, "current_price": 100.0, "key_reasons": []},
        ])
        row = db.get_recommendations(date.today())[0]
        assert row["forward_eps"] is None
