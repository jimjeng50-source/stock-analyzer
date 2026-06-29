"""
tests/test_recommendation_db.py
RecommendationDB SQLite 資料庫讀寫單元測試
"""

import pytest
import sqlite3
from datetime import date

import pandas as pd

from screener.recommendation_db import RecommendationDB


_SAMPLE_REC = {
    "rank": 1,
    "stock_id": "2330",
    "stock_name": "台積電",
    "total_score": 82.5,
    "recommendation": "強力買進",
    "current_price": 850.0,
    "key_reasons": ["理由1", "理由2", "理由3"],
    "risk_warning": "注意風險",
    "target_price_base": 950.0,
    "upside_pct": 11.8,
    "industry": "半導體",
    "score_breakdown": {
        "chips_score": 88,
        "fundamental_score": 80,
        "technical_score": 78,
        "momentum_score": 82,
    },
}

_SAMPLE_SCAN_LOG = {
    "universe_count": 200,
    "after_filter_count": 120,
    "scored_count": 118,
    "failed_count": 2,
    "scan_duration_sec": 145.6,
    "top_score": 82.5,
}


@pytest.fixture
def db(tmp_path):
    """使用臨時路徑建立 RecommendationDB。"""
    return RecommendationDB(db_path=str(tmp_path / "test_rec.db"))


class TestRecommendationDB:

    def test_init_creates_tables(self, db):
        """初始化後應建立 daily_recommendations 與 scan_logs 資料表。"""
        conn = sqlite3.connect(db.db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "daily_recommendations" in tables
        assert "scan_logs" in tables

    def test_save_and_get_recommendations(self, db):
        """存入 2 筆推薦，get_recommendations 應回傳 2 筆。"""
        today = date.today()
        rec2 = dict(_SAMPLE_REC, rank=2, stock_id="2317", stock_name="鴻海",
                    total_score=71.0, current_price=180.0)
        db.save_recommendations(today, [_SAMPLE_REC, rec2])
        result = db.get_recommendations(today)
        assert len(result) == 2
        ids = {r["stock_id"] for r in result}
        assert "2330" in ids
        assert "2317" in ids

    def test_save_duplicate_ignored(self, db):
        """同日同股票代號存入兩次，資料庫中應只有 1 筆。"""
        today = date.today()
        db.save_recommendations(today, [_SAMPLE_REC])
        db.save_recommendations(today, [_SAMPLE_REC])  # 重複
        result = db.get_recommendations(today)
        assert len(result) == 1

    def test_save_scan_log(self, db):
        """save_scan_log 不應拋出例外。"""
        db.save_scan_log(date.today(), _SAMPLE_SCAN_LOG)

    def test_get_recent_recommendations_returns_df(self, db):
        """存入資料後，get_recent_recommendations 應回傳非空 DataFrame。"""
        db.save_recommendations(date.today(), [_SAMPLE_REC])
        df = db.get_recent_recommendations(n_days=30)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_get_performance_summary_no_data(self, db):
        """無資料時，get_performance_summary 應回傳包含 total_recommendations=0 的 dict。"""
        summary = db.get_performance_summary(n_days=90)
        assert isinstance(summary, dict)
        assert summary.get("total_recommendations", 0) == 0
        assert summary.get("evaluated_count", 0) == 0

    def test_update_performance(self, db):
        """
        存入一筆推薦後，update_performance 填入 price_5d=920.0，
        重新取得後 return_5d_pct 應約等於 (920/850 - 1) * 100 ≈ 8.24%。
        """
        today = date.today()
        db.save_recommendations(today, [_SAMPLE_REC])
        db.update_performance("2330", today, price_5d=920.0)
        recs = db.get_recommendations(today)
        assert len(recs) == 1
        ret5 = recs[0].get("return_5d_pct")
        assert ret5 is not None
        expected = (920.0 / 850.0 - 1) * 100
        assert abs(ret5 - expected) < 0.01
