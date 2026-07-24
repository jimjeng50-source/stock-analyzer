"""
tests/test_jobs_tracking.py
job_track_daily 兩個月每日績效追蹤任務整合測試（mock 價格抓取）
"""

import pytest
import pandas as pd
from datetime import date
from unittest.mock import patch, MagicMock

import jobs
from screener.recommendation_db import RecommendationDB


def _seed_rec(db, today, entry=100.0):
    db.save_recommendations(today, [{
        "rank": 1, "stock_id": "2330", "stock_name": "台積電", "total_score": 80,
        "recommendation": "買進", "current_price": entry, "key_reasons": [],
        "risk_warning": "", "target_price_base": 120, "upside_pct": 20,
        "industry": "半導體", "score_breakdown": {}, "hot_tags": [],
    }])


def test_job_track_daily_fills_curve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # reports/ 寫到暫存目錄
    db_path = str(tmp_path / "rec.db")
    monkeypatch.setattr("screener.recommendation_db.RECOMMENDATION_DB_PATH", db_path)

    db = RecommendationDB(db_path=db_path)
    today = date.today()
    _seed_rec(db, today, entry=100.0)

    # 價格序列：推薦當日 close=105（day0），實務上 day0 close≈買進價
    price_df = pd.DataFrame({"date": [today.isoformat()], "close": [105.0]})
    fake_fetcher = MagicMock()
    fake_fetcher.get_price.return_value = price_df

    with patch("data.fetcher.FinMindFetcher", return_value=fake_fetcher):
        rc = jobs.job_track_daily()

    assert rc == 0
    curve = db.get_daily_returns(today, "2330")
    assert not curve.empty
    assert curve.iloc[0]["return_pct"] == pytest.approx(5.0)   # 105/100-1
    assert curve.iloc[0]["entry_price"] == pytest.approx(100.0)
    # 報告有輸出
    assert (tmp_path / "reports" / "tracking_2m.md").exists()


def test_job_track_daily_no_active(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "rec.db")
    monkeypatch.setattr("screener.recommendation_db.RECOMMENDATION_DB_PATH", db_path)
    RecommendationDB(db_path=db_path)                 # 空庫，無推薦
    with patch("data.fetcher.FinMindFetcher") as mock_f:
        rc = jobs.job_track_daily()
    assert rc == 0
    mock_f.assert_not_called()                        # 無推薦時不抓價
