"""
tests/test_historical_eval.py
Tests for screener/historical_eval.py
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from screener.historical_eval import HistoricalEvaluator, evaluate_60d_accuracy
from screener.recommendation_db import RecommendationDB


class TestHistoricalEvaluator:
    def test_evaluate_computes_returns_and_win_rate(self):
        """完整流程（mock 掉外部資料）：報酬率、勝率、大盤比較正確。"""
        universe = pd.DataFrame([
            {"stock_id": "2330", "stock_name": "台積電", "market_cap_b": 500.0},
            {"stock_id": "2317", "stock_name": "鴻海", "market_cap_b": 120.0},
            {"stock_id": "2454", "stock_name": "聯發科", "market_cap_b": 80.0},
        ])
        scored = pd.DataFrame([
            {"stock_id": "2330", "total_score": 85.0},
            {"stock_id": "2317", "total_score": 75.0},
            {"stock_id": "2454", "total_score": 65.0},
        ])
        prices = {
            "2330": (800.0, 880.0),    # +10%
            "2317": (100.0, 95.0),     # -5%
            "2454": (1000.0, 1100.0),  # +10%
            "^TWII": (20000.0, 20400.0),  # 大盤 +2%
        }

        ev = HistoricalEvaluator(universe_size=3)
        with patch("screener.universe.UniverseManager.get_universe", return_value=universe), \
             patch("screener.batch_scorer.BatchScorer.score_universe", return_value=scored), \
             patch.object(ev, "_get_entry_exit_price",
                          side_effect=lambda sid, d, h: prices.get(sid, (None, None))):
            result = ev.evaluate(days_ago=90, horizon_days=60, top_k=3, save_to_db=False)

        assert result["error"] is None
        assert len(result["picks"]) == 3
        assert result["avg_return_pct"] == 5.0            # (10 - 5 + 10) / 3
        assert result["win_rate"] == round(2 / 3, 3)
        assert result["benchmark_return_pct"] == 2.0
        assert result["alpha_pct"] == 3.0

        pick = result["picks"][0]
        assert pick["stock_id"] == "2330"
        assert pick["return_pct"] == 10.0
        assert pick["win"] is True

    def test_evaluate_error_when_universe_empty(self):
        ev = HistoricalEvaluator()
        with patch("screener.universe.UniverseManager.get_universe",
                   return_value=pd.DataFrame()):
            result = ev.evaluate(save_to_db=False)
        assert result["error"] == "無法取得候選池"

    def test_evaluate_error_when_scoring_empty(self):
        universe = pd.DataFrame([{"stock_id": "2330", "stock_name": "台積電", "market_cap_b": 1.0}])
        ev = HistoricalEvaluator()
        with patch("screener.universe.UniverseManager.get_universe", return_value=universe), \
             patch("screener.batch_scorer.BatchScorer.score_universe", return_value=pd.DataFrame()):
            result = ev.evaluate(save_to_db=False)
        assert "歷史評分無結果" in result["error"]


class TestEvaluate60dAccuracy:
    def test_report_from_db(self, tmp_path):
        """DB 中有 60 日績效的推薦 → 正確計算報告。"""
        db = RecommendationDB(db_path=str(tmp_path / "test.db"))
        rec_date = date.today() - timedelta(days=90)
        db.save_recommendations(rec_date, [
            {"rank": 1, "stock_id": "2330", "stock_name": "台積電",
             "total_score": 85.0, "current_price": 800.0, "key_reasons": [], "hot_tags": ["籌碼:法人買超"]},
            {"rank": 2, "stock_id": "2317", "stock_name": "鴻海",
             "total_score": 75.0, "current_price": 100.0, "key_reasons": []},
            {"rank": 4, "stock_id": "9999", "stock_name": "第四名",
             "total_score": 60.0, "current_price": 50.0, "key_reasons": []},
        ])
        db.update_performance("2330", rec_date, price_60d=880.0)   # +10%
        db.update_performance("2317", rec_date, price_60d=95.0)    # -5%
        db.update_performance("9999", rec_date, price_60d=100.0)   # rank 4，不入樣本

        report = evaluate_60d_accuracy(db, top_k=3)

        assert report["overall"]["evaluated"] == 2
        assert report["overall"]["avg_return_pct"] == 2.5   # (10 - 5) / 2
        assert report["overall"]["win_rate"] == 0.5
        assert report["overall"]["dates"] == 1
        assert set(report["by_date"]["stock_id"]) == {"2330", "2317"}

    def test_empty_db_returns_empty_report(self, tmp_path):
        db = RecommendationDB(db_path=str(tmp_path / "empty.db"))
        report = evaluate_60d_accuracy(db)
        assert report["overall"] == {}
        assert report["by_date"].empty


class TestDBMigrationAndHotTags:
    def test_hot_tags_roundtrip(self, tmp_path):
        db = RecommendationDB(db_path=str(tmp_path / "hot.db"))
        rec_date = date.today()
        db.save_recommendations(rec_date, [
            {"rank": 1, "stock_id": "2330", "stock_name": "台積電",
             "total_score": 85.0, "current_price": 800.0, "key_reasons": [],
             "hot_tags": ["籌碼:法人買超前十", "社群:PTT熱議(5篇)"]},
        ])
        rows = db.get_recommendations(rec_date)
        assert rows[0]["hot_tags"] == "籌碼:法人買超前十, 社群:PTT熱議(5篇)"

    def test_update_performance_60d_only(self, tmp_path):
        """只回填 60 日不影響 5/20 日欄位。"""
        db = RecommendationDB(db_path=str(tmp_path / "p60.db"))
        rec_date = date.today() - timedelta(days=70)
        db.save_recommendations(rec_date, [
            {"rank": 1, "stock_id": "2330", "stock_name": "台積電",
             "total_score": 85.0, "current_price": 100.0, "key_reasons": []},
        ])
        db.update_performance("2330", rec_date, price_60d=120.0)
        row = db.get_recommendations(rec_date)[0]
        assert row["price_60d"] == 120.0
        assert row["return_60d_pct"] == pytest.approx(20.0)
        assert row["price_5d"] is None
        assert row["return_5d_pct"] is None

    def test_performance_summary_includes_60d(self, tmp_path):
        db = RecommendationDB(db_path=str(tmp_path / "s60.db"))
        rec_date = date.today() - timedelta(days=70)
        db.save_recommendations(rec_date, [
            {"rank": 1, "stock_id": "2330", "stock_name": "台積電",
             "total_score": 85.0, "current_price": 100.0, "key_reasons": []},
        ])
        db.update_performance("2330", rec_date, price_60d=110.0)
        summary = db.get_performance_summary(n_days=90)
        assert summary["avg_return_60d"] == pytest.approx(10.0)
        assert summary["win_rate_60d"] == 1.0
