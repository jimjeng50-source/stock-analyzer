"""
tests/test_history_backfill.py
Tests for screener/history_backfill.py — HistoryBackfiller
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from screener.history_backfill import HistoryBackfiller
from screener.recommendation_db import RecommendationDB


def _price_df(start: date, n_days: int, base: float, daily_gain: float = 0.5):
    """產生連續 n_days 的日 K 資料（跳過週末）。"""
    dates, closes = [], []
    d, px = start, base
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(pd.Timestamp(d))
            closes.append(px)
            px += daily_gain
        d += timedelta(days=1)
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1_000_000] * n_days,
    })


class TestScoreAsOf:
    def test_no_lookahead(self):
        """評分只用截止日（含）之前的價格。"""
        start = date(2026, 1, 1)
        pdf = _price_df(start, 150, 100.0)
        cutoff = pdf["date"].iloc[99]

        captured = {}

        def fake_technical(p):
            captured["max_date"] = p["date"].max()
            return {}

        scorer = MagicMock()
        scorer.score.return_value = {"total_score": 75.0}

        row = HistoryBackfiller._score_as_of(
            "2330", cutoff, {"2330": pdf}, {},
            lambda a, b: {}, fake_technical, lambda a, b, c: {}, lambda p: {},
            scorer,
        )
        assert row is not None
        assert captured["max_date"] <= cutoff
        assert row["entry_price"] == pytest.approx(float(pdf["close"].iloc[99]), abs=0.01)

    def test_insufficient_history_returns_none(self):
        pdf = _price_df(date(2026, 5, 1), 30, 100.0)   # 只有 30 天 < 60
        row = HistoryBackfiller._score_as_of(
            "2330", pdf["date"].iloc[-1], {"2330": pdf}, {},
            lambda a, b: {}, lambda p: {}, lambda a, b, c: {}, lambda p: {},
            MagicMock(),
        )
        assert row is None


class TestPriceAt:
    def test_returns_last_close_before_target(self):
        pdf = _price_df(date(2026, 6, 1), 20, 100.0, daily_gain=1.0)
        target = pdf["date"].iloc[5] + pd.Timedelta(hours=12)   # 當日盤後
        px = HistoryBackfiller._price_at(pdf, target)
        assert px == pytest.approx(float(pdf["close"].iloc[5]), abs=0.01)

    def test_future_target_returns_none(self):
        """target 超過資料範圍（未來）→ None，留待排程回填。"""
        pdf = _price_df(date(2026, 6, 1), 10, 100.0)
        assert HistoryBackfiller._price_at(pdf, pdf["date"].max() + pd.Timedelta(days=30)) is None


class TestRunBackfill:
    def _setup(self, tmp_path, n_price_days=200):
        """共用 mock 環境：3 支股票、價格資料齊全。"""
        db = RecommendationDB(db_path=str(tmp_path / "bf.db"))
        universe = pd.DataFrame([
            {"stock_id": "2330", "stock_name": "台積電", "market_cap_b": 500.0},
            {"stock_id": "2317", "stock_name": "鴻海", "market_cap_b": 120.0},
            {"stock_id": "2454", "stock_name": "聯發科", "market_cap_b": 80.0},
        ])
        price_start = date.today() - timedelta(days=300)
        price_map = {
            "2330": _price_df(price_start, n_price_days, 800.0, 2.0),
            "2317": _price_df(price_start, n_price_days, 100.0, -0.2),
            "2454": _price_df(price_start, n_price_days, 1000.0, 1.0),
        }
        return db, universe, price_map

    def test_backfills_days_and_performance(self, tmp_path):
        db, universe, price_map = self._setup(tmp_path)

        # 用 db_path 讓 backfiller 內部 RecommendationDB() 指向 tmp DB
        start = date.today() - timedelta(days=40)

        with patch("screener.universe.UniverseManager.get_universe", return_value=universe), \
             patch.object(HistoryBackfiller, "_bulk_price_history", return_value=price_map), \
             patch.object(HistoryBackfiller, "_fetch_finmind_once", return_value={}), \
             patch("screener.recommendation_db.RecommendationDB", return_value=db), \
             patch("models.scorer.Scorer") as MockScorer:
            MockScorer.return_value.score.side_effect = (
                lambda *a, **k: {"total_score": float(np.random.default_rng(0).uniform(60, 80))}
            )
            result = HistoryBackfiller(universe_size=3, top_k=3).run(start=start)

        assert result["error"] is None
        assert result["days_done"] > 10          # 約 4 週的交易日
        assert result["recs_saved"] == result["days_done"] * 3

        # 抽查一天：有 3 筆、rank 排序、5 日績效已回填（20 日部分可能超出資料）
        df = db.get_recent_recommendations(n_days=45)
        assert not df.empty
        one_day = df[df["recommend_date"] == df["recommend_date"].iloc[-1]]
        assert len(one_day) == 3
        assert one_day["recommendation"].iloc[0] == "歷史回補"
        evaluated_5d = df.dropna(subset=["return_5d_pct"])
        assert len(evaluated_5d) > 0

    def test_skips_days_with_existing_recommendations(self, tmp_path):
        db, universe, price_map = self._setup(tmp_path)
        start = date.today() - timedelta(days=40)

        # 先塞一天既有推薦
        existing_day = None
        for offset in range(10):
            d = start + timedelta(days=offset)
            if d.weekday() < 5:
                existing_day = d
                break
        db.save_recommendations(existing_day, [
            {"rank": 1, "stock_id": "9999", "stock_name": "既有",
             "total_score": 99.0, "current_price": 50.0, "key_reasons": []},
        ])

        with patch("screener.universe.UniverseManager.get_universe", return_value=universe), \
             patch.object(HistoryBackfiller, "_bulk_price_history", return_value=price_map), \
             patch.object(HistoryBackfiller, "_fetch_finmind_once", return_value={}), \
             patch("screener.recommendation_db.RecommendationDB", return_value=db), \
             patch("models.scorer.Scorer") as MockScorer:
            MockScorer.return_value.score.return_value = {"total_score": 70.0}
            result = HistoryBackfiller(universe_size=3, top_k=3).run(start=start)

        assert result["days_skipped"] >= 1
        # 既有推薦未被覆蓋
        rows = db.get_recommendations(existing_day)
        assert rows[0]["stock_id"] == "9999"

    def test_error_when_universe_empty(self, tmp_path):
        with patch("screener.universe.UniverseManager.get_universe",
                   return_value=pd.DataFrame()):
            result = HistoryBackfiller().run(start=date.today() - timedelta(days=30))
        assert result["error"] == "無法取得候選池"
