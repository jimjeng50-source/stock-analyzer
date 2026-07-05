"""
tests/test_risk_monitor.py
Tests for alerts/risk_monitor.py — RiskMonitor
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alerts.risk_monitor import RiskMonitor
from screener.recommendation_db import RecommendationDB


def _recs_df(rows):
    return pd.DataFrame(rows)


class TestPositionRisk:
    def _base_row(self, sid, name, entry, target=None):
        return {
            "recommend_date": (date.today() - timedelta(days=10)).isoformat(),
            "stock_id": sid, "stock_name": name,
            "current_price": entry, "target_price": target,
        }

    def test_stop_loss_alert(self):
        df = _recs_df([self._base_row("2330", "台積電", 100.0)])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices", return_value={"2330": {"price": 87.0, "ma60": None}}):
            alerts = RiskMonitor().check_position_risk()
        assert len(alerts) == 1
        assert alerts[0]["action"] == "stop_loss"
        assert "停損" in alerts[0]["msg"]

    def test_take_profit_alert(self):
        df = _recs_df([self._base_row("2330", "台積電", 100.0)])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices", return_value={"2330": {"price": 116.0, "ma60": 100.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert alerts[0]["action"] == "take_profit"

    def test_target_hit_takes_precedence(self):
        df = _recs_df([self._base_row("2330", "台積電", 100.0, target=110.0)])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices", return_value={"2330": {"price": 112.0, "ma60": 100.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert alerts[0]["action"] == "target_hit"

    def test_no_alert_within_range(self):
        df = _recs_df([self._base_row("2330", "台積電", 100.0)])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices", return_value={"2330": {"price": 103.0, "ma60": 95.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert alerts == []

    def test_uses_latest_recommendation_per_stock(self):
        """同一股票多次推薦，以最近一次的推薦價為準。"""
        old = self._base_row("2330", "台積電", 200.0)
        old["recommend_date"] = (date.today() - timedelta(days=50)).isoformat()
        new = self._base_row("2330", "台積電", 100.0)
        df = _recs_df([old, new])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices", return_value={"2330": {"price": 103.0, "ma60": 95.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert alerts == []   # vs 100 → +3%，非 vs 200 → -48%


class TestEPSRisk:
    def test_eps_downgrade_alert(self):
        df = _recs_df([{
            "recommend_date": date.today().isoformat(),
            "stock_id": "2330", "stock_name": "台積電", "forward_eps": 40.0,
        }])
        calc = MagicMock()
        calc.calculate.return_value = {"forward_eps_1y": 34.0, "error": None}  # -15%
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch("factors.forward_eps.ForwardEPSCalculator", return_value=calc), \
             patch("data.fetcher.DataFetcher"):
            alerts = RiskMonitor().check_eps_risk()
        assert len(alerts) == 1
        assert "下修" in alerts[0]["msg"]

    def test_no_alert_when_eps_stable(self):
        df = _recs_df([{
            "recommend_date": date.today().isoformat(),
            "stock_id": "2330", "stock_name": "台積電", "forward_eps": 40.0,
        }])
        calc = MagicMock()
        calc.calculate.return_value = {"forward_eps_1y": 41.0, "error": None}
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch("factors.forward_eps.ForwardEPSCalculator", return_value=calc), \
             patch("data.fetcher.DataFetcher"):
            alerts = RiskMonitor().check_eps_risk()
        assert alerts == []


class TestRunDailyAndFormat:
    def test_run_daily_aggregates_and_flags(self):
        rm = RiskMonitor()
        with patch.object(rm, "check_market_risk", return_value=["⚠️ 大盤跌 2%"]), \
             patch.object(rm, "check_position_risk", return_value=[]), \
             patch.object(rm, "check_revenue_risk", return_value=[]), \
             patch.object(rm, "check_eps_risk", return_value=[]):
            report = rm.run_daily()
        assert report["has_alerts"] is True
        assert report["market"] == ["⚠️ 大盤跌 2%"]

    def test_run_daily_no_alerts(self):
        rm = RiskMonitor()
        with patch.object(rm, "check_market_risk", return_value=[]), \
             patch.object(rm, "check_position_risk", return_value=[]), \
             patch.object(rm, "check_revenue_risk", return_value=[]), \
             patch.object(rm, "check_eps_risk", return_value=[]):
            report = rm.run_daily()
        assert report["has_alerts"] is False

    def test_check_failure_does_not_break_others(self):
        rm = RiskMonitor()
        with patch.object(rm, "check_market_risk", side_effect=Exception("down")), \
             patch.object(rm, "check_position_risk",
                          return_value=[{"stock_id": "2330", "stock_name": "台積電",
                                         "action": "stop_loss", "msg": "🔴 停損"}]), \
             patch.object(rm, "check_revenue_risk", return_value=[]), \
             patch.object(rm, "check_eps_risk", return_value=[]):
            report = rm.run_daily()
        assert report["market"] == []
        assert report["has_alerts"] is True

    def test_format_message_sections(self):
        rm = RiskMonitor()
        report = {
            "market": ["⚠️ 大盤單日下跌 -2.0%"],
            "positions": [{"msg": "🔴 2330 停損警訊", "action": "stop_loss"}],
            "revenue": [{"msg": "📅 2317 即將公布", "type": "upcoming"}],
            "eps": [],
            "has_alerts": True,
            "checked_at": "2026-07-02",
        }
        msg = rm.format_message(report)
        assert "大盤風險" in msg
        assert "持股警訊" in msg
        assert "營收動態" in msg
        assert "Forward EPS" not in msg      # 空區塊不顯示
        assert "不構成投資建議" in msg


class TestMA60StopLoss:
    def test_below_ma60_with_loss_alerts(self):
        """未達 -12% 但跌破 60 日線且虧損 → 趨勢轉弱警訊。"""
        df = pd.DataFrame([{
            "recommend_date": (date.today() - timedelta(days=10)).isoformat(),
            "stock_id": "2330", "stock_name": "台積電",
            "current_price": 100.0, "target_price": None,
        }])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices",
                          return_value={"2330": {"price": 95.0, "ma60": 98.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert len(alerts) == 1
        assert alerts[0]["action"] == "stop_loss"
        assert "60 日線" in alerts[0]["msg"]

    def test_below_ma60_but_profitable_no_alert(self):
        """跌破 60 日線但仍獲利 → 不警示（避免噪音）。"""
        df = pd.DataFrame([{
            "recommend_date": (date.today() - timedelta(days=10)).isoformat(),
            "stock_id": "2330", "stock_name": "台積電",
            "current_price": 100.0, "target_price": None,
        }])
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch.object(RiskMonitor, "_bulk_last_prices",
                          return_value={"2330": {"price": 105.0, "ma60": 108.0}}):
            alerts = RiskMonitor().check_position_risk()
        assert alerts == []


class TestFundamentalRisk:
    def test_consecutive_revenue_decline_alerts(self):
        df = pd.DataFrame([{
            "recommend_date": date.today().isoformat(),
            "stock_id": "2330", "stock_name": "台積電",
        }])
        rev = pd.DataFrame({"revenue_yoy": [5.0, -3.0, -8.0]})
        fetcher = MagicMock()
        fetcher.get_monthly_revenue.return_value = rev
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch("data.fetcher.DataFetcher", return_value=fetcher), \
             patch.object(RiskMonitor, "_fetch_conference_schedule", return_value={}):
            alerts = RiskMonitor().check_fundamental_risk()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "revenue_decline"
        assert "連 2 個月衰退" in alerts[0]["msg"]

    def test_conference_within_3_days_alerts(self):
        df = pd.DataFrame([{
            "recommend_date": date.today().isoformat(),
            "stock_id": "2330", "stock_name": "台積電",
        }])
        fetcher = MagicMock()
        fetcher.get_monthly_revenue.return_value = pd.DataFrame({"revenue_yoy": [5.0, 8.0]})
        conf = {"2330": date.today() + timedelta(days=2)}
        with patch("screener.recommendation_db.RecommendationDB.get_recent_recommendations",
                   return_value=df), \
             patch("data.fetcher.DataFetcher", return_value=fetcher), \
             patch.object(RiskMonitor, "_fetch_conference_schedule", return_value=conf):
            alerts = RiskMonitor().check_fundamental_risk()
        assert any(a["type"] == "conference" for a in alerts)
