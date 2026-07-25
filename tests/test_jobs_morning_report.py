"""
tests/test_jobs_morning_report.py
job_morning_report 新增區塊（兩個月追蹤、推薦邏輯狀態）整合測試
"""

import pandas as pd
from unittest.mock import patch, MagicMock

import jobs


def _recent_df():
    return pd.DataFrame([{
        "recommend_date": "2026-07-24", "rank": 1, "stock_id": "2330",
        "stock_name": "台積電", "total_score": 80, "current_price": 1000,
        "forward_eps": 45.0, "eps_growth_pct": 12, "target_price": 1200, "upside_pct": 20,
    }])


def _tracking_df(n=25):
    rows = []
    for i in range(n):
        rows.append({
            "recommend_date": f"2026-07-{(i % 27) + 1:02d}",
            "stock_id": f"{2000 + i}", "stock_name": f"股{i}",
            "entry_price": 100.0, "last_close": 105.0, "last_return_pct": 5.0,
            "max_return_pct": 8.0, "min_return_pct": -2.0, "days_tracked": 10,
        })
    return pd.DataFrame(rows)


def _run_morning_report():
    mock_db = MagicMock()
    mock_db.get_performance_summary.return_value = {
        "avg_return_20d": 2.0, "win_rate_20d": 0.55,
        "avg_return_60d": 5.0, "win_rate_60d": 0.62, "evaluated_count": 30,
    }
    mock_db.get_recent_recommendations.return_value = _recent_df()
    mock_db.get_tracking_summary.return_value = _tracking_df(25)

    captured = {}
    mock_notif = MagicMock()
    mock_notif.send_telegram.side_effect = lambda msg: captured.setdefault("msg", msg) or True

    with patch("screener.recommendation_db.RecommendationDB", return_value=mock_db), \
         patch("alerts.notifier.Notifier", return_value=mock_notif), \
         patch("screener.historical_eval.evaluate_60d_accuracy", return_value={}), \
         patch("screener.universe.UniverseManager") as MU, \
         patch("screener.hot_stocks.HotStockDetector") as MH:
        MU.return_value.get_universe.return_value = pd.DataFrame()
        MH.return_value.detect_all.return_value = {}
        rc = jobs.job_morning_report()
    return rc, captured.get("msg", "")


def test_morning_report_has_tracking_and_status():
    rc, msg = _run_morning_report()
    assert rc == 0
    assert "兩個月追蹤（25 檔）" in msg
    assert "平均最新報酬 +5.0%｜勝率 100%" in msg
    assert "…完整 25 檔見 Streamlit 儀表板" in msg     # 超過 20 檔的截斷提示
    assert "推薦邏輯狀態" in msg
    assert "目標 70%" in msg
    assert "近期 60 日勝率 62%" in msg


def test_morning_report_dedupes_by_stock():
    """同股多次推薦只顯示一行（去重）。"""
    dup = pd.concat([_tracking_df(3), _tracking_df(3)], ignore_index=True)  # 每檔重複
    mock_db = MagicMock()
    mock_db.get_performance_summary.return_value = {"win_rate_60d": 0.5}
    mock_db.get_recent_recommendations.return_value = pd.DataFrame()
    mock_db.get_tracking_summary.return_value = dup
    mock_notif = MagicMock()
    cap = {}
    mock_notif.send_telegram.side_effect = lambda m: cap.setdefault("m", m) or True
    with patch("screener.recommendation_db.RecommendationDB", return_value=mock_db), \
         patch("alerts.notifier.Notifier", return_value=mock_notif), \
         patch("screener.historical_eval.evaluate_60d_accuracy", return_value={}), \
         patch("screener.universe.UniverseManager") as MU, \
         patch("screener.hot_stocks.HotStockDetector") as MH:
        MU.return_value.get_universe.return_value = pd.DataFrame()
        MH.return_value.detect_all.return_value = {}
        jobs.job_morning_report()
    assert "兩個月追蹤（3 檔）" in cap["m"]      # 6 列去重成 3 檔
