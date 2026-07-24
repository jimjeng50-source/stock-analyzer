"""
tests/test_jobs_weekly_tune.py
job_weekly_tune 週度調參任務整合測試（mock DB 與 Telegram）
"""

import pandas as pd
from unittest.mock import patch

import jobs
import config


def _evaluated_df(n=20):
    fund = [50 + 2 * i for i in range(n)]
    ret = [(f - 70) * 0.5 for f in fund]
    return pd.DataFrame({
        "fundamental_score": fund,
        "chips_score": [60] * n, "risk_score": [60] * n,
        "technical_score": [60] * n, "momentum_score": [60] * n,
        "return_60d_pct": ret,
    })


def test_weekly_tune_writes_weights_and_notifies(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", str(tmp_path / "weights.json"))
    with patch("screener.recommendation_db.RecommendationDB") as MockDB, \
         patch("alerts.notifier.Notifier") as MockNotif:
        MockDB.return_value.get_recent_recommendations.return_value = _evaluated_df(20)
        rc = jobs.job_weekly_tune()
    assert rc == 0
    MockNotif.return_value.send_telegram.assert_called_once()
    assert (tmp_path / "weights.json").exists()
    # 生效權重應反映調整（基本面加權）
    assert config.get_active_factor_weights()["fundamental"] > config.FACTOR_WEIGHTS["fundamental"]


def test_weekly_tune_insufficient_samples_no_write(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WEIGHTS_STORE_PATH", str(tmp_path / "weights.json"))
    with patch("screener.recommendation_db.RecommendationDB") as MockDB, \
         patch("alerts.notifier.Notifier") as MockNotif:
        MockDB.return_value.get_recent_recommendations.return_value = _evaluated_df(3)
        rc = jobs.job_weekly_tune()
    assert rc == 0
    MockNotif.return_value.send_telegram.assert_called_once()   # 仍推播狀態
    assert not (tmp_path / "weights.json").exists()             # 但不寫入權重
