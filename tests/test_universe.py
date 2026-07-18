"""
tests/test_universe.py
Tests for screener/universe.py — UniverseManager
"""

import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from screener.universe import UniverseManager


class TestUniverseManager:
    """Tests for UniverseManager."""

    # ── get_custom_watchlist ───────────────────────────────────────────────────

    def test_get_custom_watchlist_empty(self):
        """When WATCHLIST_CUSTOM env var is not set, returns []."""
        mgr = UniverseManager()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WATCHLIST_CUSTOM", None)
            result = mgr.get_custom_watchlist()
        assert result == []

    def test_get_custom_watchlist_parses(self):
        """When WATCHLIST_CUSTOM='2330,2317', returns ['2330', '2317']."""
        mgr = UniverseManager()
        with patch.dict(os.environ, {"WATCHLIST_CUSTOM": "2330,2317"}):
            result = mgr.get_custom_watchlist()
        assert result == ["2330", "2317"]

    # ── _load_cache ────────────────────────────────────────────────────────────

    def test_load_cache_returns_none_when_missing(self, tmp_path):
        """_load_cache() returns None when the cache file doesn't exist."""
        mgr = UniverseManager()
        mgr.CACHE_PATH = str(tmp_path / "nonexistent_cache.json")
        result = mgr._load_cache()
        assert result is None

    # ── _save_cache / _load_cache round-trip ──────────────────────────────────

    def test_save_and_load_cache(self, tmp_path):
        """save_cache then load_cache returns equivalent DataFrame."""
        mgr = UniverseManager()
        cache_path = str(tmp_path / "test_universe_cache.json")
        mgr.CACHE_PATH = cache_path

        df = pd.DataFrame([
            {
                "stock_id": "2330",
                "stock_name": "台積電",
                "market": "TWSE",
                "industry": "半導體",
                "market_cap_b": 20000.0,
                "avg_volume_k": 50000.0,
                "last_price": 850.0,
            }
        ])

        with patch("os.makedirs"):
            mgr._save_cache(df)

        loaded = mgr._load_cache()
        assert loaded is not None
        assert list(loaded["stock_id"]) == ["2330"]
        assert list(loaded["stock_name"]) == ["台積電"]

    # ── merge_with_custom ─────────────────────────────────────────────────────

    def test_merge_with_custom_adds_missing(self):
        """merge_with_custom adds stock not present in universe_df."""
        mgr = UniverseManager()

        universe_df = pd.DataFrame([
            {
                "stock_id": "2330",
                "stock_name": "台積電",
                "market": "TWSE",
                "industry": "半導體",
                "market_cap_b": 20000.0,
                "avg_volume_k": 50000.0,
                "last_price": 850.0,
            }
        ])

        with patch.dict(os.environ, {"WATCHLIST_CUSTOM": "9999"}):
            with patch.object(mgr, "_fetch_latest_price", return_value=100.0) as mock_price:
                merged = mgr.merge_with_custom(universe_df)

        stock_ids = list(merged["stock_id"])
        assert "9999" in stock_ids
        assert "2330" in stock_ids
        mock_price.assert_called_once_with("9999")

    # ── _fetch_market_snapshot fallback ────────────────────────────────────────

    def test_snapshot_falls_back_to_twse_when_finmind_unavailable(self):
        """FinMind 快照失敗時，改用 TWSE/TPEX 官方 OpenAPI。"""
        mgr = UniverseManager()
        twse_df = pd.DataFrame([
            {"stock_id": "2330", "close": 850.0,
             "Trading_Volume": 30_000_000, "Trading_money": 25_500_000_000},
        ])
        with patch.object(mgr, "_fetch_finmind_snapshot", return_value=None), \
             patch.object(mgr, "_fetch_twse_tpex_snapshot", return_value=twse_df) as mock_twse:
            result = mgr._fetch_market_snapshot()
        mock_twse.assert_called_once()
        assert result is not None
        assert list(result["stock_id"]) == ["2330"]

    def test_twse_tpex_snapshot_parses_and_cleans(self):
        """TWSE/TPEX 回應：千分位逗號和 '-' 值被正確清理，close<=0 被排除。"""
        mgr = UniverseManager()

        twse_payload = [
            {"Code": "2330", "Name": "台積電", "ClosingPrice": "850.00",
             "TradeVolume": "30,000,000", "TradeValue": "25,500,000,000"},
            {"Code": "9998", "Name": "無成交", "ClosingPrice": "-",
             "TradeVolume": "-", "TradeValue": "-"},
        ]
        tpex_payload = [
            {"SecuritiesCompanyCode": "5483", "CompanyName": "中美晶", "Close": "180.50",
             "TradingShares": "5,000,000", "TransactionAmount": "902,500,000"},
        ]

        def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = twse_payload if "twse" in url else tpex_payload
            return resp

        with patch("screener.universe.requests.get", side_effect=fake_get):
            snap = mgr._fetch_twse_tpex_snapshot()

        assert snap is not None
        ids = set(snap["stock_id"])
        assert ids == {"2330", "5483"}          # "-" 無成交列被排除
        row = snap[snap["stock_id"] == "2330"].iloc[0]
        assert row["close"] == 850.0
        assert row["Trading_Volume"] == 30_000_000

    def test_twse_tpex_snapshot_returns_none_when_both_fail(self):
        """兩個官方 API 都失敗時回傳 None。"""
        mgr = UniverseManager()
        with patch("screener.universe.requests.get", side_effect=Exception("network down")):
            snap = mgr._fetch_twse_tpex_snapshot()
        assert snap is None

    # ── FinMind 清單失敗時，用 TWSE/TPEX 快照建候選池 ──────────────────────────

    def test_universe_from_snapshot_fallback(self):
        """_fetch_stock_info 回空（FinMind 402）→ get_universe 改用快照建池。"""
        mgr = UniverseManager()
        snap = pd.DataFrame({
            "stock_id": ["2330", "2317", "9999"],
            "stock_name": ["台積電", "鴻海", ""],
            "close": [900.0, 200.0, 50.0],
            "Trading_Volume": [30_000_000, 20_000_000, 10_000_000],
            "Trading_money": [2.7e10, 4e9, 5e8],
        })
        with patch.object(mgr, "_fetch_stock_info", return_value=pd.DataFrame()), \
             patch.object(mgr, "_fetch_twse_tpex_snapshot", return_value=snap), \
             patch.object(mgr, "_load_cache", return_value=None), \
             patch.object(mgr, "_save_cache"):
            uni = mgr.get_universe()
        assert not uni.empty
        assert set(["stock_id", "stock_name", "last_price", "market_cap_b"]).issubset(uni.columns)
        row = uni[uni["stock_id"] == "2330"].iloc[0]
        assert row["stock_name"] == "台積電"
        assert row["last_price"] == 900.0
        # 無股名者以代號代替
        blank = uni[uni["stock_id"] == "9999"]
        if not blank.empty:
            assert blank.iloc[0]["stock_name"] == "9999"
