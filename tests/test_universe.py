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
