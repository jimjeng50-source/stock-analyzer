"""
tests/test_hot_stocks.py
Tests for screener/hot_stocks.py — HotStockDetector
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from screener.hot_stocks import HotStockDetector


def _universe():
    return pd.DataFrame([
        {"stock_id": "2330", "stock_name": "台積電", "market_cap_b": 500.0},
        {"stock_id": "2317", "stock_name": "鴻海", "market_cap_b": 120.0},
        {"stock_id": "2454", "stock_name": "聯發科", "market_cap_b": 80.0},
    ])


class TestChipsHot:
    def test_parses_t86_top_net_buy(self):
        """T86 回應解析：依三大法人買超排序取前 N，賣超不入榜。"""
        payload = {
            "stat": "OK",
            "fields": ["證券代號", "證券名稱", "外資買賣超股數", "三大法人買賣超股數"],
            "data": [
                ["2330", "台積電", "1,000,000", "5,000,000"],
                ["2317", "鴻海", "500,000", "3,000,000"],
                ["9999", "賣超股", "-100,000", "-2,000,000"],
            ],
        }
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = payload

        with patch("screener.hot_stocks.requests.get", return_value=resp):
            result = HotStockDetector().detect_chips_hot()

        assert "2330" in result
        assert "2317" in result
        assert "9999" not in result           # 賣超不算熱門
        assert "籌碼" in result["2330"][0]

    def test_returns_empty_on_network_failure(self):
        with patch("screener.hot_stocks.requests.get", side_effect=Exception("down")):
            result = HotStockDetector().detect_chips_hot(max_lookback_days=2)
        assert result == {}


class TestSocialHot:
    def test_counts_code_and_name_mentions(self):
        """標題中的代號與股票名稱都要被統計。"""
        titles = [
            "[標的] 2330 台積電 多",
            "[請益] 台積電還能買嗎",
            "[新聞] 2330 法說會重點",
            "[標的] 2454 聯發科 空",
        ]
        det = HotStockDetector()
        with patch.object(det, "_fetch_ptt_titles", return_value=titles):
            result = det.detect_social_hot(_universe())

        assert "2330" in result                # 3 篇（2 代號 + 1 名稱）
        assert "社群" in result["2330"][0]
        assert "2454" not in result            # 只有 1 篇，低於門檻


    def test_returns_empty_when_ptt_unreachable(self):
        det = HotStockDetector()
        with patch.object(det, "_fetch_ptt_titles", return_value=[]):
            result = det.detect_social_hot(_universe())
        assert result == {}

    def test_parse_ptt_titles(self):
        html = (
            '<div class="title">\n<a href="/bbs/Stock/M.123.html">[標的] 2330 台積電</a>'
            '</div><div class="title">\n<a href="/bbs/Stock/M.124.html">[新聞] 外資大買</a></div>'
        )
        titles = HotStockDetector._parse_ptt_titles(html)
        assert titles == ["[標的] 2330 台積電", "[新聞] 外資大買"]


class TestVolumeHot:
    def test_top_turnover_tagged(self):
        result = HotStockDetector().detect_volume_hot(_universe())
        assert "2330" in result
        assert "量能" in result["2330"][0]

    def test_empty_universe(self):
        assert HotStockDetector().detect_volume_hot(pd.DataFrame()) == {}
        assert HotStockDetector().detect_volume_hot(None) == {}


class TestDetectAll:
    def test_merges_tags_from_multiple_sources(self):
        det = HotStockDetector()
        with patch.object(det, "detect_chips_hot", return_value={"2330": ["籌碼:法人買超前十(5,000張)"]}), \
             patch.object(det, "detect_social_hot", return_value={"2330": ["社群:PTT熱議(3篇)"], "2317": ["社群:PTT熱議(2篇)"]}), \
             patch.object(det, "detect_volume_hot", return_value={}):
            result = det.detect_all(_universe())

        assert len(result["2330"]) == 2        # 兩個面向都命中
        assert result["2317"] == ["社群:PTT熱議(2篇)"]

    def test_all_sources_fail_returns_empty(self):
        det = HotStockDetector()
        with patch.object(det, "detect_chips_hot", return_value={}), \
             patch.object(det, "detect_social_hot", return_value={}), \
             patch.object(det, "detect_volume_hot", return_value={}):
            assert det.detect_all(_universe()) == {}
