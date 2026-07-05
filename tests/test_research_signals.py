"""
tests/test_research_signals.py
外資/券商報告新聞信號（公開新聞來源）
"""

from unittest.mock import patch

import pandas as pd

from screener.research_signals import fetch_research_signals, _is_research_title


def _universe():
    return pd.DataFrame([
        {"stock_id": "2330", "stock_name": "台積電"},
        {"stock_id": "2454", "stock_name": "聯發科"},
    ])


class TestTitleFilter:
    def test_requires_subject_and_action(self):
        assert _is_research_title("外資調升台積電目標價至1200元")
        assert _is_research_title("高盛重申聯發科買進評等")
        assert not _is_research_title("台積電今日大漲3%")          # 無研報主體
        assert not _is_research_title("外資今日買超500億")          # 無評等/目標價動作


class TestFetchSignals:
    def test_matches_by_name_and_code(self):
        titles = [
            "外資調升台積電目標價至1200元",
            "大摩喊進聯發科(2454-TW)，上看1500",
            "台股收盤上漲200點",
        ]
        with patch("screener.research_signals._fetch_cnyes_titles", return_value=titles):
            signals = fetch_research_signals(_universe())
        assert "2330" in signals
        assert "2454" in signals
        assert all(tag.startswith("研報:") for tags in signals.values() for tag in tags)

    def test_empty_on_fetch_failure(self):
        with patch("screener.research_signals._fetch_cnyes_titles", return_value=[]):
            assert fetch_research_signals(_universe()) == {}
