"""tests/test_realtime.py — 證交所 MIS 即時報價抓取與解析。"""

from datetime import datetime
from unittest.mock import patch

import pytest

from data import realtime
from data.realtime import (Quote, _parse_entry, _split_levels, fetch_quote,
                           fetch_quotes, is_trading_hours, session_progress)
from utils.tz import TW_TZ


def _mis_item(code="2330", **over):
    item = {
        "c": code, "n": "台積電", "ex": "tse",
        "z": "1130.0000", "o": "1135.0000", "h": "1140.0000", "l": "1125.0000",
        "y": "1130.0000", "u": "1245.0000", "w": "1020.0000",
        "v": "21545", "tv": "73", "d": "20260828", "t": "13:30:00",
        "a": "1135.0000_1140.0000_1145.0000_",
        "f": "245_346_120_",
        "b": "1130.0000_1125.0000_1120.0000_",
        "g": "561_502_310_",
        "tlong": "1787000000000",
    }
    item.update(over)
    return item


class TestParsing:

    def test_split_levels(self):
        levels = _split_levels("100.5_101.0_", "12_30_")
        assert levels == [(100.5, 12), (101.0, 30)]

    def test_split_levels_handles_missing(self):
        assert _split_levels("-", "-") == []
        assert _split_levels("", "") == []

    def test_parse_entry_full(self):
        q = _parse_entry(_mis_item())
        assert q.stock_id == "2330"
        assert q.stock_name == "台積電"
        assert q.price == 1130.0
        assert q.cum_volume == 21545
        assert q.bid1 == 1130.0
        assert q.ask1 == 1135.0
        assert q.bid_volume == 561 + 502 + 310
        assert q.ask_volume == 245 + 346 + 120
        assert q.trade_date == "20260828"
        assert q.valid

    def test_parse_entry_no_trade_falls_back_to_pz(self):
        """尚未成交時 z 為 '-'，應退回前一盤成交價 pz。"""
        q = _parse_entry(_mis_item(z="-", pz="1128.0000"))
        assert q.price == 1128.0

    def test_parse_entry_no_trade_no_pz_uses_mid(self):
        """沒有 z 也沒有 pz → 用五檔買賣中價。"""
        q = _parse_entry(_mis_item(z="-", pz="-"))
        assert q.price == pytest.approx((1130.0 + 1135.0) / 2)

    def test_parse_entry_missing_code_returns_none(self):
        assert _parse_entry({"n": "無代號"}) is None
        assert _parse_entry("not a dict") is None

    def test_parse_entry_commas_stripped(self):
        q = _parse_entry(_mis_item(v="1,234,567"))
        assert q.cum_volume == 1234567


class TestQuoteProperties:

    def test_change_pct(self):
        q = Quote(stock_id="2330", price=110.0, prev_close=100.0)
        assert q.change_pct == pytest.approx(10.0)

    def test_change_pct_no_prev_close(self):
        assert Quote(stock_id="2330", price=110.0).change_pct == 0.0

    def test_turnover(self):
        q = Quote(stock_id="2330", price=100.0, cum_volume=500)
        assert q.turnover == pytest.approx(500 * 100.0 * 1000)

    def test_limit_up_requires_empty_asks(self):
        locked = Quote(stock_id="1", price=110.0, limit_up=110.0,
                       bids=[(110.0, 900)], asks=[])
        assert locked.is_limit_up
        # 漲停價但賣單還在 → 不算鎖死
        open_ask = Quote(stock_id="1", price=110.0, limit_up=110.0,
                         bids=[(110.0, 900)], asks=[(110.0, 50)])
        assert not open_ask.is_limit_up

    def test_limit_down_requires_empty_bids(self):
        q = Quote(stock_id="1", price=90.0, limit_down=90.0, bids=[], asks=[(90.0, 800)])
        assert q.is_limit_down

    def test_invalid_when_no_price(self):
        assert not Quote(stock_id="2330", price=0.0).valid


class TestFetch:

    def setup_method(self):
        realtime.clear_channel_cache()

    def test_fetch_quotes_parses_and_caches_channel(self):
        with patch.object(realtime, "_call_mis",
                          return_value=[_mis_item("2330")]) as m:
            out = fetch_quotes(["2330"])
        assert "2330" in out
        assert out["2330"].price == 1130.0
        # 第一次不知道市場別 → 同時問 tse_ 與 otc_
        assert m.call_args[0][0] == ["tse_2330.tw", "otc_2330.tw"]

        # 第二次應只問已快取的 tse_
        with patch.object(realtime, "_call_mis",
                          return_value=[_mis_item("2330")]) as m2:
            fetch_quotes(["2330"])
        assert m2.call_args[0][0] == ["tse_2330.tw"]

    def test_fetch_quotes_otc(self):
        with patch.object(realtime, "_call_mis",
                          return_value=[_mis_item("6488", ex="otc")]):
            out = fetch_quotes(["6488"])
        assert out["6488"].exchange == "otc"

    def test_fetch_quotes_missing_stock_omitted(self):
        with patch.object(realtime, "_call_mis", return_value=[]):
            assert fetch_quotes(["9999"]) == {}

    def test_fetch_quotes_empty_input(self):
        assert fetch_quotes([]) == {}
        assert fetch_quotes(["", "  "]) == {}

    def test_fetch_quotes_dedupes_input(self):
        with patch.object(realtime, "_call_mis",
                          return_value=[_mis_item("2330")]) as m:
            fetch_quotes(["2330", "2330"])
        assert m.call_args[0][0] == ["tse_2330.tw", "otc_2330.tw"]

    def test_fetch_quote_single(self):
        with patch.object(realtime, "_call_mis", return_value=[_mis_item("2330")]):
            assert fetch_quote("2330").stock_id == "2330"

    def test_call_mis_network_failure_returns_empty(self):
        with patch.object(realtime.requests, "get", side_effect=OSError("boom")), \
             patch.object(realtime, "_throttle"):
            assert realtime._call_mis(["tse_2330.tw"]) == []

    def test_call_mis_bad_json_shape(self):
        class _R:
            def raise_for_status(self): pass
            def json(self): return ["unexpected"]
        with patch.object(realtime.requests, "get", return_value=_R()), \
             patch.object(realtime, "_throttle"):
            assert realtime._call_mis(["tse_2330.tw"]) == []


class TestTradingHours:

    def test_weekday_in_session(self):
        # 2026-08-28 是週五
        assert is_trading_hours(datetime(2026, 8, 28, 10, 30, tzinfo=TW_TZ))

    def test_weekend_is_closed(self):
        assert not is_trading_hours(datetime(2026, 8, 29, 10, 30, tzinfo=TW_TZ))

    def test_before_open_and_after_close(self):
        assert not is_trading_hours(datetime(2026, 8, 28, 8, 45, tzinfo=TW_TZ))
        assert not is_trading_hours(datetime(2026, 8, 28, 14, 0, tzinfo=TW_TZ))

    def test_preopen_flag(self):
        t = datetime(2026, 8, 28, 8, 45, tzinfo=TW_TZ)
        assert not is_trading_hours(t)
        assert is_trading_hours(t, include_preopen=True)

    def test_session_progress_bounds(self):
        assert session_progress(datetime(2026, 8, 28, 8, 0, tzinfo=TW_TZ)) == 0.0
        assert session_progress(datetime(2026, 8, 28, 14, 0, tzinfo=TW_TZ)) == 1.0
        mid = session_progress(datetime(2026, 8, 28, 11, 15, tzinfo=TW_TZ))
        assert 0.49 < mid < 0.51
