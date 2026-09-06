"""tests/test_intraday_flow.py — 盤中資金流引擎。"""

from datetime import datetime, timedelta

import pytest

from data.realtime import Quote
from factors.intraday_flow import (FlowBook, FlowTracker, classify_flow,
                                   expected_volume_ratio)
from utils.tz import TW_TZ

BASE = datetime(2026, 8, 28, 10, 0, tzinfo=TW_TZ)


def q(price, cum_volume, i=0, bid=100.0, ask=100.5, bid_v=500, ask_v=500,
      high=None, low=None, **over):
    kwargs = dict(
        stock_id="2330", stock_name="台積電", price=price, prev_close=100.0,
        open=100.0, high=high if high is not None else price,
        low=low if low is not None else price,
        limit_up=110.0, limit_down=90.0, cum_volume=cum_volume,
        bids=[(bid, bid_v)], asks=[(ask, ask_v)],
        ts=BASE + timedelta(seconds=20 * i), trade_date="20260828",
    )
    kwargs.update(over)
    return Quote(**kwargs)


class TestClassifyFlow:

    def test_at_or_above_ask_is_all_buy(self):
        prev = q(100.0, 0, bid=99.5, ask=100.5)
        assert classify_flow(100.5, prev, 100.0, 0) == 1.0
        assert classify_flow(101.0, prev, 100.0, 0) == 1.0

    def test_at_or_below_bid_is_all_sell(self):
        prev = q(100.0, 0, bid=99.5, ask=100.5)
        assert classify_flow(99.5, prev, 100.0, 0) == 0.0
        assert classify_flow(99.0, prev, 100.0, 0) == 0.0

    def test_between_bid_ask_splits_proportionally(self):
        prev = q(100.0, 0, bid=99.0, ask=101.0)
        assert classify_flow(100.0, prev, 100.0, 0) == pytest.approx(0.5)
        assert classify_flow(100.5, prev, 100.0, 0) == pytest.approx(0.75)

    def test_tick_rule_when_no_quote(self):
        assert classify_flow(101.0, None, 100.0, 0) == 1.0
        assert classify_flow(99.0, None, 100.0, 0) == 0.0

    def test_tick_rule_flat_carries_previous_direction(self):
        assert classify_flow(100.0, None, 100.0, 1) == 1.0
        assert classify_flow(100.0, None, 100.0, -1) == 0.0
        assert classify_flow(100.0, None, 100.0, 0) == 0.5

    def test_no_information_is_neutral(self):
        assert classify_flow(100.0, None, 0.0, 0) == 0.5


class TestVolumeCurve:

    def test_preopen_is_zero(self):
        assert expected_volume_ratio(datetime(2026, 8, 28, 8, 0, tzinfo=TW_TZ)) == 0.0

    def test_close_is_one(self):
        assert expected_volume_ratio(datetime(2026, 8, 28, 14, 0, tzinfo=TW_TZ)) == 1.0

    def test_monotonic_increasing(self):
        times = [datetime(2026, 8, 28, h, m, tzinfo=TW_TZ)
                 for h, m in [(9, 15), (9, 45), (10, 30), (11, 30), (12, 30), (13, 20)]]
        vals = [expected_volume_ratio(t) for t in times]
        assert vals == sorted(vals)

    def test_u_shape_front_loaded(self):
        """開盤半小時就該吃掉全日約兩成，不是線性的 1/9。"""
        r = expected_volume_ratio(datetime(2026, 8, 28, 9, 30, tzinfo=TW_TZ))
        assert 0.15 < r < 0.25


class TestFlowTracker:

    def test_first_update_has_no_delta(self):
        t = FlowTracker("2330")
        p = t.update(q(100.0, 1000, i=0))
        assert p.delta_volume == 0
        assert p.cum_net_amount == 0.0

    def test_rising_price_accumulates_buy_flow(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0, bid=99.5, ask=100.5))
        p = t.update(q(100.5, 1100, i=1, bid=100.0, ask=101.0))
        assert p.delta_volume == 100
        assert p.buy_volume == pytest.approx(100)     # 成交在前賣價 → 全外盤
        assert p.net_amount > 0

    def test_falling_price_accumulates_sell_flow(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0, bid=99.5, ask=100.5))
        p = t.update(q(99.5, 1100, i=1, bid=99.0, ask=100.0))
        assert p.sell_volume == pytest.approx(100)
        assert p.net_amount < 0

    def test_negative_volume_delta_ignored(self):
        """累積量倒退（資料異常）不該產生負成交量。"""
        t = FlowTracker("2330")
        t.update(q(100.0, 5000, i=0))
        p = t.update(q(100.0, 4000, i=1))
        assert p.delta_volume == 0

    def test_none_and_invalid_quotes_ignored(self):
        t = FlowTracker("2330")
        assert t.update(None) is None
        assert t.update(q(0.0, 100)) is None
        assert len(t.points) == 0

    def test_obi_from_book(self):
        t = FlowTracker("2330")
        p = t.update(q(100.0, 1000, bid_v=800, ask_v=200))
        assert p.obi == pytest.approx(0.6)

    def test_obi_zero_when_book_empty(self):
        t = FlowTracker("2330")
        p = t.update(q(100.0, 1000, bids=[], asks=[]))
        assert p.obi == 0.0

    def test_trade_date_change_resets(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0))
        t.update(q(100.5, 1500, i=1))
        assert len(t.points) == 2
        t.update(q(101.0, 200, i=2, trade_date="20260831"))
        assert len(t.points) == 1          # 重置後只剩新的一筆
        assert t.trade_date == "20260831"

    def test_summary_not_ready_without_data(self):
        assert FlowTracker("2330").summary()["ready"] is False

    def test_summary_flow_imbalance_all_buy(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0, bid=99.5, ask=100.5))
        for i in range(1, 6):
            t.update(q(100.0 + i * 0.5, 1000 + i * 100, i=i,
                       bid=99.5 + i * 0.5, ask=100.5 + i * 0.5))
        s = t.summary()
        assert s["ready"]
        assert s["flow_imbalance"] == pytest.approx(1.0)
        assert s["net_amount"] > 0

    def test_summary_price_position(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0, high=110.0, low=90.0))
        assert t.summary()["price_position"] == pytest.approx(0.5)

    def test_summary_vol_surge_uses_curve(self):
        t = FlowTracker("2330")
        # 10:00 依曲線約走完 32%，均量 10000 張 → 期望 3200 張
        t.update(q(100.0, 3200, i=0))
        s = t.summary(avg_volume_5d=10000)
        assert s["vol_surge"] == pytest.approx(1.0, abs=0.06)

    def test_summary_vol_surge_none_without_avg(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 3200, i=0))
        assert t.summary()["vol_surge"] is None

    def test_coverage_ratio(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 9000, i=0))     # 中途才開始追蹤
        t.update(q(100.5, 10000, i=1))
        s = t.summary()
        assert s["coverage_ratio"] == pytest.approx(0.1)

    def test_flow_acceleration_detects_speedup(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0, bid=99.5, ask=100.5))
        for i in range(1, 21):
            inc = 300 if i > 15 else 50          # 後段明顯放量
            prev_cum = t.points[-1].cum_volume
            t.update(q(100.0 + i * 0.1, prev_cum + inc, i=i,
                       bid=99.5 + i * 0.1, ask=100.5 + i * 0.1))
        assert t.flow_acceleration() > 1.0

    def test_flow_acceleration_needs_samples(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0))
        assert t.flow_acceleration() == 0.0

    def test_to_dataframe_shape(self):
        t = FlowTracker("2330")
        for i in range(4):
            t.update(q(100.0 + i, 1000 + i * 100, i=i))
        df = t.to_dataframe()
        assert len(df) == 4
        for col in ("ts", "price", "cum_net_amount", "obi", "vwap"):
            assert col in df.columns

    def test_to_dataframe_empty_has_columns(self):
        df = FlowTracker("2330").to_dataframe()
        assert df.empty
        assert "cum_net_amount" in df.columns

    def test_reset_clears_state(self):
        t = FlowTracker("2330")
        t.update(q(100.0, 1000, i=0))
        t.update(q(100.5, 1100, i=1))
        t.reset()
        assert not t.points
        assert t.summary()["ready"] is False

    def test_max_points_bounded(self):
        t = FlowTracker("2330", max_points=5)
        for i in range(20):
            t.update(q(100.0, 1000 + i, i=i))
        assert len(t.points) == 5


class TestFlowBook:

    def test_update_dispatches_per_stock(self):
        book = FlowBook(["2330", "2454"])
        book.update({"2330": q(100.0, 1000, i=0),
                     "2454": Quote(stock_id="2454", price=900.0, cum_volume=500,
                                   prev_close=890.0, ts=BASE, trade_date="20260828")})
        assert set(book.stock_ids) == {"2330", "2454"}
        assert len(book.trackers["2330"].points) == 1

    def test_add_is_idempotent(self):
        book = FlowBook()
        a = book.add("2330")
        assert book.add("2330") is a

    def test_remove(self):
        book = FlowBook(["2330"])
        book.remove("2330")
        assert book.stock_ids == []

    def test_summaries_uses_per_stock_avg_volume(self):
        book = FlowBook(["2330"])
        book.update({"2330": q(100.0, 3200, i=0)})
        out = book.summaries({"2330": 10000})
        assert out["2330"]["vol_surge"] is not None
