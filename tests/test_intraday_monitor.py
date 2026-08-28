"""tests/test_intraday_monitor.py — 盤中監控、推播去重與冷卻。"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alerts import intraday_monitor as im
from alerts.intraday_monitor import IntradayMonitor, build_daily_context
from data.realtime import Quote
from utils.tz import TW_TZ

BASE = datetime(2026, 8, 28, 10, 0, tzinfo=TW_TZ)


def quote(price, cum_volume, i=0, bid=None, ask=None, bid_v=900, ask_v=300):
    return Quote(
        stock_id="2330", stock_name="台積電", price=price, prev_close=100.0,
        open=100.0, high=price, low=99.0, limit_up=110.0, limit_down=90.0,
        cum_volume=cum_volume,
        bids=[(bid if bid is not None else price - 0.5, bid_v)],
        asks=[(ask if ask is not None else price + 0.5, ask_v)],
        ts=BASE + timedelta(seconds=30 * i), trade_date="20260828",
    )


def bullish_quotes(n=12, start_cum=5000):
    """一段量價齊揚、資金持續流入的序列。"""
    out, cum, price = [], start_cum, 100.0
    for i in range(n):
        price += 0.4
        cum += 400 if i > n // 2 else 150
        out.append(quote(round(price, 2), cum, i=i,
                         bid=round(price - 0.5, 2), ask=round(price - 0.5, 2)))
    return out


@pytest.fixture
def notifier():
    n = MagicMock()
    n.send_telegram.return_value = True
    return n


def _monitor(notifier, **kw):
    kw.setdefault("skip_market_hours_check", True)
    m = IntradayMonitor(["2330"], notifier=notifier, **kw)
    # 避免測試打到 FinMind/證交所
    m._daily_ctx = {"2330": {"avg_volume_5d": 20000, "foreign_net_5d": 5000,
                             "trust_net_5d": 800, "ma20": 95.0}}
    m._ctx_date = im.now_tw().date()
    return m


def _feed(monitor, quotes, send=True):
    """把一串快照依序餵進 poll_once。"""
    fired = []
    for q in quotes:
        with patch.object(im, "fetch_quotes", return_value={"2330": q}):
            fired.extend(monitor.poll_once(send=send))
    return fired


class TestBuildDailyContext:

    def test_units_converted_to_lots(self):
        price = pd.DataFrame({
            "date": pd.date_range("2026-06-01", periods=30),
            "close": [100.0] * 30,
            "volume": [20_000_000.0] * 30,       # 股數
        })
        inst = pd.DataFrame({
            "date": pd.to_datetime(["2026-08-26"] * 2 + ["2026-08-27"] * 2),
            "name": ["外資", "投信", "外資", "投信"],
            "net": [1_000_000, 200_000, 1_000_000, 200_000],   # 股數
        })
        fetcher = MagicMock()
        fetcher.get_price.return_value = price
        fetcher.get_institutional.return_value = inst
        with patch("data.fetcher.FinMindFetcher", return_value=fetcher):
            ctx = build_daily_context("2330")
        assert ctx["avg_volume_5d"] == pytest.approx(20_000)      # 張
        assert ctx["ma20"] == pytest.approx(100.0)
        assert ctx["foreign_net_5d"] == pytest.approx(2_000)      # 張
        assert ctx["trust_net_5d"] == pytest.approx(400)

    def test_falls_back_to_t86_when_finmind_empty(self):
        fetcher = MagicMock()
        fetcher.get_price.return_value = pd.DataFrame()
        fetcher.get_institutional.return_value = pd.DataFrame()
        t86 = pd.DataFrame({
            "date": pd.to_datetime(["2026-08-27"]),
            "name": ["外資"], "net": [3_000_000],
        })
        with patch("data.fetcher.FinMindFetcher", return_value=fetcher), \
             patch("data.twse_chips.get_t86_institutional", return_value=t86):
            ctx = build_daily_context("2330")
        assert ctx["foreign_net_5d"] == pytest.approx(3_000)

    def test_all_sources_failing_returns_none_fields(self):
        with patch("data.fetcher.FinMindFetcher", side_effect=RuntimeError("down")), \
             patch("data.twse_chips.get_t86_institutional", side_effect=RuntimeError):
            ctx = build_daily_context("2330")
        assert ctx == {"avg_volume_5d": None, "foreign_net_5d": None,
                       "trust_net_5d": None, "ma20": None}


class TestAlerting:

    def test_strong_setup_sends_alert(self, notifier):
        """冷卻期內只補推「明顯轉強」的訊號，所以把 rescore_delta 拉高後應只有一則。"""
        m = _monitor(notifier, min_score=65, rescore_delta=100)
        fired = _feed(m, bullish_quotes())
        entries = [f for f in fired if f["type"] == "entry"]
        assert len(entries) == 1
        notifier.send_telegram.assert_called_once()
        assert "2330" in notifier.send_telegram.call_args[0][0]

    def test_strengthening_signal_is_re_alerted(self, notifier):
        """分數持續走高（≥ rescore_delta）時值得再推一次。"""
        m = _monitor(notifier, min_score=65, rescore_delta=8)
        entries = [f for f in _feed(m, bullish_quotes()) if f["type"] == "entry"]
        assert len(entries) >= 2
        assert entries[-1]["signal"]["score"] >= entries[0]["signal"]["score"] + 8

    def test_dry_run_does_not_send(self, notifier):
        m = _monitor(notifier, min_score=65)
        fired = _feed(m, bullish_quotes(), send=False)
        assert [f for f in fired if f["type"] == "entry"]
        notifier.send_telegram.assert_not_called()

    def test_weak_setup_never_alerts(self, notifier):
        m = _monitor(notifier, min_score=70)
        falling, cum, price = [], 5000, 100.0
        for i in range(12):
            price -= 0.4
            cum += 300
            falling.append(quote(round(price, 2), cum, i=i,
                                 bid=round(price, 2), ask=round(price + 1, 2),
                                 bid_v=200, ask_v=900))
        assert not [f for f in _feed(m, falling) if f["type"] == "entry"]
        notifier.send_telegram.assert_not_called()

    def test_no_quotes_is_a_noop(self, notifier):
        m = _monitor(notifier)
        with patch.object(im, "fetch_quotes", return_value={}):
            assert m.poll_once() == []

    def test_empty_watchlist_is_a_noop(self, notifier):
        m = IntradayMonitor([], notifier=notifier, skip_market_hours_check=True)
        assert m.poll_once() == []

    def test_alert_log_accumulates(self, notifier):
        m = _monitor(notifier, min_score=65)
        _feed(m, bullish_quotes())
        assert len(m.alert_log) >= 1


class TestCooldown:

    def _hot_signal(self, score):
        return {"grade": "strong", "score": score}

    def test_first_alert_allowed(self, notifier):
        m = _monitor(notifier, min_score=70)
        assert m._should_alert("2330", self._hot_signal(80))

    def test_second_alert_suppressed_within_cooldown(self, notifier):
        m = _monitor(notifier, min_score=70, cooldown_minutes=30)
        m._record_alert("2330", self._hot_signal(80))
        assert not m._should_alert("2330", self._hot_signal(82))

    def test_big_improvement_breaks_cooldown(self, notifier):
        m = _monitor(notifier, min_score=70, cooldown_minutes=30, rescore_delta=8)
        m._record_alert("2330", self._hot_signal(72))
        assert m._should_alert("2330", self._hot_signal(85))

    def test_alert_allowed_after_cooldown_expires(self, notifier):
        m = _monitor(notifier, min_score=70, cooldown_minutes=30)
        m._record_alert("2330", self._hot_signal(80))
        m._last_alert["2330"]["ts"] = im.now_tw() - timedelta(minutes=31)
        assert m._should_alert("2330", self._hot_signal(80))

    def test_below_threshold_clears_cooldown(self, notifier):
        m = _monitor(notifier, min_score=70)
        m._record_alert("2330", self._hot_signal(80))
        assert not m._should_alert("2330", {"grade": "watch", "score": 55})
        assert "2330" not in m._last_alert
        # 訊號轉強可立刻重推
        assert m._should_alert("2330", self._hot_signal(80))

    def test_grade_watch_never_alerts_even_if_score_high(self, notifier):
        m = _monitor(notifier, min_score=50)
        assert not m._should_alert("2330", {"grade": "watch", "score": 99})


class TestExitAlerts:

    def test_exit_alert_on_stop_loss(self, notifier):
        m = _monitor(notifier, min_score=95,
                     positions={"2330": {"entry_price": 100.0, "stop_loss": 99.5,
                                         "target": 105.0}})
        falling = [quote(100.0, 5000, i=0), quote(98.0, 5400, i=1, bid=98.0, ask=99.0)]
        fired = _feed(m, falling)
        exits = [f for f in fired if f["type"] == "exit"]
        assert exits
        assert "出場警示" in exits[0]["message"]

    def test_no_exit_without_position(self, notifier):
        m = _monitor(notifier, min_score=95)
        fired = _feed(m, [quote(100.0, 5000, i=0), quote(90.0, 5400, i=1)])
        assert not [f for f in fired if f["type"] == "exit"]

    def test_exit_alert_respects_cooldown(self, notifier):
        m = _monitor(notifier, min_score=95, cooldown_minutes=30,
                     positions={"2330": {"entry_price": 100.0, "stop_loss": 99.5}})
        falling = [quote(100.0, 5000, i=0)] + \
                  [quote(98.0, 5000 + 400 * i, i=i, bid=98.0, ask=99.0)
                   for i in range(1, 5)]
        exits = [f for f in _feed(m, falling) if f["type"] == "exit"]
        assert len(exits) == 1

    def test_format_exit_alert_contents(self):
        msg = IntradayMonitor.format_exit_alert(
            {"stock_id": "2330", "price": 95.0, "pnl_pct": -5.0,
             "reasons": ["觸及停損"]}, "台積電")
        assert "2330" in msg and "台積電" in msg
        assert "觸及停損" in msg
        assert "-5.00%" in msg


class TestRunLoop:

    def test_run_skips_outside_market_hours(self, notifier):
        m = IntradayMonitor(["2330"], notifier=notifier)
        with patch.object(im, "is_trading_hours", return_value=False):
            assert m.run() == []
        notifier.send_telegram.assert_not_called()

    def test_run_stops_at_max_polls(self, notifier):
        m = _monitor(notifier)
        with patch.object(im, "fetch_quotes", return_value={}) as f, \
             patch.object(im.time, "sleep"):
            m.run(max_polls=3, send=False)
        assert f.call_count == 3

    def test_run_without_end_condition_is_rejected(self, notifier):
        """略過時段檢查又不給上限 → 會是無窮迴圈，應直接報錯。"""
        m = _monitor(notifier)
        with pytest.raises(ValueError):
            m.run(send=False)

    def test_run_stops_when_market_closes(self, notifier):
        m = IntradayMonitor(["2330"], notifier=notifier)
        with patch.object(im, "is_trading_hours", side_effect=[True, False]), \
             patch.object(im, "fetch_quotes", return_value={}), \
             patch.object(im.time, "sleep"):
            m.run(send=False)

    def test_poll_errors_do_not_break_loop(self, notifier):
        m = _monitor(notifier)
        with patch.object(m, "poll_once", side_effect=RuntimeError("boom")), \
             patch.object(im.time, "sleep"):
            m.run(max_polls=2, send=False)      # 例外被吞掉後正常結束

    def test_snapshot_returns_summary_and_signal(self, notifier):
        m = _monitor(notifier)
        _feed(m, bullish_quotes(), send=False)
        snap = m.snapshot()
        assert "2330" in snap
        assert snap["2330"]["summary"]["ready"]
        assert "score" in snap["2330"]["signal"]


class TestDailyContextCache:

    def test_context_fetched_once_per_day(self, notifier):
        m = IntradayMonitor(["2330"], notifier=notifier, skip_market_hours_check=True)
        with patch.object(im, "build_daily_context",
                          return_value={"avg_volume_5d": 1}) as b:
            m.daily_context("2330")
            m.daily_context("2330")
        b.assert_called_once()

    def test_context_refreshed_on_new_day(self, notifier):
        m = IntradayMonitor(["2330"], notifier=notifier, skip_market_hours_check=True)
        with patch.object(im, "build_daily_context",
                          return_value={"avg_volume_5d": 1}) as b:
            m.daily_context("2330")
            m._ctx_date = m._ctx_date - timedelta(days=1)
            m.daily_context("2330")
        assert b.call_count == 2
