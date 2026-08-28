"""tests/test_entry_signal.py — 小波段進場點評分模型。"""

import pytest

from factors.entry_signal import (MIN_INTRADAY_VOLUME, SIGNAL_WEIGHTS,
                                  evaluate_entry, evaluate_exit,
                                  format_entry_alert, score_daily_chips,
                                  score_flow_accel, score_flow_net, score_obi,
                                  score_price_position, score_vol_surge)


def summary(**over):
    """一份「中性偏多、無否決」的 flow summary。"""
    s = {
        "ready": True, "stock_id": "2330", "stock_name": "台積電",
        "price": 100.0, "prev_close": 98.0, "change_pct": 2.0,
        "open": 98.5, "high": 101.0, "low": 97.5,
        "cum_volume": 20000, "turnover": 2e9,
        "net_amount": 5e7, "net_volume": 500.0,
        "buy_volume": 6000.0, "sell_volume": 4000.0,
        "net_ratio": 0.025, "flow_imbalance": 0.20, "flow_accel": 1.5,
        "obi": 0.2, "obi_avg": 0.20, "bid_volume": 900, "ask_volume": 600,
        "vwap": 99.0, "price_vs_vwap": 1.0, "price_position": 0.70,
        "vol_surge": 1.6, "samples": 50, "coverage_ratio": 0.6,
        "is_limit_up": False, "is_limit_down": False, "ts": None,
        "trade_time": "10:30:00",
    }
    s.update(over)
    return s


CTX = {"avg_volume_5d": 20000, "foreign_net_5d": 4000,
       "trust_net_5d": 600, "ma20": 95.0}


class TestWeights:

    def test_weights_sum_to_one(self):
        assert sum(SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


class TestSubSignals:

    def test_flow_net_monotonic(self):
        low, _ = score_flow_net(-0.2)
        mid, _ = score_flow_net(0.0)
        high, _ = score_flow_net(0.25)
        assert low < mid < high
        assert mid == pytest.approx(50.0)

    def test_flow_accel_rewards_speedup(self):
        slowing, _ = score_flow_accel(0.3)
        steady, _ = score_flow_accel(1.0)
        speeding, _ = score_flow_accel(2.0)
        assert slowing < steady < speeding
        outflow, _ = score_flow_accel(-1.0)
        assert outflow < slowing

    def test_obi_symmetric_around_neutral(self):
        assert score_obi(0.0)[0] == pytest.approx(50.0)
        assert score_obi(0.3)[0] > 80
        assert score_obi(-0.3)[0] < 20

    def test_vol_surge_none_is_neutral(self):
        s, note = score_vol_surge(None)
        assert s == 50.0
        assert "無資料" in note

    def test_vol_surge_penalises_extreme_blowoff(self):
        """量能 2 倍好，但 8 倍常是末升段爆量，分數應回落。"""
        assert score_vol_surge(2.0)[0] > score_vol_surge(8.0)[0]

    def test_vol_surge_penalises_dry_volume(self):
        assert score_vol_surge(0.4)[0] < 30

    def test_price_position_peaks_in_middle_upper_band(self):
        best = score_price_position(0.70)[0]
        assert best > score_price_position(0.05)[0]
        assert best > score_price_position(0.99)[0]

    def test_price_position_penalises_chasing(self):
        assert score_price_position(0.97)[0] < 40

    def test_daily_chips_unknown_is_neutral(self):
        assert score_daily_chips(None, None, 10000)[0] == 50.0

    def test_daily_chips_buy_beats_sell(self):
        buy, _ = score_daily_chips(5000, 1000, 20000)
        sell, _ = score_daily_chips(-5000, -1000, 20000)
        assert buy > 50 > sell

    def test_daily_chips_without_avg_volume_still_scores(self):
        s, _ = score_daily_chips(8000, 0, None)
        assert s > 50


class TestEvaluateEntry:

    def test_strong_setup_scores_high(self):
        sig = evaluate_entry(summary(), CTX)
        assert sig["score"] >= 75
        assert sig["grade"] == "strong"
        assert sig["trade_plan"] is not None

    def test_not_ready_summary(self):
        sig = evaluate_entry({"ready": False, "stock_id": "2330"})
        assert sig["grade"] == "unknown"
        assert sig["trade_plan"] is None
        assert sig["score"] == 0.0

    def test_empty_summary(self):
        assert evaluate_entry({})["grade"] == "unknown"

    def test_components_cover_all_weights(self):
        sig = evaluate_entry(summary(), CTX)
        assert set(sig["components"]) == set(SIGNAL_WEIGHTS)

    # ── 否決條件 ──────────────────────────────────────────────────────────────

    def test_veto_accelerating_outflow(self):
        sig = evaluate_entry(summary(flow_imbalance=-0.2, flow_accel=-1.0), CTX)
        assert sig["score"] <= 35
        assert any("流出" in v for v in sig["vetoes"])

    def test_veto_below_vwap(self):
        sig = evaluate_entry(summary(price_vs_vwap=-2.5), CTX)
        assert sig["score"] <= 40
        assert any("VWAP" in v for v in sig["vetoes"])

    def test_veto_limit_up_lock(self):
        sig = evaluate_entry(summary(is_limit_up=True), CTX)
        assert sig["score"] <= 45
        assert any("漲停" in v for v in sig["vetoes"])

    def test_veto_limit_down_is_harshest(self):
        sig = evaluate_entry(summary(is_limit_down=True), CTX)
        assert sig["score"] <= 15

    def test_veto_illiquid(self):
        sig = evaluate_entry(summary(cum_volume=MIN_INTRADAY_VOLUME - 1), CTX)
        assert sig["score"] <= 35
        assert any("流動性" in v for v in sig["vetoes"])

    def test_veto_takes_the_strictest_cap(self):
        sig = evaluate_entry(
            summary(is_limit_down=True, cum_volume=10, price_vs_vwap=-5), CTX)
        assert sig["score"] <= 15

    def test_no_trade_plan_when_vetoed(self):
        assert evaluate_entry(summary(is_limit_down=True), CTX)["trade_plan"] is None

    # ── 警示 ──────────────────────────────────────────────────────────────────

    def test_warning_below_ma20(self):
        sig = evaluate_entry(summary(price=90.0), {**CTX, "ma20": 95.0})
        assert any("20 日均線" in w for w in sig["warnings"])

    def test_warning_chasing_high(self):
        sig = evaluate_entry(summary(price_position=0.98), CTX)
        assert any("當日最高" in w for w in sig["warnings"])

    def test_warning_low_coverage(self):
        sig = evaluate_entry(summary(coverage_ratio=0.05), CTX)
        assert any("涵蓋" in w for w in sig["warnings"])
        assert sig["confidence"] == "低"

    def test_warning_big_daily_gain(self):
        sig = evaluate_entry(summary(change_pct=9.0), CTX)
        assert any("漲幅" in w for w in sig["warnings"])

    def test_confidence_levels(self):
        assert evaluate_entry(summary(samples=3), CTX)["confidence"] == "低"
        assert evaluate_entry(summary(samples=20, coverage_ratio=0.3),
                              CTX)["confidence"] == "中"
        assert evaluate_entry(summary(samples=60, coverage_ratio=0.8),
                              CTX)["confidence"] == "高"

    def test_no_context_still_works(self):
        sig = evaluate_entry(summary())
        assert sig["score"] > 0
        assert sig["components"]["daily_chips"]["score"] == 50.0


class TestTradePlan:

    def test_plan_absent_below_threshold(self):
        sig = evaluate_entry(summary(flow_imbalance=0.0, flow_accel=0.2,
                                     obi_avg=0.0, vol_surge=0.8,
                                     price_position=0.3), CTX)
        assert sig["score"] < 65
        assert sig["trade_plan"] is None

    def test_stop_loss_below_price_and_capped(self):
        plan = evaluate_entry(summary(), CTX)["trade_plan"]
        assert plan["stop_loss"] < 100.0
        assert 1.5 <= plan["risk_pct"] <= 4.01

    def test_stop_loss_capped_when_day_low_is_far(self):
        """當日低點很遠時，停損不該拉到 -10%，要被 -4% 上限擋住。"""
        plan = evaluate_entry(summary(low=80.0, vwap=85.0, price_vs_vwap=1.0),
                              CTX)["trade_plan"]
        assert plan["stop_loss"] >= 100.0 * 0.96 - 1e-6

    def test_targets_ordered(self):
        plan = evaluate_entry(summary(), CTX)["trade_plan"]
        assert plan["entry_low"] <= plan["entry_high"]
        assert plan["stop_loss"] < plan["target_1"] < plan["target_2"]
        assert plan["reward_risk"] > 0

    def test_position_hint_scales_with_score(self):
        strong = evaluate_entry(summary(), CTX)["trade_plan"]
        assert strong["position_hint"] == "1/2 倉"


class TestEvaluateExit:

    def test_healthy_position_holds(self):
        ex = evaluate_exit(summary(), entry_price=98.0)
        assert ex["exit"] is False
        assert ex["urgency"] == "none"
        assert ex["pnl_pct"] == pytest.approx(2.04, abs=0.01)

    def test_stop_loss_hit(self):
        ex = evaluate_exit(summary(price=95.0), entry_price=100.0, stop_loss=96.0)
        assert ex["exit"] is True
        assert any("停損" in r for r in ex["reasons"])

    def test_target_hit(self):
        ex = evaluate_exit(summary(price=107.0), entry_price=100.0, target=105.0)
        assert ex["exit"] is True
        assert any("停利" in r for r in ex["reasons"])

    def test_flow_reversal_triggers_exit(self):
        ex = evaluate_exit(summary(flow_imbalance=-0.15, flow_accel=-0.8))
        assert ex["exit"] is True

    def test_mild_selling_is_medium_not_exit(self):
        ex = evaluate_exit(summary(flow_imbalance=-0.02, flow_accel=0.5))
        assert ex["urgency"] == "medium"
        assert ex["exit"] is False

    def test_broke_vwap_triggers_exit(self):
        ex = evaluate_exit(summary(price_vs_vwap=-1.6))
        assert ex["exit"] is True

    def test_heavy_ask_stack_is_medium(self):
        ex = evaluate_exit(summary(obi_avg=-0.4))
        assert ex["urgency"] in ("medium", "high")

    def test_not_ready(self):
        ex = evaluate_exit({"ready": False})
        assert ex["exit"] is False
        assert ex["pnl_pct"] is None

    def test_no_entry_price_means_no_pnl(self):
        assert evaluate_exit(summary())["pnl_pct"] is None


class TestFormatting:

    def test_alert_contains_key_sections(self):
        s = summary()
        msg = format_entry_alert(evaluate_entry(s, CTX), s)
        assert "2330" in msg
        assert "訊號拆解" in msg
        assert "小波段操作參考" in msg
        assert "不構成投資建議" in msg

    def test_alert_shows_vetoes(self):
        s = summary(is_limit_down=True)
        msg = format_entry_alert(evaluate_entry(s, CTX), s)
        assert "否決條件" in msg
        assert "小波段操作參考" not in msg

    def test_alert_fits_telegram_limit(self):
        s = summary()
        assert len(format_entry_alert(evaluate_entry(s, CTX), s)) < 4096

    def test_summary_line_mentions_veto_first(self):
        sig = evaluate_entry(summary(is_limit_down=True), CTX)
        assert "否決" in sig["summary_line"]
