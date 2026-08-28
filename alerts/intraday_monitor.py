"""
alerts/intraday_monitor.py
盤中即時監控 —— 輪詢觀察清單的資金流，達標就立刻推播 Telegram。

流程：
    每 N 秒 → data.realtime.fetch_quotes（一次請求拿全部觀察股）
           → factors.intraday_flow.FlowBook 累積資金流序列
           → factors.entry_signal.evaluate_entry 評分
           → 達門檻且未在冷卻期 → Notifier.send_telegram

去重設計（避免同一檔一直洗版）：
    * 同一檔推播後進入冷卻期（預設 30 分鐘）
    * 冷卻期內若分數再上升 ≥ rescore_delta（預設 8 分）仍會補推
      （訊號明顯轉強值得知道）
    * 分級掉下門檻後，冷卻狀態清除，下次轉強可重新推播

日線資料（5 日均量／法人買賣超／MA20）每天只抓一次並快取，
盤中輪詢只打 MIS 免費即時 API，不消耗 FinMind 配額。
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from alerts.notifier import Notifier
from data.realtime import fetch_quotes, is_trading_hours
from factors.entry_signal import (evaluate_entry, evaluate_exit,
                                  format_entry_alert)
from factors.intraday_flow import FlowBook
from utils.tz import now_tw

logger = logging.getLogger(__name__)

# 觸發推播的最低分數（可由 config / 參數覆蓋）
DEFAULT_MIN_SCORE = 70
# 同一檔的推播冷卻（分鐘）
DEFAULT_COOLDOWN_MIN = 30
# 冷卻期內要再推播需要的分數增幅
DEFAULT_RESCORE_DELTA = 8
# 輪詢間隔（秒）。MIS 約 5 秒更新一次，20~30 秒足夠且友善
DEFAULT_POLL_INTERVAL = 30

# 會觸發推播的分級
ALERT_GRADES = {"strong", "buy"}


def build_daily_context(stock_id: str) -> Dict:
    """
    取得單一股票的日線背景資料（每日抓一次即可）。

    Returns:
        {"avg_volume_5d": 張, "foreign_net_5d": 張, "trust_net_5d": 張, "ma20": 元}
        抓不到的欄位為 None（進場模型會以中性值處理）。
    """
    ctx: Dict = {"avg_volume_5d": None, "foreign_net_5d": None,
                 "trust_net_5d": None, "ma20": None}

    # ── 均量與 MA20（FinMind → yfinance 自動退回）───────────────────────────
    try:
        from data.fetcher import FinMindFetcher
        fetcher = FinMindFetcher(stock_id, days=90)
        price = fetcher.get_price()
        if price is not None and not price.empty and "close" in price.columns:
            if "volume" in price.columns and len(price) >= 5:
                # 來源的 volume 是「股數」，轉成張
                ctx["avg_volume_5d"] = float(price["volume"].tail(5).mean()) / 1000.0
            if len(price) >= 20:
                ctx["ma20"] = float(price["close"].tail(20).mean())
    except Exception as e:
        logger.debug("%s 日線資料取得失敗：%s", stock_id, e)

    # ── 法人近 5 日買賣超（先 FinMind，缺就用證交所 T86 免費源）─────────────
    inst = None
    try:
        from data.fetcher import FinMindFetcher
        inst = FinMindFetcher(stock_id, days=30).get_institutional()
    except Exception as e:
        logger.debug("%s FinMind 籌碼取得失敗：%s", stock_id, e)
    if inst is None or getattr(inst, "empty", True):
        try:
            from data.twse_chips import get_t86_institutional
            inst = get_t86_institutional(stock_id)
        except Exception as e:
            logger.debug("%s T86 籌碼取得失敗：%s", stock_id, e)
            inst = None

    if inst is not None and not getattr(inst, "empty", True):
        try:
            from factors.chips import compute_chips
            import pandas as pd
            chips = compute_chips(inst, pd.DataFrame())
            # compute_chips 回傳的是「股數」，轉成張
            ctx["foreign_net_5d"] = chips.get("fi_5d_net", 0.0) / 1000.0
            ctx["trust_net_5d"] = chips.get("it_5d_net", 0.0) / 1000.0
        except Exception as e:
            logger.debug("%s 籌碼計算失敗：%s", stock_id, e)

    return ctx


class IntradayMonitor:
    """
    觀察清單的盤中資金流監控器。

    用法（單次輪詢，適合 GitHub Actions / cron）：
        m = IntradayMonitor(["2330", "2454"])
        alerts = m.poll_once()

    用法（常駐，適合本機或 Render worker）：
        IntradayMonitor(["2330"]).run(max_minutes=270)
    """

    def __init__(
        self,
        stock_ids: List[str],
        min_score: int = DEFAULT_MIN_SCORE,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MIN,
        rescore_delta: int = DEFAULT_RESCORE_DELTA,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        notifier: Optional[Notifier] = None,
        positions: Optional[Dict[str, Dict]] = None,
        skip_market_hours_check: bool = False,
    ):
        """
        Args:
            positions: 已持有部位 {stock_id: {"entry_price":, "stop_loss":, "target":}}
                       有給的話，出場條件成立時也會推播。
            skip_market_hours_check: True 時不檢查是否盤中（測試/回放用）。
        """
        self.stock_ids = [str(s).strip() for s in stock_ids if str(s).strip()]
        self.min_score = min_score
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.rescore_delta = rescore_delta
        self.poll_interval = poll_interval
        self.notifier = notifier or Notifier()
        self.positions = positions or {}
        self.skip_market_hours_check = skip_market_hours_check

        self.book = FlowBook(self.stock_ids)
        self._daily_ctx: Dict[str, Dict] = {}
        self._ctx_date = None
        # {stock_id: {"ts": datetime, "score": float, "grade": str}}
        self._last_alert: Dict[str, Dict] = {}
        self._exit_alerted: Dict[str, datetime] = {}
        self.alert_log: List[Dict] = []

    # ── 日線背景 ──────────────────────────────────────────────────────────────

    def daily_context(self, stock_id: str) -> Dict:
        """取得（並每日快取一次）日線背景資料。"""
        today = now_tw().date()
        if self._ctx_date != today:
            self._daily_ctx = {}
            self._ctx_date = today
        if stock_id not in self._daily_ctx:
            self._daily_ctx[stock_id] = build_daily_context(stock_id)
        return self._daily_ctx[stock_id]

    # ── 推播判斷 ──────────────────────────────────────────────────────────────

    def _should_alert(self, stock_id: str, signal: Dict) -> bool:
        """是否要推播這個進場訊號（含冷卻與去重）。"""
        if signal.get("grade") not in ALERT_GRADES:
            # 訊號轉弱 → 清掉冷卻，之後轉強可立即再推
            self._last_alert.pop(stock_id, None)
            return False
        if signal.get("score", 0) < self.min_score:
            self._last_alert.pop(stock_id, None)
            return False

        prev = self._last_alert.get(stock_id)
        if prev is None:
            return True
        elapsed = now_tw() - prev["ts"]
        if elapsed >= self.cooldown:
            return True
        # 冷卻期內：訊號明顯轉強才補推
        return signal["score"] >= prev["score"] + self.rescore_delta

    def _record_alert(self, stock_id: str, signal: Dict) -> None:
        self._last_alert[stock_id] = {
            "ts": now_tw(), "score": signal.get("score", 0),
            "grade": signal.get("grade", ""),
        }

    # ── 出場警示 ──────────────────────────────────────────────────────────────

    def _check_exit(self, stock_id: str, flow_summary: Dict) -> Optional[Dict]:
        pos = self.positions.get(stock_id)
        if not pos:
            return None
        ex = evaluate_exit(
            flow_summary,
            entry_price=pos.get("entry_price"),
            stop_loss=pos.get("stop_loss"),
            target=pos.get("target"),
        )
        if not ex.get("exit"):
            return None
        last = self._exit_alerted.get(stock_id)
        if last and now_tw() - last < self.cooldown:
            return None
        self._exit_alerted[stock_id] = now_tw()
        return ex

    @staticmethod
    def format_exit_alert(exit_signal: Dict, stock_name: str = "") -> str:
        sid = exit_signal.get("stock_id", "")
        pnl = exit_signal.get("pnl_pct")
        lines = [
            "═══════════════════",
            f"🚨 出場警示：{sid} {stock_name}",
            "═══════════════════",
            f"💰 現價 {exit_signal.get('price', 0):,.2f}"
            + (f"　損益 {pnl:+.2f}%" if pnl is not None else ""),
            "",
            "原因：",
        ]
        lines += [f"  ▪ {r}" for r in exit_signal.get("reasons", [])]
        lines += ["", "⚠️ 量化模型輸出，僅供研究參考，不構成投資建議。"]
        return "\n".join(lines)

    # ── 單次輪詢 ──────────────────────────────────────────────────────────────

    def poll_once(self, send: bool = True) -> List[Dict]:
        """
        抓一輪即時報價、更新資金流、評分，回傳本輪觸發的訊號列表。

        Args:
            send: False 時只計算不推播（dry-run / 測試）。

        Returns:
            [{"type": "entry"/"exit", "stock_id":, "signal":, "message":}]
        """
        if not self.stock_ids:
            return []

        quotes = fetch_quotes(self.stock_ids)
        if not quotes:
            logger.debug("本輪未取得任何即時報價")
            return []

        self.book.update(quotes)
        triggered: List[Dict] = []

        for sid in self.stock_ids:
            tracker = self.book.trackers.get(sid)
            if tracker is None:
                continue
            ctx = self.daily_context(sid)
            summary = tracker.summary(avg_volume_5d=ctx.get("avg_volume_5d"))
            if not summary.get("ready"):
                continue

            # 出場警示優先（手上有部位比找新進場點重要）
            ex = self._check_exit(sid, summary)
            if ex:
                msg = self.format_exit_alert(ex, summary.get("stock_name", ""))
                if send:
                    self.notifier.send_telegram(msg)
                item = {"type": "exit", "stock_id": sid, "signal": ex,
                        "message": msg, "ts": now_tw()}
                triggered.append(item)
                self.alert_log.append(item)

            signal = evaluate_entry(summary, ctx)
            if self._should_alert(sid, signal):
                msg = format_entry_alert(signal, summary)
                if send:
                    self.notifier.send_telegram(msg)
                self._record_alert(sid, signal)
                item = {"type": "entry", "stock_id": sid, "signal": signal,
                        "message": msg, "ts": now_tw()}
                triggered.append(item)
                self.alert_log.append(item)
                logger.info("進場訊號推播：%s", signal.get("summary_line"))

        return triggered

    # ── 常駐迴圈 ──────────────────────────────────────────────────────────────

    def run(self, max_minutes: Optional[int] = None, send: bool = True,
            max_polls: Optional[int] = None) -> List[Dict]:
        """
        持續輪詢直到收盤（或跑滿 max_minutes / max_polls）。

        非交易時段直接返回，不空轉。

        Args:
            max_minutes: 最長執行分鐘數；None 表示跑到收盤。
            max_polls:   最多輪詢幾次；主要供測試與 dry-run 使用。

        註：略過交易時段檢查（skip_market_hours_check=True）時，
        必須給 max_minutes 或 max_polls 其中之一，否則沒有結束條件。
        """
        if not self.skip_market_hours_check and not is_trading_hours():
            logger.info("目前非台股交易時段（平日 09:00–13:30），監控不啟動")
            return []
        if self.skip_market_hours_check and not max_minutes and not max_polls:
            raise ValueError(
                "skip_market_hours_check=True 時必須指定 max_minutes 或 max_polls，"
                "否則迴圈沒有結束條件。")

        started = now_tw()
        deadline = started + timedelta(minutes=max_minutes) if max_minutes else None
        logger.info("盤中資金流監控啟動：%s（門檻 %d 分，每 %d 秒輪詢）",
                    ",".join(self.stock_ids), self.min_score, self.poll_interval)

        polls = 0
        while True:
            try:
                self.poll_once(send=send)
            except Exception as e:
                logger.error("輪詢發生錯誤（略過本輪）：%s", e)
            polls += 1

            if max_polls and polls >= max_polls:
                logger.info("達到 max_polls，監控結束")
                break
            now = now_tw()
            if deadline and now >= deadline:
                logger.info("達到 max_minutes，監控結束")
                break
            if not self.skip_market_hours_check and not is_trading_hours(now):
                logger.info("已收盤，監控結束")
                break
            time.sleep(self.poll_interval)

        return self.alert_log

    # ── 快照（給 UI / Bot 用）─────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Dict]:
        """回傳目前每檔的 {summary, signal}，不推播。"""
        out = {}
        for sid, tracker in self.book.trackers.items():
            ctx = self.daily_context(sid)
            summary = tracker.summary(avg_volume_5d=ctx.get("avg_volume_5d"))
            out[sid] = {"summary": summary, "signal": evaluate_entry(summary, ctx)}
        return out


def analyze_now(stock_id: str, samples: int = 6, interval: float = 6.0) -> Dict:
    """
    對單一股票做一次「即時取樣分析」：連抓數筆快照後評分。

    給 Telegram /flow 指令與 Streamlit 單次查詢用 —— 因為資金流需要至少兩筆
    快照才算得出量差，這裡預設抓 6 筆（約 30 秒）換取可用的方向與加速度。

    Returns:
        {"summary": ..., "signal": ..., "tracker": FlowTracker}
    """
    from factors.intraday_flow import FlowTracker

    sid = str(stock_id).strip()
    tracker = FlowTracker(sid)
    for i in range(max(2, samples)):
        q = fetch_quotes([sid]).get(sid)
        tracker.update(q)
        if i < samples - 1:
            time.sleep(interval)

    ctx = build_daily_context(sid)
    summary = tracker.summary(avg_volume_5d=ctx.get("avg_volume_5d"))
    return {"summary": summary, "signal": evaluate_entry(summary, ctx),
            "tracker": tracker, "context": ctx}
