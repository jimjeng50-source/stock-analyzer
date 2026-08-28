"""
factors/intraday_flow.py
盤中資金動態引擎 —— 把 MIS 快照序列轉成「主動買賣資金流」時間序列。

核心問題：MIS 只給「累積成交量」與「五檔委買賣」，沒有逐筆內外盤。
本引擎的作法（券商軟體外的標準近似）：

  1. 相鄰兩次快照的累積量差 Δv 就是這段時間的成交張數。
  2. 用「成交價相對於『前一次快照』五檔買一/賣一的位置」判定主動方：
        成交價 ≥ 前賣一  → 全部視為主動買（外盤）
        成交價 ≤ 前買一  → 全部視為主動賣（內盤）
        介於買賣價之間  → 依相對位置按比例拆分
     沒有前一檔報價時退回 tick rule（比前一筆貴＝買、便宜＝賣、持平＝沿用前次方向）。
  3. 資金流金額 = 張數 × 成交價 × 1000。

限制（務必理解後再用）：
  * 5 秒一張快照，期間的來回單會互相抵銷 → 絕對金額低估，但「方向」與
    「加速/減速」的資訊仍然可靠，這正是抓小波段進場點需要的。
  * 追蹤是「從開始追蹤那一刻起」累積的。中途才開始看的話，
    coverage_ratio 會告訴你只涵蓋當日成交量的多少比例。
"""

import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Deque, Dict, List, Optional

import pandas as pd

from data.realtime import Quote, session_progress
from utils.tz import now_tw

logger = logging.getLogger(__name__)

# 台股日內累積成交量分布（U 型：開盤與尾盤量大）。
# (距開盤分鐘數, 該時點累積成交量佔全日比例)
_VOLUME_CURVE = [
    (0, 0.00), (15, 0.13), (30, 0.20), (60, 0.32), (90, 0.41),
    (120, 0.49), (150, 0.56), (180, 0.625), (210, 0.69),
    (240, 0.77), (265, 0.87), (270, 1.00),
]

# 加速度取樣：最近這個比例的樣本視為「近期」
_RECENT_FRACTION = 0.25
_MIN_RECENT_POINTS = 4
_EPS = 1e-9


@dataclass
class FlowPoint:
    """單一時點的資金流量測。金額單位＝元，量單位＝張。"""

    ts: datetime
    price: float
    cum_volume: int              # 當日累積成交量（張）
    delta_volume: int            # 與上一快照的成交量差（張）
    buy_volume: float            # 本段推估主動買張數
    sell_volume: float           # 本段推估主動賣張數
    net_amount: float            # 本段淨流入金額（買-賣）
    cum_net_amount: float        # 追蹤起始至今的累計淨流入金額
    cum_buy_volume: float
    cum_sell_volume: float
    obi: float                   # 五檔委買賣不平衡 [-1, 1]
    vwap: float
    bid1: float = 0.0
    ask1: float = 0.0


def expected_volume_ratio(dt: Optional[datetime] = None) -> float:
    """
    依日內量能分布曲線，回傳「此刻應該已經走完全日成交量的幾成」。
    用於量能倍率：實際累積量 ÷（5 日均量 × 本值）。
    """
    dt = dt or now_tw()
    prog = session_progress(dt)
    minutes = prog * _VOLUME_CURVE[-1][0]
    if minutes <= 0:
        return 0.0
    for i in range(1, len(_VOLUME_CURVE)):
        m0, r0 = _VOLUME_CURVE[i - 1]
        m1, r1 = _VOLUME_CURVE[i]
        if minutes <= m1:
            span = m1 - m0
            w = (minutes - m0) / span if span else 1.0
            return r0 + (r1 - r0) * w
    return 1.0


def classify_flow(price: float, prev_quote: Optional[Quote],
                  prev_price: float, prev_direction: int) -> float:
    """
    判定這段成交量中「主動買」所佔的比例（0~1）。

    Args:
        price:          本次快照成交價
        prev_quote:     上一次快照（提供五檔買一/賣一）
        prev_price:     上一次成交價（tick rule 退路）
        prev_direction: 上次判定方向（+1 買 / -1 賣 / 0 未知），持平時沿用

    Returns:
        主動買佔比。1.0 = 全部外盤成交；0.0 = 全部內盤成交。
    """
    bid1 = prev_quote.bid1 if prev_quote else 0.0
    ask1 = prev_quote.ask1 if prev_quote else 0.0

    if bid1 > 0 and ask1 > 0 and ask1 > bid1:
        if price >= ask1:
            return 1.0
        if price <= bid1:
            return 0.0
        return max(0.0, min(1.0, (price - bid1) / (ask1 - bid1)))

    # 沒有可用的五檔 → tick rule
    if prev_price > 0:
        if price > prev_price:
            return 1.0
        if price < prev_price:
            return 0.0
        return 1.0 if prev_direction > 0 else (0.0 if prev_direction < 0 else 0.5)
    return 0.5


class FlowTracker:
    """
    單一股票的盤中資金流追蹤器。

    用法：
        t = FlowTracker("2330")
        while trading:
            q = fetch_quote("2330")
            t.update(q)
        t.summary()        # → 指標字典（餵給 entry_signal）
        t.to_dataframe()   # → 畫動態圖用
    """

    def __init__(self, stock_id: str, max_points: int = 2000):
        self.stock_id = str(stock_id)
        self.stock_name = ""
        self.points: Deque[FlowPoint] = deque(maxlen=max_points)
        self.started_at: Optional[datetime] = None
        self.trade_date: str = ""

        self._prev_quote: Optional[Quote] = None
        self._prev_direction: int = 0
        self._cum_net_amount: float = 0.0
        self._cum_buy_vol: float = 0.0
        self._cum_sell_vol: float = 0.0
        self._vwap_pv: float = 0.0        # Σ 價×量（含起始種子）
        self._vwap_v: float = 0.0
        self._tracked_volume: int = 0     # 追蹤期間內實際觀察到的成交張數
        self._last_quote: Optional[Quote] = None

    # ── 更新 ──────────────────────────────────────────────────────────────────

    def update(self, quote: Optional[Quote]) -> Optional[FlowPoint]:
        """
        餵入一筆新快照，回傳計算出的 FlowPoint。
        quote 無效（None / 無價格）回 None，不影響既有序列。
        """
        if quote is None or not quote.valid:
            return None

        # 換日（跨交易日）→ 重置，避免把昨天的流量接到今天
        if self.trade_date and quote.trade_date and quote.trade_date != self.trade_date:
            logger.info("%s 交易日切換 %s → %s，重置資金流追蹤",
                        self.stock_id, self.trade_date, quote.trade_date)
            self.reset()

        self.stock_name = quote.stock_name or self.stock_name
        self.trade_date = quote.trade_date or self.trade_date
        ts = quote.ts or now_tw()

        if self.started_at is None:
            self.started_at = ts
            # 中途才開始追蹤：用當日典型價 (高+低+現)/3 當種子，讓 VWAP 不致失真
            if quote.cum_volume > 0:
                seed_price = quote.price
                if quote.high > 0 and quote.low > 0:
                    seed_price = (quote.high + quote.low + quote.price) / 3
                self._vwap_pv = seed_price * quote.cum_volume
                self._vwap_v = float(quote.cum_volume)

        prev = self._prev_quote
        delta_v = 0
        if prev is not None:
            delta_v = int(quote.cum_volume) - int(prev.cum_volume)
            if delta_v < 0:
                # 累積量倒退（換日或資料異常）→ 視為新的一段，不計流量
                delta_v = 0

        buy_ratio = classify_flow(
            quote.price, prev,
            prev.price if prev else 0.0,
            self._prev_direction,
        )
        if delta_v > 0:
            if buy_ratio > 0.5:
                self._prev_direction = 1
            elif buy_ratio < 0.5:
                self._prev_direction = -1

        buy_v = delta_v * buy_ratio
        sell_v = delta_v * (1 - buy_ratio)
        net_amt = (buy_v - sell_v) * quote.price * 1000

        self._cum_buy_vol += buy_v
        self._cum_sell_vol += sell_v
        self._cum_net_amount += net_amt
        self._tracked_volume += delta_v
        if delta_v > 0:
            self._vwap_pv += quote.price * delta_v
            self._vwap_v += delta_v
        vwap = (self._vwap_pv / self._vwap_v) if self._vwap_v > 0 else quote.price

        denom = quote.bid_volume + quote.ask_volume
        obi = ((quote.bid_volume - quote.ask_volume) / denom) if denom > 0 else 0.0

        point = FlowPoint(
            ts=ts,
            price=quote.price,
            cum_volume=int(quote.cum_volume),
            delta_volume=delta_v,
            buy_volume=round(buy_v, 2),
            sell_volume=round(sell_v, 2),
            net_amount=round(net_amt, 2),
            cum_net_amount=round(self._cum_net_amount, 2),
            cum_buy_volume=round(self._cum_buy_vol, 2),
            cum_sell_volume=round(self._cum_sell_vol, 2),
            obi=round(obi, 4),
            vwap=round(vwap, 4),
            bid1=quote.bid1,
            ask1=quote.ask1,
        )
        self.points.append(point)
        self._prev_quote = quote
        self._last_quote = quote
        return point

    def reset(self) -> None:
        """清空序列（換日或使用者手動重置）。"""
        self.points.clear()
        self.started_at = None
        self.trade_date = ""
        self._prev_quote = None
        self._prev_direction = 0
        self._cum_net_amount = 0.0
        self._cum_buy_vol = 0.0
        self._cum_sell_vol = 0.0
        self._vwap_pv = 0.0
        self._vwap_v = 0.0
        self._tracked_volume = 0

    # ── 輸出 ──────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """回傳可直接畫圖的 DataFrame（時間序）。"""
        if not self.points:
            return pd.DataFrame(columns=[f.name for f in FlowPoint.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(p) for p in self.points])

    def _rate_per_min(self, pts: List[FlowPoint]) -> float:
        """一段點序列的淨流入速率（元/分鐘）。"""
        if len(pts) < 2:
            return 0.0
        span_min = (pts[-1].ts - pts[0].ts).total_seconds() / 60.0
        if span_min <= 0:
            return 0.0
        net = pts[-1].cum_net_amount - pts[0].cum_net_amount
        return net / span_min

    def flow_acceleration(self) -> float:
        """
        資金流加速度：近期淨流入速率 ÷ 全段平均速率的絕對值。

          > 1   近期流入比整段平均更猛（正在加速）
          0~1   仍在流入但變慢
          < 0   近期轉為流出
        沒有足夠樣本回 0.0。結果夾在 [-5, 5]。
        """
        pts = list(self.points)
        if len(pts) < _MIN_RECENT_POINTS + 1:
            return 0.0
        n_recent = max(_MIN_RECENT_POINTS, int(len(pts) * _RECENT_FRACTION))
        recent = pts[-n_recent:]
        base_rate = self._rate_per_min(pts)
        recent_rate = self._rate_per_min(recent)
        if abs(base_rate) < _EPS:
            return 0.0
        return max(-5.0, min(5.0, recent_rate / abs(base_rate)))

    def summary(self, avg_volume_5d: Optional[float] = None) -> Dict:
        """
        整段追蹤的彙總指標，供 factors.entry_signal 評分與 UI 顯示。

        Args:
            avg_volume_5d: 近 5 日均量（張）。有給才算得出量能倍率。
        """
        q = self._last_quote
        pts = list(self.points)
        if q is None or not pts:
            return {"stock_id": self.stock_id, "ready": False,
                    "reason": "尚未取得任何即時快照"}

        last = pts[-1]
        turnover = q.turnover
        net_amount = last.cum_net_amount
        net_ratio = (net_amount / turnover) if turnover > 0 else 0.0

        # 主動買賣不平衡：追蹤期間內 (主動買-主動賣)/(主動買+主動賣)，範圍 [-1, 1]。
        # 不受「中途才開始追蹤」影響（分母同樣只算追蹤到的量），是評分的主要依據。
        tracked = last.cum_buy_volume + last.cum_sell_volume
        flow_imbalance = ((last.cum_buy_volume - last.cum_sell_volume) / tracked) if tracked > 0 else 0.0

        # 日內價格位階
        if q.high > q.low > 0:
            price_position = (q.price - q.low) / (q.high - q.low)
        else:
            price_position = 0.5
        price_position = max(0.0, min(1.0, price_position))

        # 近 10 筆的平均 OBI（單筆易受掛單瞬間變動干擾）
        recent_obi = [p.obi for p in pts[-10:]]
        obi_avg = sum(recent_obi) / len(recent_obi)

        vol_surge = None
        if avg_volume_5d and avg_volume_5d > 0:
            expected = avg_volume_5d * max(expected_volume_ratio(q.ts), 0.02)
            vol_surge = q.cum_volume / expected if expected > 0 else None

        coverage = (self._tracked_volume / q.cum_volume) if q.cum_volume > 0 else 0.0

        return {
            "ready": True,
            "stock_id": self.stock_id,
            "stock_name": self.stock_name,
            "ts": last.ts,
            "trade_time": q.trade_time,
            "price": q.price,
            "prev_close": q.prev_close,
            "change_pct": round(q.change_pct, 2),
            "open": q.open,
            "high": q.high,
            "low": q.low,
            "cum_volume": q.cum_volume,
            "turnover": turnover,

            # ── 資金流 ──
            "net_amount": round(net_amount, 0),
            "net_volume": round(last.cum_buy_volume - last.cum_sell_volume, 1),
            "buy_volume": round(last.cum_buy_volume, 1),
            "sell_volume": round(last.cum_sell_volume, 1),
            "net_ratio": round(net_ratio, 4),
            "flow_imbalance": round(flow_imbalance, 4),
            "flow_accel": round(self.flow_acceleration(), 3),

            # ── 掛單 / 價格結構 ──
            "obi": round(last.obi, 4),
            "obi_avg": round(obi_avg, 4),
            "bid_volume": q.bid_volume,
            "ask_volume": q.ask_volume,
            "vwap": last.vwap,
            "price_vs_vwap": round((q.price / last.vwap - 1) * 100, 3) if last.vwap else 0.0,
            "price_position": round(price_position, 3),
            "vol_surge": round(vol_surge, 2) if vol_surge is not None else None,

            # ── 追蹤品質 ──
            "samples": len(pts),
            "started_at": self.started_at,
            "coverage_ratio": round(coverage, 3),
            "is_limit_up": q.is_limit_up,
            "is_limit_down": q.is_limit_down,
        }


class FlowBook:
    """多股票的 FlowTracker 集合（觀察清單用）。"""

    def __init__(self, stock_ids: Optional[List[str]] = None, max_points: int = 2000):
        self.max_points = max_points
        self.trackers: Dict[str, FlowTracker] = {}
        for sid in (stock_ids or []):
            self.add(sid)

    def add(self, stock_id: str) -> FlowTracker:
        sid = str(stock_id).strip()
        if sid not in self.trackers:
            self.trackers[sid] = FlowTracker(sid, max_points=self.max_points)
        return self.trackers[sid]

    def remove(self, stock_id: str) -> None:
        self.trackers.pop(str(stock_id).strip(), None)

    @property
    def stock_ids(self) -> List[str]:
        return list(self.trackers.keys())

    def update(self, quotes: Dict[str, Quote]) -> Dict[str, FlowPoint]:
        """把一批快照分派給各自的 tracker，回傳有更新的 {stock_id: FlowPoint}。"""
        out = {}
        for sid, q in (quotes or {}).items():
            pt = self.add(sid).update(q)
            if pt is not None:
                out[sid] = pt
        return out

    def summaries(self, avg_volumes: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        avg_volumes = avg_volumes or {}
        return {sid: t.summary(avg_volume_5d=avg_volumes.get(sid))
                for sid, t in self.trackers.items()}
