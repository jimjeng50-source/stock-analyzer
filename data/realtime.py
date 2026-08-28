"""
data/realtime.py
台股「即時」報價快照 —— 資金動態圖與盤中進場訊號的資料來源。

資料來源：證交所 MIS（盤中即時行情）
    https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_2330.tw|otc_6488.tw&json=1

特性與限制（誠實說明，避免誤用）：
  * MIS 是「快照」而非逐筆（tick）。盤中約每 5 秒更新一次撮合結果，
    回傳「累積成交量 v」與「最新成交價 z」＋「五檔委買賣 b/g、a/f」。
  * 因此本模組拿不到真正的逐筆內外盤，資金流是用
    「相鄰兩次快照的累積量差 × 成交價相對前一檔買賣價的位置」推估
    （見 factors/intraday_flow.py）。這是券商軟體外的可行近似，
    方向性可靠，絕對金額會有誤差。
  * 免 token、免費，但請勿高頻打（建議間隔 ≥ 5 秒；本模組預設 10 秒節流）。
  * 非交易時段呼叫會拿到最後一筆收盤快照（欄位仍在，t 為最後撮合時間）。

上市（tse_）與上櫃（otc_）前綴不同，模組會自動判斷並快取每支股票的頻道。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

import requests

from utils.tz import TW_TZ, now_tw

logger = logging.getLogger(__name__)

_MIS_API = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
_MIS_REFERER = "https://mis.twse.com.tw/stock/fibest.jsp"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": _MIS_REFERER,
}

# MIS 單次請求的頻道數上限（官方未公告，實務上 50 內穩定）
_MAX_CHANNELS_PER_CALL = 40
# 最小請求間隔（秒）：避免被 MIS 擋
_MIN_REQUEST_INTERVAL = 3.0

# 台股交易時段（台灣時間）
MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(13, 30)
# 08:30 起有試撮（模擬撮合）資料，13:30~14:30 為盤後定價
PREOPEN_START = dtime(8, 30)

# stock_id → "tse" / "otc"（解析成功後快取，省一半請求）
_CHANNEL_CACHE: Dict[str, str] = {}
_LAST_REQUEST_TS = 0.0
_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# 快照資料結構
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Quote:
    """單一股票的即時快照。價格單位＝元，量單位＝張。"""

    stock_id: str
    stock_name: str = ""
    exchange: str = "tse"                 # tse=上市 / otc=上櫃
    price: float = 0.0                    # 最新成交價（z；無成交時退回前一盤 pz / 委買賣中價）
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0               # 昨收（y）
    limit_up: float = 0.0                 # 漲停價（u）
    limit_down: float = 0.0               # 跌停價（w）
    cum_volume: int = 0                   # 當日累積成交量（張，v）
    tick_volume: int = 0                  # 當盤成交量（張，tv）
    bids: List[tuple] = field(default_factory=list)   # [(價, 量張)] 最多五檔，由高到低
    asks: List[tuple] = field(default_factory=list)   # [(價, 量張)] 最多五檔，由低到高
    trade_time: str = ""                  # 最後撮合時間 "HH:MM:SS"
    ts: Optional[datetime] = None         # 快照時間（台灣時間）
    trade_date: str = ""                  # 交易日 "YYYYMMDD"

    # ── 衍生欄位 ──────────────────────────────────────────────────────────────

    @property
    def bid1(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def ask1(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def bid_volume(self) -> int:
        """五檔委買總量（張）。"""
        return int(sum(v for _, v in self.bids))

    @property
    def ask_volume(self) -> int:
        """五檔委賣總量（張）。"""
        return int(sum(v for _, v in self.asks))

    @property
    def change_pct(self) -> float:
        """漲跌幅（%）。"""
        if not self.prev_close or not self.price:
            return 0.0
        return (self.price / self.prev_close - 1) * 100

    @property
    def turnover(self) -> float:
        """當日累積成交金額估算（元）＝ 累積張數 × 現價 × 1000。"""
        return self.cum_volume * self.price * 1000

    @property
    def is_limit_up(self) -> bool:
        """是否漲停（且賣方掛單被清空＝鎖死）。"""
        if not self.limit_up or not self.price:
            return False
        return self.price >= self.limit_up - 1e-6 and self.ask_volume == 0

    @property
    def is_limit_down(self) -> bool:
        if not self.limit_down or not self.price:
            return False
        return self.price <= self.limit_down + 1e-6 and self.bid_volume == 0

    @property
    def valid(self) -> bool:
        """快照是否可用於計算（有價格即可，開盤前無成交量也算有效）。"""
        return self.price > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 解析工具
# ═══════════════════════════════════════════════════════════════════════════════

def _num(v, default: float = 0.0) -> float:
    """MIS 用 '-' 表示無資料；同時清掉千分位。"""
    try:
        s = str(v).replace(",", "").strip()
        if s in ("", "-", "--"):
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def _split_levels(prices: str, volumes: str) -> List[tuple]:
    """把 MIS 的 '1135.0000_1140.0000_' 與 '245_346_' 併成 [(價, 量)]。"""
    if not prices or not volumes:
        return []
    ps = [p for p in str(prices).split("_") if p not in ("", "-")]
    vs = [v for v in str(volumes).split("_") if v not in ("", "-")]
    out = []
    for p, v in zip(ps, vs):
        price = _num(p)
        vol = _num(v)
        if price > 0:
            out.append((price, int(vol)))
    return out


def _parse_entry(item: dict) -> Optional[Quote]:
    """把 MIS msgArray 的一筆記錄轉為 Quote。無法解析回 None。"""
    if not isinstance(item, dict):
        return None
    stock_id = str(item.get("c", "")).strip()
    if not stock_id:
        return None

    price = _num(item.get("z"))
    if price <= 0:
        # 尚未成交（或該盤無成交）→ 退回「前一盤成交價」，再退回委買賣中價
        price = _num(item.get("pz"))
    bids = _split_levels(item.get("b"), item.get("g"))
    asks = _split_levels(item.get("a"), item.get("f"))
    if price <= 0 and bids and asks:
        price = round((bids[0][0] + asks[0][0]) / 2, 4)
    if price <= 0:
        price = _num(item.get("y"))          # 最後退回昨收

    ts = None
    tlong = item.get("tlong")
    try:
        if tlong:
            ts = datetime.fromtimestamp(int(tlong) / 1000, TW_TZ)
    except (TypeError, ValueError, OSError):
        ts = None
    if ts is None:
        ts = now_tw()

    return Quote(
        stock_id=stock_id,
        stock_name=str(item.get("n", "")).strip(),
        exchange=str(item.get("ex", "tse")).strip() or "tse",
        price=price,
        open=_num(item.get("o")),
        high=_num(item.get("h")),
        low=_num(item.get("l")),
        prev_close=_num(item.get("y")),
        limit_up=_num(item.get("u")),
        limit_down=_num(item.get("w")),
        cum_volume=int(_num(item.get("v"))),
        tick_volume=int(_num(item.get("tv"))),
        bids=bids,
        asks=asks,
        trade_time=str(item.get("t", "")).strip(),
        ts=ts,
        trade_date=str(item.get("d", "")).strip(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 抓取
# ═══════════════════════════════════════════════════════════════════════════════

def _throttle() -> None:
    """全域節流：兩次 MIS 請求至少間隔 _MIN_REQUEST_INTERVAL 秒。"""
    global _LAST_REQUEST_TS
    with _LOCK:
        wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _LAST_REQUEST_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TS = time.monotonic()


def _call_mis(channels: List[str], timeout: int = 15) -> List[dict]:
    """對 MIS 發一次請求，回傳 msgArray（失敗回空列表）。"""
    if not channels:
        return []
    _throttle()
    try:
        resp = requests.get(
            _MIS_API,
            params={"ex_ch": "|".join(channels), "json": "1", "delay": "0",
                    "_": int(time.time() * 1000)},
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        logger.warning("MIS 即時報價請求失敗：%s", e)
        return []

    if not isinstance(body, dict):
        return []
    if body.get("rtcode") not in (None, "0000"):
        logger.warning("MIS 回應非 OK：rtcode=%s msg=%s",
                       body.get("rtcode"), body.get("rtmessage"))
    return body.get("msgArray") or []


def _channels_for(stock_ids: List[str]) -> List[str]:
    """
    組出查詢頻道。已知市場別的只查一個頻道；未知的同時查 tse_ 與 otc_，
    由回應決定，並寫入快取。
    """
    chans = []
    for sid in stock_ids:
        ex = _CHANNEL_CACHE.get(sid)
        if ex:
            chans.append(f"{ex}_{sid}.tw")
        else:
            chans.append(f"tse_{sid}.tw")
            chans.append(f"otc_{sid}.tw")
    return chans


def _chunks(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def fetch_quotes(stock_ids: List[str]) -> Dict[str, Quote]:
    """
    批次取得即時快照。

    Args:
        stock_ids: 股票代號列表（如 ["2330", "6488"]）。

    Returns:
        {stock_id: Quote}。抓不到的股票不會出現在結果中（不丟例外）。
    """
    ids = [str(s).strip() for s in stock_ids if str(s).strip()]
    if not ids:
        return {}
    # 去重但保持順序
    ids = list(dict.fromkeys(ids))

    out: Dict[str, Quote] = {}
    channels = _channels_for(ids)
    for group in _chunks(channels, _MAX_CHANNELS_PER_CALL):
        for item in _call_mis(group):
            q = _parse_entry(item)
            if q is None or not q.valid:
                continue
            # 同一支股票 tse/otc 都問時，先到先得（另一邊本來就不會有資料）
            if q.stock_id in out:
                continue
            out[q.stock_id] = q
            _CHANNEL_CACHE[q.stock_id] = q.exchange

    missing = [s for s in ids if s not in out]
    if missing:
        logger.debug("即時報價查無資料：%s", ",".join(missing))
    return out


def fetch_quote(stock_id: str) -> Optional[Quote]:
    """取得單一股票的即時快照；抓不到回 None。"""
    return fetch_quotes([stock_id]).get(str(stock_id).strip())


# ═══════════════════════════════════════════════════════════════════════════════
# 交易時段判斷
# ═══════════════════════════════════════════════════════════════════════════════

def is_trading_hours(dt: Optional[datetime] = None, include_preopen: bool = False) -> bool:
    """
    是否為台股盤中（平日 09:00–13:30）。

    Args:
        include_preopen: True 時把 08:30 起的試撮時段也算進來。

    註：不含國定假日行事曆（免費資料源沒有）。休市日 MIS 會回最後交易日的
    收盤快照，資金流引擎看得到 cum_volume 不再變動，不會產生假訊號。
    """
    dt = dt or now_tw()
    if dt.weekday() >= 5:
        return False
    start = PREOPEN_START if include_preopen else MARKET_OPEN
    return start <= dt.time() <= MARKET_CLOSE


def session_progress(dt: Optional[datetime] = None) -> float:
    """
    當日交易時段已走完的比例（0~1），用於量能倍率的同時段換算。
    盤前回 0，盤後回 1。
    """
    dt = dt or now_tw()
    t = dt.time()
    if t < MARKET_OPEN:
        return 0.0
    if t >= MARKET_CLOSE:
        return 1.0
    total = (MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute) - \
            (MARKET_OPEN.hour * 60 + MARKET_OPEN.minute)
    elapsed = (t.hour * 60 + t.minute + t.second / 60) - \
              (MARKET_OPEN.hour * 60 + MARKET_OPEN.minute)
    return max(0.0, min(1.0, elapsed / total))


def clear_channel_cache() -> None:
    """清除市場別快取（測試用）。"""
    _CHANNEL_CACHE.clear()
