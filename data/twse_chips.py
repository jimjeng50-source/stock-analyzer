"""
data/twse_chips.py
三大法人買賣超（T86）免費備援 —— 證交所官方 OpenAPI / 每日報表

FinMind 免費配額用盡（402/403）時，批次每日掃描抓不到 TaiwanStock
InstitutionalInvestorsBuySell → 籌碼面（20% 權重）全給中性 0.5，
無法反映外資/投信/自營商真實買賣超。

證交所 T86「三大法人買賣超日報」一次回傳「全市場」資料，
比 FinMind 逐股（40 支 × 多請求）省太多，且免 token。
結果快取於 process 記憶體，40 支股票共用同一份市場快照。

資料來源（依序嘗試，任一成功即用；皆失敗則回空、籌碼退回中性）：
  1. openapi.twse.com.tw/v1/fund/T86 —— 最新一個交易日、單一請求
     （與 screener/universe.py 同家族主機，production 已證實可達）
  2. www.twse.com.tw/rwd/zh/fund/T86 —— 依日期查詢，可補多個交易日歷史

只用於「即時」掃描（as_of 為 None）；歷史回溯不可用（會抓到未來資料）。
"""

import logging
from datetime import timedelta
from typing import Optional

import pandas as pd
import requests

from utils.tz import now_tw

logger = logging.getLogger(__name__)

_TWSE_OPENAPI_T86 = "https://openapi.twse.com.tw/v1/fund/T86"
_TWSE_DATED_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"

_HEADERS = {"accept": "application/json", "User-Agent": "Mozilla/5.0"}

# dated 端點回補的交易日數（涵蓋 5d 因子；逐日查，控制請求數）
_TRADING_DAYS = 6
_MAX_CALENDAR_LOOKBACK = 12

# process 級快取：整個市場的長格式籌碼表（date, stock_id, name, net）
_MARKET_CACHE: Optional[pd.DataFrame] = None
_CACHE_LOADED = False


def _to_num(v) -> float:
    """清理帶千分位逗號 / '--' 的數字字串。"""
    try:
        s = str(v).replace(",", "").replace(" ", "").strip()
        if s in ("", "-", "--", "---"):
            return 0.0
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _fmt_iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _pick_columns(cols: list) -> dict:
    """
    從欄位標籤清單挑出 code / name / foreign / trust / dealer 對應欄。
    支援中文（證交所報表標籤）與英文（OpenAPI）鍵名，防禦性比對。
    回傳 {key: label}；找不到的鍵不放入。
    """
    labels = [str(c) for c in cols]

    def match(include_groups, exclude=()):
        for grp in include_groups:
            for lab in labels:
                if all(t in lab for t in grp) and not any(x in lab for x in exclude):
                    return lab
        return None

    picked = {}
    code = match([("證券代號",), ("代號",), ("Code",)])
    name = match([("證券名稱",), ("名稱",), ("Name",)])
    # 外資：中文用「外陸資」可與「外資自營商」區隔；英文用 Foreign 並排除 Dealer/Self
    foreign = match([("外陸資",), ("ForeignInvestorsandMainland",), ("Foreign",)],
                    exclude=("Dealer", "Self", "Hedge"))
    trust = match([("投信",), ("InvestmentTrust",), ("Trust",)])
    # 自營商合計：排除「外資自營商」及子項（自行買賣 / 避險）
    dealer = match([("自營商",), ("Dealer",)],
                   exclude=("外資", "(", "（", "自行買賣", "避險", "Foreign", "Self", "Hedge"))

    if code:
        picked["code"] = code
    if name:
        picked["name"] = name
    if foreign:
        picked["foreign"] = foreign
    if trust:
        picked["trust"] = trust
    if dealer:
        picked["dealer"] = dealer
    return picked


def _rows_from_records(records: list, date_str: str) -> list:
    """
    將 dict 記錄列表（OpenAPI 格式）轉為長格式列。
    每支股票拆成外資/投信/自營商三列，name 對齊 factors.chips 判斷集合。
    """
    if not records or not isinstance(records[0], dict):
        return []
    cols = _pick_columns(list(records[0].keys()))
    if "code" not in cols:
        return []
    return _emit_rows(records, cols, date_str, getter=lambda rec, lab: rec.get(lab))


def _rows_from_fields(payload: dict, date_str: str) -> list:
    """
    將證交所 fields+data 結構（dated 端點）轉為長格式列。
    """
    if not payload or payload.get("stat") != "OK":
        return []
    fields = payload.get("fields") or []
    data = payload.get("data") or []
    if not fields or not data:
        return []
    cols = _pick_columns(fields)
    if "code" not in cols:
        return []
    idx = {k: fields.index(v) for k, v in cols.items()}

    def getter(rec, key_pos):
        return rec[key_pos] if len(rec) > key_pos else None

    # data 為 list-of-list，改用位置索引
    records = [dict(enumerate(r)) for r in data if isinstance(r, (list, tuple))]
    cols_pos = {k: idx[k] for k in cols}
    return _emit_rows(records, cols_pos, date_str,
                      getter=lambda rec, pos: rec.get(pos))


def _emit_rows(records, cols, date_str, getter) -> list:
    rows = []
    for rec in records:
        raw_code = getter(rec, cols["code"])
        if raw_code is None:
            continue
        code = str(raw_code).strip()
        if not code.isdigit():          # 濾除權證等含字母代號
            continue
        name = ""
        if "name" in cols:
            nv = getter(rec, cols["name"])
            name = str(nv).strip() if nv is not None else ""
        for label, key in (("外資", cols.get("foreign")),
                           ("投信", cols.get("trust")),
                           ("自營商", cols.get("dealer"))):
            if key is None:
                continue
            val = getter(rec, key)
            if val is None:
                continue
            rows.append({
                "date": date_str, "stock_id": code, "stock_name": name,
                "name": label, "net": _to_num(val),
            })
    return rows


def _fetch_openapi() -> list:
    """證交所 OpenAPI T86：最新一個交易日、全市場、單一請求。"""
    try:
        resp = requests.get(_TWSE_OPENAPI_T86, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            # OpenAPI 未附日期 → 以「今日」標記（僅供即時掃描用）
            return _rows_from_records(payload, now_tw().strftime("%Y-%m-%d"))
        if isinstance(payload, dict):
            return _rows_from_fields(payload, now_tw().strftime("%Y-%m-%d"))
    except Exception as e:
        logger.debug("TWSE OpenAPI T86 抓取失敗：%s", e)
    return []


def _fetch_dated_day(date_str: str) -> list:
    """證交所 dated 端點：抓單一交易日（YYYYMMDD）全市場資料。"""
    try:
        resp = requests.get(
            _TWSE_DATED_T86,
            params={"date": date_str, "selectType": "ALL", "response": "json"},
            headers=_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        return _rows_from_fields(resp.json(), _fmt_iso(date_str))
    except Exception as e:
        logger.debug("TWSE dated T86 抓取失敗 %s：%s", date_str, e)
        return []


def _load_market_institutional() -> pd.DataFrame:
    """
    載入全市場三大法人買賣超（長格式）。
    先試 OpenAPI（最新一日、proven host）；若拿不到，退回 dated 端點逐日回補。
    """
    all_rows = _fetch_openapi()

    if not all_rows:
        # OpenAPI 不可用 → dated 端點逐日往回抓多個交易日
        trading_days = 0
        d = now_tw()
        for _ in range(_MAX_CALENDAR_LOOKBACK):
            if d.weekday() < 5:            # 跳過週末，省請求
                rows = _fetch_dated_day(d.strftime("%Y%m%d"))
                if rows:
                    all_rows.extend(rows)
                    trading_days += 1
                    if trading_days >= _TRADING_DAYS:
                        break
            d = d - timedelta(days=1)

    if not all_rows:
        logger.warning("T86 備援：證交所無回傳任何資料")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # 同一 (date, stock_id, name) 去重（OpenAPI 與 dated 可能重疊最新日）
    df = df.drop_duplicates(subset=["date", "stock_id", "name"], keep="first")
    logger.info("T86 備援：載入 %d 檔次法人買賣超（%d 日）",
                len(df), df["date"].nunique())
    return df.reset_index(drop=True)


def get_market_institutional(force_refresh: bool = False) -> pd.DataFrame:
    """
    取得（並快取）全市場三大法人買賣超長格式表。
    欄位：date, stock_id, stock_name, name（外資/投信/自營商）, net（股數）。
    """
    global _MARKET_CACHE, _CACHE_LOADED
    if force_refresh:
        _CACHE_LOADED = False
        _MARKET_CACHE = None
    if not _CACHE_LOADED:
        _MARKET_CACHE = _load_market_institutional()
        _CACHE_LOADED = True
    return _MARKET_CACHE if _MARKET_CACHE is not None else pd.DataFrame()


def get_t86_institutional(stock_id: str) -> pd.DataFrame:
    """
    單支股票的三大法人買賣超（長格式），供 factors.chips.compute_chips 使用。
    回傳欄位：date, name, net（與 FinMind get_institutional 相容子集）。
    抓不到回傳空 DataFrame。
    """
    market = get_market_institutional()
    if market is None or market.empty:
        return pd.DataFrame()
    sid = str(stock_id).strip()
    sub = market[market["stock_id"] == sid]
    if sub.empty:
        return pd.DataFrame()
    return sub[["date", "name", "net"]].sort_values("date").reset_index(drop=True)
