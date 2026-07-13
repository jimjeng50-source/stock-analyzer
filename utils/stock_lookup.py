"""
utils/stock_lookup.py
股票代號 ⇆ 公司名稱解析

支援使用者以「代號」或「公司名稱」輸入，統一解析成 stock_id。
清單來源：FinMind TaiwanStockInfo（全市場），快取於 data/stock_names.json（7 天）。

resolve(query) 規則：
1. 純數字 4-6 碼 → 視為代號，補上名稱
2. 完全符合的公司名 → 回傳其代號
3. 部分符合（包含關係）→ 回傳唯一符合者；多筆符合時回傳候選清單
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional

import requests

from config import get_runtime_config

logger = logging.getLogger(__name__)

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
_CACHE_PATH = "data/stock_names.json"
_CACHE_HOURS = 24 * 7


def _load_cache() -> Optional[list]:
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        cached_at = datetime.fromisoformat(cache["cached_at"])
        if datetime.now() - cached_at > timedelta(hours=_CACHE_HOURS):
            return None
        return cache["data"]
    except Exception:
        return None


def _save_cache(rows: list) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"cached_at": datetime.now().isoformat(), "data": rows},
                      f, ensure_ascii=False)
    except Exception as e:
        logger.warning("股票名稱快取儲存失敗：%s", e)


def load_stock_list(force_refresh: bool = False) -> list:
    """
    取得全市場 [{"stock_id","stock_name"}, ...]。
    無 FINMIND_TOKEN 或抓取失敗時回傳快取（若有）或空清單。
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            return cached

    token = get_runtime_config("FINMIND_TOKEN")
    if not token:
        return _load_cache() or []

    try:
        resp = requests.get(_FINMIND_API, params={
            "dataset": "TaiwanStockInfo", "token": token,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200 or not data.get("data"):
            return _load_cache() or []

        seen, rows = set(), []
        for r in data["data"]:
            sid = str(r.get("stock_id", "")).strip()
            name = str(r.get("stock_name", "")).strip()
            if not re.match(r"^\d{4,6}$", sid) or sid in seen:
                continue
            seen.add(sid)
            rows.append({"stock_id": sid, "stock_name": name})
        if rows:
            _save_cache(rows)
        return rows
    except Exception as e:
        logger.warning("股票清單抓取失敗：%s", e)
        return _load_cache() or []


def resolve(query: str) -> dict:
    """
    解析單一輸入（代號或名稱）。

    Returns:
        {
          "ok": bool,
          "stock_id": Optional[str],
          "stock_name": Optional[str],
          "candidates": [ {stock_id, stock_name}, ... ],  # 多筆符合時
          "query": str,
        }
    """
    q = (query or "").strip()
    result = {"ok": False, "stock_id": None, "stock_name": None,
              "candidates": [], "query": q}
    if not q:
        return result

    rows = load_stock_list()
    name_map = {r["stock_name"]: r["stock_id"] for r in rows}
    id_map = {r["stock_id"]: r["stock_name"] for r in rows}

    # 1) 純數字代號
    if re.match(r"^\d{4,6}$", q):
        result.update(ok=True, stock_id=q, stock_name=id_map.get(q, q))
        return result

    # 2) 完全符合公司名
    if q in name_map:
        result.update(ok=True, stock_id=name_map[q], stock_name=q)
        return result

    # 3) 部分符合（去除常見後綴後再比對）
    q_norm = q.replace("股份有限公司", "").replace("(", "").replace(")", "").strip()
    matches = [r for r in rows if q_norm and q_norm in r["stock_name"]]
    if len(matches) == 1:
        result.update(ok=True, stock_id=matches[0]["stock_id"],
                      stock_name=matches[0]["stock_name"])
    elif len(matches) > 1:
        result["candidates"] = matches[:10]

    return result


def resolve_many(queries: list) -> dict:
    """
    批次解析。
    Returns:
        {
          "resolved": [ {stock_id, stock_name, input}, ... ],
          "ambiguous": [ {input, candidates:[...]}, ... ],
          "unresolved": [ input, ... ],
        }
    """
    out = {"resolved": [], "ambiguous": [], "unresolved": []}
    seen_ids = set()
    for q in queries:
        r = resolve(q)
        if r["ok"]:
            if r["stock_id"] in seen_ids:
                continue
            seen_ids.add(r["stock_id"])
            out["resolved"].append({
                "stock_id": r["stock_id"],
                "stock_name": r["stock_name"],
                "input": r["query"],
            })
        elif r["candidates"]:
            out["ambiguous"].append({"input": r["query"], "candidates": r["candidates"]})
        else:
            out["unresolved"].append(r["query"])
    return out


def parse_pool_input(raw: str) -> list:
    """把使用者貼上的多行/逗號/空白分隔文字，拆成 query 清單。"""
    if not raw:
        return []
    tokens = re.split(r"[,\s，、\n\r\t]+", raw.strip())
    return [t.strip() for t in tokens if t.strip()]
