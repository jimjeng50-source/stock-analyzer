"""
screener/research_signals.py
外資／券商研究報告信號（公開新聞來源）

說明：
券商與外資研究報告本身是付費授權內容，直接爬取 PDF 有法律風險且技術脆弱。
此模組改用「公開新聞報導」作為替代訊號來源 — 財經媒體會公開報導
外資報告的目標價與評等調整，這些新聞是合法可取得的公開資訊。

目前來源：鉅亨網台股新聞公開 API（無需登入）。
架構可擴充：新增來源只需實作 fetch_titles() 並加入 _SOURCES。
"""

import logging
import re
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CNYES_API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"

# 判定為「研報相關」的標題關鍵字：需同時命中主體與動作
_SUBJECT_KEYWORDS = ("外資", "大摩", "小摩", "高盛", "花旗", "瑞銀", "美銀", "摩根", "券商", "研究報告")
_ACTION_KEYWORDS = ("目標價", "評等", "調升", "調降", "喊", "上看", "重申", "首次評")

RESEARCH_TOP_N = 10  # 最多回報 N 支


def fetch_research_signals(universe_df: Optional[pd.DataFrame] = None, pages: int = 2) -> dict:
    """
    掃描近期財經新聞標題，找出被外資/券商報告點名的個股。

    Returns:
        {stock_id: ["研報:外資報告新聞(標題摘要...)"], ...}
        全部失敗時回傳空 dict，不影響主流程。
    """
    titles = _fetch_cnyes_titles(pages=pages)
    if not titles:
        logger.warning("研報信號：無法取得新聞標題")
        return {}

    # 名稱 → 代號對照
    name_to_id = {}
    if universe_df is not None and "stock_name" in universe_df.columns:
        for _, row in universe_df.iterrows():
            name = str(row.get("stock_name", "")).strip()
            if len(name) >= 2:
                name_to_id[name] = str(row["stock_id"])

    signals: dict = {}
    for title in titles:
        if not _is_research_title(title):
            continue
        matched = set()
        for code in re.findall(r"[（(](\d{4,6})(?:-TW)?[）)]", title):
            matched.add(code)
        for code in re.findall(r"\b(\d{4})\b", title):
            matched.add(code)
        for name, sid in name_to_id.items():
            if name in title:
                matched.add(sid)
        for sid in matched:
            if sid not in signals:
                snippet = title[:24] + ("…" if len(title) > 24 else "")
                signals[sid] = [f"研報:外資/券商報告新聞（{snippet}）"]

    if len(signals) > RESEARCH_TOP_N:
        signals = dict(list(signals.items())[:RESEARCH_TOP_N])

    logger.info("研報信號（%d 篇標題）：%d 支", len(titles), len(signals))
    return signals


def _is_research_title(title: str) -> bool:
    """標題需同時包含「研報主體」與「評等/目標價動作」關鍵字。"""
    has_subject = any(k in title for k in _SUBJECT_KEYWORDS)
    has_action = any(k in title for k in _ACTION_KEYWORDS)
    return has_subject and has_action


def _fetch_cnyes_titles(pages: int = 2) -> list:
    """鉅亨網台股新聞公開 API 取得近期新聞標題。"""
    titles = []
    for page in range(1, pages + 1):
        try:
            resp = requests.get(
                _CNYES_API,
                params={"page": page, "limit": 30},
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0", "accept": "application/json"},
            )
            resp.raise_for_status()
            body = resp.json()
            items = (((body or {}).get("items") or {}).get("data")) or []
            for item in items:
                title = item.get("title", "")
                if title:
                    titles.append(str(title))
        except Exception as e:
            logger.debug("鉅亨網新聞第 %d 頁抓取失敗：%s", page, e)
            break
    return titles
