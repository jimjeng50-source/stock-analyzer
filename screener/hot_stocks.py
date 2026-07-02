"""
screener/hot_stocks.py
熱門個股偵測器

三個面向（每支熱門股註記判斷來源）：
1. 籌碼面 — TWSE T86 三大法人買賣超（官方 OpenAPI，免費）
2. 社群面 — PTT 股板近期文章標題討論熱度
   （FB/IG 無公開 API 可查話題熱度，以 PTT 作為台股社群討論的替代來源）
3. 量能面 — 當日成交值排行（來自候選池快照）

輸出格式：{stock_id: ["籌碼:三大法人買超前十", "社群:PTT熱議(5篇)", ...]}
"""

import logging
import re
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_T86_API = "https://www.twse.com.tw/rwd/zh/fund/T86"
_PTT_STOCK_URL = "https://www.ptt.cc/bbs/Stock/index{page}.html"

HOT_CHIPS_TOP_N = 10       # 法人買超取前 N 名
HOT_SOCIAL_TOP_N = 10      # PTT 討論取前 N 名
HOT_SOCIAL_MIN_MENTIONS = 2  # 至少被提及 N 篇才算熱議
HOT_VOLUME_TOP_N = 10      # 成交值取前 N 名
_PTT_PAGES = 5             # 爬 PTT 最新 N 頁（每頁約 20 篇）


class HotStockDetector:
    """熱門個股偵測器。全部來源失敗時回傳空 dict，不影響主流程。"""

    def detect_all(self, universe_df: pd.DataFrame = None) -> dict:
        """
        執行全部三個面向的偵測，合併結果。

        Args:
            universe_df: 候選池（需含 stock_id；量能/社群名稱比對會用到
                         market_cap_b 與 stock_name 欄位）

        Returns:
            {stock_id: [tag, ...]}
        """
        hot: dict = {}

        for tags in (
            self.detect_chips_hot(),
            self.detect_social_hot(universe_df),
            self.detect_volume_hot(universe_df),
        ):
            for sid, tag_list in tags.items():
                hot.setdefault(sid, []).extend(tag_list)

        if hot:
            logger.info("熱門股偵測：%d 支（%s）", len(hot),
                        ", ".join(list(hot.keys())[:10]))
        return hot

    # ── 面向 1：籌碼面（三大法人買超）────────────────────────────────────────

    def detect_chips_hot(self, max_lookback_days: int = 7) -> dict:
        """
        從 TWSE T86 取得最近交易日三大法人買賣超，
        依買超股數排序取前 N 名。
        """
        for offset in range(max_lookback_days):
            day = (date.today() - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                resp = requests.get(
                    _T86_API,
                    params={"date": day, "selectType": "ALLBUT0999", "response": "json"},
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("stat") != "OK" or not data.get("data"):
                    continue

                fields = data.get("fields", [])
                try:
                    code_idx = fields.index("證券代號")
                    net_idx = next(
                        i for i, f in enumerate(fields) if "三大法人買賣超" in f
                    )
                except (ValueError, StopIteration):
                    logger.warning("T86 欄位格式改變：%s", fields[:5])
                    return {}

                rows = []
                for row in data["data"]:
                    sid = str(row[code_idx]).strip()
                    if not re.match(r"^\d{4,6}$", sid):
                        continue
                    try:
                        net = float(str(row[net_idx]).replace(",", ""))
                    except (ValueError, TypeError):
                        continue
                    rows.append((sid, net))

                if not rows:
                    continue

                rows.sort(key=lambda x: x[1], reverse=True)
                result = {}
                for sid, net in rows[:HOT_CHIPS_TOP_N]:
                    if net <= 0:
                        break
                    result[sid] = [f"籌碼:法人買超前十({net/1000:,.0f}張)"]
                logger.info("籌碼熱門（%s）：%d 支", day, len(result))
                return result

            except Exception as e:
                logger.debug("T86 %s 抓取失敗：%s", day, e)
        logger.warning("籌碼熱門偵測失敗（T86 無資料）")
        return {}

    # ── 面向 2：社群面（PTT 股板討論熱度）────────────────────────────────────

    def detect_social_hot(self, universe_df: pd.DataFrame = None) -> dict:
        """
        爬取 PTT 股板最新 N 頁文章標題，統計個股被提及次數。
        比對規則：4-6 碼數字代號，或候選池中的股票名稱。
        """
        titles = self._fetch_ptt_titles()
        if not titles:
            logger.warning("社群熱門偵測失敗（PTT 無法取得）")
            return {}

        # 名稱 → 代號對照（來自候選池）
        name_to_id = {}
        if universe_df is not None and "stock_name" in universe_df.columns:
            for _, row in universe_df.iterrows():
                name = str(row.get("stock_name", "")).strip()
                if len(name) >= 2:  # 避免單字誤判
                    name_to_id[name] = str(row["stock_id"])

        mentions: dict = {}
        for title in titles:
            seen_in_title = set()
            # 代號比對
            for code in re.findall(r"\b(\d{4,6})\b", title):
                seen_in_title.add(code)
            # 名稱比對
            for name, sid in name_to_id.items():
                if name in title:
                    seen_in_title.add(sid)
            for sid in seen_in_title:
                mentions[sid] = mentions.get(sid, 0) + 1

        ranked = sorted(mentions.items(), key=lambda x: x[1], reverse=True)
        result = {
            sid: [f"社群:PTT熱議({cnt}篇)"]
            for sid, cnt in ranked[:HOT_SOCIAL_TOP_N]
            if cnt >= HOT_SOCIAL_MIN_MENTIONS
        }
        logger.info("社群熱門（PTT %d 篇標題）：%d 支", len(titles), len(result))
        return result

    def _fetch_ptt_titles(self) -> list:
        """取得 PTT 股板最新數頁的文章標題。"""
        titles = []
        try:
            session = requests.Session()
            session.cookies.set("over18", "1")
            headers = {"User-Agent": "Mozilla/5.0"}

            # 先抓 index.html 找出最新頁碼
            resp = session.get(_PTT_STOCK_URL.format(page=""), timeout=20, headers=headers)
            resp.raise_for_status()
            html = resp.text
            titles.extend(self._parse_ptt_titles(html))

            m = re.search(r'href="/bbs/Stock/index(\d+)\.html"', html)
            if m:
                latest = int(m.group(1))
                for page in range(latest, latest - _PTT_PAGES + 1, -1):
                    if page <= 0:
                        break
                    r = session.get(_PTT_STOCK_URL.format(page=page), timeout=20, headers=headers)
                    if r.status_code == 200:
                        titles.extend(self._parse_ptt_titles(r.text))
        except Exception as e:
            logger.debug("PTT 爬取失敗：%s", e)
        return titles

    @staticmethod
    def _parse_ptt_titles(html: str) -> list:
        """從 PTT 頁面 HTML 解析文章標題。"""
        return re.findall(
            r'<div class="title">\s*<a href="[^"]*">([^<]+)</a>', html
        )

    # ── 面向 3：量能面（成交值排行）───────────────────────────────────────────

    def detect_volume_hot(self, universe_df: pd.DataFrame = None) -> dict:
        """
        候選池內當日成交值前 N 名。
        （universe 的 market_cap_b 欄位即當日成交值（億元）代理）
        """
        if universe_df is None or universe_df.empty:
            return {}
        if "market_cap_b" not in universe_df.columns:
            return {}

        df = universe_df.dropna(subset=["market_cap_b"])
        df = df[df["market_cap_b"] > 0]
        if df.empty:
            return {}

        top = df.nlargest(HOT_VOLUME_TOP_N, "market_cap_b")
        result = {
            str(row["stock_id"]): [f"量能:成交值前十({row['market_cap_b']:.1f}億)"]
            for _, row in top.iterrows()
        }
        logger.info("量能熱門：%d 支", len(result))
        return result
