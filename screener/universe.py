"""
screener/universe.py
候選股票池管理

來源優先順序：
1. TWSE + TPEX 全市場上市股票清單（FinMind TaiwanStockInfo）
2. 依市值、成交量排序，取前 N 支
3. 可附加自訂 watchlist（從 .env WATCHLIST_CUSTOM 讀取）
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import (
    SCREENER_UNIVERSE_SIZE,
    FILTER_MIN_MARKET_CAP_BILLION,
    FILTER_MIN_AVG_VOLUME_K,
    FILTER_MIN_PRICE,
    FILTER_MAX_PRICE,
    FILTER_EXCLUDE_ETF,
    DEFAULT_DAYS,
    get_runtime_config,
)

logger = logging.getLogger(__name__)

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


class UniverseManager:
    """
    候選股票池管理器

    快取策略：
    - 股票清單每日快取一次，存在 data/universe_cache.json
    - 快取有效期：24 小時
    """

    CACHE_PATH = "data/universe_cache.json"

    def __init__(self, fetcher=None):
        self.fetcher = fetcher

    # ── 公開 API ───────────────────────────────────────────────────────────────

    def get_universe(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        取得候選股票清單。

        Returns DataFrame with columns:
            stock_id, stock_name, market, industry,
            market_cap_b, avg_volume_k, last_price
        排序：market_cap_b 由大到小，取前 SCREENER_UNIVERSE_SIZE 支
        """
        if not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                logger.info("使用快取股票池（%d 支）", len(cached))
                return cached

        logger.info("重新取得全市場股票清單...")
        df = self._fetch_stock_info()
        if df.empty:
            logger.warning("無法取得股票清單，回傳空 DataFrame")
            return df

        df = self._apply_basic_filters(df)
        df = self._enrich_price_volume(df)

        # 防呆（v4.1）：若快照抓取失敗（空表或股價全為 0），
        # 明確回傳空表且「不寫入快取」，避免壞資料污染 24 小時
        if df.empty or (df["last_price"] <= 0).all():
            logger.error("候選池建立失敗：無有效股價資料（請檢查 FINMIND_TOKEN 或稍後重試）")
            return pd.DataFrame()

        df = df.sort_values("market_cap_b", ascending=False).head(SCREENER_UNIVERSE_SIZE)
        df = df.reset_index(drop=True)

        self._save_cache(df)
        logger.info("股票池已更新：%d 支", len(df))
        return df

    def get_custom_watchlist(self) -> list:
        """從 .env 讀取自訂觀察名單（WATCHLIST_CUSTOM）。"""
        raw = os.getenv("WATCHLIST_CUSTOM", "")
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    def merge_with_custom(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """
        將自訂觀察名單合併進候選池（若不在其中則追加）。
        自訂名單不受市值/流動性篩選限制。
        """
        custom_ids = self.get_custom_watchlist()
        if not custom_ids:
            return universe_df

        existing = set(universe_df["stock_id"].tolist())
        missing = [sid for sid in custom_ids if sid not in existing]
        if not missing:
            return universe_df

        extra_rows = []
        for sid in missing:
            price = self._fetch_latest_price(sid)
            extra_rows.append({
                "stock_id": sid,
                "stock_name": sid,
                "market": "CUSTOM",
                "industry": "自訂",
                "market_cap_b": 0.0,
                "avg_volume_k": 0.0,
                "last_price": price,
            })

        extra_df = pd.DataFrame(extra_rows)
        merged = pd.concat([universe_df, extra_df], ignore_index=True)
        logger.info("自訂名單追加 %d 支（共 %d 支）", len(missing), len(merged))
        return merged

    # ── 內部方法 ───────────────────────────────────────────────────────────────

    def _fetch_stock_info(self) -> pd.DataFrame:
        """從 FinMind TaiwanStockInfo 取得全市場股票基本資訊。"""
        try:
            params = {
                "dataset": "TaiwanStockInfo",
                "token": get_runtime_config("FINMIND_TOKEN"),
            }
            resp = requests.get(_FINMIND_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != 200 or not data.get("data"):
                logger.warning("TaiwanStockInfo 回應異常：%s", data.get("msg", ""))
                return pd.DataFrame()

            df = pd.DataFrame(data["data"])
            # 欄位映射：FinMind 欄位名稱
            rename_map = {
                "stock_id": "stock_id",
                "stock_name": "stock_name",
                "type": "market",
                "industry_category": "industry",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # 確保必要欄位存在
            for col in ["stock_id", "stock_name", "market", "industry"]:
                if col not in df.columns:
                    df[col] = ""

            # 排除非股票標的（ETF、特殊標的）
            if FILTER_EXCLUDE_ETF and "market" in df.columns:
                df = df[~df["market"].str.contains("ETF|指數|債券|期貨", na=False)]

            # 股票代號長度過濾：台股一般為 4-6 碼
            df["stock_id"] = df["stock_id"].astype(str).str.strip()
            df = df[df["stock_id"].str.match(r"^\d{4,6}$")]

            # 初始化市值和成交量欄位
            df["market_cap_b"] = 0.0
            df["avg_volume_k"] = 0.0
            df["last_price"] = 0.0

            return df.drop_duplicates("stock_id").reset_index(drop=True)

        except Exception as e:
            logger.error("取得股票清單失敗：%s", e)
            return pd.DataFrame()

    def _apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """排除明顯不符合條件的股票（不需 API 的過濾）。"""
        # 已在 _fetch_stock_info 中排除 ETF 和非標準代號
        return df

    def _enrich_price_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批次取得各股最新收盤價與當日成交量/成交值。

        修正說明（v4.1）：
        - 原版對 TaiwanStockPrice 發出「全市場 × 30 天」的範圍查詢，
          FinMind 免費方案不支援此類大範圍批次查詢，必定失敗，
          導致 last_price 全為 0 → QuickFilter 把所有股票過濾光 →
          「過濾後無候選股票」。
        - 改為「單一交易日快照」查詢（start_date == end_date，不帶 data_id），
          一次請求即可取得全市場當日收盤價/成交量/成交值。
          從今天起往回最多找 10 天，遇到假日自動跳過。
        """
        price_df = self._fetch_market_snapshot(max_lookback_days=10)

        if price_df is None or price_df.empty:
            logger.error(
                "FinMind 全市場快照抓取失敗（可能是 FINMIND_TOKEN 未設定或配額不足），"
                "無法建立候選池"
            )
            # 回傳空 DataFrame 讓上游明確失敗，而不是帶著全 0 股價繼續跑
            return pd.DataFrame(columns=df.columns)

        try:
            price_df["stock_id"] = price_df["stock_id"].astype(str)

            summary = (
                price_df.groupby("stock_id")
                .agg(
                    last_price=("close", "last"),
                    avg_volume_k=("Trading_Volume", lambda x: float(x.iloc[-1]) / 1000),
                    turnover_b=("Trading_money", lambda x: float(x.iloc[-1]) / 1e8),
                )
                .reset_index()
            )

            # 合併回主表
            df = df.merge(
                summary[["stock_id", "last_price", "avg_volume_k", "turnover_b"]],
                on="stock_id",
                how="left",
                suffixes=("_old", ""),
            )

            # 清理合併後的重複欄位
            if "last_price_old" in df.columns:
                df["last_price"] = df["last_price"].fillna(df["last_price_old"])
                df = df.drop(columns=["last_price_old"])
            if "avg_volume_k_old" in df.columns:
                df["avg_volume_k"] = df["avg_volume_k"].fillna(df["avg_volume_k_old"])
                df = df.drop(columns=["avg_volume_k_old"])

            df["last_price"] = pd.to_numeric(df.get("last_price", 0), errors="coerce").fillna(0)
            df["avg_volume_k"] = pd.to_numeric(df.get("avg_volume_k", 0), errors="coerce").fillna(0)
            df["turnover_b"] = pd.to_numeric(df.get("turnover_b", 0), errors="coerce").fillna(0)

            # 修正說明（v4.1）：
            # 原版 market_cap_b = last_price（把「股價」當「市值」），
            # 排序時會選出「最貴的股票」而非「最大的公司」。
            # 準確市值需要股本資料（需額外 API），改以「當日成交值（億元）」
            # 作為規模/流動性代理 — 成交值大的股票同時滿足規模與流動性需求，
            # 對選股候選池而言是比股價更合理的排序依據。
            df["market_cap_b"] = df["turnover_b"]

            # 基本價格過濾
            df = df[
                (df["last_price"] >= FILTER_MIN_PRICE) &
                (df["last_price"] <= FILTER_MAX_PRICE)
            ]
            df = df[df["avg_volume_k"] >= FILTER_MIN_AVG_VOLUME_K]

            return df.reset_index(drop=True)

        except Exception as e:
            logger.warning("批次取得股價失敗：%s，使用原始清單", e)
            return df

    def _fetch_market_snapshot(self, max_lookback_days: int = 10) -> Optional[pd.DataFrame]:
        """
        取得最近一個交易日的全市場股價快照（單日、不帶 data_id）。

        FinMind 的 TaiwanStockPrice 支援「單一日期、全市場」查詢，
        這是免費方案可用的批次方式（範圍查詢則不支援）。
        從今天往回走，最多嘗試 max_lookback_days 天（跳過假日）。

        Returns:
            DataFrame（欄位含 stock_id, close, Trading_Volume, Trading_money）
            或 None（全部失敗時）
        """
        from datetime import date as _date

        for offset in range(max_lookback_days):
            day = (_date.today() - timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                resp = requests.get(
                    _FINMIND_API,
                    params={
                        "dataset": "TaiwanStockPrice",
                        "start_date": day,
                        "end_date": day,
                        "token": get_runtime_config("FINMIND_TOKEN"),
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == 200 and data.get("data"):
                    snap = pd.DataFrame(data["data"])
                    # 假日/尚未收盤時 data 可能為空 list → 繼續往前找
                    if not snap.empty and "close" in snap.columns:
                        logger.info("全市場快照日期：%s（%d 筆）", day, len(snap))
                        return snap
                else:
                    msg = str(data.get("msg", ""))
                    # 配額不足或權限問題直接中止，不需要再試更早的日期
                    if "quota" in msg.lower() or "permission" in msg.lower() or "402" in msg:
                        logger.error("FinMind 回應：%s（配額/權限問題）", msg)
                        return None
            except Exception as e:
                logger.warning("快照 %s 抓取失敗：%s", day, e)
            time.sleep(0.5)

        return None

    def _fetch_latest_price(self, stock_id: str) -> float:
        """取得單支股票最新收盤價（自訂名單補充用）。"""
        try:
            from datetime import date
            end = date.today().strftime("%Y-%m-%d")
            start = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
            params = {
                "dataset": "TaiwanStockPrice",
                "data_id": stock_id,
                "start_date": start,
                "end_date": end,
                "token": get_runtime_config("FINMIND_TOKEN"),
            }
            resp = requests.get(_FINMIND_API, params=params, timeout=15)
            data = resp.json()
            if data.get("status") == 200 and data.get("data"):
                return float(data["data"][-1].get("close", 0))
        except Exception:
            pass
        return 0.0

    # ── 快取 ───────────────────────────────────────────────────────────────────

    def _load_cache(self) -> Optional[pd.DataFrame]:
        """讀取快取，若已過期（>24h）或不存在回傳 None。"""
        try:
            if not os.path.exists(self.CACHE_PATH):
                return None
            with open(self.CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            cached_at = datetime.fromisoformat(cache["cached_at"])
            if datetime.now() - cached_at > timedelta(hours=24):
                return None
            df = pd.DataFrame(cache["data"])
            # 防呆（v4.1）：舊版 bug 可能快取了股價全 0 的壞資料，主動作廢
            if df.empty or "last_price" not in df.columns:
                return None
            if pd.to_numeric(df["last_price"], errors="coerce").fillna(0).le(0).all():
                logger.warning("快取內容無效（股價全為 0），作廢並重新抓取")
                return None
            return df
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        """將股票池快取為 JSON。"""
        try:
            os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
            cache = {
                "cached_at": datetime.now().isoformat(),
                "data": df.to_dict(orient="records"),
            }
            with open(self.CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("快取儲存失敗：%s", e)
