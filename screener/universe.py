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
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import (
    FINMIND_TOKEN,
    SCREENER_UNIVERSE_SIZE,
    FILTER_MIN_MARKET_CAP_BILLION,
    FILTER_MIN_AVG_VOLUME_K,
    FILTER_MIN_PRICE,
    FILTER_MAX_PRICE,
    FILTER_EXCLUDE_ETF,
    DEFAULT_DAYS,
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
                "token": FINMIND_TOKEN or "",
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

            # 初始化市值和成交量欄位（NaN 代表「未取得」，與真正的 0 不同）
            import numpy as np
            df["market_cap_b"] = np.nan
            df["avg_volume_k"] = np.nan
            df["last_price"] = np.nan

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
        批次取得各股最新收盤價和近 20 日均量，用以估算市值。
        以取樣方式（不超過 SCREENER_UNIVERSE_SIZE 支）抓取，降低 API 負擔。
        """
        try:
            from datetime import date
            end = date.today().strftime("%Y-%m-%d")
            start = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")

            # 使用 TaiwanStockPrice 批次取得近期股價
            params = {
                "dataset": "TaiwanStockPrice",
                "start_date": start,
                "end_date": end,
                "token": FINMIND_TOKEN or "",
            }
            resp = requests.get(_FINMIND_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != 200 or not data.get("data"):
                logger.warning("TaiwanStockPrice 批次抓取失敗，使用預設值")
                return df

            price_df = pd.DataFrame(data["data"])
            price_df["stock_id"] = price_df["stock_id"].astype(str)

            # 計算每支股票的最新收盤價和近 20 日均量
            summary = (
                price_df.sort_values("date")
                .groupby("stock_id")
                .agg(
                    last_price=("close", "last"),
                    avg_volume_k=("Trading_Volume", lambda x: x.tail(20).mean() / 1000),
                )
                .reset_index()
            )

            # 合併回主表
            df = df.merge(
                summary[["stock_id", "last_price", "avg_volume_k"]],
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

            # 以股價作為市值代理（無法取得準確股本時）
            # 粗估：market_cap_b ≈ last_price × 1e7 / 1e8（假設股本約 1 億股）
            df["market_cap_b"] = df["last_price"] * 1.0

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
                "token": FINMIND_TOKEN or "",
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
            return pd.DataFrame(cache["data"])
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
