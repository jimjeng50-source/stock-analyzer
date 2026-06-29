"""
screener/filter.py
快速過濾條件（基礎篩選，速度優先）

過濾在 UniverseManager 輸出之後、BatchScorer 評分之前執行。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import (
    FILTER_MIN_MARKET_CAP_BILLION,
    FILTER_MIN_AVG_VOLUME_K,
    FILTER_MIN_PRICE,
    FILTER_MAX_PRICE,
    FILTER_MIN_REVENUE_YOY,
    BATCH_FETCH_DELAY_SEC,
)

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """
    過濾條件設定。
    使用 config.py 中的預設值，允許覆蓋（供測試用）。
    """
    min_market_cap_b: float = FILTER_MIN_MARKET_CAP_BILLION
    min_avg_volume_k: float = FILTER_MIN_AVG_VOLUME_K
    min_price: float = FILTER_MIN_PRICE
    max_price: float = FILTER_MAX_PRICE
    min_revenue_yoy: float = FILTER_MIN_REVENUE_YOY
    require_positive_eps_ttm: bool = True
    exclude_financial_sector: bool = False
    exclude_stock_ids: list = field(default_factory=list)


class QuickFilter:
    """
    快速過濾器

    執行順序（每步均記錄過濾掉多少支）：
    Step 1: 價格範圍過濾
    Step 2: 市值 + 成交量過濾
    Step 3: 手動排除清單
    Step 4: 金融股排除（選用）
    Step 5: 月營收 YoY 過濾
    Step 6: TTM EPS 正負過濾
    """

    def __init__(self, fetcher=None, config: FilterConfig = None):
        self.fetcher = fetcher
        self.config = config or FilterConfig()

    def run(self, universe_df: pd.DataFrame) -> tuple:
        """
        執行完整過濾流程。

        Returns:
            (passed_df, filter_report)
        """
        report = {
            "total_input": len(universe_df),
            "steps": [],
            "total_passed": 0,
            "removed_ids": [],
        }

        removed_ids = set()
        df = universe_df.copy()
        initial_ids = set(df["stock_id"].tolist())

        # Step 1 — 價格範圍
        df = self._filter_price(df)
        self._log_step(report, "價格範圍", universe_df, df)

        # Step 2 — 市值 + 成交量
        df = self._filter_market_cap_volume(df)
        self._log_step(report, "市值/成交量", universe_df, df)

        # Step 3 — 手動排除清單
        if self.config.exclude_stock_ids:
            before = len(df)
            df = df[~df["stock_id"].isin(self.config.exclude_stock_ids)]
            self._log_step(report, "手動排除", universe_df, df)

        # Step 4 — 金融股（選用）
        if self.config.exclude_financial_sector and "industry" in df.columns:
            before = len(df)
            df = df[~df["industry"].str.contains("金融|銀行|保險|證券", na=False)]
            self._log_step(report, "金融股排除", universe_df, df)

        # Step 5 — 月營收 YoY
        df = self._filter_revenue_yoy(df)
        self._log_step(report, "月營收YoY", universe_df, df)

        # Step 6 — TTM EPS
        if self.config.require_positive_eps_ttm:
            df = self._filter_eps(df)
            self._log_step(report, "TTM EPS>0", universe_df, df)

        passed_ids = set(df["stock_id"].tolist())
        report["removed_ids"] = list(initial_ids - passed_ids)
        report["total_passed"] = len(df)

        logger.info(
            "過濾完成：%d → %d（移除 %d 支）",
            report["total_input"], report["total_passed"],
            len(report["removed_ids"]),
        )
        return df.reset_index(drop=True), report

    # ── 各步過濾 ───────────────────────────────────────────────────────────────

    def _filter_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if "last_price" not in df.columns:
            return df
        # 若大多數股票無價格資料（NaN），跳過此過濾避免清空候選池
        has_price_data = df["last_price"].notna().sum() > len(df) * 0.1
        if not has_price_data:
            logger.warning("超過 90%% 個股無價格資料，跳過價格過濾")
            return df
        # NaN（未取得）視為「不確定」→ 保留；只排除確定超出範圍的股票
        mask = (
            df["last_price"].isna() |
            ((df["last_price"] >= self.config.min_price) &
             (df["last_price"] <= self.config.max_price))
        )
        return df[mask]

    def _filter_market_cap_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "market_cap_b" in result.columns:
            has_data = result["market_cap_b"].notna().sum() > len(result) * 0.1
            if has_data:
                result = result[
                    result["market_cap_b"].isna() |
                    (result["market_cap_b"] >= self.config.min_market_cap_b)
                ]
            else:
                logger.warning("市值資料不足，跳過市值過濾")
        if "avg_volume_k" in result.columns:
            has_data = result["avg_volume_k"].notna().sum() > len(result) * 0.1
            if has_data:
                result = result[
                    result["avg_volume_k"].isna() |
                    (result["avg_volume_k"] >= self.config.min_avg_volume_k)
                ]
            else:
                logger.warning("成交量資料不足，跳過成交量過濾")
        return result

    def _filter_revenue_yoy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        批次抓取最新月營收 YoY，排除衰退過深個股。
        若無法取得資料，保守地保留該股（不排除）。
        """
        if df.empty:
            return df

        passed = []
        for _, row in df.iterrows():
            sid = row["stock_id"]
            try:
                yoy = self._get_latest_revenue_yoy(sid)
                if yoy is None or yoy >= self.config.min_revenue_yoy:
                    passed.append(True)
                else:
                    logger.debug("過濾 %s：月營收YoY=%.1f%%（低於門檻）", sid, yoy)
                    passed.append(False)
            except Exception as e:
                logger.debug("取得 %s 月營收失敗，保留：%s", sid, e)
                passed.append(True)
            time.sleep(BATCH_FETCH_DELAY_SEC * 0.3)  # 輕量延遲

        df = df[passed].copy()
        return df

    def _filter_eps(self, df: pd.DataFrame) -> pd.DataFrame:
        """TTM EPS > 0 過濾。"""
        if df.empty:
            return df

        passed = []
        for _, row in df.iterrows():
            sid = row["stock_id"]
            try:
                ttm_eps = self._get_ttm_eps(sid)
                if ttm_eps is None or ttm_eps > 0:
                    passed.append(True)
                else:
                    logger.debug("過濾 %s：TTM EPS=%.2f（虧損）", sid, ttm_eps)
                    passed.append(False)
            except Exception as e:
                logger.debug("取得 %s EPS 失敗，保留：%s", sid, e)
                passed.append(True)
            time.sleep(BATCH_FETCH_DELAY_SEC * 0.3)

        return df[passed].copy()

    # ── 資料取得輔助 ──────────────────────────────────────────────────────────

    def _get_latest_revenue_yoy(self, stock_id: str) -> Optional[float]:
        """取得最新月營收 YoY（%）。"""
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            rev_df = fetcher.get_monthly_revenue(stock_id, months=3)
            if rev_df is None or rev_df.empty:
                return None
            if "revenue_yoy" in rev_df.columns:
                yoy = rev_df["revenue_yoy"].dropna()
                return float(yoy.iloc[-1]) if not yoy.empty else None
        except Exception:
            pass
        return None

    def _get_ttm_eps(self, stock_id: str) -> Optional[float]:
        """取得近四季 EPS 合計（TTM）。"""
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            eps_df = fetcher.get_quarterly_eps(stock_id, n_quarters=4)
            if eps_df is None or len(eps_df) < 4:
                return None
            return float(eps_df["eps"].tail(4).sum())
        except Exception:
            pass
        return None

    # ── 報告輔助 ──────────────────────────────────────────────────────────────

    def _log_step(
        self,
        report: dict,
        step_name: str,
        original_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> None:
        prev_remaining = (
            report["steps"][-1]["remaining"] if report["steps"] else report["total_input"]
        )
        removed = prev_remaining - len(current_df)
        report["steps"].append({
            "step": step_name,
            "removed": removed,
            "remaining": len(current_df),
        })
        if removed > 0:
            logger.info("  [%s] 移除 %d 支，剩 %d 支", step_name, removed, len(current_df))
