"""
screener/batch_scorer.py
批次評分引擎

直接複用現有 factors/ 模組和 models/scorer.py，
對多支股票批次執行評分，回傳排序後的結果表。
"""

import concurrent.futures
import logging
import time
from datetime import datetime

import pandas as pd

from data.fetcher import FinMindFetcher
from factors import compute_chips, compute_technical, compute_fundamental, compute_momentum
from models.scorer import Scorer
from config import FACTOR_WEIGHTS, BATCH_FETCH_DELAY_SEC, BATCH_MAX_WORKERS

logger = logging.getLogger(__name__)


class BatchScorer:
    """
    批次評分器

    執行邏輯：
    1. 逐支（或並行）呼叫 compute_chips / compute_technical /
       compute_fundamental / compute_momentum
    2. 彙整後呼叫 Scorer.score()
    3. 失敗個股記錄 error，不影響其他個股
    4. 回傳排序好的 DataFrame

    速率限制：
    - 每支股票抓完資料後 sleep BATCH_FETCH_DELAY_SEC
    - FinMind 回傳 429 時自動等待 60 秒後重試（最多 3 次）
    """

    def __init__(self, max_workers: int = BATCH_MAX_WORKERS):
        self.max_workers = max_workers
        self.scorer = Scorer(FACTOR_WEIGHTS)
        self.failed_df: pd.DataFrame = pd.DataFrame()

    def score_universe(
        self,
        stock_ids: list,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        對候選清單批次評分。

        Returns DataFrame sorted by total_score desc, with columns:
            stock_id, total_score, recommendation, chips_score,
            fundamental_score, technical_score, momentum_score,
            risk_score, current_price, error, scored_at
        """
        if not stock_ids:
            return pd.DataFrame()

        results = []
        failed = []
        total = len(stock_ids)

        logger.info("開始批次評分：%d 支股票（%d 執行緒）", total, self.max_workers)

        if self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(self._score_single, sid): sid for sid in stock_ids}
                done_count = 0
                for future in concurrent.futures.as_completed(future_map):
                    done_count += 1
                    sid = future_map[future]
                    try:
                        row = future.result()
                    except Exception as e:
                        row = {"stock_id": sid, "error": str(e), "total_score": None}
                    row["scored_at"] = datetime.now()
                    if row.get("error"):
                        failed.append(row)
                    else:
                        results.append(row)
                    if show_progress and done_count % 10 == 0:
                        logger.info("  進度 %d/%d（%.0f%%）", done_count, total, done_count / total * 100)
        else:
            for i, sid in enumerate(stock_ids, 1):
                row = self._score_single(sid)
                row["scored_at"] = datetime.now()
                if row.get("error"):
                    failed.append(row)
                else:
                    results.append(row)
                if show_progress and i % 10 == 0:
                    logger.info("  進度 %d/%d（%.0f%%）", i, total, i / total * 100)

        self.failed_df = pd.DataFrame(failed) if failed else pd.DataFrame()
        if not results:
            logger.warning("批次評分：無成功結果")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.dropna(subset=["total_score"])
        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
        logger.info("批次評分完成：%d 成功，%d 失敗", len(df), len(failed))
        return df

    def _score_single(self, stock_id: str) -> dict:
        """
        單支股票評分（供並行呼叫）。
        不呼叫 Claude API，不輸出 print，所有例外都 catch。
        """
        for attempt in range(3):
            try:
                fetcher = FinMindFetcher(stock_id)
                price_df = fetcher.get_price()
                if price_df is None or price_df.empty:
                    return {"stock_id": stock_id, "error": "無股價資料", "total_score": None}

                institutional_df = fetcher.get_institutional()
                margin_df = fetcher.get_margin_trading()
                revenue_df = fetcher.get_monthly_revenue()
                financial_df = fetcher.get_financial_statements()
                current_price = float(price_df["close"].iloc[-1])

                chips = compute_chips(institutional_df, margin_df)
                technical = compute_technical(price_df)
                fundamental = compute_fundamental(revenue_df, financial_df, current_price)
                momentum = compute_momentum(price_df)

                result = self.scorer.score(chips, technical, fundamental, momentum)

                time.sleep(BATCH_FETCH_DELAY_SEC)

                return {
                    "stock_id": stock_id,
                    "total_score": result["total_score"],
                    "recommendation": result["recommendation"],
                    "chips_score": result["category_scores"].get("chips", 0),
                    "fundamental_score": result["category_scores"].get("fundamental", 0),
                    "technical_score": result["category_scores"].get("technical", 0),
                    "momentum_score": result["category_scores"].get("momentum", 0),
                    "risk_score": result["category_scores"].get("risk", 0),
                    "current_price": current_price,
                    "error": "",
                }

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Too Many Requests" in err_str:
                    wait = 60 * (attempt + 1)
                    logger.warning("FinMind 429 限流，等待 %ds 後重試 %s（第%d次）", wait, stock_id, attempt + 1)
                    time.sleep(wait)
                else:
                    logger.warning("評分失敗 %s: %s", stock_id, e)
                    return {"stock_id": stock_id, "error": err_str, "total_score": None}

        return {"stock_id": stock_id, "error": "重試 3 次仍失敗（API 限流）", "total_score": None}
