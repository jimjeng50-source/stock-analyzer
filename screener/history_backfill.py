"""
screener/history_backfill.py
歷史推薦回補

以真實歷史資料回補指定區間內每個交易日的推薦記錄（前 K 名），
並直接回填 5/20/60 日績效。

效率設計（關鍵）：
- 每支股票的資料只向 API 抓「一次完整區間」（FinMind 個股查詢 + yfinance 股價），
  然後在本地按日切片重算因子 → 30 支股票約 120-150 次請求，免費配額可負擔。
- 績效直接從已下載的股價序列計算，不需額外請求。

時間點正確性：每個交易日的評分只使用該日（含）之前的資料，不引入未來資料。
限制：候選池以「目前」成交值排行為基準（存活者偏差，與歷史回溯評估相同註記）。
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


class HistoryBackfiller:
    """區間歷史推薦回補器。"""

    def __init__(self, universe_size: int = 30, top_k: int = 3):
        self.universe_size = universe_size
        self.top_k = top_k

    def run(
        self,
        start: date,
        end: Optional[date] = None,
        progress_callback=None,
    ) -> dict:
        """
        回補 [start, end] 區間內每個交易日的推薦。

        Returns:
            {"days_done": int, "days_skipped": int, "recs_saved": int,
             "stocks": int, "error": Optional[str]}
        """
        def _report(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        end = end or (date.today() - timedelta(days=1))
        result = {"days_done": 0, "days_skipped": 0, "recs_saved": 0,
                  "stocks": 0, "error": None}

        try:
            from screener.recommendation_db import RecommendationDB
            from factors import (
                compute_chips, compute_technical,
                compute_fundamental, compute_momentum,
            )
            from models.scorer import Scorer
            from config import FACTOR_WEIGHTS

            db = RecommendationDB()
            scorer = Scorer(FACTOR_WEIGHTS)

            # Step 1 — 候選池
            _report(f"建立候選池（成交值前 {self.universe_size} 名）...")
            from screener.universe import UniverseManager
            universe_df = UniverseManager().get_universe()
            if universe_df.empty:
                result["error"] = "無法取得候選池"
                return result
            universe_df = universe_df.head(self.universe_size)
            stock_ids = universe_df["stock_id"].astype(str).tolist()
            name_map = dict(zip(
                universe_df["stock_id"].astype(str),
                universe_df.get("stock_name", universe_df["stock_id"]),
            ))
            result["stocks"] = len(stock_ids)

            # Step 2 — 批次抓價（一次涵蓋整段區間 + 因子回看期 + 績效前瞻期）
            _report(f"批次下載 {len(stock_ids)} 支股價（{start} 前 200 天起）...")
            price_map = self._bulk_price_history(
                stock_ids, start - timedelta(days=200), date.today()
            )
            if not price_map:
                result["error"] = "股價批次下載失敗"
                return result

            # Step 3 — 每支股票抓一次 FinMind 資料（籌碼/營收/財報）
            _report("抓取各股籌碼與基本面資料（每支一次）...")
            fm_map = self._fetch_finmind_once(stock_ids, start, _report)

            # Step 4 — 交易日清單（取自台積電股價日期，最具代表性）
            ref_prices = price_map.get("2330")
            if ref_prices is None:
                ref_prices = next(iter(price_map.values()))
            trading_days = [
                d.date() for d in ref_prices["date"]
                if start <= d.date() <= end
            ]
            _report(f"回補區間 {start} ~ {end}：{len(trading_days)} 個交易日")

            # Step 5 — 按日切片評分 + 儲存 + 績效回填
            for i, day in enumerate(trading_days, 1):
                existing = db.get_recommendations(day)
                if existing:
                    result["days_skipped"] += 1
                    continue

                day_ts = pd.Timestamp(day)
                scored = []
                for sid in stock_ids:
                    row = self._score_as_of(
                        sid, day_ts, price_map, fm_map,
                        compute_chips, compute_technical,
                        compute_fundamental, compute_momentum, scorer,
                    )
                    if row:
                        scored.append(row)

                if not scored:
                    result["days_skipped"] += 1
                    continue

                scored.sort(key=lambda x: x["total_score"], reverse=True)
                top = scored[:self.top_k]

                recs = []
                for rank, s in enumerate(top, 1):
                    recs.append({
                        "rank": rank,
                        "stock_id": s["stock_id"],
                        "stock_name": name_map.get(s["stock_id"], s["stock_id"]),
                        "total_score": round(s["total_score"], 1),
                        "recommendation": "歷史回補",
                        "current_price": s["entry_price"],
                        "key_reasons": [f"歷史回補（{day.isoformat()} 時間點評分）"],
                        "hot_tags": [],
                    })
                db.save_recommendations(day, recs)
                result["recs_saved"] += len(recs)

                # 績效回填（從已下載股價直接取，不需 API）
                for s in top:
                    perf = {}
                    for horizon, key in ((5, "price_5d"), (20, "price_20d"), (60, "price_60d")):
                        px = self._price_at(
                            price_map[s["stock_id"]],
                            day_ts + pd.Timedelta(days=horizon),
                        )
                        if px is not None:
                            perf[key] = px
                    if perf:
                        db.update_performance(s["stock_id"], day, **perf)

                result["days_done"] += 1
                if i % 5 == 0 or i == len(trading_days):
                    _report(f"進度 {i}/{len(trading_days)}（已存 {result['recs_saved']} 筆）")

        except Exception as e:
            logger.error("歷史回補異常：%s", e, exc_info=True)
            result["error"] = str(e)

        return result

    # ── 單日單股評分 ───────────────────────────────────────────────────────────

    @staticmethod
    def _score_as_of(sid, day_ts, price_map, fm_map,
                     compute_chips, compute_technical,
                     compute_fundamental, compute_momentum, scorer) -> Optional[dict]:
        """以 day_ts 為截止點切片資料並評分。資料不足回傳 None。"""
        price_df = price_map.get(sid)
        if price_df is None:
            return None
        p = price_df[price_df["date"] <= day_ts]
        if len(p) < 60:            # 技術/動能因子最少需求
            return None
        entry_price = float(p["close"].iloc[-1])

        fm = fm_map.get(sid, {})

        def _slice(df):
            if df is None or df.empty or "date" not in df.columns:
                return pd.DataFrame()
            return df[df["date"] <= day_ts]

        try:
            chips = compute_chips(_slice(fm.get("institutional")), _slice(fm.get("margin")))
            technical = compute_technical(p)
            fundamental = compute_fundamental(
                _slice(fm.get("revenue")), _slice(fm.get("financial")), entry_price
            )
            momentum = compute_momentum(p)
            score = scorer.score(chips, technical, fundamental, momentum)
            return {
                "stock_id": sid,
                "total_score": float(score["total_score"]),
                "entry_price": round(entry_price, 2),
            }
        except Exception as e:
            logger.debug("評分失敗 %s @%s：%s", sid, day_ts.date(), e)
            return None

    # ── 資料抓取（每支一次）────────────────────────────────────────────────────

    @staticmethod
    def _bulk_price_history(stock_ids: list, start: date, end: date) -> dict:
        """yfinance 批次下載整段 OHLCV。回傳 {stock_id: DataFrame(date/open/high/low/close/volume)}。"""
        if not _HAS_YFINANCE:
            return {}
        tickers = [f"{s}.TW" for s in stock_ids] + [f"{s}.TWO" for s in stock_ids]
        out = {}
        try:
            raw = yf.download(
                tickers, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                progress=False, auto_adjust=True, threads=True, group_by="ticker",
            )
            if raw.empty:
                return {}
            for sid in stock_ids:
                for suffix in (".TW", ".TWO"):
                    tkr = f"{sid}{suffix}"
                    if tkr not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[tkr].dropna(subset=["Close"])
                    if sub.empty:
                        continue
                    df = sub.reset_index()
                    df.columns = [str(c).lower() for c in df.columns]
                    df = df.rename(columns={"index": "date"})
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                    out[sid] = df[["date", "open", "high", "low", "close", "volume"]]
                    break
        except Exception as e:
            logger.warning("股價批次下載失敗：%s", e)
        return out

    @staticmethod
    def _fetch_finmind_once(stock_ids: list, start: date, _report) -> dict:
        """每支股票抓一次完整區間的籌碼/營收/財報。無 token 時回傳空（因子取中性分）。"""
        from config import get_runtime_config
        from data.fetcher import FinMindFetcher

        fm_map = {}
        if not get_runtime_config("FINMIND_TOKEN"):
            _report("⚠️ 無 FINMIND_TOKEN — 籌碼/基本面因子將取中性分（僅價格因子有效）")
            return fm_map

        days_span = (date.today() - start).days + 90
        for i, sid in enumerate(stock_ids, 1):
            try:
                fetcher = FinMindFetcher(sid, days=days_span)
                fm_map[sid] = {
                    "institutional": fetcher.get_institutional(),
                    "margin": fetcher.get_margin_trading(),
                    "revenue": fetcher.get_monthly_revenue(),
                    "financial": fetcher.get_financial_statements(),
                }
            except Exception as e:
                logger.debug("FinMind 抓取失敗 %s：%s", sid, e)
            if i % 10 == 0:
                _report(f"資料抓取進度 {i}/{len(stock_ids)}")
        return fm_map

    @staticmethod
    def _price_at(price_df: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[float]:
        """target 當日或之前最後一個交易日收盤價；target 超過資料範圍回傳 None。"""
        if price_df is None or price_df.empty:
            return None
        if target_ts > price_df["date"].max():
            return None       # 未來日期尚未到，留待排程回填
        sub = price_df[price_df["date"] <= target_ts]
        if sub.empty:
            return None
        return round(float(sub["close"].iloc[-1]), 2)
