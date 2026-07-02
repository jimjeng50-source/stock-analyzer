"""
screener/historical_eval.py
歷史推薦正確率評估

用真實歷史資料重建過去某日的評分（時間點截斷，不引入未來資料），
選出當時的前 K 名，再對照 N 天後的實際股價計算報酬率與勝率。

用途：
- 系統上線前的追溯評估（例如「90 天前選的前 3 名，60 天後表現如何」）
- 評估結果同時寫入 daily_recommendations（recommend_date=歷史日期），
  讓正確率報告可以累積歷史樣本。

限制（誠實註記）：
- 候選池以「目前」的成交值排行為基準（歷史成交值排行需付費資料），
  存在輕微的存活者偏差。
- 評分使用與線上完全相同的 factors/ + Scorer，資料截止於評估日。
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


class HistoricalEvaluator:
    """歷史時間點推薦評估器。"""

    def __init__(self, universe_size: int = 50, max_workers: int = 2):
        """
        Args:
            universe_size: 評估用候選池大小（預設 50，控制 FinMind 請求量）
            max_workers: 批次評分執行緒數
        """
        self.universe_size = universe_size
        self.max_workers = max_workers

    def evaluate(
        self,
        days_ago: int = 90,
        horizon_days: int = 60,
        top_k: int = 3,
        save_to_db: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        執行歷史評估。

        Args:
            days_ago: 評估日 = 今天 - days_ago
            horizon_days: 持有天數（評估日 + horizon_days 的股價）
            top_k: 取前 K 名
            save_to_db: 是否把歷史推薦寫入 DB（累積正確率樣本）
            progress_callback: fn(str) 進度回報（供 Streamlit 顯示）

        Returns:
            {
                "as_of": str, "horizon_days": int,
                "picks": [{stock_id, stock_name, total_score, entry_price,
                           exit_price, return_pct, win}, ...],
                "avg_return_pct": float, "win_rate": float,
                "benchmark_return_pct": float,  # 大盤同期
                "alpha_pct": float,
                "scored_count": int,
                "error": Optional[str],
            }
        """
        def _report(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        as_of_date = date.today() - timedelta(days=days_ago)
        as_of = as_of_date.isoformat()
        result = {
            "as_of": as_of, "horizon_days": horizon_days,
            "picks": [], "avg_return_pct": None, "win_rate": None,
            "benchmark_return_pct": None, "alpha_pct": None,
            "scored_count": 0, "error": None,
        }

        try:
            # Step 1 — 候選池（目前成交值排行前 N）
            _report(f"建立候選池（成交值前 {self.universe_size} 名）...")
            from screener.universe import UniverseManager
            mgr = UniverseManager()
            universe_df = mgr.get_universe()
            if universe_df.empty:
                result["error"] = "無法取得候選池"
                return result
            universe_df = universe_df.head(self.universe_size)
            name_map = dict(zip(
                universe_df["stock_id"].astype(str),
                universe_df.get("stock_name", universe_df["stock_id"]),
            ))

            # Step 2 — 以歷史時間點評分
            _report(f"以 {as_of} 為截止點批次評分 {len(universe_df)} 支（需數分鐘）...")
            from screener.batch_scorer import BatchScorer
            scorer = BatchScorer(max_workers=self.max_workers, as_of=as_of)
            scored_df = scorer.score_universe(
                universe_df["stock_id"].astype(str).tolist(), show_progress=True
            )
            result["scored_count"] = len(scored_df)
            if scored_df.empty:
                result["error"] = "歷史評分無結果（請確認 FINMIND_TOKEN 已設定）"
                return result

            top_df = scored_df.head(top_k)

            # Step 3 — 取得每支的進場價（評估日收盤）與出場價（+horizon 收盤）
            _report(f"取得 Top {top_k} 的歷史價格與 {horizon_days} 天後價格...")
            picks = []
            for _, row in top_df.iterrows():
                sid = str(row["stock_id"])
                entry, exit_ = self._get_entry_exit_price(sid, as_of_date, horizon_days)
                ret = ((exit_ / entry - 1) * 100) if (entry and exit_) else None
                picks.append({
                    "stock_id": sid,
                    "stock_name": name_map.get(sid, sid),
                    "total_score": round(float(row["total_score"]), 1),
                    "entry_price": entry,
                    "exit_price": exit_,
                    "return_pct": round(ret, 2) if ret is not None else None,
                    "win": (ret > 0) if ret is not None else None,
                })
            result["picks"] = picks

            evaluated = [p for p in picks if p["return_pct"] is not None]
            if evaluated:
                rets = [p["return_pct"] for p in evaluated]
                result["avg_return_pct"] = round(sum(rets) / len(rets), 2)
                result["win_rate"] = round(
                    sum(1 for r in rets if r > 0) / len(rets), 3
                )

            # Step 4 — 大盤基準（加權指數同期報酬）
            bench = self._get_benchmark_return(as_of_date, horizon_days)
            result["benchmark_return_pct"] = bench
            if bench is not None and result["avg_return_pct"] is not None:
                result["alpha_pct"] = round(result["avg_return_pct"] - bench, 2)

            # Step 5 — 寫入 DB 累積樣本
            if save_to_db and picks:
                self._save_to_db(as_of_date, picks, horizon_days)
                _report("已寫入推薦紀錄資料庫")

        except Exception as e:
            logger.error("歷史評估異常：%s", e, exc_info=True)
            result["error"] = str(e)

        return result

    # ── 價格取得 ───────────────────────────────────────────────────────────────

    def _get_entry_exit_price(
        self, stock_id: str, as_of_date: date, horizon_days: int
    ) -> tuple:
        """
        用 yfinance 一次取得進場價（as_of 當日或之後第一個交易日收盤）
        與出場價（as_of + horizon 當日或之前最後一個交易日收盤）。
        yfinance 免費無配額，適合歷史區間查詢。
        """
        if not _HAS_YFINANCE:
            return None, None
        start = as_of_date.strftime("%Y-%m-%d")
        end = (as_of_date + timedelta(days=horizon_days + 10)).strftime("%Y-%m-%d")
        exit_cutoff = pd.Timestamp(as_of_date + timedelta(days=horizon_days))

        for suffix in (".TW", ".TWO"):
            try:
                raw = yf.download(
                    f"{stock_id}{suffix}", start=start, end=end,
                    progress=False, auto_adjust=True,
                )
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                closes = raw["Close"].dropna()
                if closes.empty:
                    continue
                idx = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = idx
                entry = float(closes.iloc[0])
                exit_series = closes[closes.index <= exit_cutoff]
                exit_ = float(exit_series.iloc[-1]) if not exit_series.empty else None
                return round(entry, 2), (round(exit_, 2) if exit_ else None)
            except Exception as e:
                logger.debug("yfinance %s%s 歷史價格失敗：%s", stock_id, suffix, e)
        return None, None

    def _get_benchmark_return(self, as_of_date: date, horizon_days: int) -> Optional[float]:
        """加權指數（^TWII）同期報酬率。"""
        entry, exit_ = self._get_entry_exit_price("^TWII", as_of_date, horizon_days)
        if entry and exit_:
            return round((exit_ / entry - 1) * 100, 2)
        return None

    # ── DB 寫入 ────────────────────────────────────────────────────────────────

    def _save_to_db(self, as_of_date: date, picks: list, horizon_days: int) -> None:
        """把歷史評估結果寫入 daily_recommendations（含 60 日績效回填）。"""
        try:
            from screener.recommendation_db import RecommendationDB
            db = RecommendationDB()
            recs = []
            for rank, p in enumerate(picks, 1):
                recs.append({
                    "rank": rank,
                    "stock_id": p["stock_id"],
                    "stock_name": p["stock_name"],
                    "total_score": p["total_score"],
                    "recommendation": "歷史回溯",
                    "current_price": p["entry_price"],
                    "key_reasons": [f"歷史評估（{as_of_date.isoformat()} 時間點評分）"],
                    "hot_tags": [],
                })
            db.save_recommendations(as_of_date, recs)
            for p in picks:
                if p["exit_price"] is not None and horizon_days >= 60:
                    db.update_performance(
                        p["stock_id"], as_of_date, price_60d=p["exit_price"]
                    )
        except Exception as e:
            logger.warning("歷史評估寫入 DB 失敗：%s", e)


def evaluate_60d_accuracy(db=None, top_k: int = 3) -> dict:
    """
    60 日推薦正確率報告（讀取 DB 中已有 60 日績效的推薦）。

    對每個推薦日取前 top_k 名，統計 60 日報酬與勝率。

    Returns:
        {
            "by_date": DataFrame(recommend_date, stock_id, stock_name,
                                 total_score, return_60d_pct, win),
            "overall": {avg_return_pct, win_rate, evaluated, dates},
        }
    """
    from screener.recommendation_db import RecommendationDB
    db = db or RecommendationDB()

    df = db.get_recent_recommendations(n_days=365)
    if df.empty or "return_60d_pct" not in df.columns:
        return {"by_date": pd.DataFrame(), "overall": {}}

    df = df[df["rank"] <= top_k].copy()
    evaluated = df.dropna(subset=["return_60d_pct"])
    if evaluated.empty:
        return {"by_date": pd.DataFrame(), "overall": {}}

    evaluated["win"] = evaluated["return_60d_pct"] > 0
    overall = {
        "avg_return_pct": round(evaluated["return_60d_pct"].mean(), 2),
        "win_rate": round(evaluated["win"].mean(), 3),
        "evaluated": len(evaluated),
        "dates": evaluated["recommend_date"].nunique(),
    }
    cols = [c for c in ("recommend_date", "rank", "stock_id", "stock_name",
                        "total_score", "current_price", "price_60d",
                        "return_60d_pct", "win", "hot_tags") if c in evaluated.columns]
    return {"by_date": evaluated[cols].reset_index(drop=True), "overall": overall}
