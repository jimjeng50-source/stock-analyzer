"""
screener/recommendation_db.py
推薦紀錄資料庫

表結構：
- daily_recommendations：每日推薦清單 + 事後績效欄位
- scan_logs：每次掃描摘要記錄
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import RECOMMENDATION_DB_PATH

logger = logging.getLogger(__name__)

_CREATE_RECOMMENDATIONS = """
CREATE TABLE IF NOT EXISTS daily_recommendations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recommend_date    DATE NOT NULL,
    rank              INTEGER NOT NULL,
    stock_id          TEXT NOT NULL,
    stock_name        TEXT,
    total_score       REAL,
    recommendation    TEXT,
    current_price     REAL,
    reason_1          TEXT,
    reason_2          TEXT,
    reason_3          TEXT,
    risk_warning      TEXT,
    target_price      REAL,
    upside_pct        REAL,
    industry          TEXT,
    chips_score       REAL,
    fundamental_score REAL,
    technical_score   REAL,
    momentum_score    REAL,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price_5d          REAL,
    price_20d         REAL,
    return_5d_pct     REAL,
    return_20d_pct    REAL,
    price_60d         REAL,
    return_60d_pct    REAL,
    hot_tags          TEXT,
    forward_eps       REAL,
    eps_growth_pct    REAL,
    UNIQUE(recommend_date, stock_id)
);
"""

# 既有資料庫的欄位補齊（ALTER TABLE 不支援 IF NOT EXISTS，逐一檢查）
_MIGRATION_COLUMNS = {
    "price_60d": "REAL",
    "return_60d_pct": "REAL",
    "hot_tags": "TEXT",
    "forward_eps": "REAL",
    "eps_growth_pct": "REAL",
}

# 推薦名單「兩個月每日績效」追蹤表：每檔推薦、每個交易日一列
# （day_offset=0 為推薦當日買進價）。維持滾動 2 個月的績效曲線。
_CREATE_DAILY_RETURNS = """
CREATE TABLE IF NOT EXISTS recommendation_daily_returns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    recommend_date DATE NOT NULL,
    stock_id       TEXT NOT NULL,
    stock_name     TEXT,
    as_of_date     DATE NOT NULL,
    day_offset     INTEGER,
    entry_price    REAL,
    close_price    REAL,
    return_pct     REAL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(recommend_date, stock_id, as_of_date)
);
"""

_CREATE_SCAN_LOGS = """
CREATE TABLE IF NOT EXISTS scan_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_date       DATE NOT NULL,
    universe_count  INTEGER,
    after_filter    INTEGER,
    scored_count    INTEGER,
    failed_count    INTEGER,
    duration_sec    REAL,
    top_score       REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class RecommendationDB:
    """推薦紀錄資料庫存取層。"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or RECOMMENDATION_DB_PATH
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """建立資料表（若不存在）。"""
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE_RECOMMENDATIONS)
            conn.execute(_CREATE_SCAN_LOGS)
            conn.execute(_CREATE_DAILY_RETURNS)
            # migration：舊版資料庫補上新欄位
            existing_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(daily_recommendations)")
            }
            for col, col_type in _MIGRATION_COLUMNS.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE daily_recommendations ADD COLUMN {col} {col_type}")
                    logger.info("DB migration：新增欄位 %s", col)
        logger.debug("DB 初始化完成：%s", self.db_path)

    # ── 寫入 ───────────────────────────────────────────────────────────────────

    def save_recommendations(self, recommend_date: date, recommendations: list) -> None:
        """
        儲存當日推薦清單。
        若同日同股已存在則忽略（UNIQUE 約束）。
        """
        if not recommendations:
            return
        date_str = recommend_date.isoformat() if isinstance(recommend_date, date) else str(recommend_date)
        sql = """
        INSERT OR IGNORE INTO daily_recommendations
            (recommend_date, rank, stock_id, stock_name, total_score, recommendation,
             current_price, reason_1, reason_2, reason_3, risk_warning,
             target_price, upside_pct, industry,
             chips_score, fundamental_score, technical_score, momentum_score, hot_tags,
             forward_eps, eps_growth_pct)
        VALUES
            (:recommend_date, :rank, :stock_id, :stock_name, :total_score, :recommendation,
             :current_price, :reason_1, :reason_2, :reason_3, :risk_warning,
             :target_price, :upside_pct, :industry,
             :chips_score, :fundamental_score, :technical_score, :momentum_score, :hot_tags,
             :forward_eps, :eps_growth_pct)
        """
        rows = []
        for rec in recommendations:
            reasons = rec.get("key_reasons", [])
            score_bd = rec.get("score_breakdown", {})
            rows.append({
                "recommend_date": date_str,
                "rank": rec.get("rank", 0),
                "stock_id": rec.get("stock_id", ""),
                "stock_name": rec.get("stock_name", ""),
                "total_score": rec.get("total_score"),
                "recommendation": rec.get("recommendation", ""),
                "current_price": rec.get("current_price"),
                "reason_1": reasons[0] if len(reasons) > 0 else None,
                "reason_2": reasons[1] if len(reasons) > 1 else None,
                "reason_3": reasons[2] if len(reasons) > 2 else None,
                "risk_warning": rec.get("risk_warning"),
                "target_price": rec.get("target_price_base"),
                "upside_pct": rec.get("upside_pct"),
                "industry": rec.get("industry"),
                "chips_score": score_bd.get("chips_score"),
                "fundamental_score": score_bd.get("fundamental_score"),
                "technical_score": score_bd.get("technical_score"),
                "momentum_score": score_bd.get("momentum_score"),
                "hot_tags": ", ".join(rec.get("hot_tags", [])) or None,
                "forward_eps": rec.get("forward_eps"),
                "eps_growth_pct": rec.get("eps_growth_rate"),
            })
        with self._conn() as conn:
            conn.executemany(sql, rows)
        logger.info("儲存 %d 筆推薦紀錄（%s）", len(rows), date_str)

    def save_scan_log(self, scan_date: date, scan_summary: dict) -> None:
        """儲存掃描摘要。"""
        date_str = scan_date.isoformat() if isinstance(scan_date, date) else str(scan_date)
        sql = """
        INSERT INTO scan_logs
            (scan_date, universe_count, after_filter, scored_count, failed_count, duration_sec, top_score)
        VALUES
            (:scan_date, :universe_count, :after_filter, :scored_count, :failed_count, :duration_sec, :top_score)
        """
        with self._conn() as conn:
            conn.execute(sql, {
                "scan_date": date_str,
                "universe_count": scan_summary.get("universe_count"),
                "after_filter": scan_summary.get("after_filter_count"),
                "scored_count": scan_summary.get("scored_count"),
                "failed_count": scan_summary.get("failed_count"),
                "duration_sec": scan_summary.get("scan_duration_sec"),
                "top_score": scan_summary.get("top_score"),
            })

    # ── 讀取 ───────────────────────────────────────────────────────────────────

    def get_recommendations(self, recommend_date: date) -> list:
        """取得指定日期的推薦清單。"""
        date_str = recommend_date.isoformat() if isinstance(recommend_date, date) else str(recommend_date)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_recommendations WHERE recommend_date=? ORDER BY rank",
                (date_str,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_recommendations(self, n_days: int = 30) -> pd.DataFrame:
        """取得近 N 天的推薦紀錄，回傳 DataFrame。"""
        sql = """
        SELECT * FROM daily_recommendations
        WHERE recommend_date >= date('now', :offset)
        ORDER BY recommend_date DESC, rank ASC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, {"offset": f"-{n_days} days"}).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ── 兩個月每日績效追蹤 ─────────────────────────────────────────────────────

    def get_active_tracking_recommendations(self, window_days: int = 60) -> list:
        """
        取得仍在 2 個月追蹤窗內的推薦（去重 recommend_date+stock_id）。
        回傳 [{recommend_date, stock_id, stock_name, entry_price}, ...]。
        """
        sql = """
        SELECT recommend_date, stock_id,
               MAX(stock_name) AS stock_name,
               MAX(current_price) AS entry_price
        FROM daily_recommendations
        WHERE recommend_date >= date('now', :offset)
          AND current_price IS NOT NULL
        GROUP BY recommend_date, stock_id
        ORDER BY recommend_date DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, {"offset": f"-{window_days} days"}).fetchall()
        return [dict(r) for r in rows]

    def save_daily_returns(self, rows: list) -> None:
        """
        批次寫入（upsert）每日績效列。每列需含：
        recommend_date, stock_id, stock_name, as_of_date, day_offset,
        entry_price, close_price, return_pct。
        """
        if not rows:
            return
        sql = """
        INSERT INTO recommendation_daily_returns
            (recommend_date, stock_id, stock_name, as_of_date, day_offset,
             entry_price, close_price, return_pct)
        VALUES
            (:recommend_date, :stock_id, :stock_name, :as_of_date, :day_offset,
             :entry_price, :close_price, :return_pct)
        ON CONFLICT(recommend_date, stock_id, as_of_date) DO UPDATE SET
            close_price=excluded.close_price,
            return_pct=excluded.return_pct,
            day_offset=excluded.day_offset
        """
        with self._conn() as conn:
            conn.executemany(sql, rows)

    def get_daily_returns(self, recommend_date, stock_id: str) -> pd.DataFrame:
        """取得單一推薦的每日績效曲線（依 as_of_date 升序）。"""
        date_str = recommend_date.isoformat() if isinstance(recommend_date, date) else str(recommend_date)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM recommendation_daily_returns "
                "WHERE recommend_date=? AND stock_id=? ORDER BY as_of_date ASC",
                (date_str, stock_id),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def get_tracking_summary(self, window_days: int = 60) -> pd.DataFrame:
        """
        近 window_days 內每檔推薦的績效摘要（供 2 個月名單報告）：
        entry_price、最新收盤/報酬、期間最佳/最差報酬（含最大回落）、追蹤天數。
        """
        sql = """
        WITH latest AS (
            SELECT recommend_date, stock_id, MAX(as_of_date) AS last_date
            FROM recommendation_daily_returns
            WHERE recommend_date >= date('now', :offset)
            GROUP BY recommend_date, stock_id
        )
        SELECT d.recommend_date, d.stock_id,
               MAX(d.stock_name) AS stock_name,
               MAX(d.entry_price) AS entry_price,
               COUNT(*) AS days_tracked,
               MAX(d.return_pct) AS max_return_pct,
               MIN(d.return_pct) AS min_return_pct,
               (SELECT close_price FROM recommendation_daily_returns r
                 WHERE r.recommend_date=d.recommend_date AND r.stock_id=d.stock_id
                 ORDER BY r.as_of_date DESC LIMIT 1) AS last_close,
               (SELECT return_pct FROM recommendation_daily_returns r
                 WHERE r.recommend_date=d.recommend_date AND r.stock_id=d.stock_id
                 ORDER BY r.as_of_date DESC LIMIT 1) AS last_return_pct
        FROM recommendation_daily_returns d
        WHERE d.recommend_date >= date('now', :offset)
        GROUP BY d.recommend_date, d.stock_id
        ORDER BY d.recommend_date DESC, d.stock_id ASC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, {"offset": f"-{window_days} days"}).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def prune_daily_returns(self, keep_days: int = 75) -> int:
        """刪除超過保留窗（預設 75 天，2 個月+緩衝）的每日績效列，維持滾動名單。"""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM recommendation_daily_returns "
                "WHERE recommend_date < date('now', ?)",
                (f"-{keep_days} days",),
            )
            return cur.rowcount

    # ── 績效回填 ───────────────────────────────────────────────────────────────

    def update_performance(
        self,
        stock_id: str,
        recommend_date: date,
        price_5d: Optional[float] = None,
        price_20d: Optional[float] = None,
        price_60d: Optional[float] = None,
    ) -> None:
        """回填 5 / 20 / 60 日後的實際股價，並計算報酬率。只更新有提供的欄位。"""
        date_str = recommend_date.isoformat() if isinstance(recommend_date, date) else str(recommend_date)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT current_price FROM daily_recommendations WHERE recommend_date=? AND stock_id=?",
                (date_str, stock_id),
            ).fetchone()
            if not row:
                return

            entry_price = row["current_price"] or 0

            updates, params = [], []
            for price, price_col, ret_col in (
                (price_5d, "price_5d", "return_5d_pct"),
                (price_20d, "price_20d", "return_20d_pct"),
                (price_60d, "price_60d", "return_60d_pct"),
            ):
                if price is not None:
                    ret = ((price / entry_price - 1) * 100) if entry_price else None
                    updates += [f"{price_col}=?", f"{ret_col}=?"]
                    params += [price, ret]

            if not updates:
                return
            params += [date_str, stock_id]
            conn.execute(
                f"UPDATE daily_recommendations SET {', '.join(updates)} "
                "WHERE recommend_date=? AND stock_id=?",
                params,
            )

    def get_performance_summary(self, n_days: int = 90) -> dict:
        """
        取得近 N 天推薦績效摘要。

        Returns:
            {
                "avg_return_5d": float,
                "avg_return_20d": float,
                "win_rate_5d": float,    # 5 日正報酬勝率 (0-1)
                "win_rate_20d": float,
                "total_recommendations": int,
                "evaluated_count": int,
            }
        """
        sql = """
        SELECT return_5d_pct, return_20d_pct, return_60d_pct
        FROM daily_recommendations
        WHERE recommend_date >= date('now', :offset)
          AND (return_5d_pct IS NOT NULL OR return_60d_pct IS NOT NULL)
        """
        with self._conn() as conn:
            rows = conn.execute(sql, {"offset": f"-{n_days} days"}).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM daily_recommendations WHERE recommend_date >= date('now', :offset)",
                {"offset": f"-{n_days} days"},
            ).fetchone()[0]

        if not rows:
            return {
                "avg_return_5d": None, "avg_return_20d": None, "avg_return_60d": None,
                "win_rate_5d": None, "win_rate_20d": None, "win_rate_60d": None,
                "total_recommendations": total, "evaluated_count": 0,
            }

        ret5 = [r["return_5d_pct"] for r in rows if r["return_5d_pct"] is not None]
        ret20 = [r["return_20d_pct"] for r in rows if r["return_20d_pct"] is not None]
        ret60 = [r["return_60d_pct"] for r in rows if r["return_60d_pct"] is not None]

        return {
            "avg_return_5d": round(sum(ret5) / len(ret5), 2) if ret5 else None,
            "avg_return_20d": round(sum(ret20) / len(ret20), 2) if ret20 else None,
            "avg_return_60d": round(sum(ret60) / len(ret60), 2) if ret60 else None,
            "win_rate_5d": round(sum(1 for r in ret5 if r > 0) / len(ret5), 3) if ret5 else None,
            "win_rate_20d": round(sum(1 for r in ret20 if r > 0) / len(ret20), 3) if ret20 else None,
            "win_rate_60d": round(sum(1 for r in ret60 if r > 0) / len(ret60), 3) if ret60 else None,
            "total_recommendations": total,
            "evaluated_count": max(len(ret5), len(ret60)),
        }
