"""
alerts/revenue_calendar.py
月營收公告時程預測與蒐集

台灣法規：每月 10 日前公布（最晚延至 15 日）
"""

import sqlite3
import logging
import time
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from data.fetcher import DataFetcher
from utils.tz import now_tw, today_tw

logger = logging.getLogger(__name__)


class RevenueCalendar:
    """
    功能：
    1. 從 FinMind 抓取歷史月營收公告日期，推算各公司慣用公布日
    2. 生成「未來一週內預計公布」的個股名單
    3. 新月營收公布後，自動觸發分析流程
    """

    LEGAL_DEADLINE_DAY = 10
    MAX_DEADLINE_DAY = 15

    def __init__(
        self,
        fetcher: DataFetcher,
        db_path: str = "data/revenue_tracker.db",
    ):
        self.fetcher = fetcher
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_db()

    # ── 資料庫初始化 ──────────────────────────────────────────────────────────

    def initialize_db(self):
        """建立 SQLite 資料表（若不存在）。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS revenue_announcements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_id TEXT NOT NULL,
                    announce_date DATE NOT NULL,
                    revenue_month TEXT NOT NULL,
                    revenue_amount REAL,
                    yoy_pct REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_id, revenue_month)
                );

                CREATE TABLE IF NOT EXISTS announcement_patterns (
                    stock_id TEXT PRIMARY KEY,
                    avg_announce_day REAL,
                    std_announce_day REAL,
                    sample_months INTEGER,
                    last_updated DATE
                );

                CREATE TABLE IF NOT EXISTS watchlist (
                    stock_id TEXT PRIMARY KEY,
                    stock_name TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # migration：watchlist 加 source 欄位（manual=手動加入 / auto=每日推薦同步）
            cols = {row[1] for row in conn.execute("PRAGMA table_info(watchlist)")}
            if "source" not in cols:
                conn.execute("ALTER TABLE watchlist ADD COLUMN source TEXT DEFAULT 'manual'")
        logger.info("資料庫初始化完成：%s", self.db_path)

    # ── 追蹤清單管理 ─────────────────────────────────────────────────────────

    def add_to_watchlist(self, stock_id: str, stock_name: str = ""):
        """加入追蹤清單。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist(stock_id, stock_name) VALUES (?, ?)",
                (stock_id, stock_name),
            )

    def get_watchlist(self) -> list:
        """取得所有追蹤股票清單。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT stock_id, stock_name, source FROM watchlist ORDER BY stock_id"
            ).fetchall()
        return [
            {"stock_id": r[0], "stock_name": r[1], "source": r[2] or "manual"}
            for r in rows
        ]

    def sync_from_recommendations(self, n_days: int = 60) -> dict:
        """
        以「近 n_days 天每日推薦的個股」同步追蹤清單。

        規則：
        - 推薦股以 source='auto' 加入（已存在的手動股不變）
        - 超出 60 天窗口的 auto 股自動移除（手動加入的保留）

        Returns:
            {"added": int, "removed": int, "total": int}
        """
        try:
            from screener.recommendation_db import RecommendationDB
            rec_df = RecommendationDB().get_recent_recommendations(n_days=n_days)
        except Exception as e:
            logger.warning("讀取推薦紀錄失敗，跳過同步：%s", e)
            return {"added": 0, "removed": 0, "total": len(self.get_watchlist())}

        rec_map = {}
        if not rec_df.empty:
            for _, row in rec_df.iterrows():
                sid = str(row["stock_id"])
                rec_map[sid] = str(row.get("stock_name") or sid)

        added = removed = 0
        with sqlite3.connect(self.db_path) as conn:
            existing = {
                r[0]: (r[1] or "manual")
                for r in conn.execute("SELECT stock_id, source FROM watchlist")
            }
            # 移除窗口外的 auto 股
            for sid, source in existing.items():
                if source == "auto" and sid not in rec_map:
                    conn.execute("DELETE FROM watchlist WHERE stock_id=?", (sid,))
                    removed += 1
            # 加入新推薦股
            for sid, name in rec_map.items():
                if sid not in existing:
                    conn.execute(
                        "INSERT OR IGNORE INTO watchlist(stock_id, stock_name, source) VALUES (?, ?, 'auto')",
                        (sid, name),
                    )
                    added += 1
            total = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]

        if added or removed:
            logger.info("追蹤清單同步：+%d / -%d（共 %d）", added, removed, total)
        return {"added": added, "removed": removed, "total": total}

    # ── 歷史模式更新 ─────────────────────────────────────────────────────────

    def update_historical_patterns(self, stock_ids: list = None):
        """
        批次更新歷史公告日期模式。
        從 FinMind 抓取近 12 個月公告記錄，計算各公司慣用公布日。
        """
        if stock_ids is None:
            stock_ids = [s["stock_id"] for s in self.get_watchlist()]

        if not stock_ids:
            logger.info("追蹤清單為空，跳過更新")
            return

        for stock_id in stock_ids:
            try:
                self._update_stock_pattern(stock_id)
                time.sleep(0.5)
            except Exception as e:
                logger.warning("更新 %s 歷史模式失敗：%s", stock_id, e)

    def _update_stock_pattern(self, stock_id: str):
        """更新單一股票的公告日期模式。"""
        rev_df = self.fetcher.get_monthly_revenue(stock_id, months=14)
        if rev_df is None or rev_df.empty:
            return

        # FinMind 月營收的 date 欄位為公告日期
        if "date" not in rev_df.columns:
            return

        announce_days = rev_df["date"].dt.day.tolist()
        if len(announce_days) < 3:
            return

        avg_day = float(sum(announce_days) / len(announce_days))
        std_day = float(pd.Series(announce_days).std())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO announcement_patterns
                    (stock_id, avg_announce_day, std_announce_day, sample_months, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (stock_id, avg_day, std_day, len(announce_days), str(today_tw())),
            )

            # 同步更新公告記錄
            for _, row in rev_df.iterrows():
                try:
                    revenue_month = row["date"].strftime("%Y%m")
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO revenue_announcements
                            (stock_id, announce_date, revenue_month, revenue_amount, yoy_pct)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            stock_id,
                            str(row["date"].date()),
                            revenue_month,
                            float(row.get("revenue", 0)),
                            float(row.get("revenue_yoy", 0)) if pd.notna(row.get("revenue_yoy")) else None,
                        ),
                    )
                except Exception:
                    pass

    # ── 預測即將公布 ─────────────────────────────────────────────────────────

    def get_upcoming_announcements(self, days_ahead: int = 7) -> list:
        """
        預測未來 N 天內即將公布月營收的個股。

        Returns: list of dicts，依預期公布日期排序。
        """
        today = today_tw()
        target_end = today + timedelta(days=days_ahead)
        results = []

        with sqlite3.connect(self.db_path) as conn:
            patterns = conn.execute(
                "SELECT stock_id, avg_announce_day, std_announce_day, sample_months FROM announcement_patterns"
            ).fetchall()

            for stock_id, avg_day, std_day, samples in patterns:
                if avg_day is None:
                    continue

                # 估計本月預計公布日
                this_month_expected = date(today.year, today.month, min(int(avg_day), 28))
                # 如果本月已過，看下個月
                if this_month_expected < today:
                    next_month = today.replace(day=1) + timedelta(days=32)
                    this_month_expected = date(next_month.year, next_month.month, min(int(avg_day), 28))

                if today <= this_month_expected <= target_end:
                    if std_day and std_day < 1:
                        confidence = "high"
                    elif std_day and std_day < 2:
                        confidence = "medium"
                    else:
                        confidence = "low"

                    # 取上次公告資訊
                    last_row = conn.execute(
                        """SELECT yoy_pct, announce_date FROM revenue_announcements
                           WHERE stock_id = ? ORDER BY announce_date DESC LIMIT 1""",
                        (stock_id,),
                    ).fetchone()

                    stock_name = (conn.execute(
                        "SELECT stock_name FROM watchlist WHERE stock_id = ?", (stock_id,)
                    ).fetchone() or ("",))[0]

                    results.append({
                        "stock_id": stock_id,
                        "stock_name": stock_name,
                        "expected_date": this_month_expected,
                        "confidence": confidence,
                        "last_revenue_yoy": float(last_row[0]) if last_row and last_row[0] else None,
                        "last_announce_date": last_row[1] if last_row else None,
                    })

        results.sort(key=lambda x: x["expected_date"])
        return results

    # ── 新公告偵測 ────────────────────────────────────────────────────────────

    def check_new_announcements(self) -> list:
        """
        檢查今日是否有新公布的月營收。

        Returns: 新公告個股清單。
        """
        watchlist = self.get_watchlist()
        if not watchlist:
            return []

        today = today_tw()
        new_announcements = []

        for item in watchlist:
            stock_id = item["stock_id"]
            try:
                rev_df = self.fetcher.get_monthly_revenue(stock_id, months=2)
                if rev_df is None or rev_df.empty:
                    time.sleep(0.3)
                    continue

                latest = rev_df.iloc[-1]
                announce_date = latest["date"].date()

                if announce_date != today:
                    time.sleep(0.3)
                    continue

                revenue_month = latest["date"].strftime("%Y%m")

                # 檢查是否已在資料庫中
                with sqlite3.connect(self.db_path) as conn:
                    existing = conn.execute(
                        "SELECT id FROM revenue_announcements WHERE stock_id = ? AND revenue_month = ?",
                        (stock_id, revenue_month),
                    ).fetchone()

                if existing:
                    time.sleep(0.3)
                    continue

                # 新公告！更新資料庫
                yoy = float(latest["revenue_yoy"]) if pd.notna(latest.get("revenue_yoy")) else None

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO revenue_announcements
                            (stock_id, announce_date, revenue_month, revenue_amount, yoy_pct)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (stock_id, str(announce_date), revenue_month,
                         float(latest.get("revenue", 0)), yoy),
                    )

                new_announcements.append({
                    "stock_id": stock_id,
                    "stock_name": item["stock_name"],
                    "announce_date": announce_date,
                    "revenue_month": revenue_month,
                    "revenue_amount": float(latest.get("revenue", 0)),
                    "yoy_pct": yoy,
                })
                logger.info("新月營收公告：%s（%s）YoY=%s%%", stock_id, revenue_month, yoy)

            except Exception as e:
                logger.warning("檢查 %s 新公告失敗：%s", stock_id, e)
            time.sleep(0.3)

        return new_announcements

    def get_recent_announcements(self, months: int = 3) -> pd.DataFrame:
        """
        取得近 N 個月的已公布月營收，依 YoY 排序。

        Returns: DataFrame 含 stock_id, revenue_month, revenue_amount, yoy_pct
        """
        cutoff = today_tw() - timedelta(days=months * 32)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT a.stock_id, w.stock_name, a.revenue_month, a.revenue_amount, a.yoy_pct, a.announce_date
                FROM revenue_announcements a
                LEFT JOIN watchlist w ON a.stock_id = w.stock_id
                WHERE a.announce_date >= ?
                ORDER BY a.yoy_pct DESC
                """,
                (str(cutoff),),
            ).fetchall()

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=[
            "stock_id", "stock_name", "revenue_month", "revenue_amount", "yoy_pct", "announce_date"
        ])
