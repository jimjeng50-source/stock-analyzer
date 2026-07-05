#!/usr/bin/env python
"""
jobs.py — GitHub Actions 排程任務入口（單次執行後結束）

用法：
    python jobs.py scan       # 每日掃描 + Telegram 推播
    python jobs.py risk       # 每日風險警訊（有警訊才推播）
    python jobs.py backfill   # 5/20/60 日績效回填

與 alerts/scheduler.py（常駐排程器）的差異：
這裡每個任務跑一次就結束，適合 GitHub Actions cron 呼叫。
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jobs")


def job_scan() -> int:
    """每日全市場掃描 → 推薦 → Telegram 推播。"""
    from screener.recommender import DailyRecommender
    from alerts.notifier import Notifier

    result = DailyRecommender().run(dry_run=False)
    if result.get("error"):
        logger.error("掃描失敗：%s", result["error"])
        # 失敗也通知，避免默默斷更
        Notifier().send_telegram(f"❌ 每日掃描失敗：{result['error']}")
        return 1

    Notifier().send_telegram(result["message"])
    logger.info("已推播 %d 支推薦", len(result["recommendations"]))
    return 0


def job_risk() -> int:
    """每日風險警訊：有警訊才推播。"""
    from alerts.risk_monitor import RiskMonitor
    from alerts.notifier import Notifier

    monitor = RiskMonitor()
    report = monitor.run_daily()
    if report["has_alerts"]:
        Notifier().send_telegram(monitor.format_message(report))
        n = sum(len(report[k]) for k in ("market", "positions", "revenue", "eps"))
        logger.info("風險警訊已推播（%d 則）", n)
    else:
        logger.info("今日無風險警訊")
    return 0


def job_backfill() -> int:
    """回填 5/20/60 日後實際股價，計算推薦績效。"""
    from datetime import date, timedelta
    from screener.recommendation_db import RecommendationDB
    from data.fetcher import DataFetcher

    db = RecommendationDB()
    fetcher = DataFetcher()
    today = date.today()
    filled = 0

    # 每種 horizon 往回找一週內的推薦日（cron 不保證每天跑，補齊漏網）
    for offset_days, col_label in [(5, "5d"), (20, "20d"), (60, "60d")]:
        for extra in range(7):
            target_date = today - timedelta(days=offset_days + extra)
            for rec in db.get_recommendations(target_date):
                if rec.get(f"price_{col_label}") is not None:
                    continue
                sid = rec["stock_id"]
                try:
                    price = fetcher.get_market_price(sid)
                    if price:
                        db.update_performance(
                            sid, target_date, **{f"price_{col_label}": price}
                        )
                        filled += 1
                except Exception as ex:
                    logger.warning("回填 %s %s 失敗：%s", sid, col_label, ex)

    logger.info("績效回填完成（%d 筆）", filled)
    return 0


def job_backfill_history(start_str: str) -> int:
    """回補指定日期起的歷史推薦（真實歷史資料，時間點截斷）。"""
    from datetime import date as _date
    from screener.history_backfill import HistoryBackfiller

    start = _date.fromisoformat(start_str)
    backfiller = HistoryBackfiller(universe_size=30, top_k=3)
    result = backfiller.run(start=start)

    if result.get("error"):
        logger.error("歷史回補失敗：%s", result["error"])
        return 1
    logger.info(
        "歷史回補完成：%d 個交易日（跳過 %d）、儲存 %d 筆推薦",
        result["days_done"], result["days_skipped"], result["recs_saved"],
    )
    return 0


JOBS = {"scan": job_scan, "risk": job_risk, "backfill": job_backfill}


def main():
    parser = argparse.ArgumentParser(description="排程任務入口")
    parser.add_argument("job", choices=sorted(JOBS.keys()) + ["backfill-history"],
                        help="要執行的任務")
    parser.add_argument("--start", default="2026-06-01",
                        help="backfill-history 起始日（YYYY-MM-DD）")
    args = parser.parse_args()
    if args.job == "backfill-history":
        sys.exit(job_backfill_history(args.start))
    sys.exit(JOBS[args.job]())


if __name__ == "__main__":
    main()
