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
        n = sum(len(report[k]) for k in ("market", "positions", "revenue", "eps", "fundamental"))
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


def job_report_export() -> int:
    """
    產出正確率報告：
    1. reports/accuracy_report.csv（明細）+ reports/accuracy_summary.md（摘要）
       → 由 workflow commit 回 repo（方案 A）
    2. 若設定 GDRIVE_SERVICE_ACCOUNT_JSON + GDRIVE_FOLDER_ID，
       同步上傳到 Google Drive（方案 B）
    """
    import os
    from datetime import date
    from screener.recommendation_db import RecommendationDB
    from screener.historical_eval import evaluate_60d_accuracy

    db = RecommendationDB()
    os.makedirs("reports", exist_ok=True)

    # 明細 CSV：近一年推薦 + 各期報酬
    df = db.get_recent_recommendations(n_days=365)
    csv_path = "reports/accuracy_report.csv"
    if df.empty:
        logger.warning("無推薦紀錄，輸出空報告")
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write("（尚無推薦紀錄）\n")
    else:
        cols = [c for c in (
            "recommend_date", "rank", "stock_id", "stock_name", "total_score",
            "current_price", "forward_eps", "eps_growth_pct",
            "price_5d", "return_5d_pct", "price_20d", "return_20d_pct",
            "price_60d", "return_60d_pct", "hot_tags", "recommendation",
        ) if c in df.columns]
        df[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 摘要 Markdown
    perf = db.get_performance_summary(n_days=365)
    acc = evaluate_60d_accuracy(db, top_k=3)
    md_path = "reports/accuracy_summary.md"

    def _fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "—"

    lines = [
        f"# 推薦正確率報告",
        f"",
        f"產出日期：{date.today().isoformat()}",
        f"",
        f"## 整體績效（近一年全部推薦）",
        f"",
        f"| 指標 | 20 日 | 60 日 |",
        f"|------|------|------|",
        f"| 平均報酬 | {_fmt(perf.get('avg_return_20d'), '%')} | {_fmt(perf.get('avg_return_60d'), '%')} |",
        f"| 勝率 | {_fmt(perf.get('win_rate_20d'))} | {_fmt(perf.get('win_rate_60d'))} |",
        f"",
        f"總推薦數：{perf.get('total_recommendations', 0)}｜已評估：{perf.get('evaluated_count', 0)}",
        f"",
    ]
    if acc.get("overall"):
        o = acc["overall"]
        lines += [
            f"## 前 3 名 60 日正確率（主指標）",
            f"",
            f"- 平均 60 日報酬：{_fmt(o.get('avg_return_pct'), '%')}",
            f"- 正確率（正報酬比例）：{_fmt(o.get('win_rate'))}",
            f"- 樣本數：{o.get('evaluated', 0)}（{o.get('dates', 0)} 個推薦日）",
            f"",
        ]
    lines.append("*由 stock-analyzer 自動產出。僅供研究參考，不構成投資建議。*")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("報告已輸出：%s、%s", csv_path, md_path)

    # 方案 B：Google Drive 上傳（未設定憑證時自動跳過）
    try:
        from utils.gdrive import upload_file, is_configured
        if is_configured():
            upload_file(csv_path, f"accuracy_report.csv")
            upload_file(md_path, f"accuracy_summary.md")
        else:
            logger.info("Google Drive 未設定，僅輸出到 repo reports/")
    except Exception as e:
        logger.warning("Drive 上傳階段異常（報告仍在 reports/）：%s", e)

    return 0


JOBS = {"scan": job_scan, "risk": job_risk, "backfill": job_backfill,
        "report-export": job_report_export}


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
