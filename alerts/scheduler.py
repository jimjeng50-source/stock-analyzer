"""
alerts/scheduler.py
APScheduler 主排程控制器

排程任務：
1. 每月 8 日 09:00 → 掃描「本月預計公布」名單，發送週預告通知
2. 每月 10-15 日 18:30 → 檢查當日新公告，觸發分析並推播
3. 每週五 08:00 → 發送下週重點觀察個股週報
4. 每日 09:00 → 更新產業鏈信號（外資籌碼）

執行方式：
    python -m alerts.scheduler
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_components():
    """延遲匯入，避免啟動時就載入所有依賴。"""
    from data.fetcher import DataFetcher
    from alerts.revenue_calendar import RevenueCalendar
    from alerts.notifier import Notifier
    from config import FINMIND_TOKEN

    fetcher = DataFetcher()
    calendar = RevenueCalendar(fetcher)
    notifier = Notifier()
    return fetcher, calendar, notifier


def run_scheduler():
    """主排程函數。"""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.error("請安裝 apscheduler：pip install apscheduler>=3.10.0")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="Asia/Taipei")

    # ── 任務 1：月初掃描預告（每月 8 日上午 9:00）──────────────────────────
    @scheduler.scheduled_job("cron", day="8", hour="9", minute="0", id="monthly_preview")
    def monthly_preview_alert():
        """發送本月即將公布月營收的個股預告。"""
        try:
            _, calendar, notifier = _get_components()
            upcoming = calendar.get_upcoming_announcements(days_ahead=10)
            if upcoming:
                msg = notifier.format_weekly_preview(upcoming)
                notifier.send_line(msg)
                logger.info("月初預告已推播，共 %d 個股", len(upcoming))
            else:
                logger.info("本月無預計公布個股")
        except Exception as e:
            logger.error("月初預告任務失敗：%s", e)

    # ── 任務 2：月中新公告偵測（每月 10-15 日 18:30）──────────────────────
    @scheduler.scheduled_job("cron", day="10-15", hour="18", minute="30", id="detect_new_revenue")
    def detect_new_revenue():
        """偵測今日新公告並觸發分析。"""
        try:
            _, calendar, notifier = _get_components()
            new_items = calendar.check_new_announcements()
            if not new_items:
                logger.info("今日無新月營收公告")
                return
            for item in new_items:
                msg = notifier.format_revenue_alert(item)
                notifier.send_line(msg)
                logger.info("新公告推播：%s YoY=%s%%", item["stock_id"], item.get("yoy_pct"))
        except Exception as e:
            logger.error("新公告偵測任務失敗：%s", e)

    # ── 任務 3：週五週報（每週五 08:00）────────────────────────────────────
    @scheduler.scheduled_job("cron", day_of_week="fri", hour="8", minute="0", id="weekly_watchlist")
    def weekly_watchlist():
        """週五發送下週觀察重點。"""
        try:
            _, calendar, notifier = _get_components()
            upcoming = calendar.get_upcoming_announcements(days_ahead=7)
            msg = notifier.format_weekly_preview(upcoming)
            notifier.send_line(msg)
            logger.info("週報已推播，共 %d 個股", len(upcoming))
        except Exception as e:
            logger.error("週報任務失敗：%s", e)

    # ── 任務 4：每日籌碼更新（週一至五 09:00）─────────────────────────────
    @scheduler.scheduled_job("cron", day_of_week="mon-fri", hour="9", minute="0", id="daily_chain_update")
    def daily_chain_update():
        """更新產業鏈信號（外資籌碼）。"""
        try:
            from factors.supply_chain import SupplyChainAnalyzer, SUPPLY_CHAIN_MAP
            fetcher, _, notifier = _get_components()
            analyzer = SupplyChainAnalyzer(fetcher)
            for chain_key in SUPPLY_CHAIN_MAP.keys():
                try:
                    result = analyzer.analyze_chain(chain_key)
                    overall = result.get("overall_signal", 0)
                    logger.info("產業鏈信號 %s：%+.2f (%s)", chain_key, overall, result.get("signal_label", ""))
                    # 只在信號明顯時推播
                    if abs(overall) >= 0.5:
                        msg = notifier.format_chain_signal(result)
                        notifier.send_line(msg)
                except Exception as ex:
                    logger.warning("產業鏈 %s 更新失敗：%s", chain_key, ex)
        except Exception as e:
            logger.error("每日籌碼更新任務失敗：%s", e)

    logger.info("排程器啟動（Asia/Taipei），按 Ctrl+C 停止")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("排程器已停止")


if __name__ == "__main__":
    run_scheduler()
