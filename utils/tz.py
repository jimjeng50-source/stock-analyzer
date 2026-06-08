"""台灣時區工具（UTC+8）。整個專案統一從這裡取得「現在時間」與「今日日期」。"""

from datetime import datetime
from zoneinfo import ZoneInfo

TW_TZ = ZoneInfo("Asia/Taipei")


def now_tw() -> datetime:
    """回傳台灣當前時間（aware datetime, UTC+8）。"""
    return datetime.now(TW_TZ)


def today_tw():
    """回傳台灣今日 date 物件。"""
    return now_tw().date()
