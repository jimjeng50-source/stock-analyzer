"""
alerts/notifier.py
推播通知模組：支援 LINE Notify 與 Email（SMTP）
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import requests

from config import (
    LINE_NOTIFY_TOKEN,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL,
)

logger = logging.getLogger(__name__)

_LINE_API = "https://notify-api.line.me/api/notify"
_MAX_LINE_LEN = 1000


class Notifier:
    """
    推播通知器。

    支援 LINE Notify 與 Email（SMTP）。
    Token/密碼從 config.py 讀取（透過 .env 或 st.secrets）。
    """

    def __init__(self):
        self.line_token = LINE_NOTIFY_TOKEN
        self.smtp_host = SMTP_HOST
        self.smtp_port = SMTP_PORT
        self.smtp_user = SMTP_USER
        self.smtp_password = SMTP_PASSWORD
        self.alert_email = ALERT_EMAIL

    # ── LINE Notify ────────────────────────────────────────────────────────────

    def send_line(self, message: str, image_url: Optional[str] = None) -> bool:
        """
        傳送 LINE Notify 訊息。
        超過 1000 字自動截斷並附加說明。
        """
        if not self.line_token:
            logger.warning("LINE_NOTIFY_TOKEN 未設定，跳過推播")
            return False

        if len(message) > _MAX_LINE_LEN:
            message = message[:_MAX_LINE_LEN - 20] + "...（完整報告見 Streamlit）"

        data = {"message": message}
        if image_url:
            data["imageFullsize"] = image_url
            data["imageThumbnail"] = image_url

        try:
            resp = requests.post(
                _LINE_API,
                headers={"Authorization": f"Bearer {self.line_token}"},
                data=data,
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("LINE Notify 推播成功")
                return True
            logger.warning("LINE Notify 推播失敗：%s %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("LINE Notify 推播異常：%s", e)
        return False

    # ── Email ──────────────────────────────────────────────────────────────────

    def send_email(
        self, subject: str, body: str, to_email: Optional[str] = None
    ) -> bool:
        """
        使用 SMTP 傳送 HTML 格式 Email。
        to_email 若 None，使用 config 中的 ALERT_EMAIL。
        """
        if not all([self.smtp_user, self.smtp_password]):
            logger.warning("SMTP 設定不完整，跳過 Email")
            return False

        recipient = to_email or self.alert_email
        if not recipient:
            logger.warning("收件人 Email 未設定")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = recipient
            msg.attach(MIMEText(body, "html", "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, [recipient], msg.as_string())

            logger.info("Email 發送成功至 %s", recipient)
            return True
        except Exception as e:
            logger.error("Email 發送失敗：%s", e)
        return False

    # ── 訊息格式化 ─────────────────────────────────────────────────────────────

    def format_revenue_alert(self, stock_info: dict) -> str:
        """格式化月營收公告提醒訊息（LINE 格式）。"""
        stock_id = stock_info.get("stock_id", "")
        stock_name = stock_info.get("stock_name", "")
        revenue_month = stock_info.get("revenue_month", "")
        revenue_amount = stock_info.get("revenue_amount", 0)
        yoy_pct = stock_info.get("yoy_pct")
        mom_pct = stock_info.get("mom_pct")
        claude_summary = stock_info.get("claude_summary", "")
        streamlit_url = stock_info.get("streamlit_url", "")

        yoy_str = f"{yoy_pct:+.1f}%" if yoy_pct is not None else "N/A"
        mom_str = f"{mom_pct:+.1f}%" if mom_pct is not None else "N/A"
        yoy_icon = "📈" if (yoy_pct or 0) >= 0 else "📉"

        lines = [
            "═══════════════════",
            f"📊 月營收新公告：{stock_id} {stock_name}",
            "═══════════════════",
            f"📅 {revenue_month} 月營收",
            f"💰 {revenue_amount:,.0f} 千元",
            f"{yoy_icon} YoY：{yoy_str}",
            f"📊 MoM：{mom_str}",
        ]
        if claude_summary:
            lines += ["", "💡 分析摘要：", claude_summary]
        if streamlit_url:
            lines += ["", f"🔗 完整報告：{streamlit_url}"]

        return "\n".join(lines)

    def format_weekly_preview(self, upcoming_stocks: list) -> str:
        """格式化週報預覽訊息（LINE 格式）。"""
        from utils.tz import today_tw
        from datetime import timedelta
        today = today_tw()
        end_date = today + timedelta(days=6)

        lines = [
            "═══════════════════",
            f"📅 本週即將公布月營收（{today.strftime('%m/%d')} - {end_date.strftime('%m/%d')}）",
            "═══════════════════",
        ]
        if not upcoming_stocks:
            lines.append("本週無預計公布個股")
        else:
            for s in upcoming_stocks:
                sid = s.get("stock_id", "")
                name = s.get("stock_name", "")
                exp_date = s.get("expected_date")
                last_yoy = s.get("last_revenue_yoy")
                date_str = exp_date.strftime("%m/%d") if exp_date else "—"
                yoy_str = f"{last_yoy:+.1f}%" if last_yoy is not None else "—"
                conf = {"high": "★★★", "medium": "★★", "low": "★"}.get(
                    s.get("confidence", "low"), "★"
                )
                lines.append(f"▪ {sid} {name}（預期 {date_str} 公布）{conf}")
                lines.append(f"  上月 YoY {yoy_str}")

        lines.append("═══════════════════")
        return "\n".join(lines)

    def format_chain_signal(self, chain_analysis: dict) -> str:
        """格式化產業鏈信號摘要（LINE 格式）。"""
        chain_name = chain_analysis.get("chain_name", "")
        overall = chain_analysis.get("overall_signal", 0.0)
        label = chain_analysis.get("signal_label", "")
        tier_signals = chain_analysis.get("tier_signals", {})
        lead_lag = chain_analysis.get("lead_lag_months", 0)

        icon = "🟢" if overall > 0.3 else ("🔴" if overall < -0.3 else "🟡")
        lines = [
            f"🔗 產業鏈信號：{chain_name}",
            f"{icon} 整體：{label}（{overall:+.2f}）",
            f"  上游：{tier_signals.get('upstream', 0):+.2f}",
            f"  中游：{tier_signals.get('midstream', 0):+.2f}",
            f"  下游：{tier_signals.get('downstream', 0):+.2f}",
            f"  Lead-Lag：{lead_lag} 個月",
        ]
        return "\n".join(lines)
