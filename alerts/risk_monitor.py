"""
alerts/risk_monitor.py
每日風險警訊監控

以「賺錢優先、守住本金」為原則，每日檢查四類風險並主動推播：

1. 大盤風險 — 加權指數單日重挫、跌破季線（系統性風險，先於個股）
2. 持股風險 — 近 60 天推薦個股的停損/停利/達標訊號
3. 營收風險 — 追蹤清單即將公布月營收（前 3 天預告）＋新公布營收惡化
4. EPS 風險 — 推薦股 Forward EPS 相對推薦時點下修

只在有警訊時推播（避免雜訊疲勞）。
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

# 風險門檻（穩健長線配置：停損放寬避免正常波動洗出場，跌破趨勢線才示警）
MARKET_DROP_PCT = -1.5        # 大盤單日跌幅警戒
STOP_LOSS_PCT = -12.0         # 個股停損線（或跌破 60 日線）
TAKE_PROFIT_PCT = 15.0        # 個股停利提醒
REVENUE_YOY_ALERT = -10.0     # 月營收 YoY 惡化門檻
EPS_DOWNGRADE_PCT = -10.0     # Forward EPS 下修門檻
EPS_CHECK_LIMIT = 10          # 每日最多重算 N 支 EPS（控制 API 用量）
FUNDAMENTAL_CHECK_LIMIT = 10  # 每日最多檢查 N 支基本面惡化


class RiskMonitor:
    """每日風險警訊監控器。各檢查獨立 try/except，單項失敗不影響其他。"""

    def run_daily(self) -> dict:
        """
        執行全部檢查。

        Returns:
            {
                "market": [str, ...],     # 大盤警訊
                "positions": [dict, ...], # 持股警訊（含 action: stop_loss/take_profit/target_hit）
                "revenue": [dict, ...],   # 營收警訊/預告
                "eps": [dict, ...],       # EPS 下修警訊
                "has_alerts": bool,
                "checked_at": str,
            }
        """
        report = {
            "market": [], "positions": [], "revenue": [], "eps": [],
            "fundamental": [],
            "has_alerts": False, "checked_at": date.today().isoformat(),
        }

        for key, fn in (
            ("market", self.check_market_risk),
            ("positions", self.check_position_risk),
            ("revenue", self.check_revenue_risk),
            ("eps", self.check_eps_risk),
            ("fundamental", self.check_fundamental_risk),
        ):
            try:
                report[key] = fn()
            except Exception as e:
                logger.warning("風險檢查 %s 失敗：%s", key, e)

        report["has_alerts"] = any(
            report[k] for k in ("market", "positions", "revenue", "eps", "fundamental")
        )
        return report

    # ── 1. 大盤風險 ────────────────────────────────────────────────────────────

    def check_market_risk(self) -> list:
        """加權指數：單日重挫、跌破季線（MA60）。"""
        if not _HAS_YFINANCE:
            return []
        warnings = []
        try:
            raw = yf.download("^TWII", period="6mo", progress=False, auto_adjust=True)
            if raw.empty:
                return []
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            closes = raw["Close"].dropna()
            if len(closes) < 2:
                return []

            last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
            chg = (last / prev - 1) * 100
            if chg <= MARKET_DROP_PCT:
                warnings.append(
                    f"⚠️ 大盤單日下跌 {chg:.1f}%（{last:,.0f} 點）— 系統性風險升高，建議暫緩加碼"
                )

            if len(closes) >= 60:
                ma60 = float(closes.tail(60).mean())
                if last < ma60:
                    warnings.append(
                        f"⚠️ 大盤跌破季線（現值 {last:,.0f} < MA60 {ma60:,.0f}）— 中期趨勢轉弱"
                    )
        except Exception as e:
            logger.debug("大盤風險檢查失敗：%s", e)
        return warnings

    # ── 2. 持股風險（推薦股停損/停利）──────────────────────────────────────────

    def check_position_risk(self, n_days: int = 60) -> list:
        """
        近 N 天推薦個股 vs 現價：
        - 跌破停損線（-8%）→ 停損警訊
        - 漲過停利線（+15%）→ 停利提醒
        - 達到目標價 → 達標提醒
        """
        from screener.recommendation_db import RecommendationDB
        df = RecommendationDB().get_recent_recommendations(n_days=n_days)
        if df.empty:
            return []

        # 每支股票取最近一次推薦
        df = df.sort_values("recommend_date").drop_duplicates("stock_id", keep="last")
        df = df.dropna(subset=["current_price"])
        if df.empty:
            return []

        prices = self._bulk_last_prices(df["stock_id"].astype(str).tolist())
        alerts = []
        for _, row in df.iterrows():
            sid = str(row["stock_id"])
            info = prices.get(sid) or {}
            now_price = info.get("price")
            ma60 = info.get("ma60")
            entry = row["current_price"]
            if not now_price or not entry:
                continue
            chg = (now_price / entry - 1) * 100
            name = row.get("stock_name", sid)
            target = row.get("target_price")

            below_ma60 = ma60 is not None and now_price < ma60 and chg < 0
            if chg <= STOP_LOSS_PCT:
                alerts.append({
                    "stock_id": sid, "stock_name": name, "action": "stop_loss",
                    "msg": f"🔴 {sid} {name} 已跌破停損線：推薦價 {entry:.0f} → 現價 {now_price:.0f}（{chg:+.1f}%）— 建議檢視停損",
                })
            elif below_ma60:
                alerts.append({
                    "stock_id": sid, "stock_name": name, "action": "stop_loss",
                    "msg": f"🟠 {sid} {name} 跌破 60 日線（現價 {now_price:.0f} < MA60 {ma60:.0f}，{chg:+.1f}%）— 中期趨勢轉弱，建議檢視持股",
                })
            elif target and now_price >= target:
                alerts.append({
                    "stock_id": sid, "stock_name": name, "action": "target_hit",
                    "msg": f"🎯 {sid} {name} 已達目標價 {target:.0f}：現價 {now_price:.0f}（{chg:+.1f}%）— 可考慮分批停利",
                })
            elif chg >= TAKE_PROFIT_PCT:
                alerts.append({
                    "stock_id": sid, "stock_name": name, "action": "take_profit",
                    "msg": f"🟢 {sid} {name} 獲利 {chg:+.1f}%：推薦價 {entry:.0f} → 現價 {now_price:.0f} — 可考慮部分落袋",
                })
        return alerts

    # ── 3. 營收風險（追蹤清單）─────────────────────────────────────────────────

    def check_revenue_risk(self) -> list:
        """
        - 追蹤清單 3 天內即將公布月營收 → 預告（公布前留意波動）
        - 近 2 天新公布且 YoY 低於門檻 → 惡化警訊
        """
        from data.fetcher import DataFetcher
        from alerts.revenue_calendar import RevenueCalendar

        calendar = RevenueCalendar(DataFetcher())
        alerts = []

        # 即將公布（前 3 天）
        try:
            upcoming = calendar.get_upcoming_announcements(days_ahead=3)
            for s in upcoming:
                exp = s.get("expected_date")
                exp_str = exp.strftime("%m/%d") if exp else "近日"
                last_yoy = s.get("last_revenue_yoy")
                yoy_str = f"（上月 YoY {last_yoy:+.1f}%）" if last_yoy is not None else ""
                alerts.append({
                    "stock_id": s["stock_id"], "type": "upcoming",
                    "msg": f"📅 {s['stock_id']} {s.get('stock_name', '')} 預計 {exp_str} 公布月營收{yoy_str}— 公布前後留意波動",
                })
        except Exception as e:
            logger.debug("即將公布檢查失敗：%s", e)

        # 新公布且惡化
        try:
            recent = calendar.get_recent_announcements(months=1)
            if not recent.empty and "announce_date" in recent.columns:
                recent["announce_date"] = pd.to_datetime(recent["announce_date"], errors="coerce")
                cutoff = pd.Timestamp(date.today() - timedelta(days=2))
                fresh = recent[recent["announce_date"] >= cutoff]
                for _, row in fresh.iterrows():
                    yoy = pd.to_numeric(row.get("yoy_pct"), errors="coerce")
                    if pd.notna(yoy) and yoy <= REVENUE_YOY_ALERT:
                        alerts.append({
                            "stock_id": row["stock_id"], "type": "deterioration",
                            "msg": f"🔴 {row['stock_id']} {row.get('stock_name', '')} 最新月營收 YoY {yoy:+.1f}% — 營收惡化，建議檢視持股",
                        })
        except Exception as e:
            logger.debug("營收惡化檢查失敗：%s", e)

        return alerts

    # ── 4. Forward EPS 下修 ────────────────────────────────────────────────────

    def check_eps_risk(self, n_days: int = 60) -> list:
        """
        近 N 天推薦股中有存 Forward EPS 者，重算並比對：
        下修超過門檻 → 警訊。每日最多檢查 EPS_CHECK_LIMIT 支（控制 API 量）。
        """
        from screener.recommendation_db import RecommendationDB
        df = RecommendationDB().get_recent_recommendations(n_days=n_days)
        if df.empty or "forward_eps" not in df.columns:
            return []

        df = df.dropna(subset=["forward_eps"])
        df = df[df["forward_eps"] > 0]
        if df.empty:
            return []
        df = df.sort_values("recommend_date").drop_duplicates("stock_id", keep="last")
        df = df.head(EPS_CHECK_LIMIT)

        try:
            from data.fetcher import DataFetcher
            from factors.forward_eps import ForwardEPSCalculator
            calc = ForwardEPSCalculator(DataFetcher())
        except Exception as e:
            logger.debug("Forward EPS 模組載入失敗：%s", e)
            return []

        alerts = []
        for _, row in df.iterrows():
            sid = str(row["stock_id"])
            old_eps = float(row["forward_eps"])
            try:
                result = calc.calculate(sid)
                new_eps = result.get("forward_eps_1y")
                if result.get("error") or not new_eps:
                    continue
                chg = (new_eps / old_eps - 1) * 100
                if chg <= EPS_DOWNGRADE_PCT:
                    alerts.append({
                        "stock_id": sid, "stock_name": row.get("stock_name", sid),
                        "msg": f"🔻 {sid} {row.get('stock_name', sid)} Forward EPS 下修 {chg:.1f}%"
                               f"（{old_eps:.2f} → {new_eps:.2f} 元）— 基本面轉弱訊號",
                    })
            except Exception as e:
                logger.debug("EPS 重算失敗 %s：%s", sid, e)
        return alerts

    # ── 5. 基本面惡化（月營收連續衰退 + 法說會前提醒）─────────────────────────

    def check_fundamental_risk(self, n_days: int = 60) -> list:
        """
        近 N 天推薦股（最多 FUNDAMENTAL_CHECK_LIMIT 支）：
        - 月營收 YoY 連續 2 個月為負 → 基本面惡化警訊
        - 3 天內有法說會 → 提醒（法說會是 Forward EPS 修正的最大觸發點）
        """
        from screener.recommendation_db import RecommendationDB
        df = RecommendationDB().get_recent_recommendations(n_days=n_days)
        if df.empty:
            return []
        df = df.sort_values("recommend_date").drop_duplicates("stock_id", keep="last")
        df = df.head(FUNDAMENTAL_CHECK_LIMIT)

        alerts = []

        # 月營收連續衰退
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            for _, row in df.iterrows():
                sid = str(row["stock_id"])
                name = row.get("stock_name", sid)
                try:
                    rev = fetcher.get_monthly_revenue(sid, months=4)
                    if rev is None or rev.empty or "revenue_yoy" not in rev.columns:
                        continue
                    yoy = pd.to_numeric(rev["revenue_yoy"], errors="coerce").dropna()
                    if len(yoy) >= 2 and yoy.iloc[-1] < 0 and yoy.iloc[-2] < 0:
                        alerts.append({
                            "stock_id": sid, "type": "revenue_decline",
                            "msg": f"🔻 {sid} {name} 月營收連 2 個月衰退"
                                   f"（最新 YoY {yoy.iloc[-1]:+.1f}%、前月 {yoy.iloc[-2]:+.1f}%）— 基本面轉弱",
                        })
                except Exception as e:
                    logger.debug("營收惡化檢查失敗 %s：%s", sid, e)
        except Exception as e:
            logger.debug("基本面惡化檢查載入失敗：%s", e)

        # 法說會提醒（TWSE 官方 OpenAPI，失敗時靜默跳過）
        try:
            conf_map = self._fetch_conference_schedule()
            watch_ids = set(df["stock_id"].astype(str))
            for sid, conf_date in conf_map.items():
                if sid in watch_ids:
                    days_to = (conf_date - date.today()).days
                    if 0 <= days_to <= 3:
                        alerts.append({
                            "stock_id": sid, "type": "conference",
                            "msg": f"🎤 {sid} 將於 {conf_date.strftime('%m/%d')} 舉行法說會 — "
                                   "Forward EPS 可能修正，公布前後留意波動",
                        })
        except Exception as e:
            logger.debug("法說會檢查失敗：%s", e)

        return alerts

    @staticmethod
    def _fetch_conference_schedule() -> dict:
        """
        從 TWSE OpenAPI 取得上市公司法說會日程。
        Returns: {stock_id: date}。端點格式改變或失敗時回傳空 dict。
        """
        import requests as _rq
        out = {}
        try:
            resp = _rq.get(
                "https://openapi.twse.com.tw/v1/opendata/t187ap38_L",
                timeout=30, headers={"accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            rows = resp.json()
            for row in rows:
                # 防禦性欄位比對（官方欄位名稱可能是中文）
                sid = None
                conf_date = None
                for k, v in row.items():
                    if sid is None and ("公司代號" in k or k.lower() in ("code", "companycode")):
                        sid = str(v).strip()
                    if conf_date is None and ("日期" in k or "date" in k.lower()):
                        s = str(v).strip().replace("/", "-")
                        # 民國年轉西元
                        parts = s.split("-")
                        try:
                            if len(parts) == 3:
                                y = int(parts[0])
                                if y < 1000:
                                    y += 1911
                                conf_date = date(y, int(parts[1]), int(parts[2]))
                        except (ValueError, IndexError):
                            pass
                if sid and conf_date:
                    out[sid] = conf_date
        except Exception as e:
            logger.debug("法說會日程抓取失敗：%s", e)
        return out

    # ── 訊息格式化 ─────────────────────────────────────────────────────────────

    def format_message(self, report: dict) -> str:
        """組裝 Telegram 推播訊息（只在 has_alerts 時呼叫）。"""
        lines = [
            "╔══════════════════════╗",
            "║  🚨 每日風險警訊報告  ║",
            f"║  {report['checked_at']}        ║",
            "╚══════════════════════╝",
        ]

        if report["market"]:
            lines += ["", "━━ 大盤風險 ━━"]
            lines += report["market"]

        if report["positions"]:
            lines += ["", "━━ 持股警訊（近 60 天推薦股）━━"]
            lines += [a["msg"] for a in report["positions"]]

        if report["revenue"]:
            lines += ["", "━━ 營收動態（追蹤清單）━━"]
            lines += [a["msg"] for a in report["revenue"]]

        if report["eps"]:
            lines += ["", "━━ Forward EPS 下修 ━━"]
            lines += [a["msg"] for a in report["eps"]]

        if report.get("fundamental"):
            lines += ["", "━━ 基本面惡化／法說會 ━━"]
            lines += [a["msg"] for a in report["fundamental"]]

        lines += [
            "",
            "⚠️ 警訊由量化規則自動產生，僅供參考，不構成投資建議。",
        ]
        return "\n".join(lines)

    # ── 工具 ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _bulk_last_prices(stock_ids: list) -> dict:
        """
        yfinance 批次取得最新收盤價與 MA60（免 API 配額）。
        Returns: {stock_id: {"price": float, "ma60": Optional[float]}}
        """
        if not _HAS_YFINANCE or not stock_ids:
            return {}
        tickers = [f"{s}.TW" for s in stock_ids] + [f"{s}.TWO" for s in stock_ids]
        prices = {}
        try:
            raw = yf.download(tickers, period="4mo", progress=False,
                              auto_adjust=True, threads=True)
            if raw.empty:
                return {}
            closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            for sid in stock_ids:
                for suffix in (".TW", ".TWO"):
                    col = f"{sid}{suffix}"
                    if col in closes.columns:
                        series = closes[col].dropna()
                        if not series.empty:
                            prices[sid] = {
                                "price": float(series.iloc[-1]),
                                "ma60": float(series.tail(60).mean()) if len(series) >= 60 else None,
                            }
                            break
        except Exception as e:
            logger.debug("批次取價失敗：%s", e)
        return prices
