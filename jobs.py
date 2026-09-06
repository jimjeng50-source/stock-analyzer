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
    from config import get_runtime_config
    from screener.recommender import DailyRecommender
    from alerts.notifier import Notifier

    # 前置檢查：基本面佔 45% 權重，缺 FINMIND_TOKEN 會導致全部分數被低估、
    # 掃不出候選股。這是設定錯誤，明確報錯（exit 1）讓使用者去設 secret。
    if not get_runtime_config("FINMIND_TOKEN"):
        logger.error("FINMIND_TOKEN 未設定 — 無法取得財報/籌碼資料，掃描會失真。"
                     "請在 GitHub Actions Secrets 設定 FINMIND_TOKEN。")
        Notifier().send_telegram(
            "❌ 每日掃描無法執行：FINMIND_TOKEN 未設定。\n"
            "請到 repo Settings → Secrets and variables → Actions 設定 FINMIND_TOKEN。"
        )
        return 1

    result = DailyRecommender().run(dry_run=False)

    # 硬錯誤（資料抓取失敗、例外）→ exit 1 並通知
    if result.get("error"):
        logger.error("掃描失敗：%s", result["error"])
        Notifier().send_telegram(f"❌ 每日掃描失敗：{result['error']}")
        return 1

    # 軟性無候選（流程正常但今日無達標個股）→ 仍推播訊息，正常結束
    Notifier().send_telegram(result["message"])
    if result.get("no_candidates") or not result["recommendations"]:
        logger.info("今日無達標推薦（已推播說明訊息）")
    else:
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


def job_track_daily() -> int:
    """
    更新推薦名單的「兩個月每日績效」曲線（滾動 2 個月）。

    對每一檔仍在 60 天追蹤窗內的推薦：取推薦日買進價（current_price）為基準，
    抓每個交易日收盤，計算逐日報酬，upsert 到 recommendation_daily_returns，
    再清除超過保留窗的舊列。價格用免費 FinMind→yfinance 序列（get_price）。
    """
    from screener.recommendation_db import RecommendationDB
    from data.fetcher import FinMindFetcher

    db = RecommendationDB()
    active = db.get_active_tracking_recommendations(window_days=60)
    if not active:
        logger.info("每日績效追蹤：無進行中的推薦")
        return 0

    price_cache = {}
    total_rows = 0
    for item in active:
        sid = item["stock_id"]
        rec_date = str(item["recommend_date"])
        entry = item.get("entry_price")
        name = item.get("stock_name") or sid
        if not entry:
            continue
        try:
            if sid not in price_cache:
                # days=95：足以涵蓋 60 天窗（約 60 交易日≈84 日曆日）+ 緩衝
                price_cache[sid] = FinMindFetcher(sid, days=95).get_price()
            price_df = price_cache[sid]
            if price_df is None or price_df.empty or "close" not in price_df.columns:
                continue
            sub = price_df[["date", "close"]].copy()
            sub["date"] = sub["date"].astype(str)
            sub = sub[sub["date"] >= rec_date].sort_values("date")
            rows = []
            for offset, (_, r) in enumerate(sub.iterrows()):
                close = float(r["close"])
                ret = (close / entry - 1) * 100 if entry else None
                rows.append({
                    "recommend_date": rec_date, "stock_id": sid, "stock_name": name,
                    "as_of_date": str(r["date"]), "day_offset": offset,
                    "entry_price": float(entry), "close_price": close, "return_pct": ret,
                })
            db.save_daily_returns(rows)
            total_rows += len(rows)
        except Exception as ex:
            logger.warning("每日績效追蹤失敗 %s（%s）：%s", sid, rec_date, ex)

    pruned = db.prune_daily_returns(keep_days=75)
    logger.info("每日績效追蹤：更新 %d 列，清除 %d 舊列", total_rows, pruned)

    _export_tracking_report(db)
    return 0


def _export_tracking_report(db) -> None:
    """輸出兩個月推薦名單績效報告（reports/tracking_2m.csv + .md）。"""
    import os
    from datetime import date

    os.makedirs("reports", exist_ok=True)
    summary = db.get_tracking_summary(window_days=60)
    csv_path = "reports/tracking_2m.csv"
    md_path = "reports/tracking_2m.md"

    if summary is None or summary.empty:
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write("（近兩個月尚無推薦追蹤紀錄）\n")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 兩個月推薦名單績效\n\n（近兩個月尚無推薦追蹤紀錄）\n")
        logger.info("兩個月績效報告：無資料")
        return

    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    def _f(v, s=""):
        return f"{v:+.1f}{s}" if isinstance(v, (int, float)) else "—"

    lines = [
        "# 兩個月推薦名單績效追蹤",
        "",
        f"產出日期：{date.today().isoformat()}（滾動追蹤最近 60 天內的推薦）",
        "",
        "| 推薦日 | 代號 | 名稱 | 買進價 | 最新價 | 最新報酬 | 期間最佳 | 期間最差 | 追蹤天數 |",
        "|------|------|------|------|------|------|------|------|------|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['recommend_date']} | {r['stock_id']} | {r.get('stock_name','')} "
            f"| {r['entry_price']:.1f} | {(_num(r.get('last_close')))} "
            f"| {_f(r.get('last_return_pct'), '%')} | {_f(r.get('max_return_pct'), '%')} "
            f"| {_f(r.get('min_return_pct'), '%')} | {int(r['days_tracked'])} |"
        )
    lines += ["", "*由 stock-analyzer 自動產出。僅供研究參考，不構成投資建議。*"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("兩個月績效報告已輸出：%s、%s（%d 檔）", csv_path, md_path, len(summary))


def _num(v):
    """數值格式化小工具（None→—）。"""
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def job_weekly_tune() -> int:
    """
    每週推薦邏輯回顧＋調參：
    依近半年已評估推薦（60 日報酬）的各因子分與實際報酬相關性，微調因子權重，
    寫入 data/weights.json（下次掃描即生效），並推播調整說明到 Telegram。
    目標：長期把 60 日勝率拉到 70% 以上。
    """
    from datetime import date
    from screener.recommendation_db import RecommendationDB
    from screener.weight_tuner import compute_tuned_weights, format_tune_message
    from config import get_active_factor_weights, save_factor_weights
    from alerts.notifier import Notifier

    db = RecommendationDB()
    df = db.get_recent_recommendations(n_days=180)
    current = get_active_factor_weights()
    new_weights, report = compute_tuned_weights(df, current)

    if report["tuned"]:
        save_factor_weights(new_weights, {
            "updated_at": date.today().isoformat(),
            "win_rate_60d": report["win_rate"],
            "samples": report["n"],
            "correlations": report["correlations"],
        })
        logger.info("週度調參：權重已更新 %s", report["changes"])
    else:
        logger.info("週度調參：%s", report["reason"])

    try:
        Notifier().send_telegram(format_tune_message(report))
    except Exception as e:
        logger.warning("週度調參推播失敗：%s", e)
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


def job_morning_report() -> int:
    """
    晨間報告（早上 8:00）→ Telegram：
    - 推薦模型正確率（含歷史回補樣本：20/60 日報酬、勝率）
    - 最新一日推薦（或觀察名單）及其 Forward EPS / 目標價
    """
    from datetime import date
    from screener.recommendation_db import RecommendationDB
    from screener.historical_eval import evaluate_60d_accuracy
    from alerts.notifier import Notifier

    db = RecommendationDB()
    perf = db.get_performance_summary(n_days=180)
    acc = evaluate_60d_accuracy(db, top_k=3)

    def _p(v, s=""):
        return f"{v:+.1f}{s}" if isinstance(v, (int, float)) else "—"

    def _w(v):
        return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "—"

    lines = [
        "╔══════════════════════╗",
        "║  ☀️ 台股晨間報告      ║",
        f"║  {date.today().isoformat()}        ║",
        "╚══════════════════════╝",
        "",
        "📊 推薦模型績效（近半年）",
        f"　20 日：{_p(perf.get('avg_return_20d'), '%')}（勝率 {_w(perf.get('win_rate_20d'))}）",
        f"　60 日：{_p(perf.get('avg_return_60d'), '%')}（勝率 {_w(perf.get('win_rate_60d'))}）",
        f"　已評估 {perf.get('evaluated_count', 0)} 筆",
    ]
    if acc.get("overall"):
        o = acc["overall"]
        lines.append(f"　前 3 名 60 日正確率：{_w(o.get('win_rate'))}"
                     f"（平均 {_p(o.get('avg_return_pct'), '%')}）")

    # 最新一日推薦（帶 Forward EPS / 目標價）
    recent = db.get_recent_recommendations(n_days=7)
    lines += ["", "🏆 最新推薦"]
    if recent is not None and not recent.empty:
        latest_date = recent["recommend_date"].max()
        today_recs = recent[recent["recommend_date"] == latest_date].sort_values("rank")
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}
        lines.append(f"（{latest_date}）")
        for _, r in today_recs.head(5).iterrows():
            icon = medal.get(int(r.get("rank", 0)), f"#{int(r.get('rank', 0))}")
            sid = r.get("stock_id", "")
            name = r.get("stock_name", sid)
            score = r.get("total_score") or 0
            price = r.get("current_price") or 0
            lines.append(f"{icon} {sid} {name}　評分 {score:.0f}｜{price:.0f} 元")
            feps = r.get("forward_eps")
            g = r.get("eps_growth_pct")
            tp = r.get("target_price")
            up = r.get("upside_pct")
            if feps is not None and str(feps) != "nan":
                gs = f"（成長 {g:+.0f}%）" if g is not None and str(g) != "nan" else ""
                lines.append(f"　📈 Forward EPS：{feps:.2f} 元{gs}")
            if tp is not None and str(tp) != "nan":
                us = f"（{up:+.0f}%）" if up is not None and str(up) != "nan" else ""
                lines.append(f"　🎯 目標價：{tp:.0f} 元{us}")
    else:
        lines.append("　目前資料庫尚無推薦紀錄。可執行 backfill-history 回補歷史。")

    # 社群/熱門候選池（籌碼買超、社群熱議、研報點名、量能）
    try:
        from screener.universe import UniverseManager
        from screener.hot_stocks import HotStockDetector
        uni = UniverseManager().get_universe()
        hot = HotStockDetector().detect_all(uni if uni is not None else None)
        if hot:
            name_map = {}
            if uni is not None and not uni.empty and "stock_name" in uni.columns:
                name_map = dict(zip(uni["stock_id"].astype(str), uni["stock_name"]))
            lines += ["", "🔥 熱門候選池（社群/籌碼/研報）"]
            # 依標記數排序（越多來源命中越前面），取前 8
            ranked = sorted(hot.items(), key=lambda kv: len(kv[1]), reverse=True)[:8]
            for sid, tags in ranked:
                nm = name_map.get(str(sid), sid)
                lines.append(f"・{sid} {nm}｜{'、'.join(tags)}")
    except Exception as e:
        logger.warning("熱門候選池區塊產生失敗（略過）：%s", e)

    def _ok(v):
        return v is not None and v == v      # 濾掉 None 與 NaN

    # ── 兩個月推薦名單績效追蹤（去重同股、取最近 20 檔）─────────────
    try:
        track = db.get_tracking_summary(window_days=60)
        if track is not None and not track.empty:
            t = (track.sort_values("recommend_date", ascending=False)
                      .drop_duplicates(subset="stock_id", keep="first"))
            n_total = len(t)
            rets = [r for r in t["last_return_pct"].tolist() if _ok(r)]
            lines += ["", f"📅 兩個月追蹤（{n_total} 檔）"]
            if rets:
                avg_ret = sum(rets) / len(rets)
                win = sum(1 for r in rets if r > 0) / len(rets)
                lines.append(f"　平均最新報酬 {avg_ret:+.1f}%｜勝率 {win*100:.0f}%")
            for _, r in t.head(20).iterrows():
                sid = r["stock_id"]
                nm = r.get("stock_name") or sid
                entry = r.get("entry_price")
                last = r.get("last_close")
                ret = r.get("last_return_pct")
                entry_s = f"{entry:.0f}" if _ok(entry) else "—"
                last_s = f"{last:.0f}" if _ok(last) else "—"
                ret_s = f"{ret:+.1f}%" if _ok(ret) else "—"
                lines.append(f"・{sid} {nm}｜買 {entry_s}→{last_s}（{ret_s}）")
            if n_total > 20:
                lines.append(f"　…完整 {n_total} 檔見 Streamlit 儀表板")
    except Exception as e:
        logger.warning("兩個月追蹤區塊產生失敗（略過）：%s", e)

    # ── 推薦邏輯狀態：目前生效權重 + 近期勝率 vs 70% 目標 ──────────
    try:
        from config import get_active_factor_weights, FACTOR_WEIGHTS
        from screener.weight_tuner import TARGET_WIN_RATE
        aw = get_active_factor_weights()
        changed = any(abs(aw[k] - FACTOR_WEIGHTS[k]) > 0.005 for k in FACTOR_WEIGHTS)
        zh = {"fundamental": "基本面", "chips": "籌碼", "risk": "風險",
              "technical": "技術", "momentum": "動能"}
        wstr = "、".join(f"{zh[k]}{aw[k]*100:.0f}%"
                         for k in ["fundamental", "chips", "risk", "technical", "momentum"])
        wr60 = perf.get("win_rate_60d")
        wr_s = f"{wr60*100:.0f}%" if isinstance(wr60, (int, float)) else "—"
        lines += ["", "⚙️ 推薦邏輯狀態",
                  f"　目前權重{'（已調參）' if changed else '（預設）'}：{wstr}",
                  f"　近期 60 日勝率 {wr_s}（目標 {TARGET_WIN_RATE*100:.0f}%）"]
    except Exception as e:
        logger.warning("權重狀態區塊產生失敗（略過）：%s", e)

    lines += ["", "⚠️ 僅供研究參考，不構成投資建議。投資有風險，請自行評估。"]

    ok = Notifier().send_telegram("\n".join(lines))
    logger.info("晨間報告已推播" if ok else "晨間報告推播失敗（檢查 Telegram 設定）")
    return 0


def job_intraday_watch(max_minutes: int = 0) -> int:
    """
    盤中即時資金流監控：輪詢觀察清單，達進場門檻立刻推 Telegram。

    非交易時段（平日 09:00–13:30 以外）直接結束，不空轉。
    max_minutes = 0 表示跑到收盤為止。
    """
    from config import (INTRADAY_COOLDOWN_MIN, INTRADAY_MIN_SCORE,
                        INTRADAY_POLL_INTERVAL, get_intraday_watchlist)
    from alerts.intraday_monitor import IntradayMonitor
    from alerts.notifier import Notifier
    from data.realtime import is_trading_hours

    if not is_trading_hours():
        logger.info("非台股交易時段（平日 09:00–13:30），盤中監控不啟動")
        return 0

    watchlist = get_intraday_watchlist()
    if not watchlist:
        logger.warning("盤中監控清單是空的 — 請設定 INTRADAY_WATCHLIST 或 "
                       "WATCHLIST_CUSTOM，或先跑一次 scan 產生推薦名單。")
        Notifier().send_telegram(
            "⚠️ 盤中資金流監控沒有標的可看。\n"
            "請在 Secrets/Variables 設定 INTRADAY_WATCHLIST（例：2330,2454,3231），"
            "或先執行每日掃描產生推薦名單。")
        return 0

    monitor = IntradayMonitor(
        watchlist,
        min_score=INTRADAY_MIN_SCORE,
        cooldown_minutes=INTRADAY_COOLDOWN_MIN,
        poll_interval=INTRADAY_POLL_INTERVAL,
    )
    logger.info("盤中監控啟動：%s（門檻 %d 分）", ",".join(watchlist), INTRADAY_MIN_SCORE)
    alerts = monitor.run(max_minutes=max_minutes or None)

    entries = [a for a in alerts if a["type"] == "entry"]
    logger.info("盤中監控結束：共推播 %d 則進場訊號、%d 則出場警示",
                len(entries), len(alerts) - len(entries))
    return 0


JOBS = {"scan": job_scan, "risk": job_risk, "backfill": job_backfill,
        "track-daily": job_track_daily, "weekly-tune": job_weekly_tune,
        "report-export": job_report_export, "morning-report": job_morning_report,
        "intraday-watch": job_intraday_watch}


def main():
    parser = argparse.ArgumentParser(description="排程任務入口")
    parser.add_argument("job", choices=sorted(JOBS.keys()) + ["backfill-history"],
                        help="要執行的任務")
    parser.add_argument("--start", default="2026-06-01",
                        help="backfill-history 起始日（YYYY-MM-DD）")
    parser.add_argument("--max-minutes", type=int, default=0,
                        help="intraday-watch 最長執行分鐘數（0＝跑到收盤）")
    args = parser.parse_args()
    if args.job == "backfill-history":
        sys.exit(job_backfill_history(args.start))
    if args.job == "intraday-watch":
        sys.exit(job_intraday_watch(args.max_minutes))
    sys.exit(JOBS[args.job]())


if __name__ == "__main__":
    main()
