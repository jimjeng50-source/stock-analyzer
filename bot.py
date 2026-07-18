"""台股多因子評分 Telegram 機器人"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from data.fetcher import FinMindFetcher
from factors import compute_chips, compute_technical, compute_fundamental, compute_momentum
from models.scorer import Scorer
from config import FACTOR_WEIGHTS, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=3)


# ── 格式化工具 ────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 10) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _format_result(stock_id: str, result: dict, current_price: float) -> str:
    total = result["total_score"]
    rec = result["recommendation"]
    cat = result["category_scores"]

    lines = [
        f"📊 *{stock_id} 分析結果*",
        f"",
        f"💰 現價：NT$ {current_price:,.2f}",
        f"🎯 綜合評分：*{total:.1f} / 100*",
        f"📌 {rec}",
        f"",
        f"━━━ 各面向分數 ━━━",
    ]

    for key, label in [
        ("chips",       "籌碼面"),
        ("fundamental", "基本面"),
        ("technical",   "技術面"),
        ("momentum",    "動能面"),
        ("risk",        "風險面"),
    ]:
        s = cat.get(key, 0)
        lines.append(f"`{label}` {_bar(s)} {s:.0f}")

    return "\n".join(lines)


# ── 同步分析（在 executor 執行，避免阻塞事件迴圈） ──────────────────────────

def _run_analysis(stock_id: str, use_ai: bool) -> tuple:
    fetcher = FinMindFetcher(stock_id)
    price_df = fetcher.get_price()

    if price_df.empty:
        return None, 0.0, ""

    institutional_df = fetcher.get_institutional()
    margin_df        = fetcher.get_margin_trading()
    revenue_df       = fetcher.get_monthly_revenue()
    financial_df     = fetcher.get_financial_statements()

    current_price = float(price_df["close"].iloc[-1])

    chips       = compute_chips(institutional_df, margin_df)
    technical   = compute_technical(price_df)
    fundamental = compute_fundamental(revenue_df, financial_df, current_price)
    momentum    = compute_momentum(price_df)

    result = Scorer(FACTOR_WEIGHTS).score(chips, technical, fundamental, momentum)

    ai_advice = ""
    if use_ai and ANTHROPIC_API_KEY:
        from utils.claude_api import get_investment_advice
        ai_advice = get_investment_advice(result, stock_id)

    return result, current_price, ai_advice


# ── 共用分析流程 ──────────────────────────────────────────────────────────────

async def _analyze(update: Update, stock_id: str, use_ai: bool):
    stock_id = stock_id.strip()

    if not stock_id.isdigit() or not (3 <= len(stock_id) <= 6):
        await update.message.reply_text("❌ 請輸入正確的股票代號（純數字，如 2330）")
        return

    wait_msg = await update.message.reply_text(f"⏳ 正在分析 {stock_id}，請稍候...")

    try:
        loop = asyncio.get_event_loop()
        result, price, advice = await loop.run_in_executor(
            _executor, _run_analysis, stock_id, use_ai
        )

        if result is None:
            await wait_msg.edit_text(f"❌ 找不到 {stock_id} 的資料，請確認代號是否正確")
            return

        await wait_msg.edit_text(
            _format_result(stock_id, result, price),
            parse_mode="Markdown",
        )

        if advice:
            await update.message.reply_text(
                f"🤖 *AI 投資建議*\n\n{advice}",
                parse_mode="Markdown",
            )

    except Exception as e:
        logger.error(f"分析 {stock_id} 失敗：{e}")
        await wait_msg.edit_text(f"❌ 分析失敗：{e}")


# ── 指令處理器 ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 歡迎使用 *台股多因子評分機器人*！\n\n"
        "📌 *使用方式*\n"
        "直接傳股票代號即可分析：\n"
        "`2330` → 台積電\n"
        "`6213` → 聯茂\n"
        "`0050` → 元大台灣50\n\n"
        "📋 *指令*\n"
        "/analyze 2330 — 完整分析 + AI 建議\n"
        "/quick 2330 — 快速評分（不含 AI）\n"
        "/help — 使用說明",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *使用說明*\n\n"
        "直接傳*股票代號*（純數字）→ 快速評分\n\n"
        "━━━ 個股分析 ━━━\n"
        "*/analyze 代號* → 完整分析 + Claude AI 建議\n"
        "*/quick 代號* → 只看評分，速度較快\n"
        "*/eps 代號* → Forward EPS 與三情境目標價\n"
        "*/report 代號* → 完整研究報告摘要\n\n"
        "━━━ 產業與市場 ━━━\n"
        "*/chain semiconductor* → 半導體產業鏈信號\n"
        "*/chain ai\\_server* → AI 伺服器供應鏈\n"
        "*/chain ev\\_components* → 電動車零組件\n"
        "*/revenue* → 本週即將公布月營收個股\n\n"
        "━━━ 每日推薦 ━━━\n"
        "*/recommend* → 今日精選股票推薦\n"
        "*/recommend refresh* → 立即重新掃描全市場\n"
        "*/recommend history* → 近期推薦績效摘要\n\n"
        "━━━ 設定 ━━━\n"
        "*/myid* → 查詢你的 Chat ID（用於啟用主動推播）\n\n"
        "資料來源：FinMind API + yfinance\n"
        "AI 建議：Anthropic Claude",
        parse_mode="Markdown",
    )


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法：`/analyze 2330`", parse_mode="Markdown")
        return
    await _analyze(update, context.args[0], use_ai=True)


async def cmd_quick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法：`/quick 2330`", parse_mode="Markdown")
        return
    await _analyze(update, context.args[0], use_ai=False)


# ── v3 新增指令 ────────────────────────────────────────────────────────────────

async def cmd_revenue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/revenue — 顯示本週即將公布月營收的個股清單"""
    wait_msg = await update.message.reply_text("⏳ 查詢本週即將公布月營收清單...")
    try:
        from data.fetcher import DataFetcher
        from alerts.revenue_calendar import RevenueCalendar
        from alerts.notifier import Notifier

        loop = asyncio.get_event_loop()

        def _run():
            fetcher = DataFetcher()
            cal = RevenueCalendar(fetcher)
            upcoming = cal.get_upcoming_announcements(days_ahead=7)
            notifier = Notifier()
            return notifier.format_weekly_preview(upcoming)

        msg = await loop.run_in_executor(_executor, _run)
        await wait_msg.edit_text(msg)
    except Exception as e:
        logger.error(f"/revenue 失敗：{e}")
        await wait_msg.edit_text(f"❌ 查詢失敗：{e}")


async def cmd_chain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/chain [產業] — 顯示指定產業鏈的當前景氣信號"""
    valid_chains = {
        "semiconductor": "半導體產業鏈",
        "ai_server": "AI 伺服器供應鏈",
        "ev_components": "電動車零組件",
        "半導體": "semiconductor",
        "ai": "ai_server",
        "ev": "ev_components",
    }
    chain_arg = context.args[0].lower() if context.args else "semiconductor"
    # 中文別名對應
    if chain_arg in ("半導體",):
        chain_arg = "semiconductor"
    elif chain_arg in ("ai", "ai伺服器", "伺服器"):
        chain_arg = "ai_server"
    elif chain_arg in ("ev", "電動車", "ev_components"):
        chain_arg = "ev_components"

    if chain_arg not in ("semiconductor", "ai_server", "ev_components"):
        await update.message.reply_text(
            "❌ 請指定有效產業鏈：\n"
            "`/chain semiconductor` — 半導體\n"
            "`/chain ai_server` — AI 伺服器\n"
            "`/chain ev_components` — 電動車零組件",
            parse_mode="Markdown",
        )
        return

    wait_msg = await update.message.reply_text(f"⏳ 分析 {chain_arg} 產業鏈...")
    try:
        from data.fetcher import DataFetcher
        from factors.supply_chain import SupplyChainAnalyzer
        from alerts.notifier import Notifier

        loop = asyncio.get_event_loop()

        def _run():
            fetcher = DataFetcher()
            analyzer = SupplyChainAnalyzer(fetcher)
            result = analyzer.analyze_chain(chain_arg)
            notifier = Notifier()
            return notifier.format_chain_signal(result), result

        msg, chain_result = await loop.run_in_executor(_executor, _run)
        flow = chain_result.get("capital_flow_direction", "")
        full_msg = msg + (f"\n\n💡 {flow}" if flow else "")
        await wait_msg.edit_text(full_msg)
    except Exception as e:
        logger.error(f"/chain 失敗：{e}")
        await wait_msg.edit_text(f"❌ 分析失敗：{e}")


async def cmd_eps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/eps [股票代號] — 顯示指定個股的 Forward EPS 與目標價"""
    if not context.args:
        await update.message.reply_text("用法：`/eps 2330`", parse_mode="Markdown")
        return

    stock_id = context.args[0].strip()
    if not stock_id.isdigit():
        await update.message.reply_text("❌ 請輸入正確的股票代號（純數字）")
        return

    wait_msg = await update.message.reply_text(f"⏳ 計算 {stock_id} Forward EPS...")
    try:
        from data.fetcher import DataFetcher
        from factors.forward_eps import ForwardEPSCalculator

        loop = asyncio.get_event_loop()

        def _run():
            fetcher = DataFetcher()
            calc = ForwardEPSCalculator(fetcher)
            return calc.calculate(stock_id)

        eps = await loop.run_in_executor(_executor, _run)

        if eps.get("error"):
            await wait_msg.edit_text(f"❌ {eps['error']}")
            return

        tp = eps.get("target_price", {})
        growth_pct = (eps.get("eps_growth_rate") or 0) * 100
        conf = eps.get("confidence", "low")
        conf_icon = {"high": "✅", "medium": "🔶", "low": "⚠️"}.get(conf, "⚠️")
        peg = eps.get("peg_ratio")
        peg_str = f"PEG：{peg:.2f}" if peg else "PEG：N/A"

        msg = (
            f"🎯 *{stock_id} Forward EPS 分析*\n\n"
            f"💰 當前股價：NT${eps.get('current_price', 0):,.1f}\n"
            f"📊 TTM EPS：{eps.get('ttm_eps', 0):.2f} 元\n"
            f"📈 Forward EPS：{eps.get('forward_eps_1y', 0):.2f} 元（{growth_pct:+.1f}%）\n"
            f"\n━━━ 三情境目標價 ━━━\n"
            f"🐻 熊市：NT${tp.get('bear') or 0:,.1f}\n"
            f"📌 基準：NT${tp.get('base') or 0:,.1f}\n"
            f"🐂 牛市：NT${tp.get('bull') or 0:,.1f}\n"
        )
        if eps.get("upside_pct") is not None:
            msg += f"\n📐 距基準漲幅：{eps['upside_pct']:+.1f}%\n"
        msg += f"\n{peg_str}\n{conf_icon} 信心度：{conf}｜{eps.get('confidence_reason', '')}"

        await wait_msg.edit_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"/eps {stock_id} 失敗：{e}")
        await wait_msg.edit_text(f"❌ 計算失敗：{e}")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/report [股票代號] — 生成並傳送完整研究報告摘要"""
    if not context.args:
        await update.message.reply_text("用法：`/report 2330`", parse_mode="Markdown")
        return

    stock_id = context.args[0].strip()
    if not stock_id.isdigit():
        await update.message.reply_text("❌ 請輸入正確的股票代號（純數字）")
        return

    wait_msg = await update.message.reply_text(
        f"⏳ 正在為 {stock_id} 生成完整研究報告（可能需要 30-60 秒）..."
    )
    try:
        from data.fetcher import DataFetcher
        from reports.research_report import ResearchReportGenerator

        loop = asyncio.get_event_loop()

        def _run():
            fetcher = DataFetcher()
            gen = ResearchReportGenerator(fetcher)
            return gen.generate(stock_id)

        report = await loop.run_in_executor(_executor, _run)

        if report.get("error"):
            await wait_msg.edit_text(f"❌ {report['error']}")
            return

        summary = report.get("report_summary", "")
        tp = report.get("target_price", {})
        base_tp = tp.get("base")
        forward_eps = report.get("forward_eps_data", {})
        confidence = forward_eps.get("confidence", "low")
        conf_icon = {"high": "✅", "medium": "🔶", "low": "⚠️"}.get(confidence, "⚠️")

        msg = (
            f"📋 *{stock_id} 研究報告摘要*\n\n"
            f"{summary}\n\n"
        )
        if base_tp:
            msg += f"🎯 基準目標價：NT${base_tp:,.1f}\n"
        msg += f"{conf_icon} 推估信心度：{confidence}"

        await wait_msg.edit_text(msg, parse_mode="Markdown")
        await update.message.reply_text(
            "💡 完整 Markdown 報告（前 2000 字）：\n\n"
            + report.get("report_text", "")[:2000]
        )
    except Exception as e:
        logger.error(f"/report {stock_id} 失敗：{e}")
        await wait_msg.edit_text(f"❌ 報告生成失敗：{e}")


async def cmd_myid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/myid — 顯示目前的 Telegram Chat ID，用於設定 TELEGRAM_CHAT_ID 環境變數"""
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"🆔 *你的 Chat ID*\n\n`{chat_id}`\n\n"
        "請將此數字填入 `.env` 的 `TELEGRAM_CHAT_ID`，\n"
        "或在 Streamlit Cloud → Settings → Secrets 中加入：\n"
        "`TELEGRAM_CHAT_ID = \"" + str(chat_id) + "\"`\n\n"
        "設定後，排程器就能主動推播月營收與產業鏈警示給你。",
        parse_mode="Markdown",
    )


async def cmd_recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/recommend [refresh|history] — 今日推薦清單、強制重掃或歷史績效"""
    arg = (context.args[0].lower() if context.args else "").strip()
    chat_id = str(update.effective_chat.id)

    # ── /recommend history ─────────────────────────────────────────────────────
    if arg == "history":
        wait_msg = await update.message.reply_text("⏳ 查詢近 7 日推薦績效...")
        try:
            loop = asyncio.get_event_loop()

            def _run():
                from screener.recommendation_db import RecommendationDB
                db = RecommendationDB()
                perf = db.get_performance_summary(n_days=30)
                recent_df = db.get_recent_recommendations(n_days=7)
                return perf, recent_df

            perf, recent_df = await loop.run_in_executor(_executor, _run)
            avg5 = f"{perf['avg_return_5d']:+.1f}%" if perf.get("avg_return_5d") is not None else "—"
            avg20 = f"{perf['avg_return_20d']:+.1f}%" if perf.get("avg_return_20d") is not None else "—"
            wr5 = f"{perf['win_rate_5d']*100:.0f}%" if perf.get("win_rate_5d") is not None else "—"
            total = perf.get("total_recommendations", 0)
            evaluated = perf.get("evaluated_count", 0)

            msg = (
                "📊 *近 30 日推薦績效摘要*\n\n"
                f"平均 5 日報酬：{avg5}\n"
                f"平均 20 日報酬：{avg20}\n"
                f"5 日正報酬勝率：{wr5}\n"
                f"總推薦次數：{total}　已評估：{evaluated}\n\n"
            )
            if not recent_df.empty:
                msg += "━━━ 近 7 日推薦個股 ━━━\n"
                for _, row in recent_df.iterrows():
                    r5 = f"{row['return_5d_pct']:+.1f}%" if row.get("return_5d_pct") is not None else "待評估"
                    msg += f"  {row['recommend_date']} #{row['rank']} {row['stock_id']} {row.get('stock_name','')} → 5日 {r5}\n"
            await wait_msg.edit_text(msg, parse_mode="Markdown")
        except Exception as e:
            await wait_msg.edit_text(f"❌ 查詢失敗：{e}")
        return

    # ── /recommend refresh（僅自己可用）──────────────────────────────────────
    if arg == "refresh":
        from config import TELEGRAM_CHAT_ID
        if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
            await update.message.reply_text("❌ 此指令僅限管理員使用")
            return
        wait_msg = await update.message.reply_text(
            "⏳ 正在執行全市場掃描（需數分鐘，請耐心等候）..."
        )
        try:
            loop = asyncio.get_event_loop()

            def _run():
                from screener.recommender import DailyRecommender
                return DailyRecommender().run(dry_run=False)

            result = await loop.run_in_executor(_executor, _run)
            if result.get("error"):
                await wait_msg.edit_text(f"❌ 掃描失敗：{result['error']}")
            else:
                n = len(result["recommendations"])
                await wait_msg.edit_text(
                    result["message"][:4000] if len(result["message"]) > 4000 else result["message"]
                )
        except Exception as e:
            logger.error(f"/recommend refresh 失敗：{e}")
            await wait_msg.edit_text(f"❌ 掃描失敗：{e}")
        return

    # ── /recommend（預設：從 DB 讀今日推薦）─────────────────────────────────
    wait_msg = await update.message.reply_text("⏳ 查詢今日推薦清單...")
    try:
        loop = asyncio.get_event_loop()

        def _run():
            from screener.recommendation_db import RecommendationDB
            from datetime import date
            db = RecommendationDB()
            return db.get_recommendations(date.today())

        recs = await loop.run_in_executor(_executor, _run)

        if not recs:
            await wait_msg.edit_text(
                "📭 今日尚未執行掃描。\n\n"
                "使用 `/recommend refresh` 立即掃描（需數分鐘）\n"
                "或等候每日 17:30 自動推播",
                parse_mode="Markdown",
            )
            return

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}
        lines = [f"🏆 *今日推薦（{recs[0].get('recommend_date', '')}）*\n"]
        for rec in recs:
            rank = rec.get("rank", 0)
            sid = rec["stock_id"]
            name = rec.get("stock_name", sid)
            score = rec.get("total_score") or 0
            price = rec.get("current_price") or 0
            r1 = rec.get("reason_1", "")
            lines += [
                f"{medal.get(rank, f'#{rank}')} *{sid} {name}*",
                f"💰 NT${price:,.0f}｜評分 {score:.0f}/100",
                f"✅ {r1}" if r1 else "",
                "",
            ]

        lines.append("⚠️ 僅供學習參考，不構成投資建議")
        msg = "\n".join(l for l in lines if l is not None)
        await wait_msg.edit_text(msg[:4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"/recommend 失敗：{e}")
        await wait_msg.edit_text(f"❌ 查詢失敗：{e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.isdigit() and 3 <= len(text) <= 6:
        await _analyze(update, text, use_ai=False)
    else:
        await update.message.reply_text(
            "💡 直接傳股票代號（如 `2330`）開始分析，或輸入 /help 查看說明",
            parse_mode="Markdown",
        )


# ── 啟動 ──────────────────────────────────────────────────────────────────────

def main():
    from config import get_runtime_config
    token = get_runtime_config("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ 請設定 TELEGRAM_BOT_TOKEN（環境變數 / .env / Secrets）")
        return

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("quick", cmd_quick))
    # v3 新增指令
    app.add_handler(CommandHandler("revenue", cmd_revenue))
    app.add_handler(CommandHandler("chain", cmd_chain))
    app.add_handler(CommandHandler("eps", cmd_eps))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("myid", cmd_myid))
    # v4 新增指令
    app.add_handler(CommandHandler("recommend", cmd_recommend))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 台股機器人已啟動，按 Ctrl+C 停止")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
