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
        "*/analyze 代號* → 完整分析 + Claude AI 建議\n"
        "*/quick 代號* → 只看評分，速度較快\n\n"
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
    if not TELEGRAM_BOT_TOKEN:
        print("❌ 請在 .env 設定 TELEGRAM_BOT_TOKEN")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("quick", cmd_quick))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 台股機器人已啟動，按 Ctrl+C 停止")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
