"""
台股評分系統 - 命令列入口

用法：
    python main.py --stock 6213
    python main.py --stock 2330 --no-ai
    python main.py --stock 0050 --save
"""

import argparse
import sys

from data.fetcher import FinMindFetcher
from factors import compute_chips, compute_technical, compute_fundamental, compute_momentum
from models.scorer import Scorer
from config import FACTOR_WEIGHTS, FINMIND_TOKEN, ANTHROPIC_API_KEY


def _bar(score: float, width: int = 30) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _print_scores(result: dict, stock_id: str):
    total = result["total_score"]
    rec = result["recommendation"]
    cat = result["category_scores"]
    label_map = {
        "chips": "籌碼面", "fundamental": "基本面",
        "technical": "技術面", "momentum": "動能面", "risk": "風險面",
    }
    print("\n" + "=" * 50)
    print(f"  股票代號：{stock_id}")
    print(f"  綜合評分：{total:.1f} / 100")
    print(f"  投資建議：{rec}")
    print("=" * 50)
    for key, label in label_map.items():
        s = cat.get(key, 0)
        print(f"  {label:6s} [{_bar(s)}] {s:.1f}")
    print("=" * 50 + "\n")


def run(stock_id: str, use_ai: bool = True, save_report: bool = False):
    print(f"\n[1/4] 正在取得 {stock_id} 市場資料...")
    if not FINMIND_TOKEN:
        print("      ⚠️  未設定 FINMIND_TOKEN，籌碼/基本面將使用中性值")

    fetcher = FinMindFetcher(stock_id)
    price_df = fetcher.get_price()
    if price_df.empty:
        print(f"❌ 無法取得股價資料，請確認股票代號 {stock_id} 是否正確")
        sys.exit(1)

    institutional_df = fetcher.get_institutional()
    margin_df = fetcher.get_margin_trading()
    revenue_df = fetcher.get_monthly_revenue()
    financial_df = fetcher.get_financial_statements()

    current_price = float(price_df["close"].iloc[-1]) if not price_df.empty else 0.0

    print(f"[2/4] 計算因子中（股價資料：{len(price_df)} 筆）...")
    chips = compute_chips(institutional_df, margin_df)
    technical = compute_technical(price_df)
    fundamental = compute_fundamental(revenue_df, financial_df, current_price)
    momentum = compute_momentum(price_df)

    print("[3/4] 評分計算中...")
    scorer = Scorer(FACTOR_WEIGHTS)
    result = scorer.score(chips, technical, fundamental, momentum)
    _print_scores(result, stock_id)

    ai_advice = ""
    if use_ai:
        if not ANTHROPIC_API_KEY:
            print("⚠️  未設定 ANTHROPIC_API_KEY，跳過 AI 建議")
        else:
            print("[4/4] 取得 AI 投資建議...")
            from utils.claude_api import get_investment_advice
            ai_advice = get_investment_advice(result, stock_id)
            print("\n── AI 投資建議 ─────────────────────────────────")
            print(ai_advice)
            print("─" * 50 + "\n")
    else:
        print("[4/4] 已略過 AI 建議（--no-ai）")

    if save_report:
        from utils.report import save_html_report
        path = save_html_report(
            stock_id, result, price_df, institutional_df, revenue_df, ai_advice
        )
        print(f"✅ HTML 報告已儲存：{path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="台股多因子評分系統")
    parser.add_argument("--stock", "-s", help="股票代號，例如：2330")
    parser.add_argument("--no-ai", action="store_true", help="不呼叫 Claude API")
    parser.add_argument("--save", action="store_true", help="輸出 HTML 報告至 output/")
    parser.add_argument("--scan", action="store_true", help="執行全市場掃描並輸出今日推薦")
    parser.add_argument("--scan-dry-run", action="store_true", help="掃描但不推播、不寫資料庫（測試用）")
    parser.add_argument("--scan-top", type=int, default=None, help="覆蓋推薦輸出數量")
    args = parser.parse_args()

    if args.scan or args.scan_dry_run:
        from screener.recommender import DailyRecommender
        from config import SCREENER_TOP_N
        import config as _cfg

        if args.scan_top:
            _cfg.SCREENER_TOP_N = args.scan_top

        recommender = DailyRecommender()
        result = recommender.run(dry_run=args.scan_dry_run)

        if result.get("error"):
            print(f"❌ 掃描失敗：{result['error']}")
            sys.exit(1)

        print(result["message"])

        if not args.scan_dry_run and result["recommendations"]:
            from alerts.notifier import Notifier
            notifier = Notifier()
            notifier.send_telegram(result["message"])
            print(f"\n✅ 已推播 {len(result['recommendations'])} 支推薦至 Telegram")

        sys.exit(0)

    if not args.stock:
        parser.print_help()
        sys.exit(1)

    run(args.stock, use_ai=not args.no_ai, save_report=args.save)


if __name__ == "__main__":
    main()
