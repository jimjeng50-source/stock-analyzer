"""
reports/research_report.py
完整研究報告生成器

整合 ForwardEPSCalculator + SupplyChainAnalyzer +
compute_financial_quality_metrics + scorer 結果
→ 呼叫 Claude API 生成完整繁體中文研究報告
"""

import logging
from typing import Optional

from data.fetcher import DataFetcher
from factors.forward_eps import ForwardEPSCalculator
from factors.supply_chain import SupplyChainAnalyzer
from factors.fundamental import compute_financial_quality_metrics
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from utils.tz import now_tw

logger = logging.getLogger(__name__)


class ResearchReportGenerator:
    """
    整合所有分析維度，呼叫 Claude API 生成完整的自然語言研究報告。
    """

    def __init__(self, fetcher: DataFetcher, anthropic_client=None):
        self.fetcher = fetcher
        self.client = anthropic_client
        self.forward_eps_calc = ForwardEPSCalculator(fetcher)
        self.chain_analyzer = SupplyChainAnalyzer(fetcher)

    def generate(
        self, stock_id: str, existing_score_result: dict = None
    ) -> dict:
        """
        生成完整研究報告。

        Args:
            stock_id: 股票代號
            existing_score_result: 若已有 scorer 結果，直接傳入避免重複計算

        Returns:
            {
                "stock_id": str,
                "report_text": str,
                "report_summary": str,
                "forward_eps_data": dict,
                "quality_metrics": dict,
                "chain_data": dict,
                "target_price": dict,
                "generated_at": str,
                "error": Optional[str],
            }
        """
        result = {
            "stock_id": stock_id,
            "report_text": "",
            "report_summary": "",
            "forward_eps_data": {},
            "quality_metrics": {},
            "chain_data": {},
            "target_price": {},
            "generated_at": now_tw().strftime("%Y-%m-%d %H:%M"),
            "error": None,
        }

        try:
            # 計算各維度數據
            forward_eps = self.forward_eps_calc.calculate(stock_id)
            quality = compute_financial_quality_metrics(stock_id)
            chain = self.chain_analyzer.analyze_for_stock(stock_id)
            news = self._get_news_summary(stock_id)

            result["forward_eps_data"] = forward_eps
            result["quality_metrics"] = quality
            result["chain_data"] = chain
            result["target_price"] = forward_eps.get("target_price", {})

            # 呼叫 Claude API 生成報告
            if not self.client and not ANTHROPIC_API_KEY:
                result["report_text"] = self._build_fallback_report(
                    stock_id, forward_eps, quality, chain
                )
                result["report_summary"] = self._extract_summary(result["report_text"])
                return result

            prompt = self._build_claude_prompt(
                stock_id, forward_eps, quality, chain,
                existing_score_result or {}, news
            )

            client = self.client
            if client is None:
                import anthropic
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            report_text = response.content[0].text
            result["report_text"] = report_text
            result["report_summary"] = self._extract_summary(report_text)

        except Exception as e:
            logger.error("研究報告生成失敗 %s：%s", stock_id, e)
            result["error"] = f"生成失敗：{str(e)}"
            if not result["report_text"]:
                result["report_text"] = self._build_fallback_report(
                    stock_id,
                    result.get("forward_eps_data", {}),
                    result.get("quality_metrics", {}),
                    result.get("chain_data", {}),
                )

        return result

    def _build_claude_prompt(
        self,
        stock_id: str,
        forward_eps: dict,
        quality: dict,
        chain: dict,
        score_result: dict,
        news_summary: str,
    ) -> str:
        """組合傳給 Claude API 的 Prompt。"""
        ttm = forward_eps.get("ttm_eps", "N/A")
        fwd = forward_eps.get("forward_eps_1y", "N/A")
        growth = forward_eps.get("eps_growth_rate", 0)
        growth_pct = f"{growth * 100:.1f}%" if growth is not None else "N/A"
        tp = forward_eps.get("target_price", {})
        base_tp = tp.get("base", "N/A")
        bull_tp = tp.get("bull", "N/A")
        bear_tp = tp.get("bear", "N/A")
        peg = forward_eps.get("peg_ratio", "N/A")
        current_price = forward_eps.get("current_price", "N/A")
        confidence = forward_eps.get("confidence", "low")
        upside = forward_eps.get("upside_pct", "N/A")

        dsi = quality.get("dsi", "N/A")
        dso = quality.get("dso", "N/A")
        fcf_yield = quality.get("fcf_yield", "N/A")
        capex_int = quality.get("capex_intensity", "N/A")
        q_label = quality.get("quality_label", "普通")
        highlights = "\n".join(f"  - {h}" for h in quality.get("highlights", []))
        concerns = "\n".join(f"  - {c}" for c in quality.get("concerns", []))

        chain_name = chain.get("chain_name", "不在追蹤產業鏈")
        chain_sig = chain.get("chain_signal", 0.0)
        upstream_sig = chain.get("upstream_signal", 0.0)
        lead_lag_impact = chain.get("lead_lag_impact", "中性")
        expected_impact = chain.get("expected_impact_in", "N/A")
        chain_interp = chain.get("interpretation", "")

        total_score = score_result.get("total_score", "N/A")
        recommendation = score_result.get("recommendation", "N/A")

        today_str = now_tw().strftime("%Y-%m-%d")

        return f"""
你是一位台股研究分析師，請根據以下量化數據，撰寫一份完整的個股研究報告（繁體中文，Markdown 格式）。

=== 基本資訊 ===
股票代號：{stock_id}
分析日期：{today_str}
當前股價：NT${current_price}
多因子評分：{total_score} / 100　{recommendation}

=== Forward EPS 分析數據 ===
TTM EPS（近四季合計）：{ttm} 元
推估 Forward EPS（1年）：{fwd} 元
EPS 成長率假設：{growth_pct}
三情境目標價：熊市 NT${bear_tp} / 基準 NT${base_tp} / 牛市 NT${bull_tp}
距基準目標漲幅：{upside}%
PEG Ratio：{peg}
推估信心度：{confidence}

=== 財務品質指標 ===
整體品質評等：{q_label}
存貨周轉天數（DSI）：{dsi} 天
應收帳款天數（DSO）：{dso} 天
自由現金流佔營收（FCF Yield）：{fcf_yield}%
資本支出強度（Capex/Revenue）：{capex_int}%
主要亮點：
{highlights or "  （無）"}
主要隱憂：
{concerns or "  （無）"}

=== 產業鏈分析 ===
所在產業鏈：{chain_name}
整體鏈信號：{chain_sig:+.2f}
上游信號：{upstream_sig:+.2f}
Lead-Lag 影響：{lead_lag_impact}（{expected_impact}）
{chain_interp}

=== 近期新聞摘要 ===
{news_summary or "（無法取得新聞）"}

---

請依照以下 Markdown 結構撰寫報告：

## {stock_id} 投資研究報告
**生成日期**: {today_str}  |  **信心等級**: {confidence}

### 核心投資論點
（3 句話，說明為何此時值得或不值得關注）

### Forward EPS 分析
（解說 TTM EPS、成長假設、目標價三情境，說明 PEG 合理性）

### 財務品質評估
（解說 DSI/DSO/FCF Yield/Capex，點出亮點或隱憂）

### 產業鏈資金流向
（說明上游信號、Lead-Lag 推論、資金可能的移動方向）

### 主要催化劑
- 上行催化劑 1
- 上行催化劑 2

### 主要風險
- 下行風險 1
- 下行風險 2

### 操作參考
（目標價區間、時間軸、停損參考價位——請加免責聲明）

---
⚠️ 本報告由 AI 自動生成，僅供學習參考，不構成投資建議。
"""

    def _get_news_summary(self, stock_id: str) -> str:
        """抓取近 5 則新聞標題，整理成一段文字。"""
        try:
            import yfinance as yf
            for suffix in [".TW", ".TWO"]:
                ticker = yf.Ticker(f"{stock_id}{suffix}")
                news = ticker.news
                if news:
                    titles = [n.get("title", "") for n in news[:5] if n.get("title")]
                    if titles:
                        return "\n".join(f"- {t}" for t in titles)
        except Exception:
            pass
        return ""

    def _build_fallback_report(
        self,
        stock_id: str,
        forward_eps: dict,
        quality: dict,
        chain: dict,
    ) -> str:
        """無法呼叫 Claude API 時的備援靜態報告。"""
        tp = forward_eps.get("target_price", {})
        lines = [
            f"## {stock_id} 投資研究報告",
            f"**生成日期**: {now_tw().strftime('%Y-%m-%d')}  |  （未連接 AI 生成，顯示數據摘要）",
            "",
            "### Forward EPS 概覽",
            f"- TTM EPS：{forward_eps.get('ttm_eps', 'N/A')} 元",
            f"- Forward EPS：{forward_eps.get('forward_eps_1y', 'N/A')} 元",
            f"- 成長率假設：{(forward_eps.get('eps_growth_rate') or 0) * 100:.1f}%",
            f"- 目標價（基準）：NT${tp.get('base', 'N/A')}",
            "",
            "### 財務品質",
            f"- 品質評等：{quality.get('quality_label', 'N/A')}",
            f"- DSI：{quality.get('dsi', 'N/A')} 天",
            f"- DSO：{quality.get('dso', 'N/A')} 天",
            f"- FCF Yield：{quality.get('fcf_yield', 'N/A')}%",
            "",
            "### 產業鏈",
            chain.get("interpretation", "無產業鏈資料"),
            "",
            "---",
            "⚠️ 本報告數據自動計算，僅供學習參考，不構成投資建議。",
            "設定 ANTHROPIC_API_KEY 後可啟用完整 AI 分析報告。",
        ]
        return "\n".join(lines)

    def _extract_summary(self, report_text: str) -> str:
        """從報告中提取核心投資論點作為摘要（最多 100 字）。"""
        marker = "### 核心投資論點"
        if marker in report_text:
            start = report_text.index(marker) + len(marker)
            end = report_text.find("\n###", start)
            section = report_text[start:end].strip() if end > 0 else report_text[start:].strip()
            # 去掉 markdown 符號
            lines = [ln.strip("# *-").strip() for ln in section.splitlines() if ln.strip()]
            summary = " ".join(lines)
            return summary[:100] + "..." if len(summary) > 100 else summary
        return report_text[:100] + "..." if len(report_text) > 100 else report_text
