"""
screener/recommender.py
主動推薦主控器

這是 v4 的核心入口，對外提供 DailyRecommender.run() 函數。
"""

import json
import logging
import time
from datetime import date
from typing import Optional

import pandas as pd

from screener.universe import UniverseManager
from screener.filter import QuickFilter, FilterConfig
from screener.batch_scorer import BatchScorer
from screener.recommendation_db import RecommendationDB
from config import (
    SCREENER_TOP_N,
    SCREENER_QUICK_SCORE_THRESHOLD,
    SCREENER_MIN_RECOMMEND_SCORE,
    SCREENER_REGIME_FILTER,
    FORWARD_EPS_RERANK_WEIGHT,
    CLAUDE_MODEL,
    get_runtime_config,
)

logger = logging.getLogger(__name__)


class DailyRecommender:
    """
    每日主動推薦器

    完整執行流程：
    Phase 1 - 建立候選池（UniverseManager）
    Phase 2 - 快速過濾（QuickFilter）
    Phase 3 - 批次評分（BatchScorer）→ 取 Top 20
    Phase 4 - 深度分析（Forward EPS + 產業鏈）
    Phase 5 - Claude API 生成推薦理由（每支 Top N 各一份）
    Phase 6 - 儲存推薦紀錄
    Phase 7 - 組裝推播訊息
    """

    def __init__(self):
        from data.fetcher import FinMindFetcher, DataFetcher
        self.fetcher = FinMindFetcher("")
        self.data_fetcher = DataFetcher()
        self.universe_mgr = UniverseManager(self.fetcher)
        self.filter = QuickFilter(self.fetcher)
        self.batch_scorer = BatchScorer()
        self.db = RecommendationDB()

    def run(self, dry_run: bool = False) -> dict:
        """
        執行完整推薦流程。

        Args:
            dry_run: True 時不推播、不寫資料庫，只回傳結果（測試用）

        Returns:
            {
                "date": str,
                "recommendations": list[dict],
                "scan_summary": dict,
                "market_context": str,
                "message": str,       # 格式化好的 Telegram 推播訊息
                "error": Optional[str],
            }
        """
        today = date.today()
        start_time = time.time()

        result = {
            "date": today.isoformat(),
            "recommendations": [],
            "watch_list": [],
            "scan_summary": {},
            "market_context": "",
            "message": "",
            "error": None,
            "no_candidates": False,
        }

        try:
            # Phase 1 — 候選池
            logger.info("Phase 1: 建立候選股票池...")
            universe_df = self.universe_mgr.get_universe()
            universe_df = self.universe_mgr.merge_with_custom(universe_df)

            if universe_df.empty:
                result["error"] = "無法取得股票清單"
                return result

            # Phase 1.5 — 熱門股偵測（籌碼/社群/量能），併入候選池並註記來源
            hot_tags_map = self._detect_hot_stocks(universe_df)
            universe_df = self._merge_hot_stocks(universe_df, hot_tags_map)
            universe_count = len(universe_df)
            logger.info("候選池：%d 支（含 %d 支熱門股）", universe_count, len(hot_tags_map))

            # Phase 2 — 快速過濾
            logger.info("Phase 2: 快速過濾...")
            passed_df, filter_report = self.filter.run(universe_df)
            after_filter_count = len(passed_df)
            logger.info("過濾後：%d 支", after_filter_count)

            if passed_df.empty:
                # 非硬錯誤：流程正常但今日無符合過濾條件個股
                result["no_candidates"] = True
                result["scan_summary"] = {
                    "universe_count": universe_count,
                    "after_filter_count": 0,
                    "scored_count": 0, "failed_count": 0,
                    "scan_duration_sec": round(time.time() - start_time, 1),
                    "top_score": 0,
                }
                result["message"] = self._format_message(result)
                return result

            # Phase 3 — 批次評分，取 Top 20
            logger.info("Phase 3: 批次評分（%d 支）...", after_filter_count)
            scored_df = self.batch_scorer.score_universe(
                passed_df["stock_id"].tolist(), show_progress=True
            )
            scored_count = len(scored_df)
            failed_count = len(self.batch_scorer.failed_df)

            # 合併股票名稱和產業別
            if "stock_name" in passed_df.columns:
                name_map = passed_df.set_index("stock_id")[["stock_name", "industry"]].to_dict("index")
                scored_df["stock_name"] = scored_df["stock_id"].map(
                    lambda x: name_map.get(x, {}).get("stock_name", x)
                )
                scored_df["industry"] = scored_df["stock_id"].map(
                    lambda x: name_map.get(x, {}).get("industry", "")
                )

            # 依門檻篩選並取 Top 20
            top20_df = scored_df[
                scored_df["total_score"] >= SCREENER_QUICK_SCORE_THRESHOLD
            ].head(20)

            if top20_df.empty:
                # 非硬錯誤：評分完成但無個股達快篩門檻。附最高分觀察名單，正常回傳。
                result["no_candidates"] = True
                result["watch_list"] = [
                    {
                        "stock_id": str(row["stock_id"]),
                        "stock_name": row.get("stock_name", row["stock_id"]),
                        "total_score": round(float(row["total_score"]), 1),
                        "current_price": row.get("current_price"),
                        "hot_tags": hot_tags_map.get(str(row["stock_id"]), []),
                    }
                    for _, row in scored_df.head(3).iterrows()
                ]
                result["scan_summary"] = {
                    "universe_count": universe_count,
                    "after_filter_count": after_filter_count,
                    "scored_count": scored_count, "failed_count": failed_count,
                    "scan_duration_sec": round(time.time() - start_time, 1),
                    "top_score": float(scored_df["total_score"].max()) if not scored_df.empty else 0,
                }
                result["market_context"] = self._generate_market_context(scored_df)
                result["message"] = self._format_message(result)
                logger.info(
                    "無個股達快篩門檻（%d 分），最高分 %.1f",
                    SCREENER_QUICK_SCORE_THRESHOLD,
                    float(scored_df["total_score"].max()) if not scored_df.empty else 0,
                )
                return result

            # Phase 4 — 深度分析（Forward EPS + 產業鏈）
            logger.info("Phase 4: 深度分析 Top %d...", len(top20_df))
            deep_results = self._run_deep_analysis(top20_df)

            # Phase 4.5 — Forward EPS 前瞻分數參與最終排名（基本面優先）
            top20_df = self._rerank_with_forward_eps(top20_df, deep_results)

            # Phase 4.6 — 大盤狀態閘門：系統性風險時降級為觀察名單
            regime_warnings = self._check_market_regime()
            result["regime_warnings"] = regime_warnings

            # Phase 5 — Claude API 推薦理由
            logger.info("Phase 5: 生成推薦理由...")
            recommendations = []
            final_df = top20_df[
                top20_df["total_score"] >= SCREENER_MIN_RECOMMEND_SCORE
            ].head(SCREENER_TOP_N)

            for rank, (_, row) in enumerate(final_df.iterrows(), 1):
                sid = row["stock_id"]
                deep = deep_results.get(sid, {})
                stock_data = row.to_dict()
                stock_data.update(deep)

                reasons = self._generate_recommendation_reason(stock_data, stock_data)

                rec = {
                    "rank": rank,
                    "stock_id": sid,
                    "stock_name": row.get("stock_name", sid),
                    "total_score": row.get("total_score"),
                    "recommendation": row.get("recommendation", ""),
                    "current_price": row.get("current_price"),
                    "key_reasons": reasons,
                    "target_price_base": deep.get("target_price_base"),
                    "upside_pct": deep.get("upside_pct"),
                    "forward_eps": deep.get("forward_eps"),
                    "eps_growth_rate": deep.get("eps_growth_rate"),
                    "risk_warning": self._generate_risk_warning(stock_data),
                    "industry": row.get("industry", ""),
                    "score_breakdown": {
                        "chips_score": row.get("chips_score"),
                        "fundamental_score": row.get("fundamental_score"),
                        "technical_score": row.get("technical_score"),
                        "momentum_score": row.get("momentum_score"),
                    },
                    "hot_tags": hot_tags_map.get(sid, []),
                    "base_score": row.get("base_score"),
                    "forward_score": row.get("forward_score"),
                }
                recommendations.append(rec)

            # 大盤閘門啟動 → 推薦降級為觀察名單（不建倉、不寫 DB）
            if recommendations and regime_warnings:
                logger.warning("大盤閘門啟動（%d 則警訊），推薦降級為觀察名單", len(regime_warnings))
                result["watch_list"] = [
                    {
                        "stock_id": r["stock_id"],
                        "stock_name": r["stock_name"],
                        "total_score": round(float(r["total_score"]), 1),
                        "current_price": r["current_price"],
                        "hot_tags": r["hot_tags"],
                    }
                    for r in recommendations
                ]
                recommendations = []

            result["recommendations"] = recommendations

            # 無個股達推薦門檻 → 附觀察名單（明示未達標，不算正式推薦）
            if not recommendations and not top20_df.empty:
                result["watch_list"] = [
                    {
                        "stock_id": str(row["stock_id"]),
                        "stock_name": row.get("stock_name", row["stock_id"]),
                        "total_score": round(float(row["total_score"]), 1),
                        "current_price": row.get("current_price"),
                        "hot_tags": hot_tags_map.get(str(row["stock_id"]), []),
                    }
                    for _, row in top20_df.head(3).iterrows()
                ]
                logger.info(
                    "無個股達推薦門檻（%d 分），最高分 %.1f，回傳觀察名單 %d 支",
                    SCREENER_MIN_RECOMMEND_SCORE,
                    float(top20_df["total_score"].max()),
                    len(result["watch_list"]),
                )

            # Phase 6 — 大盤背景
            result["market_context"] = self._generate_market_context(scored_df)

            # 掃描摘要
            scan_duration = time.time() - start_time
            scan_summary = {
                "universe_count": universe_count,
                "after_filter_count": after_filter_count,
                "scored_count": scored_count,
                "failed_count": failed_count,
                "scan_duration_sec": round(scan_duration, 1),
                "top_score": float(scored_df["total_score"].max()) if not scored_df.empty else 0,
            }
            result["scan_summary"] = scan_summary

            # Phase 7 — 推播訊息
            result["message"] = self._format_message(result)

            # Phase 6b — 儲存紀錄
            if not dry_run and recommendations:
                self.db.save_recommendations(today, recommendations)
                self.db.save_scan_log(today, scan_summary)
                logger.info("推薦紀錄已儲存（%d 支）", len(recommendations))
                self._sync_revenue_watchlist()

        except Exception as e:
            logger.error("每日推薦流程異常：%s", e, exc_info=True)
            result["error"] = f"推薦流程異常：{str(e)}"

        return result

    def _sync_revenue_watchlist(self) -> None:
        """掃描完成後，把近 60 天推薦股同步進月營收追蹤清單。"""
        try:
            from alerts.revenue_calendar import RevenueCalendar
            calendar = RevenueCalendar(self.data_fetcher)
            stats = calendar.sync_from_recommendations(n_days=60)
            logger.info(
                "月營收追蹤清單已同步（+%d / -%d，共 %d 支）",
                stats["added"], stats["removed"], stats["total"],
            )
        except Exception as e:
            logger.warning("月營收追蹤清單同步失敗：%s", e)

    # ── Phase 1.5：熱門股偵測 ──────────────────────────────────────────────────

    def _detect_hot_stocks(self, universe_df: pd.DataFrame) -> dict:
        """偵測熱門股（籌碼/社群/量能）。失敗時回傳空 dict，不影響主流程。"""
        try:
            from screener.hot_stocks import HotStockDetector
            return HotStockDetector().detect_all(universe_df)
        except Exception as e:
            logger.warning("熱門股偵測失敗：%s", e)
            return {}

    def _merge_hot_stocks(self, universe_df: pd.DataFrame, hot_tags_map: dict) -> pd.DataFrame:
        """把不在候選池中的熱門股追加進去（與自訂名單同樣待遇）。"""
        if not hot_tags_map:
            return universe_df
        existing = set(universe_df["stock_id"].astype(str).tolist())
        missing = [sid for sid in hot_tags_map if sid not in existing]
        if not missing:
            return universe_df

        extra_rows = []
        for sid in missing:
            extra_rows.append({
                "stock_id": sid,
                "stock_name": sid,
                "market": "HOT",
                "industry": "熱門追加",
                "market_cap_b": float("nan"),
                "avg_volume_k": float("nan"),
                "last_price": float("nan"),
            })
        merged = pd.concat([universe_df, pd.DataFrame(extra_rows)], ignore_index=True)
        logger.info("熱門股追加 %d 支進候選池", len(missing))
        return merged

    # ── Phase 4.5：Forward EPS 前瞻排名 ────────────────────────────────────────

    def _rerank_with_forward_eps(
        self, top_df: pd.DataFrame, deep_results: dict
    ) -> pd.DataFrame:
        """
        用深度分析算出的 Forward EPS 成長率與上檔空間，
        計算 0-100 的「前瞻分數」並與基礎分混合重排。
        缺 Forward EPS 資料的個股取中性 50 分（不懲罰、不加分）。
        final = (1-w)*基礎分 + w*前瞻分，w = FORWARD_EPS_RERANK_WEIGHT
        """
        import math

        def _forward_score(sid: str) -> float:
            deep = deep_results.get(sid, {})
            growth = deep.get("eps_growth_rate")
            upside = deep.get("upside_pct")
            if growth is None and upside is None:
                return 50.0
            parts = []
            if growth is not None:
                # 成長率 sigmoid：+10% 成長 → 50 分中心，±15% 靈敏度
                parts.append(100 / (1 + math.exp(-(growth - 10) / 15)))
            if upside is not None:
                # 上檔空間 sigmoid：+5% → 50 分中心
                parts.append(100 / (1 + math.exp(-(upside - 5) / 15)))
            return sum(parts) / len(parts)

        df = top_df.copy()
        df["base_score"] = df["total_score"]
        df["forward_score"] = df["stock_id"].map(lambda s: round(_forward_score(s), 1))
        w = FORWARD_EPS_RERANK_WEIGHT
        df["total_score"] = (1 - w) * df["base_score"] + w * df["forward_score"]
        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
        logger.info(
            "Forward EPS 重排完成（權重 %.0f%%），新 Top 3：%s",
            w * 100, ", ".join(df["stock_id"].head(3).astype(str)),
        )
        return df

    # ── Phase 4.6：大盤狀態閘門 ────────────────────────────────────────────────

    def _check_market_regime(self) -> list:
        """大盤系統性風險檢查。停用或檢查失敗時回傳空 list（不擋推薦）。"""
        if not SCREENER_REGIME_FILTER:
            return []
        try:
            from alerts.risk_monitor import RiskMonitor
            return RiskMonitor().check_market_risk()
        except Exception as e:
            logger.warning("大盤閘門檢查失敗（不擋推薦）：%s", e)
            return []

    # ── Phase 4：深度分析 ──────────────────────────────────────────────────────

    def _run_deep_analysis(self, top_df: pd.DataFrame) -> dict:
        """對 Top 20 執行 Forward EPS + 產業鏈深度分析。"""
        results = {}
        try:
            from factors.forward_eps import ForwardEPSCalculator
            from factors.supply_chain import SupplyChainAnalyzer
            eps_calc = ForwardEPSCalculator(self.data_fetcher)
            chain_analyzer = SupplyChainAnalyzer(self.data_fetcher)
        except Exception as e:
            logger.warning("深度分析模組載入失敗：%s", e)
            return results

        for _, row in top_df.iterrows():
            sid = row["stock_id"]
            deep = {}
            try:
                eps_result = eps_calc.calculate(sid)
                if not eps_result.get("error"):
                    deep["target_price_base"] = (eps_result.get("target_price") or {}).get("base")
                    deep["upside_pct"] = eps_result.get("upside_pct")
                    deep["forward_eps"] = eps_result.get("forward_eps_1y")
                    deep["eps_growth_rate"] = eps_result.get("eps_growth_rate")
            except Exception as e:
                logger.debug("Forward EPS 失敗 %s：%s", sid, e)

            try:
                chain_result = chain_analyzer.analyze_for_stock(sid)
                deep["chain_score"] = chain_result.get("chain_score")
                deep["chain_signal"] = chain_result.get("chain_signal")
                deep["chain_name"] = chain_result.get("chain_name")
            except Exception as e:
                logger.debug("產業鏈分析失敗 %s：%s", sid, e)

            results[sid] = deep

        return results

    # ── Phase 5：Claude 推薦理由 ───────────────────────────────────────────────

    def _generate_recommendation_reason(
        self, stock_data: dict, score_result: dict
    ) -> list:
        """呼叫 Claude API 生成 3 條推薦理由。無 API key 或失敗時用結構化理由。"""
        if not get_runtime_config("ANTHROPIC_API_KEY"):
            return self._build_structured_reasons(stock_data)

        try:
            import anthropic

            prompt = (
                f"你是台股量化分析系統，請根據以下評分資料，生成恰好 3 條簡潔的推薦理由。\n\n"
                f"股票代號：{stock_data.get('stock_id', '')} {stock_data.get('stock_name', '')}\n"
                f"綜合評分：{score_result.get('total_score', 0):.1f}/100\n"
                f"籌碼分：{score_result.get('chips_score', 0):.0f}\n"
                f"基本面分：{score_result.get('fundamental_score', 0):.0f}\n"
                f"技術面分：{score_result.get('technical_score', 0):.0f}\n"
                f"動能分：{score_result.get('momentum_score', 0):.0f}\n"
                f"當前股價：{stock_data.get('current_price', 0):.1f} 元\n"
            )
            if stock_data.get("upside_pct") is not None:
                prompt += f"目標漲幅：{stock_data['upside_pct']:+.1f}%\n"
            if stock_data.get("chain_name"):
                prompt += f"所在產業鏈：{stock_data['chain_name']}\n"

            prompt += (
                "\n請輸出：\n"
                "- 恰好 3 條推薦理由，每條 20-40 字\n"
                "- 純繁體中文\n"
                '- 只輸出 JSON array，例如：["理由1", "理由2", "理由3"]\n'
                "- 不要有其他文字或 markdown\n"
            )

            client = anthropic.Anthropic(api_key=get_runtime_config("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # 去掉可能的 code block
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            reasons = json.loads(raw)
            if isinstance(reasons, list) and len(reasons) >= 3:
                return reasons[:3]
        except Exception as e:
            logger.warning("Claude 推薦理由生成失敗 %s: %s", stock_data.get("stock_id"), e)

        return self._build_structured_reasons(stock_data)

    def _build_structured_reasons(self, d: dict) -> list:
        """
        規則式多面向推薦理由：每個面向一句帶實際數據的理由。
        不依賴 Claude API，依面向強度排序取前 3-4 條。
        """
        candidates = []  # (優先分數, 理由文字)

        # 基本面（含 Forward EPS）— 基本面優先，permanently first if data exists
        f_score = d.get("fundamental_score") or 0
        feps = d.get("forward_eps")
        growth = d.get("eps_growth_rate")
        if feps is not None and growth is not None:
            candidates.append((
                f_score + 100,   # 基本面理由永遠優先
                f"基本面：Forward EPS {feps:.2f} 元、預估年成長 {growth:+.0f}%（基本面分 {f_score:.0f}/100）",
            ))
        elif f_score >= 60:
            candidates.append((f_score + 100, f"基本面：營收與獲利體質穩健（基本面分 {f_score:.0f}/100）"))

        # 目標價空間
        tp = d.get("target_price_base")
        upside = d.get("upside_pct")
        if tp and upside is not None and upside > 0:
            candidates.append((70 + min(upside, 30), f"估值：Forward EPS 推算目標價 {tp:.0f} 元，上檔空間 {upside:+.0f}%"))

        # 籌碼面
        c_score = d.get("chips_score") or 0
        if c_score >= 70:
            candidates.append((c_score, f"籌碼面：法人資金明顯偏多（籌碼分 {c_score:.0f}/100）"))
        elif c_score >= 60:
            candidates.append((c_score, f"籌碼面：法人動向溫和偏多（籌碼分 {c_score:.0f}/100）"))

        # 產業鏈
        chain = d.get("chain_name")
        chain_signal = d.get("chain_signal")
        if chain and chain_signal is not None and chain_signal > 0:
            candidates.append((65, f"產業面：所屬{chain}產業鏈信號偏多（{chain_signal:+.2f}）"))

        # 技術/動能（輔助）
        t_score = d.get("technical_score") or 0
        m_score = d.get("momentum_score") or 0
        if t_score >= 65 and m_score >= 65:
            candidates.append((min(t_score, m_score) - 10,
                               f"技術面：趨勢與動能同步向上（技術 {t_score:.0f}／動能 {m_score:.0f}）"))
        elif t_score >= 70:
            candidates.append((t_score - 10, f"技術面：中期趨勢結構偏多（技術分 {t_score:.0f}/100）"))

        candidates.sort(key=lambda x: x[0], reverse=True)
        reasons = [text for _, text in candidates[:4]]
        return reasons if reasons else self._fallback_reasons(d)

    def _fallback_reasons(self, score_result: dict) -> list:
        """從評分資料自動組合推薦理由（Claude API 失敗時使用）。"""
        factor_templates = {
            "chips_score": "籌碼面法人持續買超，機構資金積極佈局",
            "fundamental_score": "基本面營收成長且獲利改善，基本面支撐穩固",
            "technical_score": "技術面均線多頭排列，短中期趨勢向上",
            "momentum_score": "近期股價動能強勁，相對大盤表現突出",
            "risk_score": "波動度相對低，風險控制條件良好",
        }
        scores = {k: score_result.get(k, 0) or 0 for k in factor_templates}
        top3 = sorted(scores, key=lambda k: scores[k], reverse=True)[:3]
        return [factor_templates[k] for k in top3]

    def _generate_risk_warning(self, stock_data: dict) -> str:
        """生成風險提示（從最低分因子推導）。"""
        factor_risks = {
            "chips_score": "外資籌碼出現鬆動，注意追蹤法人動向",
            "fundamental_score": "基本面動能有限，需觀察後續財報表現",
            "technical_score": "技術面尚未突破關鍵壓力，建議等待確認",
            "momentum_score": "短期動能偏弱，留意回調風險",
            "risk_score": "股價波動較大，注意倉位控制",
        }
        min_factor = min(
            factor_risks,
            key=lambda k: stock_data.get(k, 50) or 50,
        )
        return factor_risks[min_factor]

    def _generate_market_context(self, scored_df: pd.DataFrame) -> str:
        """從整體評分分布生成大盤背景描述。"""
        if scored_df.empty:
            return "市場資料不足"
        avg_score = scored_df["total_score"].mean()
        high_pct = (scored_df["total_score"] >= 70).sum() / len(scored_df) * 100
        if avg_score >= 65:
            return f"整體市場偏多，{high_pct:.0f}% 個股評分在 70 分以上，多頭格局延續"
        elif avg_score >= 50:
            return f"市場中性盤整，{high_pct:.0f}% 個股達強勢門檻，選股替代擇時"
        else:
            return f"整體市場偏弱，僅 {high_pct:.0f}% 個股評分達強勢門檻，建議謹慎"

    # ── Phase 7：訊息格式化 ────────────────────────────────────────────────────

    def _format_message(self, result: dict) -> str:
        """格式化 Telegram 推播訊息。"""
        today_str = result["date"]
        recs = result["recommendations"]
        summary = result.get("scan_summary", {})
        context = result.get("market_context", "")
        n = len(recs)

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}
        lines = [
            "╔══════════════════════╗",
            f"║  📈 每日選股推薦報告  ║",
            f"║  {today_str}        ║",
            "╚══════════════════════╝",
            "",
            f"🌐 大盤背景：{context}",
        ]

        # 無達標推薦 → 說明原因 + 觀察名單
        if n == 0:
            top_score = summary.get("top_score", 0)
            failed = summary.get("failed_count", 0)
            scored = summary.get("scored_count", 0)
            regime = result.get("regime_warnings", [])
            if regime:
                lines += ["", "🚦 大盤風險閘門啟動 — 今日推薦降級為觀察名單（不建議建倉）"]
                lines += regime
            else:
                lines += [
                    "",
                    f"📭 今日無個股達推薦門檻（{SCREENER_MIN_RECOMMEND_SCORE} 分）",
                    f"最高分：{top_score:.0f} 分",
                ]
            if scored and failed and failed >= scored * 0.3:
                lines.append(
                    f"⚠️ 注意：{failed}/{scored + failed} 支評分失敗（資料取得問題），"
                    "分數可能被低估，建議稍後重新掃描"
                )
            watch = result.get("watch_list", [])
            if watch:
                lines += ["", "━━ 觀察名單（未達標，僅供參考）━━"]
                for w in watch:
                    price = w.get("current_price") or 0
                    lines.append(
                        f"👀 {w['stock_id']} {w.get('stock_name', '')}"
                        f"　{w['total_score']:.0f} 分｜{price:.0f} 元"
                    )
            lines += [
                "",
                "⚠️ 本報告由量化模型自動生成，僅供學習與研究參考，不構成任何投資建議。",
            ]
            return "\n".join(lines)

        lines += [
            "",
            f"━━━ Top {n} 推薦 ━━━",
        ]

        for rec in recs:
            rank = rec["rank"]
            icon = medal.get(rank, f"#{rank}")
            sid = rec["stock_id"]
            name = rec.get("stock_name", sid)
            price = rec.get("current_price") or 0
            score = rec.get("total_score") or 0
            reasons = rec.get("key_reasons", [])
            risk = rec.get("risk_warning", "")
            tp = rec.get("target_price_base")
            upside = rec.get("upside_pct")

            lines += ["", f"{icon} {sid} {name}"]
            lines.append(f"💰 {price:.0f} 元｜評分 {score:.0f}/100")
            hot_tags = rec.get("hot_tags", [])
            if hot_tags:
                lines.append(f"🔥 熱門：{'、'.join(hot_tags)}")
            if tp and upside is not None:
                lines.append(f"🎯 目標價：{tp:.0f} 元（{upside:+.0f}%）")
            feps = rec.get("forward_eps")
            eps_growth = rec.get("eps_growth_rate")
            if feps is not None:
                growth_str = f"（成長 {eps_growth:+.0f}%）" if eps_growth is not None else ""
                lines.append(f"📈 Forward EPS：{feps:.2f} 元{growth_str}")
            if price:
                stop_loss = price * 0.88
                take_profit = tp if tp else price * 1.15
                lines.append(f"🛡️ 停損參考：{stop_loss:.0f} 元（-12% 或跌破60日線）｜停利參考：{take_profit:.0f} 元")
            for r in reasons:
                lines.append(f"✅ {r}")
            if risk:
                lines.append(f"⚠️ {risk}")

        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            f"掃描 {summary.get('universe_count', 0)} 支"
            f" → 篩選 {summary.get('after_filter_count', 0)} 支"
            f" → 精選 {n} 支",
            "⚠️ 本推薦由量化模型自動生成，僅供學習與研究參考，不構成任何投資建議。投資涉及風險，請自行評估。",
        ]

        return "\n".join(lines)
