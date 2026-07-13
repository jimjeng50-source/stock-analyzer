"""
screener/pool_analyzer.py
候選池全工具分析

對使用者指定的一組股票（不做門檻過濾），逐一跑完整分析工具鏈：
1. 多因子評分（籌碼/基本面/技術/動能/風險）— 複用 BatchScorer
2. Forward EPS 前瞻推估與目標價 — ForwardEPSCalculator
3. 產業鏈信號 — SupplyChainAnalyzer

回傳結果 DataFrame，供 UI 表格與個股展開明細使用。
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def analyze_pool(
    stock_ids: list,
    name_map: dict = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Args:
        stock_ids: 已解析的代號清單
        name_map: {stock_id: stock_name}
        progress_callback: fn(done, total, stage)

    Returns DataFrame，每列一支股票，含欄位：
        stock_id, stock_name, total_score, recommendation,
        chips_score, fundamental_score, technical_score, momentum_score,
        risk_score, current_price,
        forward_eps, eps_growth_pct, target_price, upside_pct,
        chain_name, chain_signal, error
    """
    name_map = name_map or {}
    if not stock_ids:
        return pd.DataFrame()

    from screener.batch_scorer import BatchScorer

    total = len(stock_ids)

    # Stage 1：多因子評分（全部股票，不設門檻）
    if progress_callback:
        progress_callback(0, total, "評分")
    scorer = BatchScorer()
    scored_df = scorer.score_universe(stock_ids, show_progress=False)
    if scored_df.empty:
        # 全部失敗仍回傳骨架，讓 UI 顯示錯誤
        return pd.DataFrame([
            {"stock_id": s, "stock_name": name_map.get(s, s),
             "error": "無法取得資料", "total_score": None}
            for s in stock_ids
        ])

    # 補回失敗個股（BatchScorer 只回成功者）
    scored_ids = set(scored_df["stock_id"].astype(str))
    failed_rows = [
        {"stock_id": s, "stock_name": name_map.get(s, s),
         "error": "評分失敗（資料不足或 API 限流）", "total_score": None}
        for s in stock_ids if s not in scored_ids
    ]

    scored_df["stock_name"] = scored_df["stock_id"].map(
        lambda s: name_map.get(str(s), str(s))
    )

    # Stage 2：深度分析（Forward EPS + 產業鏈）
    deep_map = _deep_analysis(
        scored_df["stock_id"].astype(str).tolist(), progress_callback, total
    )

    rows = []
    for _, r in scored_df.iterrows():
        sid = str(r["stock_id"])
        deep = deep_map.get(sid, {})
        rows.append({
            "stock_id": sid,
            "stock_name": r.get("stock_name", sid),
            "total_score": round(float(r["total_score"]), 1) if pd.notna(r["total_score"]) else None,
            "recommendation": r.get("recommendation", ""),
            "chips_score": r.get("chips_score"),
            "fundamental_score": r.get("fundamental_score"),
            "technical_score": r.get("technical_score"),
            "momentum_score": r.get("momentum_score"),
            "risk_score": r.get("risk_score"),
            "current_price": r.get("current_price"),
            "forward_eps": deep.get("forward_eps"),
            "eps_growth_pct": deep.get("eps_growth_rate"),
            "target_price": deep.get("target_price_base"),
            "upside_pct": deep.get("upside_pct"),
            "chain_name": deep.get("chain_name"),
            "chain_signal": deep.get("chain_signal"),
            "error": "",
        })

    result_df = pd.DataFrame(rows + failed_rows)
    if "total_score" in result_df.columns:
        result_df = result_df.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)
    return result_df


def _deep_analysis(stock_ids: list, progress_callback, total: int) -> dict:
    """對每支跑 Forward EPS + 產業鏈。單支失敗不影響其他。"""
    results = {}
    try:
        from data.fetcher import DataFetcher
        from factors.forward_eps import ForwardEPSCalculator
        from factors.supply_chain import SupplyChainAnalyzer
        fetcher = DataFetcher()
        eps_calc = ForwardEPSCalculator(fetcher)
        chain_analyzer = SupplyChainAnalyzer(fetcher)
    except Exception as e:
        logger.warning("深度分析模組載入失敗：%s", e)
        return results

    for i, sid in enumerate(stock_ids, 1):
        deep = {}
        try:
            eps = eps_calc.calculate(sid)
            if not eps.get("error"):
                deep["target_price_base"] = (eps.get("target_price") or {}).get("base")
                deep["upside_pct"] = eps.get("upside_pct")
                deep["forward_eps"] = eps.get("forward_eps_1y")
                deep["eps_growth_rate"] = eps.get("eps_growth_rate")
        except Exception as e:
            logger.debug("Forward EPS 失敗 %s：%s", sid, e)
        try:
            chain = chain_analyzer.analyze_for_stock(sid)
            deep["chain_name"] = chain.get("chain_name")
            deep["chain_signal"] = chain.get("chain_signal")
        except Exception as e:
            logger.debug("產業鏈分析失敗 %s：%s", sid, e)
        results[sid] = deep
        if progress_callback:
            progress_callback(i, total, "深度分析")
    return results
