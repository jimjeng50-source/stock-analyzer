"""
data/free_fallback.py
FinMind 免費配額用盡時的共用「免費資料備援」補值層

把原本散在 BatchScorer 內的兩個補值函式抽出來共用，確保
「每日批次掃描」「個股分析 UI」「Telegram /analyze」三條路徑
在 FinMind 402/403 時，對同一支股票算出「一致」的分數
（避免某條路徑有備援、另一條沒有 → 同股不同分）。

- augment_fundamental_with_yf：用 yfinance .info 補基本面（rev_yoy/毛利/PE/EPS）
- augment_chips_with_t86：用證交所 T86 補三大法人買賣超
- augment_factors_with_free_sources：一次補齊，供各呼叫端共用
"""

import logging

import pandas as pd

from factors import compute_chips

logger = logging.getLogger(__name__)


def augment_fundamental_with_yf(stock_id: str, fundamental: dict, current_price: float) -> dict:
    """
    FinMind 財報缺漏（如免費配額 402/403）時，用免費 yfinance .info 補基本面。

    只填「FinMind 沒給到、仍是預設 0」的欄位，避免蓋掉真實資料：
        rev_yoy      ← revenue_growth × 100（小數→百分比）
        gross_margin ← gross_margins × 100
        pe_ratio     ← trailing_pe
        eps_latest   ← trailing_eps

    對應 models/scorer.py 的歸一化：gross_margin=0 只有 0.12 分、
    pe_ratio=0 當虧損只有 0.25 分，補真值可把基本面（45% 權重）
    從被壓低的狀態拉回合理區間，避免每日 0 推薦。
    """
    try:
        from data.yf_fundamentals import get_yf_fundamentals
        yf = get_yf_fundamentals(stock_id)
    except Exception:
        yf = {}
    if not yf:
        return fundamental

    if yf.get("revenue_growth") is not None and not fundamental.get("rev_yoy"):
        fundamental["rev_yoy"] = round(yf["revenue_growth"] * 100, 2)
    if yf.get("gross_margins") is not None and not fundamental.get("gross_margin"):
        fundamental["gross_margin"] = round(yf["gross_margins"] * 100, 2)
    if yf.get("trailing_pe") is not None and not fundamental.get("pe_ratio"):
        fundamental["pe_ratio"] = round(yf["trailing_pe"], 1)
    if yf.get("trailing_eps") is not None and not fundamental.get("eps_latest"):
        fundamental["eps_latest"] = round(yf["trailing_eps"], 2)

    return fundamental


def augment_chips_with_t86(stock_id: str, chips: dict, margin_df=None) -> dict:
    """
    FinMind 三大法人全缺時，用免費證交所 T86 補籌碼面。

    只覆蓋「法人相關」欄位（外資/投信/自營商），保留原融資融券欄位。
    T86 抓不到（假日/該股無資料/網路失敗）→ 原 chips 原樣返回。
    """
    try:
        from data.twse_chips import get_t86_institutional
        inst = get_t86_institutional(stock_id)
    except Exception:
        inst = None
    if inst is None or inst.empty:
        return chips

    t86 = compute_chips(inst, pd.DataFrame())
    inst_keys = ["fi_5d_net", "fi_20d_net", "fi_consecutive", "fi_trend",
                 "it_5d_net", "it_20d_net", "it_consecutive", "dealer_5d_net"]
    for k in inst_keys:
        if t86.get(k):
            chips[k] = t86[k]
    return chips


def augment_factors_with_free_sources(
    stock_id: str,
    *,
    chips: dict,
    fundamental: dict,
    current_price: float,
    institutional_df=None,
    revenue_df=None,
    financial_df=None,
    margin_df=None,
    allow_t86: bool = True,
):
    """
    一次補齊籌碼＋基本面，供批次掃描與個股分析共用（分數一致）。

    - 基本面：FinMind 月營收「或」財報任一缺漏就用 yfinance 補
      （OR 條件，避免「只缺一半」時漏補）。
    - 籌碼：FinMind 三大法人缺漏且 allow_t86 時用 T86 補。
      allow_t86 應為 False 於歷史回溯（as_of 有值），避免抓到未來資料。

    Returns: (chips, fundamental) 補值後的 dict。
    """
    fin_missing = (revenue_df is None or revenue_df.empty) or \
                  (financial_df is None or financial_df.empty)
    if fin_missing:
        fundamental = augment_fundamental_with_yf(stock_id, fundamental, current_price)

    inst_missing = institutional_df is None or institutional_df.empty
    if allow_t86 and inst_missing:
        chips = augment_chips_with_t86(stock_id, chips, margin_df)

    return chips, fundamental
