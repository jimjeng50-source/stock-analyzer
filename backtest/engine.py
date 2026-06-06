"""
回測引擎：以歷史股價 + 滾動因子評分模擬每日收盤交易。
嚴格避免前視偏差（Look-ahead Bias）：
  - 技術/動能指標使用滾動視窗，precompute 後逐日讀取
  - 籌碼因子使用滾動加總
  - 基本面使用點對時 (point-in-time) 資料
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

BUY_FEE  = 0.001425   # 手續費
SELL_FEE = 0.001425
SELL_TAX = 0.003      # 交易稅（賣方）
LOT_SIZE = 1000       # 1 張 = 1000 股


# ── 預計算評分序列（O(N)，避免重複呼叫 O(N²)） ───────────────────────────

def _safe(series: pd.Series, date, default=0.0):
    try:
        v = series.loc[date]
        return default if pd.isna(v) else float(v)
    except Exception:
        return default


def _compute_score_series(
    price_df: pd.DataFrame,
    institutional_df: pd.DataFrame,
    revenue_df: pd.DataFrame,
    financial_df: pd.DataFrame,
    weights: dict,
) -> pd.Series:
    """
    對 price_df 中每個日期計算評分（0~100），回傳 pd.Series indexed by date。
    所有指標均使用滾動視窗，不含未來資料。
    """
    from models.scorer import Scorer
    from factors.fundamental import compute_fundamental

    try:
        from ta.trend import SMAIndicator, MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        _has_ta = True
    except ImportError:
        _has_ta = False

    price = price_df.set_index("date").sort_index()
    close = price["close"].astype(float)
    volume = price["volume"].astype(float) if "volume" in price.columns else pd.Series(1.0, index=close.index)

    scorer = Scorer(weights)
    scores = {}

    # ── 技術指標 precompute ────────────────────────────────────────────────
    if _has_ta and len(close) >= 20:
        ma5  = SMAIndicator(close, 5).sma_indicator()
        ma10 = SMAIndicator(close, 10).sma_indicator()
        ma20 = SMAIndicator(close, 20).sma_indicator()
        ma60 = SMAIndicator(close, 60).sma_indicator()
        rsi  = RSIIndicator(close, 14).rsi()
        macd_obj  = MACD(close)
        macd_hist = macd_obj.macd_diff()
        bb_pos    = BollingerBands(close, 20, 2).bollinger_pband().clip(0, 1)
        vol_ma5   = SMAIndicator(volume, 5).sma_indicator()
        vol_ma20  = SMAIndicator(volume, 20).sma_indicator()
    else:
        ma5 = ma10 = ma20 = ma60 = rsi = macd_hist = bb_pos = vol_ma5 = vol_ma20 = pd.Series(dtype=float)

    # ── 動能 precompute ───────────────────────────────────────────────────
    log_ret   = np.log(close / close.shift(1))
    ret_5d    = (close / close.shift(5)  - 1) * 100
    ret_20d   = (close / close.shift(20) - 1) * 100
    ret_60d   = (close / close.shift(60) - 1) * 100
    vol_20d   = log_ret.rolling(20).std() * np.sqrt(252) * 100
    high_252  = close.rolling(252, min_periods=20).max()
    h52w_pct  = (close / high_252 - 1) * 100

    # ── 籌碼 precompute（滾動加總） ───────────────────────────────────────
    fi_daily  = pd.Series(0.0, index=close.index)
    it_daily  = pd.Series(0.0, index=close.index)
    dl_daily  = pd.Series(0.0, index=close.index)

    if not institutional_df.empty and "name" in institutional_df.columns and "net" in institutional_df.columns:
        inst = institutional_df.copy()
        inst["date"] = pd.to_datetime(inst["date"])
        inst["net"]  = pd.to_numeric(inst["net"], errors="coerce").fillna(0)

        for mask_kw, target in [("外資", fi_daily), ("投信", it_daily), ("自營商", dl_daily)]:
            grp = inst[inst["name"].str.contains(mask_kw, na=False)].groupby("date")["net"].sum()
            for d, v in grp.items():
                if d in target.index:
                    target.loc[d] = v

    fi_5d   = fi_daily.rolling(5,  min_periods=1).sum()
    fi_20d  = fi_daily.rolling(20, min_periods=1).sum()
    it_5d   = it_daily.rolling(5,  min_periods=1).sum()
    it_20d  = it_daily.rolling(20, min_periods=1).sum()
    dl_5d   = dl_daily.rolling(5,  min_periods=1).sum()

    # ── 逐日評分 ──────────────────────────────────────────────────────────
    prev_hist = None
    dates = list(close.index)

    for i, date in enumerate(dates):
        if i < 60:
            scores[date] = 50.0
            continue

        c = _safe(close, date)
        if c <= 0:
            scores[date] = 50.0
            continue

        m5  = _safe(ma5,  date, c)
        m10 = _safe(ma10, date, c)
        m20 = _safe(ma20, date, c)
        m60 = _safe(ma60, date, c)

        rsi_val  = _safe(rsi, date, 50.0)
        hist_val = _safe(macd_hist, date, 0.0)

        # MACD 交叉
        macd_cross = 0
        if prev_hist is not None:
            if prev_hist < 0 <= hist_val:
                macd_cross = 1
            elif prev_hist > 0 >= hist_val:
                macd_cross = -1
        prev_hist = hist_val

        vm5  = _safe(vol_ma5,  date, 1.0)
        vm20 = _safe(vol_ma20, date, 1.0)
        v0   = _safe(volume, date, vm20)

        technical = {
            "above_ma5":      1 if c > m5  else -1,
            "above_ma20":     1 if c > m20 else -1,
            "above_ma60":     1 if c > m60 else -1,
            "ma_alignment":   int(m5>m10) + int(m10>m20) + int(m20>m60),
            "ma20_deviation": (c - m20) / m20 * 100 if m20 > 0 else 0,
            "rsi_14":         rsi_val,
            "rsi_signal":     1 if rsi_val < 30 else (-1 if rsi_val > 70 else 0),
            "macd_histogram": hist_val,
            "macd_cross":     macd_cross,
            "bb_position":    _safe(bb_pos, date, 0.5),
            "vol_ratio":      v0 / (vm20 + 1e-9),
            "vol_trend":      1 if vm5 > vm20 else -1,
        }

        chips = {
            "fi_5d_net":      _safe(fi_5d,  date),
            "fi_20d_net":     _safe(fi_20d, date),
            "fi_consecutive": 0,
            "fi_trend":       0.0,
            "it_5d_net":      _safe(it_5d,  date),
            "it_20d_net":     _safe(it_20d, date),
            "it_consecutive": 0,
            "dealer_5d_net":  _safe(dl_5d,  date),
            "margin_chg_5d":  0.0,
            "short_chg_5d":   0.0,
        }

        momentum = {
            "ret_5d":          _safe(ret_5d,   date),
            "ret_1m":          _safe(ret_20d,  date),
            "ret_3m":          _safe(ret_60d,  date),
            "vol_20d":         _safe(vol_20d,  date, 30.0),
            "high_52w_pct":    _safe(h52w_pct, date, -10.0),
            "momentum_accel":  _safe(ret_5d, date) - _safe(ret_20d, date),
        }

        # 基本面：取當日前最新可用資料（point-in-time）
        rev_slice = revenue_df[revenue_df["date"] <= date] if not revenue_df.empty else pd.DataFrame()
        fin_slice = financial_df[financial_df["date"] <= date] if not financial_df.empty else pd.DataFrame()
        fundamental = compute_fundamental(rev_slice, fin_slice, c)

        result = scorer.score(chips, technical, fundamental, momentum)
        scores[date] = result["total_score"]

    return pd.Series(scores)


# ── 交易模擬 ─────────────────────────────────────────────────────────────────

def _simulate_trades(
    price_df: pd.DataFrame,
    score_series: pd.Series,
    buy_threshold: int,
    sell_threshold: int,
    initial_capital: float,
) -> dict:
    capital   = float(initial_capital)
    shares    = 0
    hold_cost = 0.0
    in_pos    = False
    last_buy_i = -999

    trades     = []
    equity_log = []
    entry_trade = None

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    for i, row in price_df.iterrows():
        date  = row["date"]
        price = float(row["close"])
        score = float(score_series.get(date, 50.0)) if date in score_series.index else 50.0

        if in_pos:
            if score <= sell_threshold:
                gross    = shares * price
                proceeds = gross * (1 - SELL_FEE - SELL_TAX)
                pnl      = proceeds - hold_cost
                pnl_pct  = pnl / hold_cost * 100
                hdays    = (date - entry_trade["date_obj"]).days

                capital += proceeds
                trades.append({
                    "date":         date.strftime("%Y-%m-%d"),
                    "date_obj":     date,
                    "action":       "賣出",
                    "price":        round(price, 2),
                    "shares":       shares,
                    "proceeds":     round(proceeds),
                    "pnl":          round(pnl),
                    "pnl_pct":      round(pnl_pct, 2),
                    "holding_days": hdays,
                    "score":        round(score, 1),
                })
                shares    = 0
                hold_cost = 0.0
                in_pos    = False
                entry_trade = None
        else:
            # 等待期 5 個交易日
            if score >= buy_threshold and (i - last_buy_i) >= 5:
                unit_cost = price * LOT_SIZE * (1 + BUY_FEE)
                lots      = int(capital / unit_cost)
                if lots > 0:
                    shares    = lots * LOT_SIZE
                    hold_cost = shares * price * (1 + BUY_FEE)
                    capital  -= hold_cost
                    in_pos    = True
                    last_buy_i = i
                    entry_trade = {
                        "date":     date.strftime("%Y-%m-%d"),
                        "date_obj": date,
                        "action":   "買進",
                        "price":    round(price, 2),
                        "shares":   shares,
                        "cost":     round(hold_cost),
                        "score":    round(score, 1),
                        "pnl":      None, "pnl_pct": None,
                        "holding_days": None, "proceeds": None,
                    }
                    trades.append(entry_trade)

        equity_log.append({"date": date, "strategy": capital + shares * price, "score": round(score, 1)})

    # 強制平倉（回測結束）
    if in_pos and len(price_df) > 0:
        last = price_df.iloc[-1]
        lp   = float(last["close"])
        ld   = last["date"]
        proceeds = shares * lp * (1 - SELL_FEE - SELL_TAX)
        pnl      = proceeds - hold_cost
        capital += proceeds
        trades.append({
            "date":         ld.strftime("%Y-%m-%d"),
            "date_obj":     ld,
            "action":       "賣出(到期)",
            "price":        round(lp, 2),
            "shares":       shares,
            "proceeds":     round(proceeds),
            "pnl":          round(pnl),
            "pnl_pct":      round(pnl / hold_cost * 100, 2),
            "holding_days": (ld - entry_trade["date_obj"]).days,
            "score":        0,
        })

    equity_df = pd.DataFrame(equity_log)

    # Buy & Hold 基準
    if not price_df.empty and not equity_df.empty:
        bh_p0 = float(price_df.iloc[0]["close"])
        bh_lots = int(initial_capital / (bh_p0 * LOT_SIZE * (1 + BUY_FEE)))
        bh_sh   = bh_lots * LOT_SIZE
        bh_cash = initial_capital - bh_sh * bh_p0 * (1 + BUY_FEE)
        bh_eq   = price_df[["date", "close"]].copy()
        bh_eq["date"] = pd.to_datetime(bh_eq["date"])
        bh_eq["bh"]   = bh_cash + bh_sh * bh_eq["close"].astype(float)
        equity_df = equity_df.merge(bh_eq[["date", "bh"]].rename(columns={"bh": "buyhold"}), on="date", how="left")

    final_equity = float(equity_df["strategy"].iloc[-1]) if not equity_df.empty else capital

    return {
        "trades":          trades,
        "equity_df":       equity_df,
        "final_capital":   round(final_equity),
        "initial_capital": initial_capital,
    }


# ── 公開入口 ─────────────────────────────────────────────────────────────────

def run_backtest(
    stock_id:        str,
    start_date:      str,
    end_date:        str   = None,
    buy_threshold:   int   = 65,
    sell_threshold:  int   = 45,
    use_macro:       bool  = True,
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    執行回測，回傳 trades、equity_df、final_capital、metrics。
    """
    from data.fetcher import FinMindFetcher
    from config import FACTOR_WEIGHTS
    from backtest.metrics import calc_metrics

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # 取資料（預留 6 個月暖機期）
    days_needed = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days) + 210
    fetcher = FinMindFetcher(stock_id, days=days_needed)

    print(f"  [1/4] 取得 {stock_id} 歷史股價...")
    price_df = fetcher.get_price()
    if price_df.empty:
        raise ValueError(f"無法取得 {stock_id} 的股價資料，請確認代號是否正確。")

    print("  [2/4] 取得籌碼與基本面資料...")
    institutional_df = fetcher.get_institutional()
    revenue_df       = fetcher.get_monthly_revenue()
    financial_df     = fetcher.get_financial_statements()

    price_df["date"] = pd.to_datetime(price_df["date"])

    print("  [3/4] 計算歷史評分序列...")
    score_series = _compute_score_series(
        price_df, institutional_df, revenue_df, financial_df, FACTOR_WEIGHTS
    )

    # 若啟用總體資金面，以當前 macro_score 調整分數
    if use_macro:
        try:
            from macro.macro_scorer import calc_macro_score
            macro = calc_macro_score()
            mult  = 0.7 + 0.3 * macro["macro_score"]
            score_series = (score_series * mult).clip(0, 100)
            print(f"  總體資金面乘數：{mult:.3f}（macro_score={macro['macro_score']:.2f}）")
        except Exception as e:
            print(f"  [警告] 總體資金面乘數計算失敗：{e}")

    # 篩選回測期間
    bt_start = pd.to_datetime(start_date)
    bt_end   = pd.to_datetime(end_date)
    bt_price = price_df[
        (price_df["date"] >= bt_start) & (price_df["date"] <= bt_end)
    ].copy().reset_index(drop=True)

    if bt_price.empty:
        raise ValueError(f"指定期間 {start_date}~{end_date} 沒有股價資料。")

    print(f"  [4/4] 模擬交易（{len(bt_price)} 個交易日）...")
    result = _simulate_trades(bt_price, score_series, buy_threshold, sell_threshold, initial_capital)

    # 計算績效指標
    result["metrics"] = calc_metrics(result, initial_capital)
    result["stock_id"]  = stock_id
    result["start_date"] = start_date
    result["end_date"]   = end_date

    n_trades = len([t for t in result["trades"] if t["action"] == "買進"])
    print(f"  回測完成：共 {n_trades} 筆買進，最終淨值 NT${result['final_capital']:,.0f}")

    return result
