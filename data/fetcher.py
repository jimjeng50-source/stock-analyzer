import time
import requests
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

from config import FINMIND_TOKEN, DEFAULT_DAYS, FUNDAMENTAL_DAYS
from utils.tz import now_tw

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


class FinMindFetcher:
    """從 FinMind 取得台股資料，無 Token 時以 yfinance 取得股價替代。"""

    def __init__(self, stock_id: str, days: int = DEFAULT_DAYS):
        self.stock_id = stock_id
        self.days = days
        self.use_finmind = bool(FINMIND_TOKEN)
        self._end = now_tw()
        self._start_short = (self._end - timedelta(days=days)).strftime("%Y-%m-%d")
        self._start_long = (self._end - timedelta(days=FUNDAMENTAL_DAYS)).strftime("%Y-%m-%d")
        self._end_str = self._end.strftime("%Y-%m-%d")

    # ── 內部工具 ──────────────────────────────────────────

    def _fm_get(self, dataset: str, start: str) -> pd.DataFrame:
        """向 FinMind API 發出請求並回傳 DataFrame；失敗時回傳空 DataFrame。"""
        try:
            resp = requests.get(
                _FINMIND_API,
                params={
                    "dataset": dataset,
                    "data_id": self.stock_id,
                    "start_date": start,
                    "end_date": self._end_str,
                    "token": FINMIND_TOKEN,
                },
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("status") == 200 and body.get("data"):
                df = pd.DataFrame(body["data"])
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            print(f"[警告] FinMind {dataset} 請求失敗：{e}")
        return pd.DataFrame()

    # ── 公開方法 ──────────────────────────────────────────

    def get_price(self) -> pd.DataFrame:
        """取得日K OHLCV，回傳欄位：date, open, high, low, close, volume。"""
        if self.use_finmind:
            df = self._fm_get("TaiwanStockPrice", self._start_short)
            if not df.empty:
                rename = {"max": "high", "min": "low", "Trading_Volume": "volume", "trading_volume": "volume"}
                df = df.rename(columns=rename)
                cols = ["date", "open", "high", "low", "close", "volume"]
                available = [c for c in cols if c in df.columns]
                df = df[available].sort_values("date").reset_index(drop=True)
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                return df

        # fallback：yfinance
        if _HAS_YFINANCE:
            for suffix in [".TW", ".TWO"]:
                try:
                    raw = yf.download(
                        f"{self.stock_id}{suffix}",
                        start=self._start_short,
                        end=self._end_str,
                        progress=False,
                        auto_adjust=True,
                    )
                    if not raw.empty:
                        # yfinance >= 0.2 可能回傳 MultiIndex
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = raw.columns.get_level_values(0)
                        raw.columns = [c.lower() for c in raw.columns]
                        raw = raw.reset_index().rename(columns={"index": "date", "Date": "date"})
                        raw["date"] = pd.to_datetime(raw["date"])
                        # 統一欄位名稱
                        raw = raw.rename(columns={"adj close": "close"})
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col in raw.columns:
                                raw[col] = pd.to_numeric(raw[col], errors="coerce")
                        return raw[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
                except Exception as e:
                    print(f"[警告] yfinance {suffix} 取得失敗：{e}")
        return pd.DataFrame()

    def get_institutional(self) -> pd.DataFrame:
        """取得三大法人買賣超，回傳 FinMind TaiwanStockInstitutionalInvestorsBuySell。"""
        if not self.use_finmind:
            return pd.DataFrame()
        df = self._fm_get("TaiwanStockInstitutionalInvestorsBuySell", self._start_short)
        if df.empty:
            return df
        for col in ["buy", "sell"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["net"] = df["buy"] - df["sell"]
        return df.sort_values("date").reset_index(drop=True)

    def get_monthly_revenue(self) -> pd.DataFrame:
        """取得月營收，回傳 FinMind TaiwanStockMonthRevenue。"""
        if not self.use_finmind:
            return pd.DataFrame()
        df = self._fm_get("TaiwanStockMonthRevenue", self._start_long)
        if df.empty:
            return df
        if "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def get_financial_statements(self) -> pd.DataFrame:
        """取得季財報，回傳 FinMind TaiwanStockFinancialStatements（long format）。"""
        if not self.use_finmind:
            return pd.DataFrame()
        df = self._fm_get("TaiwanStockFinancialStatements", self._start_long)
        if df.empty:
            return df
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def get_margin_trading(self) -> pd.DataFrame:
        """取得融資融券，回傳 FinMind TaiwanStockMarginPurchaseShortSale。"""
        if not self.use_finmind:
            return pd.DataFrame()
        df = self._fm_get("TaiwanStockMarginPurchaseShortSale", self._start_short)
        if df.empty:
            return df
        for col in df.columns:
            if col != "date" and col != "stock_id":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DataFetcher：v3 多股票支援介面（以 stock_id 為參數的方法）
# ─────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """
    v3 多股票資料抓取介面。
    方法皆以 stock_id 為第一個參數，供 ForwardEPSCalculator、
    SupplyChainAnalyzer、ResearchReportGenerator 共用同一個實例。
    """

    def _fm_request(
        self,
        dataset: str,
        stock_id: str,
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        if not FINMIND_TOKEN:
            return pd.DataFrame()
        if end is None:
            end = now_tw().strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                _FINMIND_API,
                params={
                    "dataset": dataset,
                    "data_id": stock_id,
                    "start_date": start,
                    "end_date": end,
                    "token": FINMIND_TOKEN,
                },
                timeout=30,
            )
            body = resp.json()
            if body.get("status") == 200 and body.get("data"):
                df = pd.DataFrame(body["data"])
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def get_quarterly_eps(
        self, stock_id: str, n_quarters: int = 8
    ) -> Optional[pd.DataFrame]:
        """
        取得近 n 季 EPS（每股盈餘）。
        Returns DataFrame with columns ["date", "eps"]，按日期升序。
        失敗時 return None。
        """
        # 估算起始日期（每季約 91 天，多取一些緩衝）
        days_needed = (n_quarters + 2) * 100
        start = (now_tw() - timedelta(days=days_needed)).strftime("%Y-%m-%d")

        df = self._fm_request("TaiwanStockFinancialStatements", stock_id, start)
        if not df.empty and "type" in df.columns and "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            eps_keywords = ["EPS", "每股盈餘", "eps"]
            for kw in eps_keywords:
                mask = df["type"].str.contains(kw, case=False, na=False)
                if mask.any():
                    eps_df = df[mask].copy()
                    eps_q = (
                        eps_df.groupby("date")["value"]
                        .first()
                        .sort_index()
                        .dropna()
                        .tail(n_quarters)
                        .reset_index()
                    )
                    eps_q.columns = ["date", "eps"]
                    if len(eps_q) >= 4:
                        return eps_q

        # 備援：yfinance 季報
        if _HAS_YFINANCE:
            try:
                for suffix in [".TW", ".TWO"]:
                    ticker = yf.Ticker(f"{stock_id}{suffix}")
                    fin = ticker.quarterly_income_stmt
                    if fin is not None and not fin.empty:
                        shares = ticker.info.get("sharesOutstanding", None)
                        if shares and "Net Income" in fin.index:
                            net_income = fin.loc["Net Income"].dropna()
                            eps_vals = (net_income / shares).sort_index()
                            result = pd.DataFrame({
                                "date": pd.to_datetime(eps_vals.index),
                                "eps": eps_vals.values,
                            }).tail(n_quarters)
                            if len(result) >= 4:
                                return result.reset_index(drop=True)
            except Exception:
                pass
        return None

    def get_quarterly_gross_margin(
        self, stock_id: str, n_quarters: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        取得近 n 季毛利率（%）。
        Returns DataFrame with columns ["date", "gross_margin"]。
        """
        days_needed = (n_quarters + 2) * 100
        start = (now_tw() - timedelta(days=days_needed)).strftime("%Y-%m-%d")

        df = self._fm_request("TaiwanStockFinancialStatements", stock_id, start)
        if df.empty or "type" not in df.columns or "value" not in df.columns:
            return None

        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # 嘗試取 GrossProfit 和 Revenue
        gp_df, rv_df = pd.DataFrame(), pd.DataFrame()
        for kw in ["GrossProfit", "毛利", "gross_profit"]:
            mask = df["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                gp_df = df[mask].copy()
                break
        for kw in ["OperatingRevenue", "Revenue", "營業收入"]:
            mask = df["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                rv_df = df[mask].copy()
                break

        if not gp_df.empty and not rv_df.empty:
            gp_q = gp_df.groupby("date")["value"].first().sort_index().dropna()
            rv_q = rv_df.groupby("date")["value"].first().sort_index().dropna()
            common = gp_q.index.intersection(rv_q.index)
            if len(common) >= 3:
                gm = (gp_q[common] / (rv_q[common] + 1e-9) * 100).replace(
                    [float("inf"), float("-inf")], float("nan")
                ).dropna().tail(n_quarters)
                result = gm.reset_index()
                result.columns = ["date", "gross_margin"]
                return result

        # 嘗試直接取毛利率欄位
        for kw in ["GrossMargin", "毛利率"]:
            mask = df["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                gm_df = df[mask].copy()
                gm_q = gm_df.groupby("date")["value"].first().sort_index().dropna().tail(n_quarters)
                result = gm_q.reset_index()
                result.columns = ["date", "gross_margin"]
                if len(result) >= 3:
                    return result
        return None

    def get_historical_pe(
        self, stock_id: str, years: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        取得近 n 年日頻本益比（P/E Ratio）。
        Returns DataFrame with columns ["date", "pe_ratio"]。
        """
        days_needed = years * 365 + 90  # 多加 90 天以取得足夠季報
        start_price = (now_tw() - timedelta(days=days_needed)).strftime("%Y-%m-%d")
        start_eps = (now_tw() - timedelta(days=days_needed + 365)).strftime("%Y-%m-%d")

        # 取股價
        price_df = self._fm_request("TaiwanStockPrice", stock_id, start_price)
        if price_df.empty or "close" not in price_df.columns:
            return None

        price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
        price_df = price_df[["date", "close"]].dropna().sort_values("date")

        # 取季 EPS
        eps_df_raw = self._fm_request("TaiwanStockFinancialStatements", stock_id, start_eps)
        if eps_df_raw.empty or "type" not in eps_df_raw.columns:
            return None

        eps_df_raw["value"] = pd.to_numeric(eps_df_raw["value"], errors="coerce")
        eps_series = None
        for kw in ["EPS", "每股盈餘"]:
            mask = eps_df_raw["type"].str.contains(kw, case=False, na=False)
            if mask.any():
                eps_series = (
                    eps_df_raw[mask]
                    .groupby("date")["value"]
                    .first()
                    .sort_index()
                    .dropna()
                )
                break

        if eps_series is None or len(eps_series) < 4:
            return None

        # 對每個價格日期，計算 TTM EPS（最近4季加總）
        rows = []
        eps_dates = eps_series.index.tolist()
        eps_vals = eps_series.values.tolist()

        for _, row in price_df.iterrows():
            price_date = row["date"]
            close = row["close"]
            # 找到 price_date 之前的所有季報
            past_eps = [(d, v) for d, v in zip(eps_dates, eps_vals) if d <= price_date]
            if len(past_eps) < 4:
                continue
            ttm = sum(v for _, v in past_eps[-4:])
            if ttm > 0:
                rows.append({"date": price_date, "pe_ratio": round(close / ttm, 2)})

        if len(rows) < 60:
            return None
        return pd.DataFrame(rows)

    def get_market_price(self, stock_id: str) -> float:
        """取得最新收盤價；失敗時回傳 0.0。"""
        start = (now_tw() - timedelta(days=10)).strftime("%Y-%m-%d")
        df = self._fm_request("TaiwanStockPrice", stock_id, start)
        if not df.empty and "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            val = df["close"].dropna()
            if not val.empty:
                return float(val.iloc[-1])

        if _HAS_YFINANCE:
            try:
                for suffix in [".TW", ".TWO"]:
                    raw = yf.download(
                        f"{stock_id}{suffix}", period="5d",
                        progress=False, auto_adjust=True,
                    )
                    if not raw.empty:
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = raw.columns.get_level_values(0)
                        col = "Close" if "Close" in raw.columns else raw.columns[0]
                        return float(raw[col].dropna().iloc[-1])
            except Exception:
                pass
        return 0.0

    def get_monthly_revenue(
        self, stock_id: str, months: int = 6
    ) -> Optional[pd.DataFrame]:
        """
        取得近 months 個月月營收，含 YoY（年增率%）。
        Returns DataFrame with columns ["date", "revenue", "revenue_yoy"]。
        """
        # 取 13 個月以計算 YoY
        days_needed = (months + 14) * 32
        start = (now_tw() - timedelta(days=days_needed)).strftime("%Y-%m-%d")

        df = self._fm_request("TaiwanStockMonthRevenue", stock_id, start)
        if df.empty or "revenue" not in df.columns:
            return None

        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        df = df[["date", "revenue"]].dropna().sort_values("date").reset_index(drop=True)

        if len(df) < 3:
            return None

        # 計算 YoY：找 12 個月前的資料
        yoy_vals = []
        for i, row in df.iterrows():
            target_month = row["date"] - pd.DateOffset(months=12)
            # 找最近的月份（允許 45 天誤差）
            diffs = abs(df["date"] - target_month)
            idx = diffs.idxmin()
            if diffs[idx].days <= 45:
                base = df.loc[idx, "revenue"]
                if base > 0:
                    yoy_vals.append(round((row["revenue"] - base) / base * 100, 2))
                else:
                    yoy_vals.append(float("nan"))
            else:
                yoy_vals.append(float("nan"))

        df["revenue_yoy"] = yoy_vals
        # 只回傳最近 months 筆
        return df.tail(months).reset_index(drop=True)

    def get_institutional_net(
        self, stock_id: str, days: int = 30
    ) -> pd.Series:
        """
        取得外資近 days 日每日淨買賣超（張），供產業鏈分析使用。
        Returns pd.Series indexed by date；失敗時回傳空 Series。
        """
        start = (now_tw() - timedelta(days=days + 5)).strftime("%Y-%m-%d")
        df = self._fm_request(
            "TaiwanStockInstitutionalInvestorsBuySell", stock_id, start
        )
        if df.empty or "name" not in df.columns:
            return pd.Series(dtype=float)

        fi_names = {"Foreign_Investor", "外資及陸資", "外資"}
        fi_df = df[df["name"].isin(fi_names)].copy()
        if fi_df.empty:
            return pd.Series(dtype=float)

        for col in ["buy", "sell"]:
            if col in fi_df.columns:
                fi_df[col] = pd.to_numeric(fi_df[col], errors="coerce").fillna(0)
        if "net" not in fi_df.columns and "buy" in fi_df.columns:
            fi_df["net"] = fi_df["buy"] - fi_df["sell"]
        if "net" not in fi_df.columns:
            return pd.Series(dtype=float)

        return fi_df.groupby("date")["net"].sum().sort_index()
