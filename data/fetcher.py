import requests
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

from config import FINMIND_TOKEN, DEFAULT_DAYS, FUNDAMENTAL_DAYS

_FINMIND_API = "https://api.finmindtrade.com/api/v4/data"


class FinMindFetcher:
    """從 FinMind 取得台股資料，無 Token 時以 yfinance 取得股價替代。"""

    def __init__(self, stock_id: str, days: int = DEFAULT_DAYS):
        self.stock_id = stock_id
        self.days = days
        self.use_finmind = bool(FINMIND_TOKEN)
        self._end = datetime.today()
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
