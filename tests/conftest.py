"""Shared pytest fixtures for the stock-analyzer test suite."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── Price data ─────────────────────────────────────────────────────────────────

def _make_dates(n: int) -> pd.Series:
    base = datetime(2024, 1, 2)
    return pd.Series([base + timedelta(days=i) for i in range(n)])


@pytest.fixture
def price_df_90() -> pd.DataFrame:
    """90-row OHLCV DataFrame with a mild uptrend."""
    n = 90
    rng = np.random.default_rng(42)
    close = 500 + np.cumsum(rng.normal(0.5, 3, n))
    open_ = close - rng.uniform(0, 3, n)
    high  = close + rng.uniform(0, 5, n)
    low   = close - rng.uniform(0, 5, n)
    vol   = rng.integers(5_000, 50_000, n).astype(float)
    return pd.DataFrame({
        "date":   _make_dates(n),
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
    })


@pytest.fixture
def price_df_short() -> pd.DataFrame:
    """10-row price DataFrame — tests that need minimal data."""
    n = 10
    close = np.arange(100.0, 100 + n)
    return pd.DataFrame({
        "date":   _make_dates(n),
        "open":   close - 0.5,
        "high":   close + 1,
        "low":    close - 1,
        "close":  close,
        "volume": np.ones(n) * 10_000,
    })


@pytest.fixture
def price_df_with_drops() -> pd.DataFrame:
    """365-row price DataFrame containing ≥8 single-day drops of ≥5%."""
    n = 365
    rng = np.random.default_rng(7)
    close = 500 + np.cumsum(rng.normal(0.3, 2, n))
    # Inject large drops at known positions
    for idx in [30, 60, 90, 120, 150, 180, 210, 240]:
        if idx < n:
            close[idx] = close[idx - 1] * 0.93   # -7%
    open_ = close - rng.uniform(0, 2, n)
    high  = close + rng.uniform(0, 4, n)
    low   = close - rng.uniform(0, 4, n)
    vol   = rng.integers(10_000, 100_000, n).astype(float)
    return pd.DataFrame({
        "date":   _make_dates(n),
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
    })


# ── Institutional investor data ────────────────────────────────────────────────

@pytest.fixture
def institutional_df_bullish() -> pd.DataFrame:
    """30-day institutional data: foreign investors net buying."""
    n = 30
    dates = _make_dates(n)
    rows = []
    for d in dates:
        for name, net in [("外資", 5000), ("投信", 1000), ("自營商", -200)]:
            rows.append({"date": d, "name": name, "buy": net + 100, "sell": 100, "net": net})
    return pd.DataFrame(rows)


@pytest.fixture
def institutional_df_bearish() -> pd.DataFrame:
    """30-day institutional data: foreign investors net selling."""
    n = 30
    dates = _make_dates(n)
    rows = []
    for d in dates:
        for name, net in [("外資", -4000), ("投信", -800), ("自營商", 300)]:
            rows.append({"date": d, "name": name, "buy": 100, "sell": abs(net) + 100, "net": net})
    return pd.DataFrame(rows)


@pytest.fixture
def institutional_df_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "name", "buy", "sell", "net"])


# ── Margin trading data ────────────────────────────────────────────────────────

@pytest.fixture
def margin_df() -> pd.DataFrame:
    n = 30
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": _make_dates(n),
        "MarginPurchaseTodayBalance": rng.integers(100_000, 200_000, n).astype(float),
        "ShortSaleTodayBalance":      rng.integers(10_000,  30_000,  n).astype(float),
    })


# ── Financial statements / revenue ────────────────────────────────────────────

@pytest.fixture
def revenue_df() -> pd.DataFrame:
    dates = _make_dates(15)
    return pd.DataFrame({
        "date":    dates,
        "revenue": np.linspace(5e9, 8e9, 15),
    })


@pytest.fixture
def financial_df() -> pd.DataFrame:
    """Quarterly financial statements in FinMind long format."""
    quarters = [datetime(2023, 3, 31), datetime(2023, 6, 30),
                datetime(2023, 9, 30), datetime(2023, 12, 31)]
    rows = []
    for d in quarters:
        for name, val in [("EPS", 3.5), ("Revenue", 1e10),
                          ("GrossProfit", 4e9), ("NetIncome", 2e9)]:
            rows.append({"date": d, "type": name, "value": val})
    return pd.DataFrame(rows)
