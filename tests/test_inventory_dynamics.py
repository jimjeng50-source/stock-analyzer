"""tests/test_inventory_dynamics.py — 存貨動態與庫存重估信號"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from factors.inventory_dynamics import (
    compute_inventory_dynamics, apply_product_price_scenario,
)


def _long(rows):
    """rows: list of (date, type, value) → FinMind long-format df."""
    return pd.DataFrame(rows, columns=["date", "type", "value"])


def _fetcher_with(bs_rows, is_rows):
    f = MagicMock()
    def _req(dataset, sid, start, end=None):
        if dataset == "TaiwanStockBalanceSheet":
            return _long(bs_rows)
        if dataset == "TaiwanStockFinancialStatements":
            return _long(is_rows)
        return pd.DataFrame()
    f._fm_request.side_effect = _req
    return f


def _quarters(n=6):
    return [f"2025-{m:02d}-01" for m in range(1, n * 3 + 1, 3)][:n]


class TestRevaluationSignal:
    def test_adata_style_margin_surge(self):
        """毛利率加速上升 + 存貨偏高 → 正的庫存重估信號（威剛型）。"""
        qs = _quarters(6)
        # 毛利率序列（GP/Rev）：13,13,14,16,19,24 → 加速上升
        gms = [13, 13, 14, 16, 19, 24]
        rev = 1000
        bs, is_ = [], []
        for q, gm in zip(qs, gms):
            inv = 600 if q >= qs[3] else 300      # 後段存貨偏高
            bs.append((q, "存貨", inv))
            is_.append((q, "營業收入", rev))
            is_.append((q, "GrossProfit", rev * gm / 100))
            is_.append((q, "銷貨成本", rev * (1 - gm / 100)))
        r = compute_inventory_dynamics(_fetcher_with(bs, is_), "3260")
        assert r["error"] is None
        assert r["revaluation_score"] > 0.3
        assert r["signal_label"] == "庫存重估順風"
        assert r["gm_acceleration"] is not None and r["gm_acceleration"] > 0

    def test_downcycle_margin_falling_inventory_bloat(self):
        """毛利率下滑 + 週轉天數上升 → 負信號（去化風險）。"""
        qs = _quarters(6)
        gms = [25, 22, 18, 15, 12, 9]             # 毛利崩落
        bs, is_ = [], []
        cogs = 800
        for i, (q, gm) in enumerate(zip(qs, gms)):
            inv = 300 + i * 150                    # 存貨堆積
            bs.append((q, "存貨", inv))
            is_.append((q, "營業收入", 1000))
            is_.append((q, "GrossProfit", 1000 * gm / 100))
            is_.append((q, "銷貨成本", cogs))
        r = compute_inventory_dynamics(_fetcher_with(bs, is_), "xxxx")
        assert r["revaluation_score"] < -0.3
        assert r["signal_label"] == "庫存去化風險"

    def test_missing_data_neutral(self):
        f = MagicMock()
        f._fm_request.return_value = pd.DataFrame()
        r = compute_inventory_dynamics(f, "2330")
        assert r["revaluation_score"] == 0.0
        assert r["error"] is not None


class TestPriceScenario:
    def test_positive_price_boosts_eps(self):
        out = apply_product_price_scenario(
            forward_eps=10.0, ttm_eps=8.0, inv_to_rev=0.5,
            gm_latest=20.0, product_price_chg_pct=30,
        )
        assert out["scenario_eps"] > 10.0
        assert out["eps_delta_pct"] > 0
        assert "報價" in out["note"]

    def test_zero_price_no_change(self):
        out = apply_product_price_scenario(10.0, 8.0, 0.5, 20.0, 0)
        assert out["scenario_eps"] == 10.0
        assert out["eps_delta_pct"] == 0.0

    def test_higher_inventory_stronger_passthrough(self):
        low = apply_product_price_scenario(10.0, 8.0, 0.2, 20.0, 30)
        high = apply_product_price_scenario(10.0, 8.0, 0.6, 20.0, 30)
        assert high["eps_delta_pct"] > low["eps_delta_pct"]
