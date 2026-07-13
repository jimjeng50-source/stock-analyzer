"""tests/test_stock_lookup.py — 代號⇆名稱解析"""
from unittest.mock import patch

import utils.stock_lookup as sl


_ROWS = [
    {"stock_id": "2330", "stock_name": "台積電"},
    {"stock_id": "2317", "stock_name": "鴻海"},
    {"stock_id": "2454", "stock_name": "聯發科"},
    {"stock_id": "3711", "stock_name": "日月光投控"},
]


def _patch_rows():
    return patch.object(sl, "load_stock_list", return_value=_ROWS)


class TestResolve:
    def test_pure_code(self):
        with _patch_rows():
            r = sl.resolve("2330")
        assert r["ok"] and r["stock_id"] == "2330" and r["stock_name"] == "台積電"

    def test_exact_name(self):
        with _patch_rows():
            r = sl.resolve("台積電")
        assert r["ok"] and r["stock_id"] == "2330"

    def test_partial_unique(self):
        with _patch_rows():
            r = sl.resolve("日月光")
        assert r["ok"] and r["stock_id"] == "3711"

    def test_unknown(self):
        with _patch_rows():
            r = sl.resolve("不存在公司")
        assert not r["ok"] and not r["candidates"]

    def test_unknown_code_passthrough(self):
        """未知代號仍視為代號（可能是新上市未進快取）。"""
        with _patch_rows():
            r = sl.resolve("9999")
        assert r["ok"] and r["stock_id"] == "9999"

    def test_ambiguous_returns_candidates(self):
        """部分符合多筆、且無完全符合 → 回傳候選清單。"""
        rows = [
            {"stock_id": "2308", "stock_name": "台達電"},
            {"stock_id": "6121", "stock_name": "新普台達"},
        ]
        with patch.object(sl, "load_stock_list", return_value=rows):
            r = sl.resolve("台達")     # 非任一完整名稱，但為兩者子字串
        assert not r["ok"]
        assert len(r["candidates"]) == 2


class TestParseAndMany:
    def test_parse_mixed_separators(self):
        raw = "2330, 台積電\n2317 鴻海\n聯發科"
        assert sl.parse_pool_input(raw) == ["2330", "台積電", "2317", "鴻海", "聯發科"]

    def test_resolve_many_dedup(self):
        # 2330 與 台積電 應去重為一支
        with _patch_rows():
            out = sl.resolve_many(["2330", "台積電", "鴻海", "亂碼XYZ"])
        ids = [r["stock_id"] for r in out["resolved"]]
        assert ids == ["2330", "2317"]
        assert out["unresolved"] == ["亂碼XYZ"]
