"""
tests/test_twse_chips.py
證交所 T86 三大法人買賣超免費備援單元測試（全程 mock，無真實網路）
"""

import pandas as pd
from unittest.mock import patch, MagicMock

import data.twse_chips as tc


# ── dated 端點：fields + data 結構（中文標籤，帶千分位逗號）───────────────
_T86_FIELDS = [
    "證券代號", "證券名稱",
    "外陸資買賣超股數(不含外資自營商)", "外資自營商買賣超股數",
    "投信買賣超股數",
    "自營商買賣超股數(自行買賣)", "自營商買賣超股數(避險)", "自營商買賣超股數",
    "三大法人買賣超股數",
]

_T86_PAYLOAD = {
    "stat": "OK",
    "date": "20260723",
    "fields": _T86_FIELDS,
    "data": [
        ["2330", "台積電", "1,000,000", "0", "200,000", "-10,000", "-40,000", "-50,000", "1,150,000"],
        ["2317", "鴻海", "-500,000", "0", "--", "0", "0", "30,000", "-470,000"],
        ["0050", "元大台灣50", "123", "0", "0", "0", "0", "0", "123"],   # ETF 純數字代號 → 保留
        ["03001P", "權證", "1", "0", "0", "0", "0", "0", "1"],           # 含字母 → 濾除
    ],
}

# ── OpenAPI 端點：list-of-dicts（中文鍵）───────────────────────────────
_OPENAPI_RECORDS_ZH = [
    {"證券代號": "2330", "證券名稱": "台積電",
     "外陸資買賣超股數(不含外資自營商)": "1,000,000", "外資自營商買賣超股數": "0",
     "投信買賣超股數": "200,000",
     "自營商買賣超股數(自行買賣)": "-10,000", "自營商買賣超股數(避險)": "-40,000",
     "自營商買賣超股數": "-50,000", "三大法人買賣超股數": "1,150,000"},
]

# ── OpenAPI 端點：list-of-dicts（英文鍵）───────────────────────────────
_OPENAPI_RECORDS_EN = [
    {"Code": "2454", "Name": "聯發科",
     "ForeignInvestorsandMainlandInvestorsBuySell": "300000",
     "ForeignDealersSelf": "0",
     "InvestmentTrustBuySell": "80000",
     "DealersBuySell": "-12000"},
]


class TestRowsFromFields:
    def test_parses_three_investor_rows_per_stock(self):
        rows = tc._rows_from_fields(_T86_PAYLOAD, "2026-07-23")
        tsmc = [r for r in rows if r["stock_id"] == "2330"]
        assert len(tsmc) == 3
        by_name = {r["name"]: r["net"] for r in tsmc}
        assert by_name["外資"] == 1_000_000
        assert by_name["投信"] == 200_000
        assert by_name["自營商"] == -50_000        # 合計欄，非子項，非外資自營商
        assert tsmc[0]["date"] == "2026-07-23"

    def test_cleans_dash_and_commas(self):
        rows = tc._rows_from_fields(_T86_PAYLOAD, "2026-07-23")
        foxconn = {r["name"]: r["net"] for r in rows if r["stock_id"] == "2317"}
        assert foxconn["外資"] == -500_000
        assert foxconn["投信"] == 0.0               # "--" → 0

    def test_filters_non_numeric_codes(self):
        rows = tc._rows_from_fields(_T86_PAYLOAD, "2026-07-23")
        ids = {r["stock_id"] for r in rows}
        assert "03001P" not in ids
        assert "0050" in ids

    def test_non_ok_stat_returns_empty(self):
        assert tc._rows_from_fields({"stat": "無資料"}, "2026-07-23") == []

    def test_missing_fields_returns_empty(self):
        assert tc._rows_from_fields({"stat": "OK", "data": [["2330"]]}, "2026-07-23") == []


class TestRowsFromRecords:
    def test_openapi_zh_keys(self):
        rows = tc._rows_from_records(_OPENAPI_RECORDS_ZH, "2026-07-24")
        by_name = {r["name"]: r["net"] for r in rows}
        assert by_name["外資"] == 1_000_000
        assert by_name["投信"] == 200_000
        assert by_name["自營商"] == -50_000
        assert all(r["date"] == "2026-07-24" for r in rows)

    def test_openapi_en_keys(self):
        rows = tc._rows_from_records(_OPENAPI_RECORDS_EN, "2026-07-24")
        by_name = {r["name"]: r["net"] for r in rows}
        assert by_name["外資"] == 300000      # Foreign，排除 ForeignDealersSelf
        assert by_name["投信"] == 80000
        assert by_name["自營商"] == -12000

    def test_empty_records_returns_empty(self):
        assert tc._rows_from_records([], "2026-07-24") == []


class TestNumberCleaning:
    def test_to_num_variants(self):
        assert tc._to_num("1,234,567") == 1234567.0
        assert tc._to_num("--") == 0.0
        assert tc._to_num("-") == 0.0
        assert tc._to_num("") == 0.0
        assert tc._to_num(None) == 0.0
        assert tc._to_num("-12,000") == -12000.0


class TestPickColumns:
    def test_zh_fields(self):
        cols = tc._pick_columns(_T86_FIELDS)
        assert cols["code"] == "證券代號"
        assert cols["foreign"] == "外陸資買賣超股數(不含外資自營商)"
        assert cols["trust"] == "投信買賣超股數"
        assert cols["dealer"] == "自營商買賣超股數"     # 合計欄

    def test_dealer_excludes_foreign_dealer_and_subitems(self):
        cols = tc._pick_columns(_T86_FIELDS)
        assert "外資" not in cols["dealer"]
        assert "(" not in cols["dealer"]


class TestGetT86Institutional:
    def _market(self):
        return pd.DataFrame([
            {"date": "2026-07-23", "stock_id": "2330", "stock_name": "台積電", "name": "外資", "net": 1_000_000},
            {"date": "2026-07-23", "stock_id": "2330", "stock_name": "台積電", "name": "投信", "net": 200_000},
            {"date": "2026-07-23", "stock_id": "2317", "stock_name": "鴻海", "name": "外資", "net": -500_000},
        ])

    def test_filters_single_stock(self):
        with patch.object(tc, "get_market_institutional", return_value=self._market()):
            df = tc.get_t86_institutional("2330")
        assert set(df.columns) == {"date", "name", "net"}
        assert len(df) == 2
        assert set(df["name"]) == {"外資", "投信"}

    def test_unknown_stock_returns_empty(self):
        with patch.object(tc, "get_market_institutional", return_value=self._market()):
            assert tc.get_t86_institutional("9999").empty

    def test_empty_market_returns_empty(self):
        with patch.object(tc, "get_market_institutional", return_value=pd.DataFrame()):
            assert tc.get_t86_institutional("2330").empty


class TestFetchAndCache:
    def setup_method(self):
        tc._MARKET_CACHE = None
        tc._CACHE_LOADED = False

    def teardown_method(self):
        tc._MARKET_CACHE = None
        tc._CACHE_LOADED = False

    def test_fetch_openapi_parses_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _OPENAPI_RECORDS_ZH
        mock_resp.raise_for_status.return_value = None
        with patch("data.twse_chips.requests.get", return_value=mock_resp):
            rows = tc._fetch_openapi()
        assert any(r["stock_id"] == "2330" and r["name"] == "外資" for r in rows)

    def test_fetch_dated_day_parses_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _T86_PAYLOAD
        mock_resp.raise_for_status.return_value = None
        with patch("data.twse_chips.requests.get", return_value=mock_resp):
            rows = tc._fetch_dated_day("20260723")
        assert rows[0]["date"] == "2026-07-23"       # YYYYMMDD → ISO

    def test_fetch_swallows_errors(self):
        with patch("data.twse_chips.requests.get", side_effect=RuntimeError("net down")):
            assert tc._fetch_openapi() == []
            assert tc._fetch_dated_day("20260723") == []

    def test_openapi_primary_skips_dated(self):
        """OpenAPI 有資料時不應再打 dated 端點。"""
        with patch.object(tc, "_fetch_openapi", return_value=[
                {"date": "2026-07-24", "stock_id": "2330", "stock_name": "台積電",
                 "name": "外資", "net": 1}]), \
             patch.object(tc, "_fetch_dated_day") as mock_dated:
            df = tc._load_market_institutional()
        assert not df.empty
        mock_dated.assert_not_called()

    def test_falls_back_to_dated_when_openapi_empty(self):
        with patch.object(tc, "_fetch_openapi", return_value=[]), \
             patch.object(tc, "_fetch_dated_day", return_value=[
                {"date": "2026-07-23", "stock_id": "2330", "stock_name": "台積電",
                 "name": "外資", "net": 1}]):
            df = tc._load_market_institutional()
        assert not df.empty
        assert "2330" in df["stock_id"].values

    def test_dedup_overlapping_day(self):
        dup = [
            {"date": "2026-07-24", "stock_id": "2330", "stock_name": "台積電", "name": "外資", "net": 1},
            {"date": "2026-07-24", "stock_id": "2330", "stock_name": "台積電", "name": "外資", "net": 1},
        ]
        with patch.object(tc, "_fetch_openapi", return_value=dup):
            df = tc._load_market_institutional()
        assert len(df) == 1

    def test_market_cache_reused(self):
        call_count = {"n": 0}

        def fake_load():
            call_count["n"] += 1
            return pd.DataFrame([{"date": "2026-07-23", "stock_id": "2330",
                                  "stock_name": "台積電", "name": "外資", "net": 1}])

        with patch.object(tc, "_load_market_institutional", side_effect=fake_load):
            tc.get_market_institutional()
            tc.get_market_institutional()
        assert call_count["n"] == 1

    def test_fmt_iso(self):
        assert tc._fmt_iso("20260723") == "2026-07-23"
