import json
import os
from dotenv import load_dotenv

load_dotenv()

_LOCAL_CONFIG_PATH = "data/local_config.json"


def _get_secret(key: str, default: str = "") -> str:
    """
    讀取 API 金鑰，依序嘗試：
      1. 環境變數 / .env 檔（本地開發、Docker、Render）
      2. st.secrets（Streamlit Community Cloud）
    """
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_runtime_config(key: str, default: str = "") -> str:
    """
    在執行時（非模組載入時）讀取設定值。
    優先序：data/local_config.json → 環境變數 → .env → st.secrets
    供需要動態讀取的場景使用（例如透過 Streamlit 設定頁面更新 API key）。
    """
    try:
        with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            local = json.load(f)
        val = local.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return _get_secret(key, default)


def save_local_config(updates: dict) -> None:
    """將 API key 更新寫入 data/local_config.json。"""
    os.makedirs(os.path.dirname(_LOCAL_CONFIG_PATH), exist_ok=True)
    try:
        with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        existing = {}
    existing.update(updates)
    with open(_LOCAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


FINMIND_TOKEN = _get_secret("FINMIND_TOKEN")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get_secret("TELEGRAM_CHAT_ID")   # 推播目標 chat_id（向 Bot 發 /start 後取得）

# v3 新增
SMTP_HOST = _get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_get_secret("SMTP_PORT", "587") or "587")
SMTP_USER = _get_secret("SMTP_USER")
SMTP_PASSWORD = _get_secret("SMTP_PASSWORD")
ALERT_EMAIL = _get_secret("ALERT_EMAIL")
CLAUDE_MODEL = "claude-sonnet-4-6"

# 預設取資料天數（股價、籌碼）
DEFAULT_DAYS = 90
# 基本面資料需更長回溯（年增率比較）
FUNDAMENTAL_DAYS = 450

# 因子權重（總和 = 1.0）
# v5 穩健配置：基本面優先（長線投資導向），籌碼保留領先性、技術/動能降為輔助
FACTOR_WEIGHTS = {
    "chips":       0.20,
    "fundamental": 0.45,
    "technical":   0.10,
    "momentum":    0.10,
    "risk":        0.15,
}

# 評分門檻（對應投資建議標誌）
SCORE_THRESHOLDS = {
    "strong_buy": 80,
    "buy":        65,
    "hold":       45,
    "sell":       30,
}

# ===== v4 Screener 設定 =====

SCREENER_UNIVERSE_SIZE = int(_get_secret("SCREENER_UNIVERSE_SIZE", "200"))
SCREENER_TOP_N = int(_get_secret("SCREENER_TOP_N", "5"))

FILTER_MIN_MARKET_CAP_BILLION = 5
FILTER_MIN_AVG_VOLUME_K = 500
FILTER_MIN_PRICE = 10.0
FILTER_MAX_PRICE = 2000.0
FILTER_EXCLUDE_ETF = True
FILTER_MIN_REVENUE_YOY = -30.0

SCREENER_QUICK_SCORE_THRESHOLD = int(_get_secret("SCREENER_QUICK_SCORE_THRESHOLD", "60"))
SCREENER_MIN_RECOMMEND_SCORE = int(_get_secret("SCREENER_MIN_RECOMMEND_SCORE", "70"))

# 大盤狀態閘門：大盤有系統性風險警訊時，當日推薦降級為觀察名單（設 "0" 停用）
SCREENER_REGIME_FILTER = _get_secret("SCREENER_REGIME_FILTER", "1") == "1"

# Forward EPS 在最終排名的權重（0~1）：final = (1-w)*基礎分 + w*前瞻分
FORWARD_EPS_RERANK_WEIGHT = float(_get_secret("FORWARD_EPS_RERANK_WEIGHT", "0.25"))

BATCH_FETCH_DELAY_SEC = 0.8
BATCH_MAX_WORKERS = 3

RECOMMENDATION_DB_PATH = "data/recommendations.db"

SCREENER_SCHEDULE_HOUR = 17
SCREENER_SCHEDULE_MINUTE = 30
