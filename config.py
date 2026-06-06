import os
from dotenv import load_dotenv

load_dotenv()


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


FINMIND_TOKEN = _get_secret("FINMIND_TOKEN")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN")
CLAUDE_MODEL = "claude-sonnet-4-6"

# 預設取資料天數（股價、籌碼）
DEFAULT_DAYS = 90
# 基本面資料需更長回溯（年增率比較）
FUNDAMENTAL_DAYS = 450

# 因子權重（總和 = 1.0）
FACTOR_WEIGHTS = {
    "chips":       0.30,
    "fundamental": 0.25,
    "technical":   0.20,
    "momentum":    0.15,
    "risk":        0.10,
}

# 評分門檻（對應投資建議標誌）
SCORE_THRESHOLDS = {
    "strong_buy": 80,
    "buy":        65,
    "hold":       45,
    "sell":       30,
}
