import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


def get_investment_advice(score_result: dict, stock_id: str) -> str:
    """
    呼叫 Claude API，輸入評分結果，輸出繁體中文投資建議。
    字數限制 350 字以內。失敗時回傳錯誤訊息。
    """
    if not ANTHROPIC_API_KEY:
        return "⚠️ 未設定 ANTHROPIC_API_KEY，無法取得 AI 建議。"

    cat = score_result.get("category_scores", {})
    raw = score_result.get("raw_factors", {})
    total = score_result.get("total_score", 0)
    rec = score_result.get("recommendation", "")

    prompt = f"""你是一位資深台股分析師，請根據以下量化評分，以繁體中文提供簡潔的投資建議。

【股票代號】{stock_id}
【綜合評分】{total:.1f} / 100
【投資建議】{rec}

【各面向分數（0~100）】
- 籌碼面：{cat.get('chips', 0):.1f}
- 基本面：{cat.get('fundamental', 0):.1f}
- 技術面：{cat.get('technical', 0):.1f}
- 動能面：{cat.get('momentum', 0):.1f}
- 風險面：{cat.get('risk', 0):.1f}

【關鍵指標】
- 外資近5日買賣超：{raw.get('fi_5d_net', 0):,.0f} 張
- 外資連續買賣超：{raw.get('fi_consecutive', 0)} 天
- 月營收年增率：{raw.get('rev_yoy', 0):.1f}%
- 最近季EPS：{raw.get('eps_latest', 0):.2f} 元
- RSI(14)：{raw.get('rsi_14', 50):.1f}
- 本益比：{raw.get('pe_ratio', 0):.1f}
- 近20日年化波動：{raw.get('vol_20d', 0):.1f}%

請依以下順序輸出，總字數 350 字以內：
1. 各面向評述（一句話）
2. 投資建議（含進出場考量）
3. 主要風險提示
4. 免責聲明（一句話）"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except anthropic.AuthenticationError:
        return "⚠️ API Key 驗證失敗，請確認 ANTHROPIC_API_KEY 設定是否正確。"
    except anthropic.RateLimitError:
        return "⚠️ API 請求頻率過高，請稍後再試。"
    except Exception as e:
        return f"⚠️ AI 建議取得失敗：{e}"
