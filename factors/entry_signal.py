"""
factors/entry_signal.py
盤中「小波段進場點」訊號模型。

═══════════════════════════════════════════════════════════════════════════════
模型設計理念
═══════════════════════════════════════════════════════════════════════════════
小波段（持有 1~5 個交易日）要賺的是「一段資金推動的行情」，所以進場點的
判準不是「便宜」，而是：

    ① 資金正在流進來（不是流出）
    ② 而且是「剛開始流」不是「流完了」
    ③ 掛單結構站在買方（下檔有承接）
    ④ 有量（沒量的上漲留不住，也出不掉）
    ⑤ 價格還沒噴到當日最高（不追末端）
    ⑥ 日線級別的法人籌碼沒有反向

六個子訊號各自 0~100 分，加權合成總分；另有「否決條件」會直接把分數壓下來
（例如資金加速流出、跌破 VWAP、漲停鎖死、流動性不足）——這些情境不管其他
分數多漂亮都不該進場。

    子訊號          權重   看什麼                        為什麼重要
    ─────────────────────────────────────────────────────────────────────
    flow_net        25%   主動買賣不平衡                 主力方向
    flow_accel      20%   資金流入的加速度               進場時機（早/晚）
    obi             15%   五檔委買賣不平衡               下檔承接力
    vol_surge       15%   量能倍率（vs 5日均量同時段）   行情續航力
    price_pos       15%   日內價格位階                   避免追高
    daily_chips     10%   外資＋投信近 5 日買超           日線資金方向一致性

分級：
    ≥ 75  🟢 強勢進場訊號
    ≥ 65  🟡 分批進場（建議 1/3 倉）
    ≥ 50  ⚪ 觀察，等訊號轉強
    ≥ 35  🔵 偏弱，不進場
    < 35  🔴 資金流出／被否決，勿進

⚠️ 本模型輸出為量化研究結果，不構成投資建議。
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 子訊號權重（總和 1.0）
SIGNAL_WEIGHTS = {
    "flow_net":    0.25,
    "flow_accel":  0.20,
    "obi":         0.15,
    "vol_surge":   0.15,
    "price_pos":   0.15,
    "daily_chips": 0.10,
}

# 分級門檻
GRADE_THRESHOLDS = [
    (75, "strong", "🟢 強勢進場訊號"),
    (65, "buy",    "🟡 分批進場"),
    (50, "watch",  "⚪ 觀察"),
    (35, "weak",   "🔵 偏弱，不進場"),
    (0,  "avoid",  "🔴 資金流出，勿進"),
]

# 否決條件的分數上限
VETO_CAPS = {
    "limit_down":     15,
    "limit_up_lock":  45,
    "flow_reversal":  35,
    "below_vwap":     40,
    "illiquid":       35,
}

# 流動性下限（張）：低於此值小波段進出容易滑價
MIN_INTRADAY_VOLUME = 500
# 樣本數低於此值，訊號信心標為「低」
MIN_SAMPLES_FOR_CONFIDENCE = 8


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _interp(x: float, anchors: List[tuple]) -> float:
    """依 (x, y) 錨點做分段線性內插；超出範圍取端點值。"""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for i in range(1, len(anchors)):
        x0, y0 = anchors[i - 1]
        x1, y1 = anchors[i]
        if x <= x1:
            span = x1 - x0
            w = (x - x0) / span if span else 1.0
            return y0 + (y1 - y0) * w
    return anchors[-1][1]


def _logistic(x: float, scale: float) -> float:
    """把任意實數壓到 0~100，x=0 → 50。scale 決定敏感度。"""
    import math
    try:
        return 100.0 / (1.0 + math.exp(-x / scale))
    except OverflowError:
        return 0.0 if x < 0 else 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# 子訊號
# ═══════════════════════════════════════════════════════════════════════════════

def score_flow_net(flow_imbalance: float) -> tuple:
    """
    主動買賣不平衡 (買-賣)/(買+賣)。
    ±0.10 已是明顯偏向，±0.25 是強勢單邊。
    """
    s = _logistic(flow_imbalance, 0.10)
    if flow_imbalance >= 0.20:
        note = f"主動買盤壓倒性領先（{flow_imbalance:+.0%}）"
    elif flow_imbalance >= 0.06:
        note = f"買方主動（{flow_imbalance:+.0%}）"
    elif flow_imbalance > -0.06:
        note = f"買賣力道均衡（{flow_imbalance:+.0%}）"
    else:
        note = f"賣壓主導（{flow_imbalance:+.0%}）"
    return _clamp(s), note


def score_flow_accel(flow_accel: float) -> tuple:
    """
    資金流加速度＝近期淨流入速率 ÷ 全段平均速率。
    >1 加速中（最想要的進場時機）；0~1 減速；<0 近期轉流出。
    """
    s = _interp(flow_accel, [(-2.0, 0), (-0.5, 15), (0.0, 35), (0.5, 48),
                             (1.0, 62), (1.6, 82), (2.5, 95), (4.0, 100)])
    if flow_accel >= 1.6:
        note = f"資金正在加速流入（{flow_accel:.1f}×）— 行情剛啟動"
    elif flow_accel >= 1.0:
        note = f"流入速度高於均值（{flow_accel:.1f}×）"
    elif flow_accel >= 0:
        note = f"流入趨緩（{flow_accel:.1f}×）— 動能在退"
    else:
        note = f"近期轉為淨流出（{flow_accel:.1f}×）"
    return _clamp(s), note


def score_obi(obi_avg: float) -> tuple:
    """五檔委買賣不平衡：買盤掛得多代表下檔有承接。"""
    s = _clamp(50 + obi_avg * 110)
    if obi_avg >= 0.25:
        note = f"委買遠多於委賣（{obi_avg:+.0%}）— 下檔承接強"
    elif obi_avg >= 0.08:
        note = f"掛單偏買方（{obi_avg:+.0%}）"
    elif obi_avg > -0.08:
        note = f"掛單均衡（{obi_avg:+.0%}）"
    else:
        note = f"委賣壓大於委買（{obi_avg:+.0%}）"
    return s, note


def score_vol_surge(vol_surge: Optional[float]) -> tuple:
    """
    量能倍率＝當日累積量 ÷（5 日均量 × 該時段應有比例）。
    1.0 = 與近期常態同步；<0.7 量縮；>4 常常是「行情已經走完」的爆量。
    """
    if vol_surge is None:
        return 50.0, "量能倍率無資料（缺 5 日均量）— 以中性計"
    s = _interp(vol_surge, [(0.3, 12), (0.6, 25), (1.0, 50), (1.3, 68),
                            (1.8, 85), (2.5, 96), (4.0, 100), (6.0, 78), (10.0, 60)])
    if vol_surge >= 4:
        note = f"爆量 {vol_surge:.1f}×（留意是否已在末升段）"
    elif vol_surge >= 1.3:
        note = f"量能放大 {vol_surge:.1f}× — 有資金關注"
    elif vol_surge >= 0.8:
        note = f"量能正常 {vol_surge:.1f}×"
    else:
        note = f"量縮 {vol_surge:.1f}× — 人氣不足"
    return _clamp(s), note


def score_price_position(price_position: float) -> tuple:
    """
    日內價格位階 (現價-最低)/(最高-最低)。
    最佳進場帶在 0.45~0.85：買方掌控但還沒噴到頂。
    貼底（<0.2）代表資金流入沒推動價格；貼頂（>0.92）是追高。
    """
    s = _interp(price_position, [(0.0, 18), (0.20, 35), (0.45, 75), (0.70, 90),
                                 (0.85, 80), (0.92, 50), (1.0, 25)])
    if price_position >= 0.92:
        note = f"貼近當日最高（位階 {price_position:.0%}）— 追高風險"
    elif price_position >= 0.45:
        note = f"價格位階 {price_position:.0%} — 買方掌控且未過熱"
    elif price_position >= 0.2:
        note = f"價格位階偏低 {price_position:.0%} — 資金尚未推動"
    else:
        note = f"貼近當日最低（{price_position:.0%}）— 弱勢"
    return _clamp(s), note


def score_daily_chips(foreign_net_5d: Optional[float],
                      trust_net_5d: Optional[float],
                      avg_volume_5d: Optional[float]) -> tuple:
    """
    日線籌碼：外資＋投信近 5 日買超佔近 5 日總成交量的比例。
    日內熱度若與日線資金方向相反，小波段勝率會明顯下降。
    """
    if foreign_net_5d is None and trust_net_5d is None:
        return 50.0, "法人籌碼無資料 — 以中性計"
    net = (foreign_net_5d or 0) + (trust_net_5d or 0)
    if not avg_volume_5d or avg_volume_5d <= 0:
        s = _logistic(net, 3000)
        note = f"法人近 5 日合計 {net:+,.0f} 張"
        return _clamp(s), note
    ratio = net / (avg_volume_5d * 5)
    s = _logistic(ratio, 0.03)
    if ratio >= 0.05:
        note = f"外資＋投信近 5 日大幅買超（{net:+,.0f} 張，佔量 {ratio:+.1%}）"
    elif ratio >= 0.01:
        note = f"法人溫和買超（{net:+,.0f} 張）"
    elif ratio > -0.01:
        note = f"法人態度中性（{net:+,.0f} 張）"
    else:
        note = f"法人近 5 日賣超（{net:+,.0f} 張）— 日線資金在退"
    return _clamp(s), note


# ═══════════════════════════════════════════════════════════════════════════════
# 主評分
# ═══════════════════════════════════════════════════════════════════════════════

def _grade(score: float) -> tuple:
    for threshold, key, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return key, label
    return "avoid", "🔴 資金流出，勿進"


def _build_trade_plan(summary: Dict, score: float) -> Optional[Dict]:
    """
    產出小波段操作參數。分數未達 65 不給計畫（不該進場就不要規劃進場）。

    停損：VWAP 下方 1.5% 與當日低點下方 0.5% 取較低者，但最多不超過 -4%
          （小波段的停損要小，超過 4% 代表這檔波動不適合這套打法）。
    停利：+4% 先出一半、+7% 出完（台股小波段常見的波幅）。
    時間停損：2 個交易日內沒啟動就退出，資金另尋標的。
    """
    if score < 65:
        return None
    price = summary.get("price") or 0
    if price <= 0:
        return None
    vwap = summary.get("vwap") or price
    low = summary.get("low") or price

    stop = min(vwap * 0.985, low * 0.995)
    stop = max(stop, price * 0.96)          # 停損不超過 -4%
    stop = min(stop, price * 0.985)         # 但也至少要有 1.5% 空間
    risk_pct = (price - stop) / price * 100
    t1, t2 = price * 1.04, price * 1.07
    avg_gain = ((t1 - price) + (t2 - price)) / 2
    rr = (avg_gain / (price - stop)) if price > stop else None

    return {
        "entry_low": round(min(vwap, price * 0.995), 2),
        "entry_high": round(price * 1.005, 2),
        "stop_loss": round(stop, 2),
        "risk_pct": round(risk_pct, 2),
        "target_1": round(t1, 2),
        "target_2": round(t2, 2),
        "reward_risk": round(rr, 2) if rr else None,
        "time_stop": "2 個交易日內未啟動即退出",
        "position_hint": "1/2 倉" if score >= 75 else "1/3 倉",
    }


def evaluate_entry(flow_summary: Dict, daily_context: Optional[Dict] = None) -> Dict:
    """
    綜合盤中資金流與日線籌碼，評估「現在是不是小波段進場點」。

    Args:
        flow_summary:  factors.intraday_flow.FlowTracker.summary() 的輸出
        daily_context: 選填的日線資料
            {
              "avg_volume_5d":  近 5 日均量（張），量能倍率與籌碼比例都要用
              "foreign_net_5d": 外資近 5 日買賣超（張）
              "trust_net_5d":   投信近 5 日買賣超（張）
              "ma20":           20 日均線（跌破會列為警示）
            }

    Returns:
        {
          "score":       0~100 綜合分,
          "grade":       strong/buy/watch/weak/avoid,
          "label":       中文分級標籤,
          "components":  {子訊號: {"score":, "weight":, "note":}},
          "vetoes":      [觸發的否決條件說明],
          "warnings":    [不否決但需留意的事],
          "confidence":  高/中/低（取決於樣本數與追蹤覆蓋率）,
          "trade_plan":  分數 ≥65 時的進出場參數，否則 None,
          "summary_line": 一行摘要（推播用）
        }
    """
    ctx = daily_context or {}

    if not flow_summary or not flow_summary.get("ready"):
        return {
            "score": 0.0, "grade": "unknown", "label": "⏳ 資料不足",
            "components": {}, "vetoes": [], "warnings": ["尚未取得足夠的即時快照"],
            "confidence": "低", "trade_plan": None,
            "summary_line": "資料不足，無法評估",
            "stock_id": (flow_summary or {}).get("stock_id", ""),
        }

    components = {}
    s, note = score_flow_net(flow_summary.get("flow_imbalance", 0.0))
    components["flow_net"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["flow_net"],
                              "label": "主動買賣力道", "note": note}

    s, note = score_flow_accel(flow_summary.get("flow_accel", 0.0))
    components["flow_accel"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["flow_accel"],
                                "label": "資金流加速度", "note": note}

    s, note = score_obi(flow_summary.get("obi_avg", 0.0))
    components["obi"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["obi"],
                         "label": "委買委賣結構", "note": note}

    s, note = score_vol_surge(flow_summary.get("vol_surge"))
    components["vol_surge"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["vol_surge"],
                               "label": "量能倍率", "note": note}

    s, note = score_price_position(flow_summary.get("price_position", 0.5))
    components["price_pos"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["price_pos"],
                               "label": "日內價格位階", "note": note}

    s, note = score_daily_chips(ctx.get("foreign_net_5d"), ctx.get("trust_net_5d"),
                                ctx.get("avg_volume_5d"))
    components["daily_chips"] = {"score": round(s, 1), "weight": SIGNAL_WEIGHTS["daily_chips"],
                                 "label": "日線法人籌碼", "note": note}

    raw_score = sum(c["score"] * c["weight"] for c in components.values())

    # ── 否決條件：不管加權分多高，這些情境都不該進場 ──────────────────────────
    vetoes: List[str] = []
    caps: List[int] = []

    if flow_summary.get("is_limit_down"):
        vetoes.append("跌停鎖死 — 無法評估承接，且隔日跳空風險高")
        caps.append(VETO_CAPS["limit_down"])

    if flow_summary.get("is_limit_up"):
        vetoes.append("漲停鎖死 — 追價成交不易，且已無小波段空間")
        caps.append(VETO_CAPS["limit_up_lock"])

    imb = flow_summary.get("flow_imbalance", 0.0)
    accel = flow_summary.get("flow_accel", 0.0)
    if imb < -0.05 and accel < 0:
        vetoes.append("資金淨流出且正在加速 — 主力在調節")
        caps.append(VETO_CAPS["flow_reversal"])

    pvw = flow_summary.get("price_vs_vwap", 0.0)
    if pvw < -1.5:
        vetoes.append(f"跌破當日均價（VWAP）{pvw:.1f}% — 當日買方已失守")
        caps.append(VETO_CAPS["below_vwap"])

    cum_vol = flow_summary.get("cum_volume", 0)
    if cum_vol < MIN_INTRADAY_VOLUME:
        vetoes.append(f"成交量僅 {cum_vol:,} 張 — 流動性不足，進出易滑價")
        caps.append(VETO_CAPS["illiquid"])

    score = min([raw_score] + caps) if caps else raw_score
    score = _clamp(score)

    # ── 警示（不否決，但要讓人看到）─────────────────────────────────────────
    warnings: List[str] = []
    ma20 = ctx.get("ma20")
    price = flow_summary.get("price") or 0
    if ma20 and price and price < ma20:
        warnings.append(f"股價在 20 日均線之下（MA20 {ma20:.2f}）— 逆勢單，建議減碼或觀望")
    if 0 < pvw < 0.3:
        warnings.append("價格貼著 VWAP — 方向尚未確立，可等站穩再進")
    if pvw > 3:
        warnings.append(f"已高於當日均價 {pvw:.1f}% — 追價成本偏高，建議等回測 VWAP")
    cov = flow_summary.get("coverage_ratio", 0)
    if cov < 0.15:
        warnings.append(f"本次追蹤僅涵蓋當日成交量的 {cov:.0%} — 訊號代表性有限")
    pos = flow_summary.get("price_position", 0.5)
    if pos >= 0.95:
        warnings.append("正好在當日最高附近 — 屬突破買法，務必守緊停損，或等回測 VWAP 再進")
    if flow_summary.get("change_pct", 0) > 7:
        warnings.append("今日漲幅已逾 7% — 小波段的空間被吃掉大半")

    samples = flow_summary.get("samples", 0)
    if samples < MIN_SAMPLES_FOR_CONFIDENCE or cov < 0.15:
        confidence = "低"
    elif samples < 30 or cov < 0.4:
        confidence = "中"
    else:
        confidence = "高"

    grade, label = _grade(score)
    plan = _build_trade_plan(flow_summary, score)

    top = sorted(components.items(), key=lambda kv: -kv[1]["score"] * kv[1]["weight"])[0]
    summary_line = (f"{flow_summary.get('stock_id','')} "
                    f"{flow_summary.get('stock_name','')} "
                    f"{score:.0f} 分 {label}；主要支撐：{top[1]['label']}")
    if vetoes:
        summary_line = (f"{flow_summary.get('stock_id','')} "
                        f"{flow_summary.get('stock_name','')} "
                        f"{score:.0f} 分 {label}；否決：{vetoes[0]}")

    return {
        "stock_id": flow_summary.get("stock_id", ""),
        "stock_name": flow_summary.get("stock_name", ""),
        "score": round(score, 1),
        "raw_score": round(raw_score, 1),
        "grade": grade,
        "label": label,
        "components": components,
        "vetoes": vetoes,
        "warnings": warnings,
        "confidence": confidence,
        "trade_plan": plan,
        "summary_line": summary_line,
        "price": price,
        "change_pct": flow_summary.get("change_pct", 0.0),
        "ts": flow_summary.get("ts"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 出場 / 減碼警示
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_exit(flow_summary: Dict, entry_price: Optional[float] = None,
                  stop_loss: Optional[float] = None,
                  target: Optional[float] = None) -> Dict:
    """
    已進場後的出場警示。小波段最怕「賺回吐」，所以資金一轉向就該減碼。

    Args:
        entry_price: 進場價（有給才算得出損益與停損停利觸價）
        stop_loss / target: 進場時規劃的停損停利價

    Returns:
        {"exit": bool, "urgency": "high"/"medium"/"none", "reasons": [...],
         "pnl_pct": 損益%（沒給 entry_price 則 None）}
    """
    if not flow_summary or not flow_summary.get("ready"):
        return {"exit": False, "urgency": "none", "reasons": ["資料不足"], "pnl_pct": None}

    price = flow_summary.get("price") or 0
    reasons: List[str] = []
    urgency = "none"

    pnl_pct = None
    if entry_price and entry_price > 0 and price > 0:
        pnl_pct = round((price / entry_price - 1) * 100, 2)

    if stop_loss and price and price <= stop_loss:
        reasons.append(f"觸及停損 {stop_loss:.2f}（現價 {price:.2f}）")
        urgency = "high"
    if target and price and price >= target:
        reasons.append(f"達停利目標 {target:.2f}（現價 {price:.2f}）— 建議至少減半")
        urgency = "high"

    imb = flow_summary.get("flow_imbalance", 0.0)
    accel = flow_summary.get("flow_accel", 0.0)
    if imb < -0.08 and accel < 0:
        reasons.append(f"資金由買轉賣且加速流出（不平衡 {imb:+.0%}）")
        urgency = "high"
    elif imb < 0:
        reasons.append(f"主動賣壓轉強（不平衡 {imb:+.0%}）")
        urgency = urgency if urgency == "high" else "medium"

    pvw = flow_summary.get("price_vs_vwap", 0.0)
    if pvw < -1.0:
        reasons.append(f"跌破當日均價 {pvw:.1f}% — 當日多方失守")
        urgency = "high"

    obi = flow_summary.get("obi_avg", 0.0)
    if obi < -0.25:
        reasons.append(f"委賣大量堆疊（OBI {obi:+.0%}）— 上方壓力重")
        urgency = urgency if urgency == "high" else "medium"

    if flow_summary.get("is_limit_down"):
        reasons.append("跌停鎖死")
        urgency = "high"

    return {
        "stock_id": flow_summary.get("stock_id", ""),
        "exit": urgency == "high",
        "urgency": urgency,
        "reasons": reasons or ["資金結構仍健康，續抱"],
        "pnl_pct": pnl_pct,
        "price": price,
    }


def format_entry_alert(signal: Dict, flow_summary: Optional[Dict] = None) -> str:
    """把進場訊號格式化成 Telegram 推播文字。"""
    fs = flow_summary or {}
    sid = signal.get("stock_id", "")
    name = signal.get("stock_name", "")
    score = signal.get("score", 0)
    price = signal.get("price", 0)
    chg = signal.get("change_pct", 0)

    lines = [
        "═══════════════════",
        f"⚡ 盤中資金訊號：{sid} {name}",
        "═══════════════════",
        f"{signal.get('label','')}　綜合 {score:.0f}/100（信心 {signal.get('confidence','—')}）",
        f"💰 現價 {price:,.2f}（{chg:+.2f}%）　{fs.get('trade_time','')}",
    ]

    net = fs.get("net_amount")
    if net is not None:
        lines.append(f"💵 主動買賣淨額 {net/1e8:+.2f} 億"
                     f"（{fs.get('flow_imbalance', 0):+.0%} 不平衡）")
    if fs.get("vwap"):
        lines.append(f"📉 均價 VWAP {fs['vwap']:.2f}（現價 {fs.get('price_vs_vwap', 0):+.1f}%）")
    if fs.get("vol_surge") is not None:
        lines.append(f"📊 量能 {fs['vol_surge']:.1f}×　累積 {fs.get('cum_volume', 0):,} 張")

    lines.append("")
    lines.append("📌 訊號拆解：")
    for c in signal.get("components", {}).values():
        lines.append(f"  ▪ {c['label']} {c['score']:.0f}　{c['note']}")

    if signal.get("vetoes"):
        lines += ["", "⛔ 否決條件："] + [f"  ▪ {v}" for v in signal["vetoes"]]
    if signal.get("warnings"):
        lines += ["", "⚠️ 注意："] + [f"  ▪ {w}" for w in signal["warnings"]]

    plan = signal.get("trade_plan")
    if plan:
        lines += [
            "",
            "🎯 小波段操作參考：",
            f"  進場帶　{plan['entry_low']:.2f} ~ {plan['entry_high']:.2f}（{plan['position_hint']}）",
            f"  停損　　{plan['stop_loss']:.2f}（-{plan['risk_pct']:.1f}%）",
            f"  停利　　{plan['target_1']:.2f} / {plan['target_2']:.2f}"
            + (f"　盈虧比 {plan['reward_risk']:.1f}" if plan.get("reward_risk") else ""),
            f"  時間停損　{plan['time_stop']}",
        ]

    lines += ["", "⚠️ 量化模型輸出，僅供研究參考，不構成投資建議。"]
    return "\n".join(lines)
