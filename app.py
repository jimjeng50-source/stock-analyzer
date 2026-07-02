"""台股多因子評分系統 v2 — Streamlit 介面（個股分析 / 總體資金面 / 回測驗證）"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_TW_TZ = ZoneInfo("Asia/Taipei")


def _now_tw() -> datetime:
    """台灣當前時間（UTC+8）。"""
    return datetime.now(_TW_TZ)


def _today_tw():
    """台灣今日日期。"""
    return _now_tw().date()

from data.fetcher import FinMindFetcher
from factors import compute_chips, compute_technical, compute_fundamental, compute_momentum
from models.scorer import Scorer
from config import get_runtime_config, save_local_config

st.set_page_config(page_title="台股多因子評分系統", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.score-big{font-size:3.2rem;font-weight:bold;color:#ffd93d;line-height:1}
.rec-label{font-size:1.2rem;padding:6px 14px;border-radius:8px;background:#1e1e2e;display:inline-block;margin-top:6px}
.macro-card{background:#161b22;border-radius:10px;padding:14px 18px;margin:4px 0}
.warn{background:#2d1b00;border:1px solid #ff9800;border-radius:8px;padding:10px 14px;margin:6px 0}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 側欄控制
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ 分析設定")

    # ── Tab1：個股分析 ────────────────────────────────────────────────────────
    stock_id = st.text_input("股票代號", value="2330", max_chars=10).strip()

    st.markdown("---")
    st.subheader("因子權重（%）")
    w_chips = st.slider("籌碼面", 0, 100, 30, step=5)
    w_fund  = st.slider("基本面", 0, 100, 25, step=5)
    w_tech  = st.slider("技術面", 0, 100, 20, step=5)
    w_mom   = st.slider("動能面", 0, 100, 15, step=5)
    w_risk  = st.slider("風險面", 0, 100, 10, step=5)

    total_w = w_chips + w_fund + w_tech + w_mom + w_risk
    if total_w != 100:
        st.markdown(f'<div class="warn">⚠️ 權重總和 {total_w}%，系統將自動正規化</div>', unsafe_allow_html=True)
    else:
        st.success(f"權重總和：{total_w}%")

    use_macro_adj = st.checkbox("啟用總體資金面乘數", value=True)
    use_ai        = st.checkbox("啟用 AI 投資建議", value=bool(ANTHROPIC_API_KEY))
    analyze_btn   = st.button("🔍 開始個股分析", type="primary", use_container_width=True)

    # ── Tab3：回測設定 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("回測設定")
    bt_stock = st.text_input("回測標的", value="0050", max_chars=10).strip()
    bt_start = st.date_input("開始日期", value=_today_tw() - timedelta(days=730))
    bt_end   = st.date_input("結束日期", value=_today_tw())
    bt_buy   = st.slider("買進閾值", 50, 90, 65, step=5)
    bt_sell  = st.slider("賣出閾值", 20, 60, 45, step=5)
    bt_macro = st.checkbox("加入總體資金面乘數", value=True)
    bt_cap   = st.number_input("初始資金（元）", value=1_000_000, step=100_000, min_value=100_000)
    bt_btn   = st.button("📈 執行回測", type="secondary", use_container_width=True)

    st.markdown("---")
    if not FINMIND_TOKEN:
        st.warning("⚠️ 未設定 FinMind Token")
    if use_ai and not ANTHROPIC_API_KEY:
        st.warning("⚠️ 未設定 Anthropic API Key")


# ═══════════════════════════════════════════════════════════════════════════════
# 頁籤
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 個股分析", "🌐 總體資金面", "📈 回測驗證", "📚 因子說明", "🛡️ 風險監控",
    "🎯 Forward EPS", "🔗 產業鏈分析", "📅 月營收追蹤", "🏆 每日推薦", "⚙️ 設定",
])


# ───────────────────────────────────────────────────────────────────────────────
# Tab 1：個股分析
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    if analyze_btn and stock_id:
        weights = {k: v / 100 for k, v in zip(
            ["chips","fundamental","technical","momentum","risk"],
            [w_chips, w_fund, w_tech, w_mom, w_risk]
        )}

        with st.spinner(f"正在取得 {stock_id} 市場資料..."):
            fetcher = FinMindFetcher(stock_id)
            price_df = fetcher.get_price()

        if price_df.empty:
            st.error(f"❌ 無法取得 **{stock_id}** 股價資料，請確認代號是否正確。")
            st.stop()

        with st.spinner("計算因子與評分..."):
            institutional_df = fetcher.get_institutional()
            margin_df        = fetcher.get_margin_trading()
            revenue_df       = fetcher.get_monthly_revenue()
            financial_df     = fetcher.get_financial_statements()
            current_price    = float(price_df["close"].iloc[-1])

            chips       = compute_chips(institutional_df, margin_df)
            technical   = compute_technical(price_df)
            fundamental = compute_fundamental(revenue_df, financial_df, current_price)
            momentum    = compute_momentum(price_df)
            result      = Scorer(weights).score(chips, technical, fundamental, momentum)

        # 儲存到 session_state 供因子說明頁使用
        st.session_state["last_raw"]      = result["raw_factors"]
        st.session_state["last_result"]   = result
        st.session_state["last_stock_id"] = stock_id

        total = result["total_score"]
        rec   = result["recommendation"]
        cat   = result["category_scores"]
        raw   = result["raw_factors"]

        # 總體資金面乘數
        macro_info = None
        if use_macro_adj:
            try:
                from macro.macro_scorer import calc_macro_score
                macro_info = calc_macro_score()
                adj_total  = round(total * macro_info["multiplier"], 1)
            except Exception:
                adj_total = total
        else:
            adj_total = total

        # ── Row 1：分數 + 雷達 + 長條 ────────────────────────────────────────
        c1, c2, c3 = st.columns([1, 1.5, 1.5])
        with c1:
            st.markdown(f'<div class="score-big">{total:.1f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rec-label">{rec}</div>', unsafe_allow_html=True)
            if macro_info:
                mult = macro_info["multiplier"]
                st.metric("總體資金面調整後", f"{adj_total}",
                          delta=f"乘數 ×{mult:.3f}  {macro_info['signal'][:2]}")
            st.metric("現價", f"NT$ {current_price:,.2f}")
            st.caption(f"資料筆數：{len(price_df)} 日")

        label_map = {"chips":"籌碼面","fundamental":"基本面","technical":"技術面","momentum":"動能面","risk":"風險面"}
        cats   = list(label_map.keys())
        labels = [label_map[c] for c in cats]
        values = [cat.get(c, 0) for c in cats]
        vc = values + [values[0]]
        lc = labels + [labels[0]]

        with c2:
            fig_radar = go.Figure(go.Scatterpolar(r=vc, theta=lc, fill="toself",
                line=dict(color="#00d4ff", width=2), fillcolor="rgba(0,212,255,0.18)"))
            fig_radar.update_layout(template="plotly_dark",
                polar=dict(radialaxis=dict(range=[0,100])),
                margin=dict(l=40,r=40,t=30,b=30), height=280, title="各面向雷達圖")
            st.plotly_chart(fig_radar, use_container_width=True)

        with c3:
            colors = ["#ff6b6b" if v<40 else "#ffd93d" if v<65 else "#6bcb77" for v in values]
            fig_bar = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                marker_color=colors, text=[f"{v:.1f}" for v in values], textposition="outside"))
            fig_bar.update_layout(template="plotly_dark", xaxis=dict(range=[0,115]),
                margin=dict(l=10,r=50,t=30,b=10), height=280, title="類別分數")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── K 線圖 ─────────────────────────────────────────────────────────
        st.markdown("---")
        if all(c in price_df.columns for c in ["open","high","low","close"]):
            try:
                from ta.trend import SMAIndicator
                cls = price_df["close"].astype(float)
                ma5_s  = SMAIndicator(cls, 5).sma_indicator()
                ma20_s = SMAIndicator(cls, 20).sma_indicator()
                ma60_s = SMAIndicator(cls, 60).sma_indicator()
                _ta_ok = True
            except Exception:
                _ta_ok = False

            fig_k = go.Figure()
            fig_k.add_trace(go.Candlestick(x=price_df["date"],
                open=price_df["open"], high=price_df["high"],
                low=price_df["low"],  close=price_df["close"],
                name="K線", increasing_line_color="#ff4b4b", decreasing_line_color="#00c087"))
            if _ta_ok:
                for s, name, color in [(ma5_s,"MA5","#ffd93d"),(ma20_s,"MA20","#00d4ff"),(ma60_s,"MA60","#ff9f43")]:
                    fig_k.add_trace(go.Scatter(x=price_df["date"], y=s, name=name,
                                               line=dict(color=color, width=1.5)))
            fig_k.update_layout(template="plotly_dark", title=f"{stock_id} 股價走勢",
                                 xaxis_rangeslider_visible=False, height=420)
            st.plotly_chart(fig_k, use_container_width=True)

        # ── 三大法人 ──────────────────────────────────────────────────────
        if not institutional_df.empty and "name" in institutional_df.columns and "net" in institutional_df.columns:
            st.markdown("---")
            fi_d = institutional_df[institutional_df["name"].str.contains("外資",na=False)].groupby("date")["net"].sum().reset_index()
            if not fi_d.empty:
                fig_i = go.Figure(go.Bar(x=fi_d["date"], y=fi_d["net"],
                    marker_color=["#ff4b4b" if v<0 else "#00c087" for v in fi_d["net"]], name="外資買賣超"))
                fig_i.update_layout(template="plotly_dark", title="外資每日買賣超（張）", height=300)
                st.plotly_chart(fig_i, use_container_width=True)
        elif not FINMIND_TOKEN:
            st.info("📌 設定 FinMind Token 後可顯示三大法人圖表")

        # ── 月營收 ────────────────────────────────────────────────────────
        if not revenue_df.empty and "revenue" in revenue_df.columns:
            st.markdown("---")
            rv = revenue_df.copy().sort_values("date")
            rv["rev_b"] = rv["revenue"] / 1e8
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=rv["date"], y=rv["rev_b"], marker_color="#58a6ff", name="月營收（億）"))
            if len(rv) >= 6:
                rv["ma6"] = rv["rev_b"].rolling(6, min_periods=1).mean()
                fig_r.add_trace(go.Scatter(x=rv["date"], y=rv["ma6"], name="6月均線",
                                           line=dict(color="#ffd93d", width=2)))
            fig_r.update_layout(template="plotly_dark", title="月營收趨勢（億元）", height=300)
            st.plotly_chart(fig_r, use_container_width=True)

        # ── 原始因子 ──────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 原始因子數值", expanded=False):
            rows = [{"因子": k, "數值": f"{v:,.2f}" if isinstance(v, float) else v}
                    for k, v in sorted(raw.items())]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── 個股法人異常警示 ──────────────────────────────────────────────
        if not institutional_df.empty:
            st.markdown("---")
            st.subheader("🚨 三大法人異常警示（個股）")
            from macro.institutional_alert import per_stock_alerts
            stock_alerts = per_stock_alerts(institutional_df, stock_id)
            if stock_alerts:
                for alr in stock_alerts:
                    level = alr.get("level", "⚪")
                    bg = {"🔴": "#2d1010", "🟢": "#0d2d1a", "🟠": "#2d1b00"}.get(level, "#1a1a2e")
                    border = {"🔴": "#ff4b4b", "🟢": "#00c087", "🟠": "#ff9f43"}.get(level, "#888")
                    st.markdown(
                        f'<div style="background:{bg};border-left:4px solid {border};'
                        f'padding:8px 14px;border-radius:6px;margin:4px 0">'
                        f'{level} <b>{alr["type"]}</b>　{alr["msg"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ 未偵測到法人異常行為")

        # ── 風險相關係數分析 ──────────────────────────────────────────────
        if not price_df.empty:
            st.markdown("---")
            with st.expander("📉 歷史跌幅 ≥5% 風險相關分析", expanded=False):
                with st.spinner("計算歷史風險相關係數..."):
                    try:
                        from utils.risk_correlation import compute_risk_correlations
                        lookback = min(len(price_df) - 1, 365)
                        risk_res = compute_risk_correlations(
                            price_df,
                            institutional_df if not institutional_df.empty else None,
                            drop_threshold=-5.0,
                            lookback_days=lookback,
                        )
                        _risk_ok = True
                    except Exception as e:
                        st.warning(f"風險分析失敗：{e}")
                        _risk_ok = False

                if _risk_ok:
                    if risk_res.get("message"):
                        st.warning(risk_res["message"])
                    else:
                        dc, ad, md = risk_res["drop_count"], risk_res["avg_drop"], risk_res["max_drop"]
                        rs, rl = risk_res["risk_score"], risk_res["risk_level"]
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("大跌次數（≥5%）", f"{dc} 次")
                        rc2.metric("平均跌幅", f"{ad:.1f}%")
                        rc3.metric("最大單日跌幅", f"{md:.1f}%")
                        rc4.metric("當前風險分數", f"{rs:.0f}/100", delta=rl)

                        if risk_res["top_risk_factors"]:
                            st.markdown("**⚠️ 統計相關係數最高的風險指標（當大跌發生時這些因子常見異常）**")
                            risk_rows = [
                                {
                                    "風險指標": r["factor_label"],
                                    "相關係數": f"{r['correlation']:.3f}",
                                    "統計意義": r["interpretation"],
                                }
                                for r in risk_res["top_risk_factors"]
                            ]
                            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

                        if risk_res["correlations"]:
                            all_rows = [
                                {
                                    "指標": r["factor_label"],
                                    "r": r["correlation"],
                                    "解讀": r["interpretation"],
                                }
                                for r in risk_res["correlations"]
                            ]
                            corr_df = pd.DataFrame(all_rows)
                            corr_colors = ["#ff4b4b" if v < -0.3 else "#ffd93d" if v < 0 else "#6bcb77"
                                           for v in corr_df["r"]]
                            fig_corr = go.Figure(go.Bar(
                                x=corr_df["r"], y=corr_df["指標"], orientation="h",
                                marker_color=corr_colors,
                                text=[f"{v:.3f}" for v in corr_df["r"]], textposition="outside",
                            ))
                            fig_corr.update_layout(
                                template="plotly_dark",
                                title="各指標與當日跌幅的 Pearson 相關係數（負值=跌跌相關）",
                                xaxis=dict(range=[-1.1, 1.1]),
                                height=380, margin=dict(l=10, r=60, t=40, b=10),
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)

        # ── AI 建議 ───────────────────────────────────────────────────────
        if use_ai:
            st.markdown("---")
            st.subheader("🤖 AI 投資建議")
            if not ANTHROPIC_API_KEY:
                st.warning("未設定 ANTHROPIC_API_KEY。")
            else:
                with st.spinner("向 Claude 取得投資建議..."):
                    from utils.claude_api import get_investment_advice
                    advice = get_investment_advice(result, stock_id)
                st.info(advice)

    elif not analyze_btn:
        st.markdown("## 👈 請在左側輸入股票代號並按「開始個股分析」")
        st.markdown("測試代號：`2330`（台積電）、`6213`（聯茂）、`0050`（ETF）")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 2：總體資金面
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🌐 總體資金面儀表板")
    st.caption(f"更新時間：{_now_tw().strftime('%Y-%m-%d %H:%M')} (台灣時間)")

    with st.spinner("載入總體資金面資料..."):
        try:
            from macro.macro_scorer import calc_macro_score
            from macro.vix import get_vix_series, vix_to_label
            from macro.fx import get_fx_series
            macro = calc_macro_score()
            _macro_ok = True
        except Exception as e:
            st.error(f"總體資金面資料載入失敗：{e}")
            _macro_ok = False

    if _macro_ok:
        raw_m = macro["raw"]
        comp  = macro["components"]

        # 預先取得原始序列（供3日趨勢使用）
        fx_s  = get_fx_series(10)
        vix_s = get_vix_series(10)

        # ── 信號標語 + 乘數說明 ────────────────────────────────────────────
        ms   = macro["macro_score"]
        mult = macro["multiplier"]
        sig  = macro["signal"]
        st.markdown(f"### {sig}")

        col_score, col_mult = st.columns([1, 2])
        with col_score:
            st.markdown(f"**總體評分：{ms:.2f} / 1.00**")
            st.progress(ms)
        with col_mult:
            adj_ex = round(70 * mult, 1)
            st.info(
                f"**個股評分乘數 ×{mult:.3f}** 的意義：\n"
                f"若個股原始評分為 70 分，調整後 = 70 × {mult:.3f} = **{adj_ex} 分**。\n"
                f"乘數範圍 **×0.70**（總體極差）～ **×1.00**（總體極佳），"
                f"反映當前總體環境對股市的加分或減分效果。"
            )

        # ── 四指標卡 ─────────────────────────────────────────────────────
        def _score_badge(score: float, good: str, mid: str, bad: str) -> str:
            if score >= 0.65: return f"🟢 {good}"
            elif score >= 0.45: return f"🟡 {mid}"
            return f"🔴 {bad}"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            twd5 = raw_m.get("twd_5d_chg", 0)
            fx_now = float(fx_s.iloc[-1]) if not fx_s.empty else 0
            st.metric("台幣匯率", f"{fx_now:.2f} TWD/USD",
                      delta=f"5日 {twd5:+.2f}%（{'升值' if twd5>0 else '貶值'}）",
                      delta_color="normal" if twd5 > 0 else "inverse")
            s = comp["fx_score"]
            interp = _score_badge(s, "升值趨勢，外資匯入利多", "走勢中性，觀望為主", "貶值趨勢，外資匯出壓力")
            st.caption(f"匯率分數：{s*100:.0f}/100　{interp}")

        with mc2:
            vix_v = raw_m.get("vix_level", 20)
            vix_c = raw_m.get("vix_5d_chg", 0)
            st.metric("VIX 恐慌指數", f"{vix_v:.1f} 點",
                      delta=f"5日 {vix_c:+.1f} 點（{'恐慌升溫' if vix_c>0 else '恐慌降溫'}）",
                      delta_color="inverse" if vix_c > 0 else "normal")
            s = comp["vix_score"]
            interp = _score_badge(s, "市場平靜，利於做多", "波動中性，保持警覺", "恐慌升溫，謹慎操作")
            st.caption(f"VIX分數：{s*100:.0f}/100　{interp}")

        with mc3:
            fi_net = raw_m.get("fi_future_net", 0)
            fi_5d  = raw_m.get("fi_future_5d_chg", 0)
            direction = "增加多單" if fi_5d >= 0 else "增加空單"
            st.metric("外資台指期淨多單", f"{fi_net:,} 口",
                      delta=f"5日 {fi_5d:+,} 口（{direction}）",
                      delta_color="normal" if fi_5d >= 0 else "inverse")
            s = comp["futures_score"]
            net_interp = "大量做多，看漲台股" if fi_net>50000 else \
                         "小幅做多" if fi_net>0 else \
                         "小幅做空，謹慎" if fi_net>-50000 else "大量做空，看跌台股"
            interp = _score_badge(s, "外資看多台股", "期貨部位中性", "外資大量放空")
            st.caption(f"期貨分數：{s*100:.0f}/100　{interp}（{net_interp}）")

        with mc4:
            fi5d = raw_m.get("fi_total_5d", 0)
            cons  = raw_m.get("fi_consecutive_days", 0)
            fi5d_b = fi5d / 1e8
            cons_txt = f"連續{'買超' if cons>0 else '賣超'} {abs(cons)} 天" if cons != 0 else "昨日持平"
            st.metric("外資現貨近5日合計",
                      f"{fi5d_b:+.1f} 億元" if fi5d != 0 else "資料載入中",
                      delta=cons_txt,
                      delta_color="normal" if fi5d >= 0 else "inverse")
            s = comp["flow_score"]
            interp = _score_badge(s, "資金積極流入", "資金小幅流動", "資金持續流出")
            st.caption(f"資金流分數：{s*100:.0f}/100　{interp}")

        # ── 近3日關鍵指標變化 ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📅 近3日關鍵指標變化")

        if not fx_s.empty and not vix_s.empty:
            trend_rows = []
            today = _today_tw()
            for i, (fx_d, fx_v) in enumerate(reversed(list(fx_s.tail(3).items()))):
                # 依實際日期決定標籤，而非假設最新筆就是「今日」
                data_date = fx_d.date() if hasattr(fx_d, "date") else fx_d
                days_ago = (today - data_date).days
                if days_ago == 0:
                    day_lbl = "今日"
                elif days_ago == 1:
                    day_lbl = "昨日"
                elif days_ago == 2:
                    day_lbl = "前日"
                else:
                    day_lbl = f"{days_ago}天前"
                vix_v_d = float(vix_s.iloc[-(i+1)]) if len(vix_s) > i else None
                # 計算台幣升貶
                if i < len(fx_s) - 1:
                    prev_fx = float(list(fx_s.items())[-(i+2)][1])
                    fx_chg = round((float(fx_v) - prev_fx) / prev_fx * 100, 3)
                    fx_delta = f"{'升' if fx_chg<0 else '貶'}{abs(fx_chg):.3f}%"
                else:
                    fx_delta = "—"

                trend_rows.append({
                    "日期": f"{fx_d.strftime('%m/%d')} {day_lbl}",
                    "台幣匯率 (TWD/USD)": f"{float(fx_v):.2f}　{fx_delta}",
                    "VIX (點)": f"{vix_v_d:.1f}" if vix_v_d else "—",
                    "外資期貨淨多單 (口)": f"{raw_m.get('fi_future_net',0):,}" if i == 0 else "—",
                    "外資現貨5日 (億)": f"{fi5d_b:+.1f}" if i == 0 else "—",
                })
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)
            latest_date = list(fx_s.tail(1).index)[0]
            latest_days_ago = (today - latest_date.date()).days
            lag_note = "" if latest_days_ago == 0 else f"（最新資料為 {latest_date.strftime('%m/%d')}，因週末或資料延遲，非今日）"
            st.caption(f"※ 期貨與資金流向資料當日更新一次，歷史日資料需付費方案　{lag_note}")
        else:
            st.warning("無法取得近3日資料（yfinance 可能暫時無法連線）")

        # ── 面向評分長條圖 ────────────────────────────────────────────────
        st.markdown("---")
        comp_labels = ["匯率面","VIX面","期貨面","資金流"]
        comp_values = [comp["fx_score"], comp["vix_score"], comp["futures_score"], comp["flow_score"]]
        comp_desc   = [
            f"台幣升貶 + 趨勢",
            f"VIX={raw_m.get('vix_level',20):.1f}點，{vix_to_label(raw_m.get('vix_level',20))}",
            f"期貨淨多單 {raw_m.get('fi_future_net',0):,} 口",
            f"外資現貨近5日 {fi5d_b:+.1f} 億元",
        ]
        comp_colors = ["#ff6b6b" if v<0.4 else "#ffd93d" if v<0.65 else "#6bcb77" for v in comp_values]
        fig_comp = go.Figure(go.Bar(
            x=[v*100 for v in comp_values], y=comp_labels, orientation="h",
            marker_color=comp_colors,
            text=[f"{v*100:.1f}　{d}" for v, d in zip(comp_values, comp_desc)],
            textposition="outside",
        ))
        fig_comp.update_layout(template="plotly_dark", xaxis=dict(range=[0,160]),
                               height=250, title="各面向評分（0~100）與當前數值",
                               margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── 走勢圖（60日） ────────────────────────────────────────────────
        fx_s60  = get_fx_series(60)
        vix_s60 = get_vix_series(60)
        col_a, col_b = st.columns(2)
        with col_a:
            if not fx_s60.empty:
                fig_fx = go.Figure(go.Scatter(x=fx_s60.index, y=fx_s60.values,
                                              line=dict(color="#00d4ff", width=2)))
                fig_fx.update_layout(template="plotly_dark",
                                     title="USDTWD 匯率（近60日，數值越低台幣越強）",
                                     yaxis_title="TWD/USD",
                                     height=300, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_fx, use_container_width=True)

        with col_b:
            if not vix_s60.empty:
                fig_vix = go.Figure(go.Scatter(x=vix_s60.index, y=vix_s60.values,
                                               line=dict(color="#ffd93d", width=2),
                                               fill="tozeroy", fillcolor="rgba(255,211,61,0.1)"))
                for level, color, lbl in [(15,"#6bcb77","平靜<15"),(20,"#ffd93d","注意>20"),(30,"#ff6b6b","警戒>30")]:
                    fig_vix.add_hline(y=level, line_dash="dot", line_color=color,
                                      annotation_text=lbl, annotation_position="right")
                fig_vix.update_layout(template="plotly_dark",
                                      title="VIX 恐慌指數（近60日）",
                                      yaxis_title="點",
                                      height=300, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_vix, use_container_width=True)

        # ── 巴菲特指標 ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 台灣巴菲特指標")
        st.caption("台股總市值 / 台灣 GDP（%）— 衡量股市相對經濟體量是否過度高估")
        with st.spinner("計算巴菲特指標..."):
            try:
                from macro.buffett import compute_buffett
                bf = compute_buffett(FINMIND_TOKEN)
                _bf_ok = True
            except Exception as e:
                st.warning(f"巴菲特指標計算失敗：{e}")
                _bf_ok = False

        if _bf_ok:
            bf1, bf2, bf3 = st.columns(3)
            with bf1:
                ratio = bf["ratio"]
                color_hex = {"🟢": "#6bcb77", "🟡": "#ffd93d", "🟠": "#ff9f43", "🔴": "#ff4b4b"}.get(bf["color"], "#aaa")
                st.markdown(
                    f'<div style="text-align:center;background:#161b22;border-radius:10px;padding:18px">'
                    f'<div style="font-size:2.8rem;font-weight:bold;color:{color_hex}">{ratio:.1f}%</div>'
                    f'<div style="font-size:0.95rem;color:#aaa;margin-top:4px">市值 / GDP</div>'
                    f'<div style="margin-top:8px;font-size:1rem;color:{color_hex}">{bf["signal"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with bf2:
                st.metric("台股總市值（估算）", f"{bf['market_cap']:.1f} 兆台幣")
                st.metric("台灣 GDP", f"{bf['gdp']:.1f} 兆台幣")
                st.metric("評分（0=貴,1=便宜）", f"{bf['score']:.2f}")
            with bf3:
                # 巴菲特指標進度條（0-200%）
                gauge_val = min(ratio / 200, 1.0)
                st.markdown("**估值水位**")
                st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;height:24px;overflow:hidden">
  <div style="background:{color_hex};width:{gauge_val*100:.0f}%;height:100%;transition:0.3s"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-top:2px">
  <span>0%（極低估）</span><span>100%（合理）</span><span>200%（嚴重高估）</span>
</div>""", unsafe_allow_html=True)
                st.caption(f"📚 {bf['historical_context']}")

            # 指標對照表
            with st.expander("📊 巴菲特指標判讀標準", expanded=False):
                st.markdown("""
| 指標比率 | 判斷 | 巴菲特原話 |
|---------|------|-----------|
| < 80% | 🟢 大幅低估，長線做多機會 | "股市非常便宜" |
| 80~100% | 🟡 小幅低估，中性偏多 | "合理估值偏低" |
| 100~120% | 🟡 合理區間 | "公平價值附近" |
| 120~150% | 🟠 偏高估，降低持倉 | "開始令人擔憂" |
| > 150% | 🔴 嚴重高估，極度謹慎 | "在玩火！" |

> 巴菲特曾說：「如果你需要判斷市場估值的單一最佳指標，那可能就是這個比率。」
> 台灣股市因外資占比高、出口導向，指標值通常比美國偏低。
""")

        # ── 三大法人五日多空指標 ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏦 三大法人五日多空指標")
        st.caption("以代理股票加總模擬全市場外資、投信、自營商流向（需 FinMind Token）")

        if not FINMIND_TOKEN:
            st.info("📌 設定 FinMind Token 後可顯示三大法人市場指標")
        else:
            with st.spinner("載入三大法人五日指標..."):
                try:
                    from macro.institutional_alert import compute_institutional_signals
                    inst_sig = compute_institutional_signals()
                    _inst_ok = inst_sig.get("available", False)
                except Exception as e:
                    st.warning(f"法人指標載入失敗：{e}")
                    _inst_ok = False

            if _inst_ok:
                combined = inst_sig["combined_signal"]
                combined_color = {"多頭": "#6bcb77", "空頭": "#ff4b4b", "分歧": "#ffd93d"}.get(combined, "#aaa")
                st.markdown(
                    f'<div style="background:#161b22;border-radius:8px;padding:8px 16px;'
                    f'display:inline-block;margin-bottom:12px">'
                    f'三大法人綜合方向：<b style="color:{combined_color};font-size:1.1rem">{combined}</b></div>',
                    unsafe_allow_html=True,
                )

                # 三欄：外資 / 投信 / 自營商
                ic1, ic2, ic3 = st.columns(3)
                for col, key, label in [(ic1, "fi", "外資"), (ic2, "it", "投信"), (ic3, "dealer", "自營商")]:
                    stats = inst_sig[key]
                    d = stats["direction"]
                    d_color = "#6bcb77" if d == "多" else ("#ff4b4b" if d == "空" else "#aaa")
                    net5  = stats["5d_net"]
                    net20 = stats["20d_net"]
                    cons  = stats["consecutive"]
                    z     = stats["z_score"]
                    with col:
                        st.markdown(
                            f'<div class="macro-card">'
                            f'<div style="font-size:1.1rem;font-weight:bold">{label}</div>'
                            f'<div style="color:{d_color};font-size:1.4rem;font-weight:bold">{d}方</div>'
                            f'<table style="width:100%;font-size:0.85rem;margin-top:6px">'
                            f'<tr><td style="color:#888">5日淨額</td><td style="text-align:right">'
                            f'<b style="color:{"#6bcb77" if net5>=0 else "#ff4b4b"}">{net5:+,}</b> 張</td></tr>'
                            f'<tr><td style="color:#888">20日淨額</td><td style="text-align:right">'
                            f'<b style="color:{"#6bcb77" if net20>=0 else "#ff4b4b"}">{net20:+,}</b> 張</td></tr>'
                            f'<tr><td style="color:#888">連續天數</td><td style="text-align:right">'
                            f'{"買超" if cons>0 else "賣超"} <b>{abs(cons)}</b> 天</td></tr>'
                            f'<tr><td style="color:#888">Z 分數</td><td style="text-align:right">'
                            f'<b style="color:{"#6bcb77" if z>1 else "#ff4b4b" if z<-1 else "#ffd93d"}">'
                            f'{z:+.1f}σ</b></td></tr>'
                            f'</table></div>',
                            unsafe_allow_html=True,
                        )

                # 異常警示
                alerts = inst_sig.get("alerts", [])
                if alerts:
                    st.markdown("#### 🚨 異常警示")
                    for alr in alerts:
                        level = alr.get("level", "⚪")
                        bg = {"🔴": "#2d1010", "🟢": "#0d2d1a", "🟠": "#2d1b00"}.get(level, "#1a1a2e")
                        border = {"🔴": "#ff4b4b", "🟢": "#00c087", "🟠": "#ff9f43"}.get(level, "#888")
                        st.markdown(
                            f'<div style="background:{bg};border-left:4px solid {border};'
                            f'padding:8px 14px;border-radius:6px;margin:4px 0">'
                            f'{level} <b>{alr["type"]}</b>（{alr["name"]}）　{alr["msg"]}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("✅ 近期無法人異常警示")

                # 歷史趨勢圖（外資5日累計）
                series_data = inst_sig.get("series", {})
                fi_series = series_data.get("fi", pd.Series(dtype=float))
                if not fi_series.empty and len(fi_series) >= 5:
                    st.markdown("---")
                    fig_fi = go.Figure()
                    colors_fi = ["#00c087" if v >= 0 else "#ff4b4b" for v in fi_series.values]
                    fig_fi.add_trace(go.Bar(x=fi_series.index, y=fi_series.values,
                                            marker_color=colors_fi, name="外資每日淨買賣超"))
                    fi_roll5 = fi_series.rolling(5, min_periods=1).sum()
                    fig_fi.add_trace(go.Scatter(x=fi_roll5.index, y=fi_roll5.values,
                                                name="5日累計", line=dict(color="#ffd93d", width=2)))
                    fig_fi.update_layout(template="plotly_dark",
                                         title="外資（代理股票）每日買賣超 vs 5日累計",
                                         height=280, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_fi, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# Tab 3：回測驗證
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📈 策略回測驗證")

    if bt_btn and bt_stock:
        if bt_sell >= bt_buy:
            st.error("❌ 賣出閾值必須低於買進閾值")
            st.stop()

        with st.spinner(f"正在對 {bt_stock} 執行回測（{bt_start} ~ {bt_end}）..."):
            try:
                from backtest.engine import run_backtest
                from backtest.visualizer import build_charts

                bt_result = run_backtest(
                    stock_id       = bt_stock,
                    start_date     = str(bt_start),
                    end_date       = str(bt_end),
                    buy_threshold  = bt_buy,
                    sell_threshold = bt_sell,
                    use_macro      = bt_macro,
                    initial_capital= float(bt_cap),
                )
                _bt_ok = True
            except Exception as e:
                st.error(f"❌ 回測失敗：{e}")
                _bt_ok = False

        if _bt_ok:
            metrics = bt_result.get("metrics", {})
            trades  = bt_result.get("trades", [])

            # ── 績效卡片 ──────────────────────────────────────────────────
            st.markdown("#### 績效摘要")
            r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
            r1c1.metric("策略總報酬",   f"{metrics.get('total_return',0):.1f}%")
            r1c2.metric("年化報酬",     f"{metrics.get('annual_return',0):.1f}%")
            r1c3.metric("最大回撤",     f"{metrics.get('max_drawdown',0):.1f}%")
            r1c4.metric("Sharpe",       f"{metrics.get('sharpe_ratio',0):.2f}")
            r1c5.metric("勝率",         f"{metrics.get('win_rate',0):.1f}%")
            r1c6.metric("vs Buy&Hold",  f"{metrics.get('excess_return',0):+.1f}%",
                        delta_color="normal" if metrics.get('excess_return',0) >= 0 else "inverse")

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            r2c1.metric("交易次數",     f"{metrics.get('trade_count',0)} 筆")
            r2c2.metric("獲利因子",     f"{metrics.get('profit_factor',0):.2f}")
            r2c3.metric("平均持有天數", f"{metrics.get('avg_holding_days',0):.1f} 天")
            r2c4.metric("Buy&Hold報酬", f"{metrics.get('bh_return',0):.1f}%")

            # ── 圖表 ─────────────────────────────────────────────────────
            charts = build_charts(bt_result, float(bt_cap))
            st.plotly_chart(charts["equity"],    use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(charts["score_dist"], use_container_width=True)
            with cc2:
                st.plotly_chart(charts["pnl"],        use_container_width=True)

            st.plotly_chart(charts["monthly"], use_container_width=True)

            # ── 交易明細 ─────────────────────────────────────────────────
            with st.expander("📋 交易明細", expanded=False):
                show_cols = ["date","action","price","shares","score","pnl","pnl_pct","holding_days"]
                tdf = pd.DataFrame([{c: t.get(c) for c in show_cols} for t in trades])
                tdf.columns = ["日期","方向","價格","股數","評分","損益(元)","損益%","持有天數"]
                st.dataframe(tdf, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.caption(
                "> ⚠️ **免責聲明**：回測結果僅供參考，過去績效不代表未來表現。"
                "本系統不構成任何投資建議，投資人應自行評估並承擔投資風險。"
            )

    else:
        st.info("👈 在左側設定回測參數後按「執行回測」，建議先用 `0050` 搭配 2 年期間測試。")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 4：因子說明
# ───────────────────────────────────────────────────────────────────────────────
def _factor_signal(key: str, val: float) -> tuple:
    """回傳 (顏色emoji, 解讀文字) 給定因子名稱與數值。"""
    def _v(n=0): return f"{val:,.{n}f}"
    if key == "fi_5d_net":
        if val > 5000:   return "🟢", f"近5日外資淨買超 **{_v()}** 張，主力積極布局"
        elif val > 0:    return "🟡", f"近5日外資淨買超 **{_v()}** 張，小幅流入"
        elif val > -5000:return "🟡", f"近5日外資淨賣超 **{_v()}** 張，小幅觀望"
        else:            return "🔴", f"近5日外資淨賣超 **{_v()}** 張，主力明顯撤退"
    if key == "fi_20d_net":
        if val > 20000:  return "🟢", f"近20日外資累計買超 **{_v()}** 張，中期看好"
        elif val > 0:    return "🟡", f"近20日外資小幅買超 **{_v()}** 張"
        else:            return "🔴", f"近20日外資累計賣超 **{_v()}** 張，中期偏空"
    if key == "fi_consecutive":
        n = int(val)
        if n > 5:    return "🟢", f"外資連續買超 **{n}** 天，趨勢強勁"
        elif n > 0:  return "🟡", f"外資連續買超 **{n}** 天"
        elif n < -5: return "🔴", f"外資連續賣超 **{abs(n)}** 天，趨勢偏空"
        elif n < 0:  return "🟡", f"外資連續賣超 **{abs(n)}** 天"
        else:        return "⚪", "外資昨日未有明顯買賣超"
    if key == "rsi_14":
        if val > 70:   return "🔴", f"RSI={_v(1)}，**超買區**（>70），短線注意拉回風險"
        elif val > 50: return "🟢", f"RSI={_v(1)}，健康多頭區間（50~70），動能充足"
        elif val > 30: return "🟡", f"RSI={_v(1)}，中性偏弱（30~50），等待反彈確認"
        else:          return "🟢", f"RSI={_v(1)}，**超賣區**（<30），潛在反彈機會"
    if key == "macd_histogram":
        if val > 0.5:  return "🟢", f"MACD柱狀值={_v(3)}，多方力道強勁且持續擴張"
        elif val > 0:  return "🟡", f"MACD柱狀值={_v(3)}，多方略佔優勢"
        elif val > -0.5:return "🟡", f"MACD柱狀值={_v(3)}，空方略佔優勢"
        else:          return "🔴", f"MACD柱狀值={_v(3)}，空方力道強勁"
    if key == "macd_cross":
        if val == 1:   return "🟢", "MACD 黃金交叉，中線買進訊號"
        elif val == -1:return "🔴", "MACD 死亡交叉，中線賣出訊號"
        else:          return "⚪", "MACD 無交叉訊號，維持現有趨勢"
    if key == "above_ma20":
        if val == 1:   return "🟢", "股價站上月線（MA20），多頭格局確立"
        else:          return "🔴", "股價跌破月線（MA20），短線走弱"
    if key == "above_ma60":
        if val == 1:   return "🟢", "股價站上季線（MA60），長線多頭"
        else:          return "🔴", "股價跌破季線（MA60），中長線偏空"
    if key == "ma_alignment":
        labels = {0:"均線空頭排列", 1:"僅1條均線多排", 2:"2條均線多排", 3:"完整多頭排列（最強）"}
        colors = {0:"🔴", 1:"🟡", 2:"🟡", 3:"🟢"}
        return colors[int(val)], f"多頭排列分數 **{int(val)}/3**：{labels[int(val)]}"
    if key == "rev_yoy":
        if val > 30:   return "🟢", f"月營收年增率 **+{_v(1)}%**，高速成長"
        elif val > 10: return "🟢", f"月營收年增率 **+{_v(1)}%**，穩健成長"
        elif val > 0:  return "🟡", f"月營收年增率 **+{_v(1)}%**，小幅成長"
        elif val > -10:return "🟡", f"月營收年增率 **{_v(1)}%**，略微衰退"
        else:          return "🔴", f"月營收年增率 **{_v(1)}%**，明顯衰退，需注意"
    if key == "eps_latest":
        if val > 5:    return "🟢", f"最近季EPS **{_v(2)} 元**，獲利優異"
        elif val > 2:  return "🟢", f"最近季EPS **{_v(2)} 元**，獲利良好"
        elif val > 0:  return "🟡", f"最近季EPS **{_v(2)} 元**，小幅獲利"
        elif val == 0: return "⚪", "最近季EPS 為 0（損益兩平）"
        else:          return "🔴", f"最近季EPS **{_v(2)} 元**，當季虧損"
    if key == "pe_ratio":
        if val <= 0:   return "🔴", "本益比為負（虧損股），估值無意義"
        elif val < 15: return "🟢", f"本益比 **{_v(1)} 倍**，估值偏低，具吸引力"
        elif val < 25: return "🟡", f"本益比 **{_v(1)} 倍**，估值合理（台股均值約15~25倍）"
        elif val < 40: return "🟡", f"本益比 **{_v(1)} 倍**，估值偏高"
        else:          return "🔴", f"本益比 **{_v(1)} 倍**，估值過高，需高成長支撐"
    if key == "gross_margin":
        if val > 50:   return "🟢", f"毛利率 **{_v(1)}%**，極強競爭護城河"
        elif val > 30: return "🟢", f"毛利率 **{_v(1)}%**，良好盈利能力"
        elif val > 15: return "🟡", f"毛利率 **{_v(1)}%**，中等水準"
        else:          return "🔴", f"毛利率 **{_v(1)}%**，偏低，需留意競爭壓力"
    if key == "ret_1m":
        if val > 10:   return "🟢", f"近月報酬 **+{_v(1)}%**，強勢股"
        elif val > 3:  return "🟢", f"近月報酬 **+{_v(1)}%**，穩健上漲"
        elif val > 0:  return "🟡", f"近月報酬 **+{_v(1)}%**，小漲整理"
        elif val > -5: return "🟡", f"近月報酬 **{_v(1)}%**，小幅回落"
        else:          return "🔴", f"近月報酬 **{_v(1)}%**，明顯下跌"
    if key == "vol_20d":
        if val < 20:   return "🟢", f"年化波動度 **{_v(1)}%**，低波動穩定型"
        elif val < 35: return "🟡", f"年化波動度 **{_v(1)}%**，一般水準"
        elif val < 50: return "🟡", f"年化波動度 **{_v(1)}%**，波動偏高"
        else:          return "🔴", f"年化波動度 **{_v(1)}%**，高波動，風險較大"
    if key == "high_52w_pct":
        if val > -5:   return "🟢", f"距52週高點 **{_v(1)}%**，接近歷史高點，強勢突破形態"
        elif val > -15:return "🟡", f"距52週高點 **{_v(1)}%**，中段整理"
        else:          return "🔴", f"距52週高點 **{_v(1)}%**，離高點較遠，動能偏弱"
    if key == "vol_ratio":
        if val > 2.0:  return "🟢", f"量比 **{_v(2)}**，今日大幅放量，市場高度關注"
        elif val > 1.2:return "🟡", f"量比 **{_v(2)}**，溫和放量"
        elif val > 0.8:return "⚪", f"量比 **{_v(2)}**，成交量正常"
        else:          return "🔴", f"量比 **{_v(2)}**，明顯縮量，市場觀望"
    if key == "bb_position":
        if val > 0.8:  return "🔴", f"布林通道位置 **{val:.0%}**，接近上軌，短線過熱"
        elif val > 0.5:return "🟢", f"布林通道位置 **{val:.0%}**，中段偏上，多頭強勢"
        elif val > 0.2:return "🟡", f"布林通道位置 **{val:.0%}**，中段偏下，偏弱整理"
        else:          return "🟢", f"布林通道位置 **{val:.0%}**，接近下軌，潛在超賣反彈"
    # 其他因子通用解讀
    return "⚪", f"數值：{_v(2)}"


with tab4:
    st.subheader("📚 因子說明與評分機制")
    st.markdown("本系統所有因子先正規化為 0~1 分，再依類別平均後加權合計，乘以 100 得出最終評分。")

    # ── 當前個股數值解讀 ─────────────────────────────────────────────────────
    if "last_raw" in st.session_state:
        raw_now = st.session_state["last_raw"]
        sid_now = st.session_state.get("last_stock_id", "")
        result_now = st.session_state.get("last_result", {})
        cat_now = result_now.get("category_scores", {})

        st.markdown(f"### 🔍 **{sid_now}** 當前數值解讀")
        st.caption("以下為最近一次個股分析的因子數值及其投資意義")

        show_factors = [
            ("籌碼面", "chips", [
                ("fi_5d_net","外資近5日買賣超"),("fi_consecutive","外資連續天數"),
                ("it_5d_net","投信近5日買賣超"),
            ]),
            ("技術面", "technical", [
                ("above_ma20","站上月線"),("above_ma60","站上季線"),
                ("ma_alignment","均線排列"),("rsi_14","RSI"),
                ("macd_histogram","MACD柱狀"),("macd_cross","MACD交叉"),
                ("bb_position","布林位置"),("vol_ratio","量比"),
            ]),
            ("基本面", "fundamental", [
                ("rev_yoy","月營收年增率"),("eps_latest","最近季EPS"),
                ("pe_ratio","本益比"),("gross_margin","毛利率"),
            ]),
            ("動能面", "momentum", [
                ("ret_1m","近月報酬"),("vol_20d","波動度"),("high_52w_pct","距高點%"),
            ]),
        ]

        for cat_label, cat_key, factors in show_factors:
            cat_score = cat_now.get(cat_key, 50)
            color = "#6bcb77" if cat_score >= 65 else "#ffd93d" if cat_score >= 45 else "#ff6b6b"
            st.markdown(f"**{cat_label}**　<span style='color:{color};font-size:1.1rem'>{cat_score:.1f} 分</span>",
                        unsafe_allow_html=True)
            rows = []
            for fkey, fname in factors:
                if fkey in raw_now:
                    val = raw_now[fkey]
                    icon, interp = _factor_signal(fkey, float(val))
                    rows.append({"因子": fname, "數值": f"{val:,.2f}" if isinstance(val, float) else val,
                                 "解讀": f"{icon} {interp}"})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                             column_config={"解讀": st.column_config.TextColumn(width="large")})
        st.markdown("---")
    else:
        st.info("💡 請先在「個股分析」頁籤執行分析，這裡就會顯示當前各因子數值與解讀。")
        st.markdown("---")

    # ── 評分門檻 ─────────────────────────────────────────────────────────────
    st.markdown("### 🎯 評分與投資建議對照")
    st.markdown("""
| 評分範圍 | 建議 | 說明 |
|----------|------|------|
| 80 ~ 100 | ⭐⭐ 強力買進 | 多因子強烈共振，高確信度 |
| 65 ~ 79  | ⭐ 買進      | 多數因子偏多，可考慮進場 |
| 45 ~ 64  | ◆ 持有觀望  | 訊號中性，持倉待訊號確認 |
| 30 ~ 44  | ▼ 減碼      | 因子偏空，宜降低持倉 |
| 0 ~ 29   | ✕ 賣出      | 多因子共同示警，建議出場 |
""")

    # ── 權重說明 ─────────────────────────────────────────────────────────────
    st.markdown("### ⚖️ 預設因子權重")
    st.markdown("""
| 面向 | 權重 | 核心邏輯 |
|------|------|---------|
| 籌碼面 | **30%** | 法人動向是最直接的主力意圖指標 |
| 基本面 | **25%** | 企業獲利成長是股價長期支撐 |
| 技術面 | **20%** | 價量結構反映市場多空力道 |
| 動能面 | **15%** | 趨勢持續性與加速度 |
| 風險面 | **10%** | 以波動度衡量投資風險 |
""")

    # ── 籌碼面 ───────────────────────────────────────────────────────────────
    with st.expander("🔵 籌碼面因子（權重 30%）", expanded=True):
        st.markdown("""
| 因子代碼 | 名稱 | 計算方式 | 正面訊號 |
|----------|------|----------|---------|
| `fi_5d_net` | 外資近5日買賣超 | 外資近5交易日買進-賣出加總（張） | 正值 = 外資持續淨買入 |
| `fi_20d_net` | 外資近20日買賣超 | 外資近20日累計淨買賣超 | 大正值 = 外資中期布局 |
| `fi_consecutive` | 外資連續買賣超天數 | 正數=連買天數，負數=連賣天數 | 連買天數越多越強 |
| `fi_trend` | 外資10日趨勢斜率 | 外資日買賣超的線性迴歸斜率 | 正斜率 = 外資買超加速 |
| `it_5d_net` | 投信近5日買賣超 | 投信近5日淨買賣超（張） | 投信持續買入代表基金建倉 |
| `it_20d_net` | 投信近20日買賣超 | 投信近20日累計 | 中期基金資金流向 |
| `it_consecutive` | 投信連續天數 | 同外資計算邏輯 | 投信連買代表長線看好 |
| `dealer_5d_net` | 自營商近5日買賣超 | 自營商近5日淨買賣超（張） | 自營商通常短線操作 |
| `margin_chg_5d` | 融資5日變化率 | (今日餘額-5日前餘額)/5日前餘額 | 負值（融資減少）為正面：散戶去槓桿 |
| `short_chg_5d` | 融券5日變化率 | 同上，以融券計算 | 負值（融券減少）為正面：空方回補 |
""")
        st.info("📌 **投資意義**：外資是台股最大機構投資人，持續買超通常代表外資看好該股中長期前景。"
                "投信買超則常見於業績股或籌碼整理完畢的個股。融資增加需注意散戶過熱風險。")

    # ── 技術面 ───────────────────────────────────────────────────────────────
    with st.expander("📈 技術面因子（權重 20%）", expanded=False):
        st.markdown("""
| 因子代碼 | 名稱 | 計算方式 | 正面訊號 |
|----------|------|----------|---------|
| `above_ma5` | 站上MA5 | 收盤 > 5日均線 → +1，否則 -1 | +1（收盤在均線上方） |
| `above_ma20` | 站上MA20 | 收盤 > 20日均線 → +1，否則 -1 | +1（月線支撐確立） |
| `above_ma60` | 站上MA60 | 收盤 > 60日均線 → +1，否則 -1 | +1（季線多頭格局） |
| `ma_alignment` | 多頭排列分數 | MA5>MA10>MA20>MA60 各算1分，0~3分 | 3分 = 完整多頭排列 |
| `ma20_deviation` | 距MA20偏離% | (收盤-MA20)/MA20×100 | 正值代表站上月線，過大需注意超漲 |
| `rsi_14` | RSI(14) | 相對強弱指標，0~100 | 50~70 為健康多頭區間 |
| `rsi_signal` | RSI轉折訊號 | RSI<30超賣=+1，RSI>70超買=-1，其他=0 | +1（超賣反彈訊號） |
| `macd_histogram` | MACD柱狀值 | MACD線 - 訊號線 | 正值且擴大 = 多方動能增強 |
| `macd_cross` | MACD交叉 | 柱狀值由負轉正=黃金交叉(+1)，反之=-1 | +1（黃金交叉買點） |
| `bb_position` | 布林通道位置 | (收盤-下軌)/(上軌-下軌)，0~1 | 0.5~0.8 為多頭強勢區 |
| `vol_ratio` | 量比 | 今日成交量/20日均量 | >1.5 為放量，配合上漲更佳 |
| `vol_trend` | 量能趨勢 | 5日均量>20日均量→+1，否則→-1 | +1（量能擴張） |
""")
        st.info("📌 **投資意義**：技術面反映市場供需與投資人情緒。均線多頭排列代表趨勢健康，"
                "MACD黃金交叉提供買進時機參考，布林通道幫助判斷價格相對位置，"
                "量能配合價格上漲才是真正的強勢訊號。")

    # ── 基本面 ───────────────────────────────────────────────────────────────
    with st.expander("💰 基本面因子（權重 25%）", expanded=False):
        st.markdown("""
| 因子代碼 | 名稱 | 計算方式 | 正面訊號 |
|----------|------|----------|---------|
| `rev_yoy` | 月營收年增率 | (本月營收-去年同月)/去年同月×100% | 高正值 = 業績高速成長 |
| `rev_mom` | 月營收月增率 | (本月-上月)/上月×100% | 正值 = 業績環比改善 |
| `rev_3m_trend` | 近3月營收趨勢 | 連3月成長→+1，連3月下滑→-1 | +1（成長趨勢確立） |
| `rev_12m_high` | 創12月營收新高 | 本月營收超越近12月最高→1，否則→0 | 1（業績創新高動能強） |
| `eps_latest` | 最近季EPS | 最新一季每股盈餘（元） | 越高代表獲利越好 |
| `eps_qoq` | EPS季增 | 本季EPS - 上季EPS | 正值 = 季度環比改善 |
| `eps_yoy` | EPS年增 | 本季EPS - 去年同季EPS | 正值 = 年度成長 |
| `gross_margin` | 毛利率 | 毛利/營收×100% | 越高代表產品競爭力越強 |
| `gpm_trend` | 毛利率季變化 | 本季毛利率 - 上季（pp） | 正值 = 毛利率擴張 |
| `pe_ratio` | 本益比 | 股價/(近四季EPS加總) | 越低代表估值越便宜 |
""")
        st.info("📌 **投資意義**：基本面代表企業的真實獲利能力。"
                "月營收年增率高於20%通常為高成長股特徵；毛利率持續擴張代表產品定價能力提升；"
                "本益比低於15倍通常被視為低估值，但需注意產業特性差異。")

    # ── 動能面 ───────────────────────────────────────────────────────────────
    with st.expander("⚡ 動能面因子（權重 15%）", expanded=False):
        st.markdown("""
| 因子代碼 | 名稱 | 計算方式 | 正面訊號 |
|----------|------|----------|---------|
| `ret_5d` | 近5日報酬率 | (今日收盤/5日前收盤-1)×100% | 正值且適度（3~10%）最佳 |
| `ret_1m` | 近20日報酬率 | (今日收盤/20日前收盤-1)×100% | 正值代表月線上漲動能 |
| `ret_3m` | 近60日報酬率 | (今日收盤/60日前收盤-1)×100% | 正值代表季線趨勢向上 |
| `high_52w_pct` | 距52週高點% | (收盤/52週最高-1)×100%（負值） | 越接近0代表越靠近高點（強勢） |
| `momentum_accel` | 動能加速度 | 近5日報酬 - 近20日報酬 | 正值 = 短期動能加速 |
""")
        st.info("📌 **投資意義**：動能因子反映股票的相對強勢程度。"
                "研究顯示，過去3~12個月強勢的股票往往在未來3~6個月繼續跑贏大盤（動能效應）。"
                "接近52週高點通常代表股票正處於強勢突破階段。")

    # ── 風險面 ───────────────────────────────────────────────────────────────
    with st.expander("🛡️ 風險面因子（權重 10%）", expanded=False):
        st.markdown("""
| 因子代碼 | 名稱 | 計算方式 | 正面訊號 |
|----------|------|----------|---------|
| `vol_20d` | 近20日年化波動度 | 20日日報酬標準差×√252×100% | 越低越好（低波動 = 高風險分數） |
""")
        st.info("📌 **投資意義**：年化波動度衡量股價的波動程度。"
                "波動度低於20%通常為穩定型股票；20~40%為一般；超過40%則屬高波動股。"
                "在相同報酬率下，波動越低代表風險調整後報酬（Sharpe Ratio）越佳。")

    # ── 總體資金面 ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 總體資金面因子（個股評分乘數）")
    st.markdown("""
> 總體資金面不直接計入個股評分，而是作為**乘數**調整最終分數：
> `最終評分 = 個股評分 × (0.7 + 0.3 × macro_score)`
> - macro_score = 0 時：乘數 = 0.7（最多降低 30 分）
> - macro_score = 1 時：乘數 = 1.0（不加乘，避免過度推薦）
""")

    with st.expander("🌐 總體資金面因子詳細說明", expanded=False):
        st.markdown("""
**匯率面（25%）**

| 指標 | 說明 | 正面訊號 |
|------|------|---------|
| `twd_5d_chg` | 台幣5日升貶幅% | 正值（台幣升值） = 外資匯入 |
| `twd_20d_chg` | 台幣20日升貶幅% | 正值代表中期台幣走強 |
| `twd_trend` | 台幣10日趨勢 | 升值趨勢 = 外資持續流入 |
| `twd_vs_ma20` | 台幣距MA20偏離 | 正值（台幣強於均線） |

---

**VIX恐慌指數面（25%）**

| 指標 | 說明 | 得分 |
|------|------|------|
| VIX < 15 | 市場平靜 | 1.00（滿分） |
| 15 ≤ VIX < 20 | 輕微波動 | 0.75 |
| 20 ≤ VIX < 25 | 中度緊張 | 0.50 |
| 25 ≤ VIX < 30 | 明顯恐慌 | 0.25 |
| VIX ≥ 30 | 極度恐慌 | 0.00（最差） |

---

**期貨面（30%）**

| 指標 | 說明 | 正面訊號 |
|------|------|---------|
| `fi_future_net` | 外資台指期淨多單（口） | 正值（外資持多） = 看漲台股 |
| `fi_future_5d_chg` | 近5日淨多單變化 | 正值 = 外資增加多單 |
| `fi_future_trend` | 10日趨勢斜率 | 正斜率 = 外資持續加多 |

> 外資台指期淨多單 > 5萬口：極度看多（滿分）

---

**資金流向面（20%）**

| 指標 | 說明 | 正面訊號 |
|------|------|---------|
| `fi_total_5d` | 外資近5日全市場買賣超 | 正值（整體外資淨買入） |
| `fi_consecutive_days` | 外資連續買超天數 | 天數越多代表趨勢越強 |
| `it_total_5d` | 投信近5日買賣超 | 投信買入代表本土機構看好 |
""")

    # ── 標準化方法 ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔧 因子標準化方法")
    st.markdown("""
所有原始因子在計入評分前，會先透過以下方式轉換為 **0~1 分**：

| 方法 | 適用因子 | 說明 |
|------|---------|------|
| **Sigmoid 函數** | 連續性因子（買賣超、報酬率等） | 以0為中心，正值趨近1，負值趨近0 |
| **線性對映** | 二元因子（above_ma5 = ±1） | (-1, 0, +1) → (0, 0.5, 1) |
| **反向Sigmoid** | 越小越好的因子（PE、波動度） | 低值得高分 |
| **固定映射** | VIX分段評分 | 依區間直接對應分數 |
""")
    st.caption("💡 設計目標：讓每個因子的量綱統一，避免數值較大的因子主導評分結果。")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 5：風險監控
# ───────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🛡️ 市場風險監控儀表板")
    st.caption(f"更新時間：{_now_tw().strftime('%Y-%m-%d %H:%M')} (台灣時間)")

    # ── 巴菲特指標 ────────────────────────────────────────────────
    st.markdown("### 📐 台灣巴菲特指標（大盤估值）")
    with st.spinner("載入巴菲特指標..."):
        try:
            from macro.buffett import compute_buffett
            bf5 = compute_buffett(FINMIND_TOKEN)
            _bf5_ok = True
        except Exception as e:
            st.warning(f"巴菲特指標失敗：{e}")
            _bf5_ok = False

    if _bf5_ok:
        bfa, bfb, bfc = st.columns(3)
        ratio5 = bf5["ratio"]
        color5_hex = {"🟢": "#6bcb77", "🟡": "#ffd93d", "🟠": "#ff9f43", "🔴": "#ff4b4b"}.get(bf5["color"], "#aaa")
        with bfa:
            st.markdown(
                f'<div style="text-align:center;background:#161b22;border-radius:10px;padding:18px">'
                f'<div style="font-size:2.8rem;font-weight:bold;color:{color5_hex}">{ratio5:.1f}%</div>'
                f'<div style="color:#aaa;margin-top:4px">市值/GDP</div>'
                f'<div style="margin-top:8px;color:{color5_hex}">{bf5["signal"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with bfb:
            st.metric("台股總市值（估算）", f"{bf5['market_cap']:.1f} 兆台幣")
            st.metric("台灣 GDP",           f"{bf5['gdp']:.1f} 兆台幣")
        with bfc:
            st.metric("巴菲特評分（0=貴, 1=便宜）", f"{bf5['score']:.2f}")
            st.caption(bf5["historical_context"])

    # ── 三大法人市場警示 ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏦 三大法人五日多空指標與警示")

    if not FINMIND_TOKEN:
        st.info("📌 設定 FinMind Token 後可顯示三大法人市場指標")
    else:
        with st.spinner("載入三大法人數據..."):
            try:
                from macro.institutional_alert import compute_institutional_signals
                inst5 = compute_institutional_signals()
                _inst5_ok = inst5.get("available", False)
            except Exception as e:
                st.warning(f"法人指標失敗：{e}")
                _inst5_ok = False

        if _inst5_ok:
            combined5 = inst5["combined_signal"]
            c5_color = {"多頭": "#6bcb77", "空頭": "#ff4b4b", "分歧": "#ffd93d"}.get(combined5, "#aaa")
            st.markdown(
                f'<div style="background:#161b22;border-radius:8px;padding:10px 16px;'
                f'margin-bottom:12px;display:inline-block">'
                f'三大法人綜合方向：<b style="color:{c5_color};font-size:1.2rem">{combined5}</b></div>',
                unsafe_allow_html=True,
            )
            t5c1, t5c2, t5c3 = st.columns(3)
            for col5, key5, lbl5 in [(t5c1, "fi", "外資"), (t5c2, "it", "投信"), (t5c3, "dealer", "自營商")]:
                s5 = inst5[key5]
                d5 = s5["direction"]
                dc5 = "#6bcb77" if d5 == "多" else ("#ff4b4b" if d5 == "空" else "#aaa")
                with col5:
                    st.markdown(
                        f'<div class="macro-card">'
                        f'<b>{lbl5}</b> <span style="color:{dc5};font-size:1.2rem">{d5}方</span><br>'
                        f'5日淨額 <b style="color:{"#6bcb77" if s5["5d_net"]>=0 else "#ff4b4b"}">'
                        f'{s5["5d_net"]:+,}</b> 張<br>'
                        f'20日淨額 <b style="color:{"#6bcb77" if s5["20d_net"]>=0 else "#ff4b4b"}">'
                        f'{s5["20d_net"]:+,}</b> 張<br>'
                        f'連續 <b>{abs(s5["consecutive"])}</b> 天'
                        f'{"買超" if s5["consecutive"]>0 else "賣超" if s5["consecutive"]<0 else "持平"}<br>'
                        f'Z分數 <b style="color:{"#6bcb77" if s5["z_score"]>1 else "#ff4b4b" if s5["z_score"]<-1 else "#ffd93d"}">'
                        f'{s5["z_score"]:+.1f}σ</b></div>',
                        unsafe_allow_html=True,
                    )

            alerts5 = inst5.get("alerts", [])
            if alerts5:
                st.markdown("#### 🚨 即時異常警示")
                for alr5 in alerts5:
                    lv5 = alr5.get("level", "⚪")
                    bg5 = {"🔴": "#2d1010", "🟢": "#0d2d1a", "🟠": "#2d1b00"}.get(lv5, "#1a1a2e")
                    bd5 = {"🔴": "#ff4b4b", "🟢": "#00c087", "🟠": "#ff9f43"}.get(lv5, "#888")
                    st.markdown(
                        f'<div style="background:{bg5};border-left:4px solid {bd5};'
                        f'padding:8px 14px;border-radius:6px;margin:4px 0">'
                        f'{lv5} <b>{alr5["type"]}</b>（{alr5["name"]}）　{alr5["msg"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ 目前無三大法人異常警示")

    # ── 全市場跌幅統計分析 ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 大盤歷史跌幅風險相關分析")
    st.caption("以台灣 50 ETF（0050）作為市場代理，分析歷史大跌的相關因子")

    if st.button("🔍 執行大盤風險相關分析", key="mkt_risk_btn"):
        with st.spinner("下載 0050 歷史資料並計算相關係數..."):
            try:
                from data.fetcher import FinMindFetcher
                from utils.risk_correlation import compute_risk_correlations
                _fetcher = FinMindFetcher("0050", days=365)
                _mkt_price = _fetcher.get_price()
                _mkt_inst  = _fetcher.get_institutional()
                mkt_risk = compute_risk_correlations(
                    _mkt_price, _mkt_inst, drop_threshold=-3.0, lookback_days=365
                )
                _mkt_ok = True
            except Exception as e:
                st.error(f"大盤風險分析失敗：{e}")
                _mkt_ok = False

        if _mkt_ok:
            if mkt_risk.get("message"):
                st.warning(mkt_risk["message"])
            else:
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("大跌次數（≥3%）", f"{mkt_risk['drop_count']} 次")
                rm2.metric("平均跌幅", f"{mkt_risk['avg_drop']:.1f}%")
                rm3.metric("最大單日跌幅", f"{mkt_risk['max_drop']:.1f}%")
                rm4.metric("當前風險分數", f"{mkt_risk['risk_score']:.0f}/100",
                            delta=mkt_risk["risk_level"])

                if mkt_risk["top_risk_factors"]:
                    st.markdown("**⚠️ 歷史大跌時最相關的風險因子（統計識別）**")
                    risk_tbl = [
                        {
                            "風險指標": r["factor_label"],
                            "相關係數": f"{r['correlation']:.3f}",
                            "統計意義": r["interpretation"],
                        }
                        for r in mkt_risk["top_risk_factors"]
                    ]
                    st.dataframe(pd.DataFrame(risk_tbl), use_container_width=True, hide_index=True)

                if mkt_risk["correlations"]:
                    mkt_corr_df = pd.DataFrame([
                        {"指標": r["factor_label"], "r": r["correlation"]}
                        for r in mkt_risk["correlations"]
                    ])
                    mkt_colors = ["#ff4b4b" if v < -0.3 else "#ffd93d" if v < 0 else "#6bcb77"
                                  for v in mkt_corr_df["r"]]
                    fig_mkt = go.Figure(go.Bar(
                        x=mkt_corr_df["r"], y=mkt_corr_df["指標"], orientation="h",
                        marker_color=mkt_colors,
                        text=[f"{v:.3f}" for v in mkt_corr_df["r"]], textposition="outside",
                    ))
                    fig_mkt.update_layout(
                        template="plotly_dark",
                        title="0050 各指標與單日跌幅的相關係數（負值代表跌跌相關，為風險因子）",
                        xaxis=dict(range=[-1.1, 1.1]),
                        height=400, margin=dict(l=10, r=60, t=40, b=10),
                    )
                    st.plotly_chart(fig_mkt, use_container_width=True)

                # 大跌事件列表
                with st.expander("📋 歷史大跌事件明細", expanded=False):
                    evts = mkt_risk["drop_events"]
                    if evts:
                        ev_df = pd.DataFrame(evts)
                        ev_df.columns = ["日期", "跌幅(%)"]
                        ev_df["跌幅(%)"] = ev_df["跌幅(%)"].round(2)
                        ev_df = ev_df.sort_values("跌幅(%)")
                        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 點擊按鈕載入大盤歷史風險相關分析")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 6：Forward EPS & 目標價
# ───────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("🎯 Forward EPS 推估與目標價分析")
    st.caption("以 TTM EPS × 營收成長率 × 毛利趨勢修正，推估未來 12 個月 EPS 與三情境目標價")

    if not FINMIND_TOKEN:
        st.warning("📌 需要 FinMind Token 才能取得財報資料，請在 .env 或 Streamlit Secrets 中設定 FINMIND_TOKEN")
    else:
        eps_stock = st.text_input(
            "分析股票代號", value=stock_id or "2330", max_chars=10, key="eps_stock_input"
        ).strip()
        eps_btn = st.button("📊 計算 Forward EPS", type="primary", key="eps_calc_btn")

        if eps_btn and eps_stock:
            with st.spinner(f"正在計算 {eps_stock} Forward EPS..."):
                try:
                    from data.fetcher import DataFetcher
                    from factors.forward_eps import ForwardEPSCalculator

                    df = DataFetcher()
                    calc = ForwardEPSCalculator(df)
                    eps_result = calc.calculate(eps_stock)
                    _eps_ok = True
                except Exception as e:
                    st.error(f"計算失敗：{e}")
                    _eps_ok = False

            if _eps_ok:
                if eps_result.get("error"):
                    st.error(f"❌ {eps_result['error']}")
                else:
                    # ── EPS 卡片 ────────────────────────────────────────────
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    ttm = eps_result.get("ttm_eps", 0) or 0
                    fwd = eps_result.get("forward_eps_1y", 0) or 0
                    growth_pct = (eps_result.get("eps_growth_rate") or 0) * 100
                    conf = eps_result.get("confidence", "low")
                    conf_color = {"high": "#6bcb77", "medium": "#ffd93d", "low": "#ff6b6b"}.get(conf, "#aaa")
                    upside = eps_result.get("upside_pct")
                    cur_price = eps_result.get("current_price", 0) or 0

                    ec1.metric("TTM EPS（近四季）", f"{ttm:.2f} 元")
                    ec2.metric(
                        "Forward EPS（1年推估）",
                        f"{fwd:.2f} 元",
                        delta=f"{growth_pct:+.1f}% 成長",
                        delta_color="normal" if growth_pct >= 0 else "inverse",
                    )
                    ec3.metric("當前股價", f"NT${cur_price:,.1f}")
                    if upside is not None:
                        ec4.metric(
                            "基準目標漲幅",
                            f"{upside:+.1f}%",
                            delta_color="normal" if upside >= 0 else "inverse",
                        )

                    # 信心度標籤
                    st.markdown(
                        f'<div style="margin:8px 0;display:inline-block;padding:4px 12px;'
                        f'border-radius:20px;background:#161b22;border:1px solid {conf_color}">'
                        f'推估信心度：<b style="color:{conf_color}">{conf.upper()}</b>'
                        f'　{eps_result.get("confidence_reason", "")}</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 三情境目標價長條圖 ────────────────────────────────
                    st.markdown("---")
                    tp = eps_result.get("target_price", {})
                    bull_tp = tp.get("bull") or 0
                    base_tp = tp.get("base") or 0
                    bear_tp = tp.get("bear") or 0

                    if any([bull_tp, base_tp, bear_tp]):
                        fig_tp = go.Figure()
                        for name, val, color in [
                            ("牛市目標", bull_tp, "#6bcb77"),
                            ("基準目標", base_tp, "#00d4ff"),
                            ("熊市目標", bear_tp, "#ff6b6b"),
                            ("當前股價", cur_price, "#ffd93d"),
                        ]:
                            if val:
                                fig_tp.add_trace(go.Bar(
                                    name=name, x=[name], y=[val],
                                    marker_color=color,
                                    text=[f"NT${val:,.1f}"], textposition="outside",
                                ))
                        fig_tp.update_layout(
                            template="plotly_dark",
                            title=f"{eps_stock} 三情境目標價 vs 當前股價",
                            yaxis_title="股價（元）",
                            height=350, showlegend=False,
                        )
                        st.plotly_chart(fig_tp, use_container_width=True)

                    # ── P/E 歷史分位數 ────────────────────────────────────
                    pe_band = eps_result.get("pe_band", {})
                    if pe_band.get("p25") and pe_band.get("p75"):
                        st.markdown("#### 📊 本益比歷史分位數")
                        pb1, pb2, pb3, pb4 = st.columns(4)
                        pb1.metric("P/E 25 分位", f"{pe_band['p25']:.1f}x")
                        pb2.metric("P/E 中位數", f"{pe_band['median']:.1f}x")
                        pb3.metric("P/E 75 分位", f"{pe_band['p75']:.1f}x")
                        if pe_band.get("current"):
                            pb4.metric("當前 P/E", f"{pe_band['current']:.1f}x")

                    # ── PEG Ratio ─────────────────────────────────────────
                    peg = eps_result.get("peg_ratio")
                    if peg is not None:
                        st.markdown("---")
                        peg_color = "#6bcb77" if peg < 1.0 else "#ffd93d" if peg < 2.0 else "#ff6b6b"
                        peg_label = "合理偏低" if peg < 1.0 else "合理" if peg < 2.0 else "偏高"
                        st.markdown(
                            f'<div style="background:#161b22;border-radius:10px;padding:12px 18px;display:inline-block">'
                            f'PEG Ratio：<b style="font-size:1.6rem;color:{peg_color}">{peg:.2f}</b>'
                            f'　<span style="color:{peg_color}">（{peg_label}）</span><br>'
                            f'<span style="color:#888;font-size:0.85rem">PEG = 本益比 ÷ 成長率%，<1 通常視為合理/低估</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        elif not eps_btn:
            st.info("👈 輸入股票代號並按「計算 Forward EPS」開始分析")
            st.markdown("""
**計算方法說明：**
1. **TTM EPS** = 最近四季 EPS 加總
2. **成長率** = 近 6 個月月營收 YoY 平均
3. **毛利趨勢修正** = 毛利率偏離均值的比例（±5% 上限）
4. **Forward EPS** = TTM EPS × (1 + 調整後成長率)
5. **目標價** = Forward EPS × 歷史 P/E 分位數
""")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 7：產業鏈分析
# ───────────────────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("🔗 產業鏈 Lead-Lag 景氣信號分析")
    st.caption("根據上中下游月營收 YoY 與外資籌碼，推估資金流向與 Lead-Lag 效應")

    from factors.supply_chain import SUPPLY_CHAIN_MAP

    chain_options = {
        "semiconductor": "🔬 半導體產業鏈",
        "ai_server": "🤖 AI 伺服器供應鏈",
        "ev_components": "⚡ 電動車零組件",
    }
    selected_chain = st.selectbox(
        "選擇產業鏈",
        options=list(chain_options.keys()),
        format_func=lambda x: chain_options[x],
        key="chain_select",
    )

    chain_btn = st.button("📡 分析產業鏈信號", type="primary", key="chain_analyze_btn")

    if chain_btn:
        if not FINMIND_TOKEN:
            st.warning("📌 需要 FinMind Token 才能取得籌碼資料")
        else:
            with st.spinner(f"正在分析 {chain_options[selected_chain]}..."):
                try:
                    from data.fetcher import DataFetcher
                    from factors.supply_chain import SupplyChainAnalyzer

                    chain_fetcher = DataFetcher()
                    chain_analyzer = SupplyChainAnalyzer(chain_fetcher)
                    chain_result = chain_analyzer.analyze_chain(selected_chain)
                    _chain_ok = True
                except Exception as e:
                    st.error(f"分析失敗：{e}")
                    _chain_ok = False

            if _chain_ok:
                overall_sig = chain_result["overall_signal"]
                sig_label = chain_result["signal_label"]
                tier_sigs = chain_result["tier_signals"]
                lead_lag = chain_result["lead_lag_months"]

                # ── 整體信號卡 ────────────────────────────────────────────
                sig_color = "#6bcb77" if overall_sig > 0.3 else "#ff6b6b" if overall_sig < -0.3 else "#ffd93d"
                st.markdown(
                    f'<div style="background:#161b22;border-radius:12px;padding:16px 24px;margin:8px 0">'
                    f'<div style="font-size:0.9rem;color:#888">{chain_result["chain_name"]} 整體信號</div>'
                    f'<div style="font-size:2.4rem;font-weight:bold;color:{sig_color}">{overall_sig:+.2f}</div>'
                    f'<div style="font-size:1.1rem;color:{sig_color}">{sig_label}</div>'
                    f'<div style="font-size:0.8rem;color:#666;margin-top:6px">Lead-Lag：上游領先下游約 {lead_lag} 個月</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── 三層信號儀表板 ────────────────────────────────────────
                st.markdown("#### 上中下游信號")
                tc1, tc2, tc3 = st.columns(3)
                chain_info = SUPPLY_CHAIN_MAP[selected_chain]
                tier_label_map = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}

                for col, tier_key in zip([tc1, tc2, tc3], ["upstream", "midstream", "downstream"]):
                    tier_sig = tier_sigs.get(tier_key, 0.0)
                    tier_color = "#6bcb77" if tier_sig > 0.2 else "#ff6b6b" if tier_sig < -0.2 else "#aaa"
                    tier_label = tier_label_map[tier_key]
                    tier_desc = chain_info["tiers"][tier_key]["description"]
                    tier_stocks = ", ".join(chain_info["tiers"][tier_key]["stocks"][:3])

                    with col:
                        st.markdown(
                            f'<div style="background:#0d1117;border:1px solid {tier_color};'
                            f'border-radius:10px;padding:14px;text-align:center">'
                            f'<div style="font-size:1rem;font-weight:bold;color:#ddd">{tier_label}</div>'
                            f'<div style="font-size:0.8rem;color:#888;margin-bottom:8px">{tier_desc}</div>'
                            f'<div style="font-size:2rem;font-weight:bold;color:{tier_color}">{tier_sig:+.2f}</div>'
                            f'<div style="font-size:0.75rem;color:#666;margin-top:6px">代表股：{tier_stocks}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── 資金流向推論 ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("#### 💡 資金流向推論")
                flow_text = chain_result.get("capital_flow_direction", "")
                if flow_text:
                    bg = "#0d2d1a" if overall_sig > 0.2 else "#2d1010" if overall_sig < -0.2 else "#1a1a2e"
                    border = "#00c087" if overall_sig > 0.2 else "#ff4b4b" if overall_sig < -0.2 else "#888"
                    st.markdown(
                        f'<div style="background:{bg};border-left:4px solid {border};'
                        f'padding:12px 16px;border-radius:6px;line-height:1.8">{flow_text}</div>',
                        unsafe_allow_html=True,
                    )

                # ── 建議關注個股 ──────────────────────────────────────────
                watch_stocks = chain_result.get("top_stocks_to_watch", [])
                if watch_stocks:
                    st.markdown("---")
                    st.markdown("#### 🔍 建議關注個股（尚未充分反映上游信號）")
                    st.markdown(
                        "、".join(f"`{s}`" for s in watch_stocks)
                        + "　（上游信號強但下游尚未跟進，存在 Lead-Lag 卡位機會）"
                    )

    else:
        st.info("👆 選擇產業鏈後按「分析產業鏈信號」")
        for chain_key, chain_info in SUPPLY_CHAIN_MAP.items():
            with st.expander(f"{chain_options.get(chain_key, chain_key)} 追蹤標的", expanded=False):
                rows = []
                for tier_key, tier_info in chain_info["tiers"].items():
                    tier_desc_map = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}
                    rows.append({
                        "層級": tier_desc_map.get(tier_key, tier_key),
                        "描述": tier_info["description"],
                        "追蹤個股": ", ".join(tier_info["stocks"]),
                        "權重": f"{tier_info['weight'] * 100:.0f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────────────
# Tab 8：月營收追蹤
# ───────────────────────────────────────────────────────────────────────────────
with tab8:
    st.subheader("📅 月營收追蹤與公告提醒")
    st.caption("追蹤個股月營收公告時程，自動偵測新公告，依法規每月 10 日前公布")

    if not FINMIND_TOKEN:
        st.warning("📌 需要 FinMind Token 才能取得月營收資料")
    else:
        try:
            from data.fetcher import DataFetcher
            from alerts.revenue_calendar import RevenueCalendar

            _rc_fetcher = DataFetcher()
            _rc = RevenueCalendar(_rc_fetcher)
            _rc_ok = True
        except Exception as e:
            st.error(f"月營收追蹤模組載入失敗：{e}")
            _rc_ok = False

        if _rc_ok:
            # ── 追蹤清單管理 ─────────────────────────────────────────────
            with st.expander("⚙️ 追蹤清單管理", expanded=False):
                add_col, del_col = st.columns(2)
                with add_col:
                    new_stock_id = st.text_input("加入追蹤", placeholder="股票代號（如 2330）", key="rc_add")
                    new_stock_name = st.text_input("股票名稱（選填）", placeholder="如 台積電", key="rc_name")
                    if st.button("➕ 加入", key="rc_add_btn"):
                        if new_stock_id:
                            _rc.add_to_watchlist(new_stock_id.strip(), new_stock_name.strip())
                            st.success(f"已加入：{new_stock_id}")
                            st.rerun()

                watchlist = _rc.get_watchlist()
                if watchlist:
                    st.markdown(f"**目前追蹤 {len(watchlist)} 檔個股**")
                    wl_df = pd.DataFrame(watchlist)
                    st.dataframe(wl_df, use_container_width=True, hide_index=True)
                else:
                    st.info("追蹤清單為空，請加入個股")

            # ── 即將公布月營收 ────────────────────────────────────────────
            st.markdown("### 📆 本週即將公布月營收")
            with st.spinner("預測即將公布個股..."):
                try:
                    upcoming = _rc.get_upcoming_announcements(days_ahead=7)
                except Exception:
                    upcoming = []

            if upcoming:
                rows = []
                for s in upcoming:
                    conf_icon = {"high": "★★★", "medium": "★★", "low": "★"}.get(
                        s.get("confidence", "low"), "★"
                    )
                    exp_date = s.get("expected_date")
                    rows.append({
                        "股票代號": s["stock_id"],
                        "股票名稱": s.get("stock_name", ""),
                        "預期公布日": exp_date.strftime("%m/%d") if exp_date else "—",
                        "預測信心度": conf_icon,
                        "上次 YoY": f"{s['last_revenue_yoy']:+.1f}%" if s.get("last_revenue_yoy") else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("本週無預計公布月營收個股（追蹤清單可能為空，或尚未建立歷史模式）")

            st.markdown("---")

            # ── 手動檢查今日新公告 ────────────────────────────────────────
            st.markdown("### 🔔 手動觸發：檢查今日新公告")
            if st.button("🔍 立即檢查今日新公告", key="rc_check_btn"):
                with st.spinner("正在檢查今日月營收公告..."):
                    try:
                        new_items = _rc.check_new_announcements()
                    except Exception as e:
                        st.error(f"檢查失敗：{e}")
                        new_items = []

                if new_items:
                    st.success(f"✅ 發現 {len(new_items)} 個新公告！")
                    for item in new_items:
                        yoy = item.get("yoy_pct")
                        yoy_str = f"{yoy:+.1f}%" if yoy is not None else "N/A"
                        yoy_color = "#6bcb77" if (yoy or 0) >= 0 else "#ff4b4b"
                        st.markdown(
                            f'<div style="background:#161b22;border-radius:8px;padding:12px 16px;margin:6px 0">'
                            f'<b>{item["stock_id"]} {item.get("stock_name", "")}</b>'
                            f'　{item.get("revenue_month", "")} 月營收'
                            f'　YoY：<b style="color:{yoy_color}">{yoy_str}</b>'
                            f'　金額：{item.get("revenue_amount", 0):,.0f} 千元'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("今日尚無新公告（或追蹤清單為空）")

            # ── 近3月已公布排行榜 ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 近 3 個月月營收 YoY 排行榜")
            with st.spinner("載入近期公告..."):
                try:
                    recent_df = _rc.get_recent_announcements(months=3)
                except Exception:
                    recent_df = pd.DataFrame()

            if not recent_df.empty:
                recent_df["yoy_pct"] = pd.to_numeric(recent_df["yoy_pct"], errors="coerce")
                recent_df = recent_df.sort_values("yoy_pct", ascending=False)
                recent_df["YoY (%)"] = recent_df["yoy_pct"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                )
                st.dataframe(
                    recent_df[["stock_id", "stock_name", "revenue_month", "YoY (%)", "announce_date"]].rename(columns={
                        "stock_id": "代號", "stock_name": "名稱",
                        "revenue_month": "月份", "announce_date": "公告日",
                    }),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("尚無歷史公告記錄（請先加入追蹤清單並執行「更新歷史模式」）")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 9：每日推薦
# ───────────────────────────────────────────────────────────────────────────────
with tab9:
    st.markdown("## 🏆 每日選股推薦")
    st.markdown("由量化模型掃描全市場，多維篩選後精選高分個股。")

    from screener.recommendation_db import RecommendationDB
    from datetime import date as _date
    import pandas as pd

    _rec_db = RecommendationDB()

    # ── 今日推薦 ──────────────────────────────────────────────────────────────
    today_recs = _rec_db.get_recommendations(_date.today())

    col_info, col_action = st.columns([3, 1])
    with col_info:
        if today_recs:
            scan_date_str = today_recs[0].get("recommend_date", str(_date.today()))
            st.caption(f"推薦日期：{scan_date_str}　共 {len(today_recs)} 支")
        else:
            st.info("今日尚未執行掃描，可點右側按鈕手動觸發。")

    with col_action:
        if st.button("🔄 立即掃描（需數分鐘）", key="scan_btn"):
            with st.spinner("掃描中，請稍候（依候選池大小約需 3-10 分鐘）..."):
                try:
                    from screener.recommender import DailyRecommender
                    recommender = DailyRecommender()
                    scan_result = recommender.run(dry_run=False)
                    if scan_result.get("error"):
                        st.error(f"掃描失敗：{scan_result['error']}")
                    else:
                        st.success(f"掃描完成！推薦 {len(scan_result['recommendations'])} 支")
                        today_recs = scan_result["recommendations"]
                except Exception as _e:
                    st.error(f"掃描異常：{_e}")

    # ── 推薦卡片 ──────────────────────────────────────────────────────────────
    if today_recs:
        medal_map = {1: "🥇", 2: "🥈", 3: "🥉"}

        for rec in today_recs:
            rank = rec.get("rank", 0)
            sid = rec.get("stock_id", "")
            name = rec.get("stock_name", sid)
            score = rec.get("total_score") or 0
            price = rec.get("current_price") or 0
            tp = rec.get("target_price")
            upside = rec.get("upside_pct")
            r1 = rec.get("reason_1") or rec.get("key_reasons", [None])[0] if isinstance(rec.get("key_reasons"), list) and rec["key_reasons"] else rec.get("reason_1", "")
            r2 = rec.get("reason_2") or (rec.get("key_reasons", [None, None])[1] if isinstance(rec.get("key_reasons"), list) and len(rec.get("key_reasons", [])) > 1 else "")
            r3 = rec.get("reason_3") or (rec.get("key_reasons", [None, None, None])[2] if isinstance(rec.get("key_reasons"), list) and len(rec.get("key_reasons", [])) > 2 else "")
            risk = rec.get("risk_warning", "")

            medal = medal_map.get(rank, f"#{rank}")
            score_color = "#6bcb77" if score >= 75 else ("#ffd166" if score >= 60 else "#ff4b4b")
            tp_str = f"NT${tp:,.0f}（{upside:+.0f}%）" if (tp and upside is not None) else "—"

            st.markdown(
                f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px 20px;margin:10px 0">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
    <span style="font-size:1.15em;font-weight:700">{medal} {sid} {name}</span>
    <span style="background:{score_color};color:#0d1117;border-radius:6px;padding:2px 10px;font-weight:700">評分 {score:.0f}/100</span>
  </div>
  <div style="color:#8b949e;margin-bottom:8px">
    現價 NT${price:,.0f}　|　目標價 {tp_str}
  </div>
  <div style="margin:6px 0">✅ {r1 or "—"}</div>
  <div style="margin:6px 0">✅ {r2 or "—"}</div>
  <div style="margin:6px 0">✅ {r3 or "—"}</div>
  {"<div style='color:#f0a500;margin-top:8px'>⚠️ " + risk + "</div>" if risk else ""}
</div>
""",
                unsafe_allow_html=True,
            )

        # ── 下載 CSV ──────────────────────────────────────────────────────────
        rec_df = pd.DataFrame(today_recs)
        csv = rec_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥 下載推薦清單 CSV", data=csv, file_name=f"recommend_{_date.today()}.csv", mime="text/csv")

    st.markdown("---")

    # ── 績效追蹤 ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 歷史推薦績效追蹤")
    perf = _rec_db.get_performance_summary(n_days=90)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("平均 5 日報酬", f"{perf['avg_return_5d']:+.1f}%" if perf.get("avg_return_5d") is not None else "—")
    p2.metric("平均 20 日報酬", f"{perf['avg_return_20d']:+.1f}%" if perf.get("avg_return_20d") is not None else "—")
    p3.metric("5 日勝率", f"{perf['win_rate_5d']*100:.0f}%" if perf.get("win_rate_5d") is not None else "—")
    p4.metric("已評估推薦數", str(perf.get("evaluated_count", 0)))

    recent_df = _rec_db.get_recent_recommendations(n_days=30)
    if not recent_df.empty:
        st.markdown("#### 近 30 日推薦記錄")
        show_cols = [c for c in ["recommend_date","rank","stock_id","stock_name","total_score","current_price","return_5d_pct","return_20d_pct"] if c in recent_df.columns]
        st.dataframe(recent_df[show_cols].rename(columns={
            "recommend_date": "日期", "rank": "排名", "stock_id": "代號",
            "stock_name": "名稱", "total_score": "評分", "current_price": "推薦價",
            "return_5d_pct": "5日報酬%", "return_20d_pct": "20日報酬%",
        }), use_container_width=True, hide_index=True)
    else:
        st.info("尚無歷史推薦記錄（每日掃描後自動填入）")

    st.markdown("---")
    st.caption("⚠️ 本推薦由量化模型自動生成，僅供學習與研究參考，不構成任何投資建議。投資涉及風險，請自行評估。")


# ───────────────────────────────────────────────────────────────────────────────
# Tab 10：設定
# ───────────────────────────────────────────────────────────────────────────────
with tab10:
    st.markdown("## ⚙️ API 金鑰設定")
    st.markdown(
        "在這裡輸入你的 API 金鑰，儲存後立即生效（不需要重啟）。"
        "金鑰存在伺服器本機的 `data/local_config.json`，**不會進入版本庫**。"
    )

    _KEY_META = [
        {
            "key": "FINMIND_TOKEN",
            "label": "FinMind Token",
            "help": "全市場掃描必填。至 finmindtrade.com 免費註冊取得。",
            "required": True,
        },
        {
            "key": "ANTHROPIC_API_KEY",
            "label": "Anthropic API Key",
            "help": "選填。設定後每日推薦會用 Claude 生成個股推薦理由。",
            "required": False,
        },
        {
            "key": "TELEGRAM_BOT_TOKEN",
            "label": "Telegram Bot Token",
            "help": "選填。設定後可推播每日推薦到 Telegram。",
            "required": False,
        },
        {
            "key": "TELEGRAM_CHAT_ID",
            "label": "Telegram Chat ID",
            "help": "選填。向你的 Bot 傳送 /myid 取得。",
            "required": False,
        },
    ]

    st.markdown("### 目前狀態")
    _status_cols = st.columns(len(_KEY_META))
    for _col, _m in zip(_status_cols, _KEY_META):
        _val = get_runtime_config(_m["key"])
        _icon = "✅" if _val else ("🔴" if _m["required"] else "⚪")
        _col.metric(_m["label"], _icon + (" 已設定" if _val else " 未設定"))

    st.markdown("---")
    st.markdown("### 編輯金鑰")

    with st.form("api_key_form"):
        _inputs = {}
        for _m in _KEY_META:
            _cur = get_runtime_config(_m["key"])
            _placeholder = "（已設定，輸入新值以覆蓋）" if _cur else "（未設定）"
            _inputs[_m["key"]] = st.text_input(
                f"{_m['label']}{'  ＊必填' if _m['required'] else '  （選填）'}",
                value="",
                placeholder=_placeholder,
                type="password",
                help=_m["help"],
            )

        _submitted = st.form_submit_button("💾 儲存", type="primary")
        if _submitted:
            _to_save = {k: v for k, v in _inputs.items() if v.strip()}
            if _to_save:
                try:
                    save_local_config(_to_save)
                    st.success(f"已儲存 {len(_to_save)} 個金鑰：{', '.join(_to_save.keys())}")
                    st.rerun()
                except Exception as _e:
                    st.error(f"儲存失敗：{_e}")
            else:
                st.info("未填入任何值，無需儲存。")

    st.markdown("---")
    st.markdown("### 清除金鑰")
    _clear_key = st.selectbox(
        "選擇要清除的金鑰",
        options=[""] + [_m["key"] for _m in _KEY_META],
        format_func=lambda x: "（請選擇）" if x == "" else x,
    )
    if st.button("🗑️ 清除選取的金鑰", disabled=not _clear_key):
        try:
            import json as _json
            with open("data/local_config.json", "r", encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            _cfg.pop(_clear_key, None)
            with open("data/local_config.json", "w", encoding="utf-8") as _f:
                _json.dump(_cfg, _f, ensure_ascii=False, indent=2)
            st.success(f"已清除 {_clear_key}")
            st.rerun()
        except Exception as _e:
            st.error(f"清除失敗：{_e}")

    st.markdown("---")
    st.caption(
        "💡 金鑰儲存在伺服器本機 `data/local_config.json`（已加入 .gitignore）。"
        "若同時設定了環境變數（.env / Render 環境設定），local_config.json 優先。"
    )
