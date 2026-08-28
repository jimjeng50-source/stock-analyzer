# 台股多因子選股評分系統

整合籌碼、基本面、技術面、動能面多維度市場資料，對個股進行評分，並透過 Claude AI 生成繁體中文投資建議報告。

---

## 專案簡介

| 功能 | 說明 |
|------|------|
| 多因子評分 | 籌碼（30%）、基本面（25%）、技術面（20%）、動能面（15%）、風險面（10%） |
| 資料來源 | FinMind API（主）+ yfinance（備援） |
| 介面 | Streamlit 互動網頁 / Python 命令列 |
| AI 建議 | Anthropic Claude 生成投資分析報告 |
| 盤中即時資金流 | 證交所 MIS 即時快照 → 主動買賣資金流動態圖 + 小波段進場訊號 + Telegram 即時通知 |

---

## 安裝步驟

### Windows

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## API Token 設定

複製 `.env.example` 為 `.env`，填入您的 Token：

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

編輯 `.env`：

```
FINMIND_TOKEN=你的FinMind Token
ANTHROPIC_API_KEY=你的Anthropic API Key
```

### 申請 FinMind Token

1. 前往 [https://finmindtrade.com/](https://finmindtrade.com/) 免費註冊
2. 登入後至「個人資料」取得 API Token
3. 免費方案每日有請求次數限制，請避免頻繁重複查詢

> **注意：** 未設定 FinMind Token 時，系統仍可運作，但籌碼面、基本面資料將設為中性值，僅技術面與動能面有效。

---

## 啟動方式

### Streamlit 互動介面（建議）

```bash
streamlit run app.py
```

開啟後在瀏覽器訪問 `http://localhost:8501`

### 命令列模式

```bash
# 基本分析
python main.py --stock 2330

# 不呼叫 Claude AI
python main.py --stock 6213 --no-ai

# 輸出 HTML 報告至 output/ 資料夾
python main.py --stock 0050 --save
```

---

## 盤中即時資金流（小波段進場訊號）

### 這是什麼

抓證交所 MIS 的盤中即時快照（免費、免 Token，約 5 秒更新一次），
把「累積成交量的變化」還原成**主動買賣資金流**，畫成動態圖，
再用六個子訊號評估「現在是不是小波段的進場點」，達門檻就推 Telegram。

### 資金流怎麼推估

MIS 只給累積成交量與五檔委買賣，**沒有逐筆內外盤**。作法是：

1. 相鄰兩次快照的累積量差 = 這段時間的成交張數
2. 用成交價相對「前一次快照」買一／賣一的位置判定主動方：
   成交價 ≥ 前賣一 → 主動買（外盤）；≤ 前買一 → 主動賣（內盤）；中間按比例拆分
3. 金額 = 張數 × 成交價 × 1000

因為 5 秒一張快照，期間來回單會互相抵銷 → **絕對金額低估，但方向與加速度可靠**，
而抓小波段要的正是方向與時機。系統會顯示 `coverage_ratio`（本次追蹤涵蓋當日成交量的比例），
讓你知道訊號的代表性。

### 進場模型：六個子訊號

| 子訊號 | 權重 | 看什麼 | 為什麼重要 |
|---|---|---|---|
| 主動買賣力道 | 25% | (主動買-主動賣)/(買+賣) | 主力方向；±10% 已算明顯 |
| 資金流加速度 | 20% | 近期流入速率 ÷ 全段平均 | 「剛開始流」勝過「早上流完了」，>1× 代表加速中 |
| 委買委賣結構 | 15% | 五檔 (委買-委賣)/(買+賣) | 買盤厚＝下檔有承接，回檔有人接 |
| 量能倍率 | 15% | 累積量 ÷（5 日均量 × 該時段應有比例） | 沒量的漲留不住，也出不掉 |
| 日內價格位階 | 15% | (現價-最低)/(最高-最低) | 最佳帶 45–85%；貼近當日最高會扣分（不追末端） |
| 日線法人籌碼 | 10% | 外資＋投信近 5 日買超佔量比 | 日內熱度和日線資金反向時勝率明顯下降 |

量能倍率用的是**日內量能 U 型分布曲線**（開盤與尾盤量大），
不是把時間線性攤平，所以早盤不會被誤判成爆量。

### 否決條件

不管加權分數多漂亮，以下情境會直接把分數壓下來：

| 條件 | 分數上限 | 理由 |
|---|---|---|
| 資金淨流出且加速 | 35 | 主力在調節 |
| 跌破當日 VWAP > 1.5% | 40 | 當日買方已失守 |
| 漲停鎖死 | 45 | 追價買不到，也沒有小波段空間了 |
| 當日成交量 < 500 張 | 35 | 流動性不足，進出易滑價 |
| 跌停鎖死 | 15 | 隔日跳空風險高 |

### 分級與操作參數

| 分數 | 分級 | 建議 |
|---|---|---|
| ≥ 75 | 🟢 強勢進場訊號 | 可進 1/2 倉 |
| ≥ 65 | 🟡 分批進場 | 1/3 倉 |
| ≥ 50 | ⚪ 觀察 | 等訊號轉強 |
| ≥ 35 | 🔵 偏弱 | 不進場 |
| < 35 | 🔴 資金流出／被否決 | 勿進 |

分數 ≥ 65 才會產出操作參數（不該進場就不規劃進場）：

- **進場帶**：VWAP ~ 現價 +0.5%
- **停損**：VWAP -1.5% 與當日低點 -0.5% 取較低者，且不超過 -4%
- **停利**：+4%（減半）／+7%（出清）
- **時間停損**：2 個交易日內未啟動即退出

另有**出場警示**：資金由買轉賣且加速、跌破 VWAP 1%、委賣大量堆疊、觸及停損停利，
都會即時推播。

### 怎麼用

**① Streamlit 動態圖** — `streamlit run app.py` → 「⚡ 即時資金流」頁籤

三段式動態圖：累計主動買賣淨額（億元）／股價 vs VWAP／每次取樣的淨流入（萬元），
勾「自動更新」即可持續刷新（Streamlit ≥1.33 只重跑該片段，不影響其他頁籤）。

**② Telegram 查詢** — 對 Bot 傳 `/flow 2330`（取樣約 30 秒後回訊號拆解與操作參數）

**③ 盤中自動通知**

```bash
# 本機／Render worker：跑到收盤為止
python jobs.py intraday-watch

# 只跑 60 分鐘
python jobs.py intraday-watch --max-minutes 60
```

GitHub Actions 用 `Intraday Watch` workflow（`.github/workflows/intraday-watch.yml`）。
**預設關閉**，要啟用請到 Settings → Secrets and variables → Actions → Variables
新增 `INTRADAY_WATCH_ENABLED = 1`。

> ⚠️ 這個 workflow 會連續執行約 4.5 小時。公開 repo 的 Actions 免費；
> **私有 repo 每月免費 2000 分鐘會不夠用**（跑滿 22 個交易日約需 6000 分鐘），
> 建議改用 Render worker 或自己的機器執行（見 `DEPLOY_BOT.md`）。

### 相關設定（.env / Secrets / Variables）

| 變數 | 預設 | 說明 |
|---|---|---|
| `INTRADAY_WATCHLIST` | 空 | 監控清單，逗號分隔。留空 → `WATCHLIST_CUSTOM` → 最近 7 日推薦名單 |
| `INTRADAY_MIN_SCORE` | 70 | 觸發推播的最低分數 |
| `INTRADAY_COOLDOWN_MIN` | 30 | 同一檔的推播冷卻分鐘數（分數再漲 8 分以上會補推） |
| `INTRADAY_POLL_INTERVAL` | 30 | 輪詢間隔（秒） |
| `INTRADAY_MAX_WATCH` | 15 | 監控清單上限 |

### 限制（用之前請先理解）

- **不是逐筆資料**：5 秒快照的近似，絕對金額會低估
- **沒有國定假日行事曆**：休市日 MIS 回最後交易日的收盤快照，累積量不變 → 曲線水平，不會產生假訊號
- **中途才開始追蹤**：資金流從你開始看的那一刻起算，`coverage_ratio` 會告訴你涵蓋度
- **模型未經長期實盤驗證**：所有門檻是依台股一般波動特性設定的先驗值，請自行回測與調整

---

## 因子說明

### 籌碼面（權重 30%）

| 因子 | 說明 |
|------|------|
| fi_5d_net | 外資近 5 日買賣超合計（張） |
| fi_20d_net | 外資近 20 日買賣超合計（張） |
| fi_consecutive | 外資連續買超天數（正）或賣超天數（負） |
| fi_trend | 外資近 10 日買賣超的線性迴歸趨勢斜率 |
| it_5d_net | 投信近 5 日買賣超合計（張） |
| it_20d_net | 投信近 20 日買賣超合計（張） |
| it_consecutive | 投信連續買賣超天數 |
| dealer_5d_net | 自營商近 5 日買賣超合計（張） |
| margin_chg_5d | 融資餘額近 5 日變化率（%，負值為減少，偏正面） |
| short_chg_5d | 融券餘額近 5 日變化率（%，負值為減少，偏正面） |

### 技術面（權重 20%）

| 因子 | 說明 |
|------|------|
| above_ma5 / ma20 / ma60 | 股價站上均線（1）或跌破（-1） |
| ma_alignment | 多頭排列分數（MA5>MA10>MA20>MA60 各算 1 分，共 0~3） |
| ma20_deviation | 股價距 MA20 偏離百分比（%） |
| rsi_14 | RSI(14) 數值 |
| rsi_signal | RSI 轉折信號（超賣=1，超買=-1，其它=0） |
| macd_histogram | MACD 柱狀值（正 = 多方力道強） |
| macd_cross | 黃金交叉=1，死亡交叉=-1，其它=0 |
| bb_position | 布林通道位置（0=下軌，0.5=中軌，1=上軌） |
| vol_ratio | 量比（今日量 / 20 日均量） |
| vol_trend | 5 日均量 > 20 日均量為 1，否則 -1 |

### 基本面（權重 25%）

| 因子 | 說明 |
|------|------|
| rev_yoy | 月營收年增率（%） |
| rev_mom | 月營收月增率（%） |
| rev_3m_trend | 近 3 個月營收趨勢（逐月成長=1，逐月下滑=-1） |
| rev_12m_high | 當月營收是否創近 12 個月新高（1 或 0） |
| eps_latest | 最近一季 EPS（元） |
| eps_qoq | EPS 季增（元） |
| eps_yoy | EPS 年增（與去年同季比，元） |
| gross_margin | 最近一季毛利率（%） |
| gpm_trend | 毛利率季變化（pp） |
| pe_ratio | 本益比（股價 / 近四季 EPS 加總） |

### 動能面（權重 15%）

| 因子 | 說明 |
|------|------|
| ret_5d | 近 5 日報酬率（%） |
| ret_1m | 近 20 日報酬率（%） |
| ret_3m | 近 60 日報酬率（%） |
| high_52w_pct | 距 52 週高點百分比（%，負值） |
| momentum_accel | 動能加速度（ret_5d - ret_1m） |

### 風險面（權重 10%）

| 因子 | 說明 |
|------|------|
| vol_20d | 近 20 日年化波動度（%，越低得分越高） |

---

## 常見問題（Q&A）

**Q：安裝 `ta` 套件時出錯怎麼辦？**
A：`ta` 為純 Python 套件，直接 `pip install ta` 即可，無需編譯 C 函式庫。若與 TA-Lib 混淆，請注意套件名稱不同。

**Q：FinMind 回傳 status 非 200 怎麼辦？**
A：可能是 Token 失效或超過當日 API 限制。請至 FinMind 官網確認 Token 狀態，或等隔日重試。

**Q：yfinance 取不到 ETF（如 0050）資料？**
A：yfinance 使用 Yahoo Finance，部分 ETF 需加上 `.TW` 後綴，系統已自動嘗試 `.TW` 與 `.TWO`。

**Q：Claude AI 建議顯示「驗證失敗」？**
A：請確認 `.env` 中的 `ANTHROPIC_API_KEY` 填寫正確，並確認帳號有足夠額度。

**Q：評分結果全部是 50 分（中性值）？**
A：通常是 FinMind Token 未設定，導致籌碼面與基本面均採用中性值。設定 Token 後重新分析即可。

**Q：Streamlit 畫面空白或報錯？**
A：確認已安裝所有 requirements.txt 套件，並在 `stock_analyzer/` 目錄下執行 `streamlit run app.py`。

---

## 專案結構

```
stock_analyzer/
├── .env.example       # API 金鑰範本
├── .gitignore
├── README.md
├── requirements.txt
├── config.py          # 全域設定（權重、門檻、API 設定）
├── main.py            # 命令列入口
├── app.py             # Streamlit 互動介面
├── data/
│   ├── fetcher.py     # FinMind + yfinance 資料抓取
│   └── realtime.py    # 證交所 MIS 盤中即時快照（五檔＋累積量）
├── factors/
│   ├── chips.py       # 籌碼面因子
│   ├── intraday_flow.py  # 盤中資金流引擎（主動買賣、VWAP、OBI）
│   ├── entry_signal.py   # 小波段進場點評分模型
│   ├── technical.py   # 技術面因子
│   ├── fundamental.py # 基本面因子
│   └── momentum.py    # 動能面因子
├── models/
│   └── scorer.py      # 加權評分模型
├── utils/
│   ├── claude_api.py  # Claude AI 投資建議
│   └── report.py      # HTML 報告輸出
└── output/            # 儲存 HTML 報告
```

---

## 免責聲明

本系統僅供學習與研究用途，輸出結果不構成任何投資建議。股票投資涉及風險，請自行評估並承擔投資決策責任。
