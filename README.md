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
│   └── fetcher.py     # FinMind + yfinance 資料抓取
├── factors/
│   ├── chips.py       # 籌碼面因子
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
