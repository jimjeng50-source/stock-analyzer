# 讓 Telegram 機器人 24/7 上線（隨時對話）

`bot.py` 是互動機器人（`/analyze`、`/eps`、`/recommend`、直接傳代號分析）。
要能隨時對話，需要一台**常駐主機**持續執行 `python bot.py`（輪詢 Telegram）。

> Streamlit Cloud 和 GitHub Actions **都不能**跑常駐程式 —— 這就是為什麼指令沒反應。
> 每日推薦「推播」不受影響（那是 GitHub Actions 定時跑），這裡談的是「即時對話」。

## 需要的環境變數

| 變數 | 必要 | 說明 |
|------|------|------|
| `TELEGRAM_BOT_TOKEN` | ✅ | BotFather 給的 token |
| `FINMIND_TOKEN` | ✅ | 個股財報資料 |
| `ANTHROPIC_API_KEY` | 選填 | AI 投資建議 |

---

## 方案比較（依「免費且省事」排序）

| 方案 | 費用 | 24/7 穩定度 | 難度 |
|------|------|------------|------|
| **家用電腦 / 樹莓派** | 免費（電費） | 看你的機器有沒有關 | ★ 最簡單 |
| **Oracle Cloud Always Free** | 永久免費 | ★★★ 真 24/7 | ★★ 需開 VM |
| **Railway / Render / Fly.io** | 免費額度有限* | ★★ | ★★ 綁 Dockerfile |
| **廉價 VPS（Linode/Vultr/DO）** | ~US$4-5/月 | ★★★ 最穩 | ★★ |

\* 免費額度常會睡眠或用完，長期跑建議 Oracle 免費 VM 或便宜 VPS。

---

## A. 最簡單：家用電腦 / 樹莓派

```bash
git clone https://github.com/jimjeng50-source/stock-analyzer.git
cd stock-analyzer
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="你的token"
export FINMIND_TOKEN="你的token"
export ANTHROPIC_API_KEY="你的key"   # 選填

python bot.py
```

想關掉終端機仍持續跑：
```bash
nohup python bot.py > bot.log 2>&1 &    # Linux/Mac 背景執行
```
或用 `tmux` / `screen`。缺點：電腦關機或斷網就停。

---

## B. 真正 24/7 免費：Oracle Cloud Always Free

1. 註冊 Oracle Cloud → 建一台 **Always Free** ARM VM（Ubuntu）
2. SSH 進去，跑上面 A 的步驟
3. 用 `systemd` 讓它開機自動啟動、掛掉自動重啟：

建立 `/etc/systemd/system/stock-bot.service`：
```ini
[Unit]
Description=Stock Telegram Bot
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/stock-analyzer
Environment=TELEGRAM_BOT_TOKEN=你的token
Environment=FINMIND_TOKEN=你的token
Environment=ANTHROPIC_API_KEY=你的key
ExecStart=/usr/bin/python3 bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
啟用：
```bash
sudo systemctl enable --now stock-bot
sudo systemctl status stock-bot      # 看狀態
journalctl -u stock-bot -f           # 看即時 log
```

---

## C. 容器平台（Railway / Fly.io / Render / 任何 VPS with Docker）

專案已附 `Dockerfile.bot`：

```bash
docker build -f Dockerfile.bot -t stock-bot .
docker run -d --restart=always \
  -e TELEGRAM_BOT_TOKEN=你的token \
  -e FINMIND_TOKEN=你的token \
  -e ANTHROPIC_API_KEY=你的key \
  --name stock-bot stock-bot
```

Railway/Render：新專案指向此 repo，Dockerfile 選 `Dockerfile.bot`，
在平台的 Variables/Environment 填上述環境變數即可。

---

## 驗證

啟動後在 Telegram 對你的 Bot：
- 傳 `/help` → 應回指令清單
- 傳 `2330` → 應回台積電分析
- 傳 `/eps 3260` → 威剛 Forward EPS

沒反應先看 log（`bot.log` 或 `journalctl`），最常見是 Token 錯或沒設環境變數。
