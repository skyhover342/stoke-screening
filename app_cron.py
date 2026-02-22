# 版本號碼：v1.2.1
print(">>> [系統啟動] 正在執行 v1.2.1：全指標線型修復與自動化營運...")

import os, time, datetime, io, base64, requests, glob
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

try:
    from google import genai
except ImportError:
    print("❌ 錯誤：找不到 google-genai 套件。")

# ==========================================
# 1. 核心參數
# ==========================================
VERSION = "v1.2.1"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = True  # 正式執行請改為 False

# ==========================================
# 2. 休市偵測系統
# ==========================================
def is_market_open_today():
    if TEST_MODE: return True
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d")
        if hist.empty: return False
        last_trade_date = hist.index[-1].date()
        ny_tz = pytz.timezone('America/New_York')
        today_ny = datetime.datetime.now(ny_tz).date()
        return last_trade_date == today_ny
    except: return True

# ==========================================
# 3. 數據抓取
# ==========================================
def fetch_and_filter_stocks():
    print(f">>> [步驟 1] 抓取數據 ({VERSION})...")
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        rows = soup.find_all('tr', valign="top")
        data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 11: continue
            try:
                data.append({
                    "Ticker": tds[1].text.strip(), "Company": tds[2].text.strip(),
                    "Sector": tds[3].text.strip(), "Industry": tds[4].text.strip(),
                    "MarketCap": tds[6].text.strip(), "PE": tds[7].text.strip(),
                    "Price": float(tds[8].text.strip()), "Change": float(tds[9].text.strip('%')),
                    "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        return df.head(2) if TEST_MODE else df.head(10)
    except: return pd.DataFrame()

# ==========================================
# 4. 專業繪圖 (補回所有遺失線型)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 繪製 {ticker} 完整指標圖表...")
    try:
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        
        # 均線
        df_all['SMA20'] = df_all['Close'].rolling(20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(200).mean()
        # MACD
        exp1 = df_all['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_all['Close'].ewm(span=26, adjust=False).mean()
        df_all['MACD'] = exp1 - exp2
        df_all['Signal'] = df_all['MACD'].ewm(span=9, adjust=False).mean()
        df_all['Hist'] = df_all['MACD'] - df_all['Signal']
        # RSI
        delta = df_all['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))
        df_1y = df_all.tail(252)

        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        
        # Row 1: K線 + 三均線 + 成交量疊加
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(210, 210, 210, 0.6)', name="Vol"), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="K"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2.2), name="SMA200"), row=1, col=1)
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*4], secondary_y=True, showgrid=False, row=1)

        # Row 2: MACD 雙線 + 柱狀圖
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['MACD'], line=dict(color='white', width=1.2), name="MACD"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['Signal'], line=dict(color='yellow', width=1.2), name="Signal"), row=2, col=1)
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Hist'], marker_color=['lime' if v>=0 else 'red' for v in df_1y['Hist']], name="Hist"), row=2, col=1)

        # Row 3: RSI 14
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#00ff00', width=1.5), name="RSI14"), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=70, x1=df_1y.index[-1], y1=70, line=dict(color="red", dash="dash"), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=30, x1=df_1y.index[-1], y1=30, line=dict(color="red", dash="dash"), row=3, col=1)

        fig1.update_layout(height=650, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(l=10, r=10, t=30, b=10))

        # 1分鐘圖 (雷達依舊)
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(210, 210, 210, 0.7)'), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            fig2.update_yaxes(range=[0, df_1m['Volume'].max()*4], secondary_y=True)
            fig2.update_layout(height=400, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except: return None, None, False

# AI 分析、HTML 生成邏輯維持全導航功能 (省略重複代碼，請參照 v1.2.0)
# ... [create_html_report 與 get_ai_insight 同 v1.2.0，請務必完整複製] ...

if __name__ == "__main__":
    if not TEST_MODE and not is_market_open_today():
        print("🛑 市場未開，終止執行。")
    else:
        df = fetch_and_filter_stocks()
        if not df.empty: create_html_report(df)
