# 版本號碼：v1.2.7
print(">>> [系統啟動] 正在執行 v1.2.7：大幅強化 Volume 與 MACD 視覺佔比...")

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
VERSION = "v1.2.7"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = True  # 正式執行請改 False

# ==========================================
# 2. 環境與數據檢查
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

def fetch_and_filter_stocks():
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
        return pd.DataFrame(data).head(2) if TEST_MODE else pd.DataFrame(data).head(10)
    except: return pd.DataFrame()

# ==========================================
# 3. 專業繪圖 (視覺比例大幅調整)
# ==========================================
def generate_stock_images(ticker):
    try:
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        
        # 指標計算
        df_all['SMA20'] = df_all['Close'].rolling(20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(200).mean()
        exp1 = df_all['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_all['Close'].ewm(span=26, adjust=False).mean()
        df_all['MACD'] = exp1 - exp2
        df_all['Signal'] = df_all['MACD'].ewm(span=9, adjust=False).mean()
        df_all['Hist'] = df_all['MACD'] - df_all['Signal']
        delta = df_all['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))
        df_1y = df_all.tail(252)

        # 調整 row_heights 使 MACD (Row 2) 更高
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                            row_heights=[0.52, 0.28, 0.2], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        
        # Row 1: K線、均線、成交量 (強化高度)
        # 視覺增強：成交量透明度調高，Y軸範圍縮小
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(210, 210, 210, 0.8)', name="Volume", showlegend=False), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="Price"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2.2), name="SMA200"), row=1, col=1)
        
        # Row 2: MACD [柱狀圖飽和度增強]
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Hist'], marker_color=['rgba(0,255,0,0.8)' if v>=0 else 'rgba(255,0,0,0.8)' for v in df_1y['Hist']], name="Histogram"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['MACD'], line=dict(color='#00FF00', width=1.8), name="MACD"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['Signal'], line=dict(color='#A020F0', width=1.8), name="Signal"), row=2, col=1)
        
        # Row 3: RSI [亮紫 + 加粗]
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#E0B0FF', width=2.2), name="RSI14"), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=70, x1=df_1y.index[-1], y1=70, line=dict(color="red", dash="dash", width=1), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=30, x1=df_1y.index[-1], y1=30, line=dict(color="red", dash="dash", width=1), row=3, col=1)

        # 視覺增強：成交量 Y 軸範圍從 4 倍縮小到 1.8 倍，讓柱子變長
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*1.8], secondary_y=True, showgrid=False, row=1)
        fig1.update_layout(height=750, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))

        # 1分鐘線圖：同步視覺強化
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(210, 210, 210, 0.8)', showlegend=False), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close'], name="1m Price"), secondary_y=False)
            spikes = df_1m[df_1m['Volume'] > df_1m['Vol_Avg']*3]
            for idx, row in spikes.iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, showarrow=True, arrowhead=1, arrowcolor=t_color, font=dict(size=11, color=t_color, weight='bold'), bgcolor="black", opacity=0.9, yshift=10)
            fig2.update_yaxes(range=[0, df_1m['Volume'].max()*1.8], secondary_y=True, showgrid=False)
            fig2.update_layout(height=450, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖異常: {e}"); return None, None, False

# AI 分析、HTML 生成邏輯與 v1.2.6 一致
# ... (省略重複的 get_ai_insight 與 create_html_report 代碼以節省空間)
# (請務必在實作時保留完整函式，包含響應式 CSS 與歷史導覽功能)

if __name__ == "__main__":
    if not TEST_MODE and not is_market_open_today():
        print("🛑 今日未開盤")
    else:
        df_stocks = fetch_and_filter_stocks()
        if not df_stocks.empty: create_html_report(df_stocks)
