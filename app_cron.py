print(">>> [系統啟動] 正在執行量價疊加、1m 爆量雷達與 SMA 200 邏輯修正...")

import os, time, datetime, io, base64, requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai

# ==========================================
# 1. 核心參數與測試開關
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = True  # 開發階段維持 True 

# ==========================================
# 2. 數據抓取 (9 大欄位)
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 抓取數據...")
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
                    "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        return df.head(2) if TEST_MODE else df.head(10) [cite: 1-905]
    except: return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (量價疊加 + SMA 修正)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 處理 {ticker} (量價疊加 + SMA 200 修正)...")
    try:
        # --- 1. 日線圖 (抓取 2y 以確保 SMA 200 完整) ---
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False) [cite: 1-905]
        if df_all.empty: return None, None, 0, False
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        
        # 計算三均線
        df_all['SMA20'] = df_all['Close'].rolling(window=20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(window=50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(window=200).mean() [cite: 1-905]
        
        # RSI
        delta = df_all['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))

        # 僅取最近 1 年用於顯示
        df_1y = df_all.last('365D') [cite: 1-905]

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]) [cite: 1-905]
        
        # 疊加成交量 (灰色、置於底層)
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(150, 150, 150, 0.3)', 
                             name="Volume", showlegend=False), row=1, col=1, secondary_y=True) [cite: 1-905]
        # K線
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], 
                                     name="Price"), row=1, col=1, secondary_y=False)
        # 三均線
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2), name="SMA200"), row=1, col=1)
        # RSI
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#00ff00', width=1.2)), row=2, col=1)
        
        # 座標軸優化：將成交量次標軸範圍拉大，使其看起來在底部
        fig1.update_yaxes(range=[0, df_1y['Volume'].max() * 4], secondary_y=True, showgrid=False, row=1, col=1) [cite: 1-905]
        fig1.update_layout(height=450, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, 
                          legend=dict(orientation="h", y=1.05, x=1, xanchor="right"), margin=dict(l=10, r=10, t=40, b=10))
        
        # --- 2. 1分鐘圖 (量價疊加) ---
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False) [cite: 1-905]
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            df_1m['Spike'] = df_1m['Volume'] > (df_1m['Vol_Avg'] * 3)
            
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            # 疊加成交量
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(150, 150, 150, 0.4)', 
                                 name="Volume", showlegend=False), secondary_y=True) [cite: 1-905]
            # K線
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            
            # 爆量提醒
            for idx, row in df_1m[df_1m['Spike']].iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, showarrow=True, arrowhead=1, arrowcolor=t_color, 
                                    font=dict(size=10, color=t_color), bgcolor="black", opacity=0.8, yshift=10)

            fig2.update_yaxes(range=[0, df_1m['Volume'].max() * 4], secondary_y=True, showgrid=False) [cite: 1-905]
            fig2.update_layout(height=350, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, 
                              title=dict(text=f"{ticker} 1m Intraday Spike Radar", font=dict(size=14)), margin=dict(l=10, r=10, t=40, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, float(df_1y['RSI'].iloc[-1]), bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖異常: {e}"); return None, None, 0, False

# HTML 渲染部分維持「全欄位總表」與「垂直堆疊佈局」
def create_html_report(df):
    print(">>> [步驟 3] 渲染量價疊加版 HTML 報表...")
    # ... (HTML 模板內容與上一版相同) ... 
    pass

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        # 由於篇幅，請在此處延用上一版完整的 create_html_report 邏輯
        pass
