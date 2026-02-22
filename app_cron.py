print(">>> [系統啟動] 正在執行量價疊加、三均線系統與 1m 爆量雷達優化...")

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
TEST_MODE = True  # 維持測試模式以節省額度

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
        return df.head(2) if TEST_MODE else df.head(10)
    except: return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (量價疊加 + SMA 200 修正)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 繪製 {ticker} (SMA 20/50/200 + Volume Overlay)...")
    try:
        # --- 1. 日線圖：抓取 2年數據確保 SMA 200 完整 ---
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df_all.empty: return None, None, 0, False
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        
        # 計算三均線
        df_all['SMA20'] = df_all['Close'].rolling(window=20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(window=50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df_all['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))

        # 截取最後一年顯示
        df_1y = df_all.tail(252)

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # 疊加成交量 (灰色、稍粗)
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(150, 150, 150, 0.4)', 
                             name="Volume", showlegend=False), row=1, col=1, secondary_y=True)
        # K線
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], 
                                     name="Price"), row=1, col=1, secondary_y=False)
        # 三均線
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2), name="SMA200"), row=1, col=1)
        # RSI
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#00ff00', width=1)), row=2, col=1)
        
        # 將成交量 Y軸範圍拉大，使其看起來在底部
        fig1.update_yaxes(range=[0, df_1y['Volume'].max() * 4], secondary_y=True, showgrid=False, row=1, col=1)
        fig1.update_layout(height=500, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, 
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                          margin=dict(l=10, r=10, t=30, b=10))
        
        # --- 2. 當日 1 分鐘圖 (1m + Volume Overlay) ---
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            df_1m['Spike'] = df_1m['Volume'] > (df_1m['Vol_Avg'] * 3)
            
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            # 疊加成交量
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(150, 150, 150, 0.5)', 
                                 name="Vol", showlegend=False), secondary_y=True)
            # K線
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            
            # 爆量標註
            for idx, row in df_1m[df_1m['Spike']].iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, showarrow=True, arrowhead=1, arrowcolor=t_color, 
                                    font=dict(size=10, color=t_color), bgcolor="black", opacity=0.8, yshift=10)

            fig2.update_yaxes(range=[0, df_1m['Volume'].max() * 4], secondary_y=True, showgrid=False)
            fig2.update_layout(height=400, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, 
                              title=dict(text=f"{ticker} 1m Intraday Spike Radar", font=dict(size=14)), 
                              margin=dict(l=10, r=10, t=40, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, float(df_1y['RSI'].iloc[-1]), bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖異常: {e}"); return None, None, 0, False

# ==========================================
# 4. HTML 渲染 (佈局垂直放大)
# ==========================================
def create_html_report(df):
    print(">>> [步驟 3] 渲染量價疊加放大版 HTML 報表...")
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; background: #f0f2f5; padding: 15px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 40px; font-size: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        .summary-table th {{ background: #003366; color: white; padding: 12px; position: sticky; top: 0; }}
        .summary-table td {{ border-bottom: 1px solid #ddd; text-align: center; padding: 10px; cursor: pointer; }}
        .summary-table tr:hover {{ background: #eef2f7; }}
        .stock-card {{ background: white; border-radius: 10px; margin-bottom: 50px; padding: 25px; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 15px; }}
        .card-header {{ border-bottom: 2px solid #003366; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }}
        .chart-grid {{ display: flex; flex-direction: column; gap: 25px; align-items: center; background: #1a1a1a; padding: 20px; border-radius: 8px; }}
        .chart-grid img {{ width: 100%; max-width: 980px; height: auto; border: 1px solid #444; border-radius: 4px; }}
        .analysis-box {{ margin-top: 20px; line-height: 1.7; background: #f8fafc; padding: 20px; border-radius: 6px; font-size: 14px; border-left: 5px solid #003366; }}
        .back-link {{ background: #003366; color: white; text-decoration: none; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
    </style></head>
    <body><div class="container" id="top">
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 深度研究週報 (量價疊加放大版)</h1>
        <table class="summary-table">
            <thead><tr><th>代碼</th><th>公司</th><th>板塊</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead>
            <tbody>
    """
    for _, row in df.iterrows():
        html_header += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Sector']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    
    cards = ""
    for _, row in df.iterrows():
        img1, img2, rsi, is_above = generate_stock_images(row['Ticker'])
        if img1:
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header">
                    <span style="font-size:24px; font-weight:bold; color:#003366;">{row['Ticker']} - {row['Company']}</span>
                    <a href="#top" class="back-link">⬆ 返回總表</a>
                </div>
                <div class="chart-grid">
                    <div style="width:100%; color:white; font-size:12px;">1Y Daily: SMA 20/50/200 & Volume Overlay</div>
                    <img src="data:image/png;base64,{img1}">
                    <div style="width:100%; color:white; font-size:12px; margin-top:10px;">Intraday 1m: Spike Radar & Volume Overlay</div>
                    <img src="data:image/png;base64,{img2}">
                </div>
                <div class="analysis-box">
                    <strong>🛡️ AI 策略師分析：</strong><br>
                    【測試模式】{row['Ticker']} 日線已整合量價疊加與三均線。RSI 為 {rsi:.2f}。
                </div>
            </div>"""
    
    with open("report.html", "w", encoding="utf-8") as f: f.write(html_header + "</tbody></table>" + cards + "</div></body></html>")
    print(">>> ✅ 報告已成功產出至 report.html")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty: create_html_report(df_stocks)
