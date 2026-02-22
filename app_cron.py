# 版本號碼：v1.2.4
print(">>> [系統啟動] 正在執行 v1.2.4：全指標修復與行動端響應式排版優化...")

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
VERSION = "v1.2.4"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = True  # 正式上線請改 False

# ==========================================
# 2. 環境與數據檢查 (休市偵測)
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
# 3. 專業繪圖 (指標與標籤全修復)
# ==========================================
def generate_stock_images(ticker):
    try:
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        
        # 指標計算
        df_all['SMA20'] = df_all['Close'].rolling(20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(200).mean()
        # MACD
        exp1 = df_all['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_all['Close'].ewm(span=26, adjust=False).mean()
        df_all['MACD'] = exp1 - exp2
        df_all['Signal'] = df_all['MACD'].ewm(span=9, adjust=False).mean()
        df_all['Hist'] = df_all['MACD'] - df_all['Signal']
        # RSI 14
        delta = df_all['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))
        df_1y = df_all.tail(252)

        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        # Row 1: K線、三均線、量
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(210, 210, 210, 0.6)', name="Vol"), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="K"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2.2), name="SMA200"), row=1, col=1)
        
        # Row 2: MACD 雙線與柱狀圖 
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['MACD'], line=dict(color='white', width=1.2), name="MACD"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['Signal'], line=dict(color='yellow', width=1.2), name="Signal"), row=2, col=1)
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Hist'], marker_color=['lime' if v>=0 else 'red' for v in df_1y['Hist']], name="Hist"), row=2, col=1)
        
        # Row 3: RSI
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#00ff00', width=1.5), name="RSI14"), row=3, col=1)
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*4], secondary_y=True, showgrid=False, row=1)
        fig1.update_layout(height=650, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))

        # 1分鐘線圖與標籤 
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(210, 210, 210, 0.7)'), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            
            # 重新實作向上買向下賣標籤
            spikes = df_1m[df_1m['Volume'] > df_1m['Vol_Avg']*3]
            for idx, row in spikes.iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, showarrow=True, arrowhead=1, arrowcolor=t_color,
                                    font=dict(size=10, color=t_color), bgcolor="black", opacity=0.8, yshift=10)

            fig2.update_yaxes(range=[0, df_1m['Volume'].max()*4], secondary_y=True, showgrid=False)
            fig2.update_layout(height=400, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        return base64.b64encode(fig1.to_image(format="png")).decode('utf-8'), fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖異常: {e}"); return None, None, False

# ==========================================
# 4. HTML 渲染與響應式佈局
# ==========================================
def create_html_report(df):
    today_str = datetime.date.today().strftime("%Y%m%d")
    os.makedirs("history", exist_ok=True)
    history_files = sorted(glob.glob("history/report_*.html"), reverse=True)
    history_links = "".join([f'<a href="history/report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])
    ICON_URL = "https://cdn-icons-png.flaticon.com/512/2422/2422796.png"

    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW"><head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <link rel="icon" href="{ICON_URL}" type="image/png">
        <link rel="apple-touch-icon" href="{ICON_URL}">
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; margin: 0; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .history-bar {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; overflow-x: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .history-item {{ text-decoration: none; color: #666; padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; font-size: 12px; white-space: nowrap; }}
            
            .summary-table-wrapper {{ overflow-x: auto; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 40px; font-size: 11px; min-width: 800px; }}
            .summary-table th {{ background: #003366; color: white; padding: 10px; }}
            .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; cursor: pointer; }}
            .summary-table tr:hover {{ background: #d1dce5; }}

            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
            
            /* 數據標題列 - 支援行動端響應式 */
            .card-header-row {{ 
                background: #003366; color: white; padding: 12px; 
                display: grid; 
                grid-template-columns: 70px 180px 100px 140px 90px 70px 70px 70px 1fr;
                text-align: center; font-size: 12px; font-weight: bold; align-items: center;
            }}

            @media (max-width: 768px) {{
                .card-header-row {{
                    grid-template-columns: repeat(3, 1fr);
                    font-size: 10px;
                    gap: 8px;
                    padding: 15px;
                }}
                .history-bar {{ padding: 10px; }}
                h1 {{ font-size: 18px; }}
            }}

            .chart-stack {{ display: flex; flex-direction: column; gap: 15px; align-items: center; background: #1a1a1a; padding: 15px; }}
            .chart-stack img {{ width: 100%; height: auto; border: 1px solid #444; }}
            .analysis-box {{ padding: 20px; line-height: 1.7; background: #f8fafc; font-size: 14px; border-top: 1px solid #eee; }}
            .back-btn {{ display: inline-block; background: #003366; color: white; text-decoration: none; padding: 8px 20px; border-radius: 4px; font-size: 12px; float: right; font-weight: bold; }}
        </style>
    </head>
    <body><div class="container" id="top">
        <div class="history-bar"><div style="font-weight:bold;margin-right:10px;color:#003366;white-space:nowrap;">📅 歷史：</div>{history_links}</div>
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 全指標深度報告 {VERSION}</h1>
        <div class="summary-table-wrapper">
            <table class="summary-table">
                <thead><tr><th>代碼</th><th>公司</th><th>板塊</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead>
                <tbody>
    """
    for _, row in df.iterrows():
        html_header += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Sector']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    
    cards = ""
    for _, row in df.iterrows():
        img1, img2, is_above = generate_stock_images(row['Ticker'])
        if img1:
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header-row">
                    <div>{row['Ticker']}</div><div>{row['Company']}</div><div>{row['Sector']}</div><div>{row['Industry']}</div>
                    <div>{row['MarketCap']}</div><div>{row['PE']}</div><div>${row['Price']}</div><div style="color:#ffcccc;">+{row['Change']}%</div><div>{row['Volume']}</div>
                </div>
                <div class="chart-stack"><img src="data:image/png;base64,{img1}"><img src="data:image/png;base64,{img2}"></div>
                <div class="analysis-box"><strong>🛡️ AI 策略師診斷：</strong><br>測試模式中...<a href="#top" class="back-btn">⬆ 返回總表</a><div style="clear:both;"></div></div>
            </div>"""
    
    full_html = html_header + "</tbody></table></div>" + cards + "</div></body></html>"
    with open(f"history/report_{today_str}.html", "w", encoding="utf-8") as f: f.write(full_html)
    with open("index.html", "w", encoding="utf-8") as f: f.write(full_html)

if __name__ == "__main__":
    if not TEST_MODE and not is_market_open_today():
        print("🛑 今日未開盤")
    else:
        df_stocks = fetch_and_filter_stocks()
        if not df_stocks.empty: create_html_report(df_stocks)
