# 版本號碼：v1.2.2
print(">>> [系統啟動] 正在執行 v1.2.2：全功能完整修復版 (指標、導航、歷史、AI)...")

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
# 1. 核心參數與測試開關
# ==========================================
VERSION = "v1.2.2"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = True  # 正式執行請改為 False 以呼叫真實 AI 與執行休市檢查

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
    print(f">>> [步驟 1] 正在抓取 Finviz 數據 ({VERSION})...")
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
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
    except:
        mock = [{"Ticker": "AAPL", "Company": "Apple Inc.", "Sector": "Tech", "Industry": "Electronics", "MarketCap": "3T", "PE": "30", "Price": 185.0, "Change": 1.2, "Volume": "50M"}]
        return pd.DataFrame(mock)

# ==========================================
# 4. 專業繪圖 (全指標線型修復)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 繪製 {ticker} 全指標技術圖表...")
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

        # 建立三層日線圖
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        
        # 第一層：K線、三均線、量疊加
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(210, 210, 210, 0.6)', name="Vol"), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="K"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2.2), name="SMA200"), row=1, col=1)
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*4], secondary_y=True, showgrid=False, row=1)

        # 第二層：MACD 雙線與柱狀圖
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['MACD'], line=dict(color='white', width=1.2), name="MACD"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['Signal'], line=dict(color='yellow', width=1.2), name="Signal"), row=2, col=1)
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Hist'], marker_color=['lime' if v>=0 else 'red' for v in df_1y['Hist']], name="Hist"), row=2, col=1)

        # 第三層：RSI 14
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#00ff00', width=1.5), name="RSI14"), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=70, x1=df_1y.index[-1], y1=70, line=dict(color="red", dash="dash"), row=3, col=1)
        fig1.add_shape(type="line", x0=df_1y.index[0], y0=30, x1=df_1y.index[-1], y1=30, line=dict(color="red", dash="dash"), row=3, col=1)

        fig1.update_layout(height=650, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True, margin=dict(l=10, r=10, t=30, b=10))

        # 1分鐘線圖
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(210, 210, 210, 0.7)'), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            for idx, row in df_1m[df_1m['Volume'] > df_1m['Vol_Avg']*3].iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                fig2.add_annotation(x=idx, y=row['High'], text="▲ BUY" if row['Close'] > row['Open'] else "▼ SELL", font=dict(size=10, color=t_color), bgcolor="black", yshift=10)
            fig2.update_yaxes(range=[0, df_1m['Volume'].max()*4], secondary_y=True)
            fig2.update_layout(height=400, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except: return None, None, False

# ==========================================
# 5. AI 分析
# ==========================================
def get_ai_insight(row, is_above_200):
    status = "站上" if is_above_200 else "低於"
    prompt = f"分析美股 {row['Ticker']}。目前價格 {row['Price']}, 今日漲幅 {row['Change']}%, 目前{status} SMA200。請以繁體中文提供 MACD 與 RSI14 走勢分析及操作評分。150 字內。"
    if TEST_MODE: return f"<p style='color:#666; font-size:12px;'>[AI 指令預覽]: {prompt}</p>"
    if not GEMINI_KEY: return "❌ 缺少 API KEY"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(45)
        return response.text.replace('\n', '<br>')
    except: return "⚠️ AI 失敗"

# ==========================================
# 6. HTML 生成 (全功能導航與歷史門戶)
# ==========================================
def create_html_report(df):
    today_str = datetime.date.today().strftime("%Y%m%d")
    os.makedirs("history", exist_ok=True)
    
    # 掃描歷史檔案
    history_files = sorted(glob.glob("history/report_*.html"), reverse=True)
    history_links = "".join([f'<a href="history/report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])

    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .history-bar {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; overflow-x: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .history-item {{ text-decoration: none; color: #666; padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; font-size: 13px; white-space: nowrap; }}
        .history-item:hover {{ background: #003366; color: white; }}
        .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 50px; font-size: 12px; }}
        .summary-table th {{ background: #003366; color: white; padding: 12px; }}
        .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; cursor: pointer; }}
        .summary-table tr:hover {{ background: #d1dce5; cursor: pointer; }}
        .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
        .card-header-row {{ background: #003366; color: white; padding: 12px; display: grid; grid-template-columns: 80px 180px 120px 150px 100px 80px 80px 80px 1fr; text-align: center; font-size: 13px; font-weight: bold; align-items: center; }}
        .chart-stack {{ display: flex; flex-direction: column; gap: 20px; align-items: center; background: #1a1a1a; padding: 20px; }}
        .chart-stack img {{ width: 100%; max-width: 1000px; border: 1px solid #444; }}
        .analysis-box {{ padding: 25px; line-height: 1.8; background: #f8fafc; font-size: 14px; position: relative; }}
        .back-btn {{ display: inline-block; background: #003366; color: white; text-decoration: none; padding: 8px 20px; border-radius: 4px; font-size: 12px; float: right; font-weight: bold; }}
    </style></head>
    <body><div class="container" id="top">
        <div class="history-bar"><div style="font-weight:bold;margin-right:15px;color:#003366;">📅 歷史報告傳送門：</div>{history_links}</div>
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 全指標深度研究報告 {VERSION}</h1>
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
                <div class="analysis-box"><strong>🛡️ AI 策略師分析：</strong><br>{get_ai_insight(row, is_above)}<a href="#top" class="back-btn">⬆ 返回總表</a><div style="clear:both;"></div></div>
            </div>"""
    
    full_html = html_header + "</tbody></table>" + cards + "</div></body></html>"
    with open(f"history/report_{today_str}.html", "w", encoding="utf-8") as f: f.write(full_html)
    with open("index.html", "w", encoding="utf-8") as f: f.write(full_html)
    print(f"✅ {VERSION} 報告產出成功。")

# ==========================================
# 7. 主程式入口
# ==========================================
if __name__ == "__main__":
    if not TEST_MODE and not is_market_open_today():
        print("🛑 今日未開盤，程式終止。")
    else:
        df_stocks = fetch_and_filter_stocks()
        if not df_stocks.empty:
            create_html_report(df_stocks)
