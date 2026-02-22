# 版本號碼：v1.3.1
print(">>> [系統啟動] 正在執行 v1.3.1：2支一組批量 AI 分析、縮減 MACD 高度與標題簡化...")

import os, time, datetime, io, base64, requests, glob, json
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
VERSION = "v1.3.1"
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
    print(f">>> [步驟 1] 正在抓取數據...")
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
                    "Ticker": tds[1].text.strip(),
                    "Company": tds[2].text.strip(),
                    "Industry": tds[4].text.strip(),
                    "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(),
                    "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')),
                    "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        return df.head(2) if TEST_MODE else df.head(10)
    except: return pd.DataFrame()

# ==========================================
# 3. 專業繪圖 (MACD 比例調回 0.25)
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

        # 改回較精幹的比例 
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.55, 0.25, 0.2], 
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
        
        # Row 1: K線、均線、成交量
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(210, 210, 210, 0.8)', name="Vol", showlegend=False), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="K"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="MA20"), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2.2), name="MA200"), row=1, col=1)
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*1.8], secondary_y=True, showgrid=False, row=1)

        # Row 2: MACD
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Hist'], marker_color=['rgba(0,255,0,0.8)' if v>=0 else 'rgba(255,0,0,0.8)' for v in df_1y['Hist']], name="Hist"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['MACD'], line=dict(color='#00FF00', width=1.8), name="MACD"), row=2, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['Signal'], line=dict(color='#A020F0', width=1.8), name="Signal"), row=2, col=1)
        
        # Row 3: RSI
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='#E0B0FF', width=2.2), name="RSI"), row=3, col=1)

        fig1.update_layout(height=800, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))

        # 1分鐘圖
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(210, 210, 210, 0.8)'), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            fig2.update_layout(height=450, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        return base64.b64encode(fig1.to_image(format="png")).decode('utf-8'), fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except: return None, None, False

# ==========================================
# 4. 批量 AI 分析 (2支一組 + JSON Mode)
# ==========================================
def get_batch_ai_insights(df_subset):
    """
    接收 2 支股票的 DataFrame，回傳 {Ticker: Insight} 的字典
    """
    tickers = df_subset['Ticker'].tolist()
    if TEST_MODE:
        return {t: f"<p style='color:#666;'>[測試] 批量分析中 - {t}</p>" for t in tickers}
    if not GEMINI_KEY:
        return {t: "❌ 缺少 API KEY" for t in tickers}

    # 建立數據摘要供 AI 參考
    data_summary = ""
    for _, row in df_subset.iterrows():
        data_summary += f"- {row['Ticker']}: 價格 ${row['Price']}, 漲幅 {row['Change']}%\n"

    prompt = f"""
    你是一位專業的美股分析師。請分析以下 {len(tickers)} 支股票的技術趨勢：
    {data_summary}
    要求：
    1. 結合 MACD 與 RSI14 走勢給予簡短建議（100字內/支）。
    2. 使用繁體中文。
    3. **必須**回傳純 JSON 格式，不含 Markdown 標籤，格式如下：
       {{"Ticker1": "分析內容1", "Ticker2": "分析內容2"}}
    """
    
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        # 嘗試解析 JSON
        raw_text = response.text.strip().replace('```json', '').replace('```', '')
        insights = json.loads(raw_text)
        time.sleep(30) # 批量處理可以稍微縮短等待時間
        return insights
    except Exception as e:
        print(f"⚠️ 批量 AI 異常: {e}")
        return {t: "⚠️ 分析暫時無法產出" for t in tickers}

# ==========================================
# 5. HTML 生成 (簡化標題卡)
# ==========================================
def create_html_report(df):
    today_str = datetime.date.today().strftime("%Y%m%d")
    os.makedirs("history", exist_ok=True)
    history_files = sorted(glob.glob("history/report_*.html"), reverse=True)
    history_links = "".join([f'<a href="history/report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])
    
    # --- 執行批量 AI 分析 ---
    print(f">>> [步驟 2] 正在進行 2 支一組的批量 AI 分析...")
    all_insights = {}
    for i in range(0, len(df), 2):
        subset = df.iloc[i:i+2]
        batch_results = get_batch_ai_insights(subset)
        all_insights.update(batch_results)

    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW"><head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; margin: 0; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .history-bar {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; overflow-x: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .history-item {{ text-decoration: none; color: #666; padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; font-size: 12px; white-space: nowrap; }}
            .summary-table-wrapper {{ overflow-x: auto; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 40px; font-size: 12px; min-width: 800px; }}
            .summary-table th {{ background: #003366; color: white; padding: 12px; }}
            .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
            
            /* 更新：7 欄位佈局 (移除 Company) */
            .card-header-row {{ 
                background: #003366; color: white; padding: 12px; 
                display: grid; 
                grid-template-columns: 80px 200px 100px 80px 80px 80px 1fr;
                text-align: center; font-size: 13px; font-weight: bold; align-items: center;
            }}
            @media (max-width: 768px) {{ .card-header-row {{ grid-template-columns: repeat(2, 1fr); font-size: 11px; gap: 8px; }} }}
            .chart-stack {{ display: flex; flex-direction: column; gap: 20px; align-items: center; background: #1a1a1a; padding: 15px; }}
            .chart-stack img {{ width: 100%; height: auto; border: 1px solid #444; }}
            .analysis-box {{ padding: 25px; line-height: 1.8; background: #f8fafc; font-size: 14px; border-top: 1px solid #eee; }}
            .back-btn {{ display: inline-block; background: #003366; color: white; text-decoration: none; padding: 8px 20px; border-radius: 4px; font-size: 12px; float: right; font-weight: bold; }}
        </style>
    </head>
    <body><div class="container" id="top">
        <div class="history-bar"><div style="font-weight:bold;margin-right:10px;color:#003366;white-space:nowrap;">📅 歷史存檔：</div>{history_links}</div>
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 全指標深度報告 {VERSION}</h1>
        <div class="summary-table-wrapper">
            <table class="summary-table">
                <thead><tr><th>代碼</th><th>公司</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead>
                <tbody>
    """
    for _, row in df.iterrows():
        html_header += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    
    cards = ""
    for _, row in df.iterrows():
        img1, img2, is_above = generate_stock_images(row['Ticker'])
        insight = all_insights.get(row['Ticker'], "分析產出中...")
        if img1:
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header-row">
                    <div>{row['Ticker']}</div><div>{row['Industry']}</div>
                    <div>{row['MarketCap']}</div><div>{row['PE']}</div><div>${row['Price']}</div>
                    <div style="color:#ffcccc;">+{row['Change']}%</div><div>{row['Volume']}</div>
                </div>
                <div class="chart-stack"><img src="data:image/png;base64,{img1}"><img src="data:image/png;base64,{img2}"></div>
                <div class="analysis-box"><strong>🛡️ AI 策略師診斷：</strong><br>{insight}<a href="#top" class="back-btn">⬆ 返回總表</a><div style="clear:both;"></div></div>
            </div>"""
    
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_header + "</tbody></table></div>" + cards + "</div></body></html>")
    with open(f"history/report_{today_str}.html", "w", encoding="utf-8") as f: f.write(html_header + "</tbody></table></div>" + cards + "</div></body></html>")

if __name__ == "__main__":
    if not TEST_MODE and not is_market_open_today():
        print("🛑 今日未開盤")
    else:
        df_stocks = fetch_and_filter_stocks()
        if not df_stocks.empty: create_html_report(df_stocks)
