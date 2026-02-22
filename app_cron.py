print(">>> [系統啟動] 正在初始化 HTML 報告生成環境...")

import os, time, datetime, io, base64, requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai

# ==========================================
# 1. 核心參數
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']

# ==========================================
# 2. 數據與圖表 (產出嵌入式 Base64)
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 抓取 Finviz 數據...")
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
                    "Industry": tds[4].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        return pd.DataFrame(data)[~pd.DataFrame(data)['Industry'].isin(EXCLUDE_INDUSTRIES)]
    except: return pd.DataFrame()

def generate_charts_base64(ticker):
    print(f">>> [圖表] 正在為 {ticker} 生成技術指標...")
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty or len(df) < 30: return None, 0, False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df['200MA'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.1, 0.4])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['200MA'], line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2)), row=3, col=1)
        
        fig.update_layout(height=500, width=900, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode('utf-8'), float(df['RSI'].iloc[-1]), bool(df['Close'].iloc[-1] > df['200MA'].iloc[-1])
    except: return None, 0, False

# ==========================================
# 3. AI 分析 (間隔與繁中優化)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if not GEMINI_KEY: return "未偵測到 API Key"
    client = genai.Client(api_key=GEMINI_KEY)
    status = "站上" if is_above_200 else "低於"
    prompt = f"分析美股 {row['Ticker']} ({row['Company']})。現價 {row['Price']}, RSI {rsi_val:.2f}, 目前{status} 200MA。請以繁體中文分點給出技術總結、贏面分數(1-100)與策略建議。字數150字內。"

    for _ in range(2):
        try:
            response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
            # 嚴格冷卻 45 秒防止 429 
            time.sleep(45) 
            return response.text.replace('\n', '<br>')
        except Exception as e:
            if "429" in str(e):
                print("⚠️ API 限制中，等待 60s 重試...")
                time.sleep(60)
            else: return f"分析失敗: {e}"
    return "API 忙碌中。"

# ==========================================
# 4. HTML 報告渲染與主執行
# ==========================================
def create_html_report(df):
    print(">>> [步驟 3] 正在生成 HTML 報告...")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; background: #f4f7f9; padding: 20px; }}
            .container {{ max-width: 950px; margin: 0 auto; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 30px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .summary-table th {{ background: #003366; color: white; padding: 12px; }}
            .summary-table td {{ padding: 10px; border-bottom: 1px solid #eee; text-align: center; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #e0e6ed; overflow: hidden; }}
            .card-header {{ background: #003366; color: white; padding: 15px; font-size: 20px; font-weight: bold; display: flex; justify-content: space-between; }}
            .chart-box {{ background: #1a1a1a; padding: 10px; text-align: center; }}
            .chart-box img {{ max-width: 100%; height: auto; border-radius: 5px; }}
            .analysis-box {{ padding: 20px; line-height: 1.8; color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align:center; color:#003366;">📊 美股 AI 技術分析報告</h1>
            <p style="text-align:right;">日期：{datetime.date.today()}</p>
            <table class="summary-table">
                <thead><tr><th>代碼</th><th>產業</th><th>現價</th><th>漲幅</th><th>成交量</th></tr></thead>
                <tbody>
    """
    for _, row in df.iterrows():
        html_content += f"<tr><td><b>{row['Ticker']}</b></td><td>{row['Industry']}</td><td>{row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    html_content += "</tbody></table>"

    for i, (_, row) in enumerate(df.head(10).iterrows()):
        img_b64, rsi, is_above = generate_charts_base64(row['Ticker'])
        if img_b64:
            ai_text = get_ai_insight(row, rsi, is_above)
            html_content += f"""
            <div class="stock-card">
                <div class="card-header"><span>{row['Ticker']} - {row['Company']}</span><span>RSI: {rsi:.2f}</span></div>
                <div class="chart-box"><img src="data:image/png;base64,{img_b64}"></div>
                <div class="analysis-box"><h3>🛡️ AI 策略師分析：</h3><p>{ai_text}</p></div>
            </div>
            """
    
    html_content += "</div></body></html>"
    with open("report.html", "w", encoding="utf-8") as f: f.write(html_content)
    print("✅ HTML 報告已儲存。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        create_html_report(df_stocks)
    else:
        print("今日無符合標的。")
