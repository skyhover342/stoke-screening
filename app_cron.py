print(">>> [系統啟動] 切換至 HTML 報告模式，正在初始化...")

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

# ==========================================
# 2. 數據與圖表 (產出 Base64 圖片)
# ==========================================
def generate_charts_base64(ticker):
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
        
        # 轉換為 Base64 字串
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode('utf-8'), float(df['RSI'].iloc[-1]), bool(df['Close'].iloc[-1] > df['200MA'].iloc[-1])
    except: return None, 0, False

# ==========================================
# 3. AI 分析 (強化間隔)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    client = genai.Client(api_key=GEMINI_KEY)
    status = "站上" if is_above_200 else "低於"
    prompt = f"請以專家身份分析美股 {row['Ticker']} ({row['Company']})。現價 {row['Price']}, RSI {rsi_val:.2f}, 目前{status} 200MA。請以繁體中文給出：1. 技術總結 2. 贏面評分(1-100) 3. 具體策略。總字數150字內。"

    for _ in range(2):
        try:
            response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
            time.sleep(45) # 嚴格遵守 45 秒間隔 
            return response.text.replace('\n', '<br>') # 轉換換行為 HTML 標籤
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ 觸發限制，等待 60s 重試...")
                time.sleep(60)
            else: return f"分析不可用: {e}"
    return "API 繁忙中。"

# ==========================================
# 4. 生成 HTML 報告
# ==========================================
def create_html_report(df):
    print(">>> [步驟 3] 正在整合 HTML 報告...")
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <title>美股 AI 深度研究報告 - {datetime.date.today()}</title>
        <style>
            body {{ font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif; background: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            h1 {{ text-align: center; color: #003366; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            .summary-table th {{ background: #003366; color: white; padding: 12px; font-size: 14px; }}
            .summary-table td {{ padding: 10px; border-bottom: 1px solid #eee; text-align: center; font-size: 13px; }}
            .stock-card {{ background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 40px; overflow: hidden; border: 1px solid #e0e6ed; }}
            .card-header {{ background: #003366; color: white; padding: 15px 20px; font-size: 20px; font-weight: bold; display: flex; justify-content: space-between; }}
            .chart-box {{ padding: 10px; background: #1a1a1a; text-align: center; }}
            .chart-box img {{ max-width: 100%; height: auto; }}
            .analysis-box {{ padding: 20px; line-height: 1.7; }}
            .score-tag {{ background: #ffefef; color: #d93025; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 美股 AI 技術分析週報</h1>
            <p style="text-align: right; color: #666;">生成日期：{datetime.date.today()}</p>
            
            <table class="summary-table">
                <thead>
                    <tr><th>代碼</th><th>產業</th><th>現價</th><th>漲幅</th><th>成交量</th></tr>
                </thead>
                <tbody>
    """
    
    # 填充總表內容
    for _, row in df.iterrows():
        html_template += f"<tr><td><b>{row['Ticker']}</b></td><td>{row['Industry']}</td><td>{row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    
    html_template += "</tbody></table>"

    # 填充個股卡片
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        img_b64, rsi, is_above = generate_charts_base64(row['Ticker'])
        if img_b64:
            ai_text = get_ai_insight(row, rsi, is_above)
            html_template += f"""
            <div class="stock-card">
                <div class="card-header">
                    <span>{row['Ticker']} - {row['Company']}</span>
                    <span class="score-tag">RSI: {rsi:.2f}</span>
                </div>
                <div class="chart-box">
                    <img src="data:image/png;base64,{img_b64}">
                </div>
                <div class="analysis-box">
                    <h3 style="color:#003366; margin-top:0;">🛡️ AI 策略師分析：</h3>
                    <p>{ai_text}</p>
                </div>
            </div>
            """

    html_template += "</div></body></html>"
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("✅ 任務完成：HTML 報告已生成，外觀精美且無亂碼問題。")

if __name__ == "__main__":
    # 這裡放 fetch_and_filter_stocks 的邏輯...
    # 執行 create_html_report(df)
