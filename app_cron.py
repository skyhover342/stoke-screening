print(">>> [系統啟動] 正在初始化全欄位導航 HTML 報告環境...")

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
# 2. 數據抓取 (抓取 9 大核心欄位)
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 正在從 Finviz 抓取完整數據表...")
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
                # 依照 v=111 索引精確抓取
                data.append({
                    "Ticker": tds[1].text.strip(),
                    "Company": tds[2].text.strip(),
                    "Sector": tds[3].text.strip(),
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
    except Exception as e:
        print(f"❌ 抓取失敗: {e}")
        return pd.DataFrame()

def generate_charts_base64(ticker):
    print(f">>> [圖表] 生成 {ticker} 技術圖表...")
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty: return None, 0, False
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

def get_ai_insight(row, rsi_val, is_above_200):
    if TEST_MODE:
        return f"【測試模式】{row['Ticker']} RSI: {rsi_val:.2f}。趨勢穩定。建議於回測支撐時佈局。"
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"分析 {row['Ticker']}。RSI {rsi_val:.2f}, 200MA。繁體中文策略。"
    try:
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(45) 
        return response.text.replace('\n', '<br>')
    except Exception as e: return f"AI 異常: {e}"

# ==========================================
# 4. HTML 報告 (包含完整總表與跳轉)
# ==========================================
def create_html_report(df):
    print(">>> [步驟 3] 正在生成完整欄位 HTML 報告...")
    
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; background: #f4f7f9; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 50px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); font-size: 12px; }}
            .summary-table th {{ background: #003366; color: white; padding: 10px; }}
            .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; }}
            .summary-table tr:hover {{ background: #f8fafc; }}
            .row-link {{ text-decoration: none; color: inherit; display: table-row; }}
            .ticker-btn {{ font-weight: bold; color: #003366; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; box-shadow: 0 6px 15px rgba(0,0,0,0.15); overflow: hidden; scroll-margin-top: 20px; }}
            .card-header {{ background: #003366; color: white; padding: 15px 25px; font-size: 20px; display: flex; justify-content: space-between; }}
            .chart-box {{ background: #1a1a1a; padding: 10px; text-align: center; }}
            .analysis-box {{ padding: 25px; line-height: 1.8; }}
            .back-btn {{ display: inline-block; margin-top: 15px; padding: 8px 15px; background: #cbd5e1; color: #334155; text-decoration: none; border-radius: 5px; font-size: 13px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container" id="top">
            <h1 style="text-align:center; color:#003366;">📊 美股 AI 深度掃描報告 {"(測試)" if TEST_MODE else ""}</h1>
            
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>代碼</th><th>公司</th><th>產業</th><th>市值</th><th>P/E</th><th>現價</th><th>漲幅</th><th>成交量</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # 填充總表內容 (點擊儲存格皆可跳轉)
    for _, row in df.iterrows():
        anchor = f"#{row['Ticker']}"
        html_header += f"""
        <tr onclick="window.location='{anchor}';" style="cursor:pointer;">
            <td class="ticker-btn">{row['Ticker']}</td>
            <td>{row['Company']}</td>
            <td>{row['Industry']}</td>
            <td>{row['MarketCap']}</td>
            <td>{row['PE']}</td>
            <td>${row['Price']}</td>
            <td style="color:red;">+{row['Change']}%</td>
            <td>{row['Volume']}</td>
        </tr>
        """
    
    html_header += "</tbody></table>"

    # 填充分析卡片
    cards = ""
    for _, row in df.iterrows():
        img_b64, rsi, is_above = generate_charts_base64(row['Ticker'])
        if img_b64:
            ai_text = get_ai_insight(row, rsi, is_above)
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header">
                    <span>{row['Ticker']} - {row['Company']}</span>
                    <span style="font-size: 14px;">Sector: {row['Sector']} | RSI: {rsi:.2f}</span>
                </div>
                <div class="chart-box"><img src="data:image/png;base64,{img_b64}" width="100%"></div>
                <div class="analysis-box">
                    <h3 style="margin-top:0; color:#003366;">🛡️ AI 策略師分析：</h3>
                    <p>{ai_text}</p>
                    <a href="#top" class="back-btn">⬆ 返回總表</a>
                </div>
            </div>
            """
    
    full_html = html_header + cards + "</div></body></html>"
    with open("report.html", "w", encoding="utf-8") as f: 
        f.write(full_html)
    print(f"✅ HTML 報告已產出至 report.html。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        create_html_report(df_stocks)
