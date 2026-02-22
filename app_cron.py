print(">>> [系統啟動] 正在初始化具備快捷導航功能的 HTML 報告環境...")

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
TEST_MODE = True  # 開發階段維持 True，不消耗 API

# ==========================================
# 2. 數據抓取與圖表生成
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
                    "Industry": tds[4].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        # 測試模式下只取 2 支，加快測試速度
        return df.head(2) if TEST_MODE else df.head(10)
    except: return pd.DataFrame()

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

# ==========================================
# 3. AI 分析 (具備 Mock 模擬模式)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if TEST_MODE:
        # 修正：移除誤植的引用標籤
        return f"【測試模式模擬文字】{row['Ticker']} 技術面分析。RSI 為 {rsi_val:.2f}，價格{'高於' if is_above_200 else '低於'} 200MA。建議根據支撐位進行操作。"

    if not GEMINI_KEY: return "無 API Key"
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"分析 {row['Ticker']}。RSI {rsi_val:.2f}, 200MA 趨勢。繁體中文策略。"
    try:
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(45) 
        return response.text.replace('\n', '<br>')
    except Exception as e: return f"AI 請求失敗: {e}"

# ==========================================
# 4. HTML 報告渲染 (加入導航功能)
# ==========================================
def create_html_report(df):
    print(">>> [步驟 3] 整合 HTML 報告與導航鏈結...")
    
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; background: #f4f7f9; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            #summary-table-top {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 50px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            #summary-table-top th {{ background: #003366; color: white; padding: 12px; text-align: center; }}
            #summary-table-top td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: center; }}
            #summary-table-top tr:hover {{ background: #f1f5f9; cursor: pointer; }}
            .ticker-link {{ color: #003366; text-decoration: none; font-weight: bold; display: block; width: 100%; height: 100%; }}
            .ticker-link:hover {{ color: #d93025; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; box-shadow: 0 6px 15px rgba(0,0,0,0.15); overflow: hidden; position: relative; scroll-margin-top: 20px; }}
            .card-header {{ background: #003366; color: white; padding: 15px 25px; font-size: 22px; display: flex; justify-content: space-between; }}
            .chart-box {{ background: #1a1a1a; padding: 10px; text-align: center; }}
            .chart-box img {{ max-width: 100%; border-radius: 5px; }}
            .analysis-box {{ padding: 25px; line-height: 1.8; }}
            .back-to-top {{ display: inline-block; margin-top: 15px; padding: 8px 15px; background: #e2e8f0; color: #475569; text-decoration: none; border-radius: 5px; font-size: 14px; font-weight: bold; transition: background 0.2s; }}
            .back-to-top:hover {{ background: #cbd5e1; }}
            .badge-test {{ background: #fbbf24; color: #78350f; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container" id="top">
            <h1 style="text-align:center; color:#003366;">📊 美股 AI 技術分析報告 {"<span class='badge-test'>TEST MODE</span>" if TEST_MODE else ""}</h1>
            <p style="text-align:right; color:#666;">生成日期：{datetime.date.today()}</p>
            
            <table id="summary-table-top">
                <thead>
                    <tr><th>代碼 (點擊跳轉)</th><th>產業</th><th>現價</th><th>漲幅</th></tr>
                </thead>
                <tbody>
    """
    
    summary_rows = ""
    for _, row in df.iterrows():
        summary_rows += f"""
        <tr>
            <td><a href="#{row['Ticker']}" class="ticker-link">{row['Ticker']}</a></td>
            <td><a href="#{row['Ticker']}" class="ticker-link">{row['Industry']}</a></td>
            <td><a href="#{row['Ticker']}" class="ticker-link">${row['Price']}</a></td>
            <td style="color:red;"><a href="#{row['Ticker']}" class="ticker-link">+{row['Change']}%</a></td>
        </tr>
        """
    
    html_middle = "</tbody></table>"

    stock_cards = ""
    for _, row in df.iterrows():
        img_b64, rsi, is_above = generate_charts_base64(row['Ticker'])
        if img_b64:
            ai_text = get_ai_insight(row, rsi, is_above)
            stock_cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header">
                    <span>{row['Ticker']} - {row['Company']}</span>
                    <span style="font-size: 16px;">RSI: {rsi:.2f}</span>
                </div>
                <div class="chart-box"><img src="data:image/png;base64,{img_b64}"></div>
                <div class="analysis-box">
                    <h3 style="margin-top:0; color:#003366;">🛡️ AI 策略師分析：</h3>
                    <p>{ai_text}</p>
                    <a href="#top" class="back-to-top">⬆ 返回總表</a>
                </div>
            </div>
            """
    
    html_footer = "</div></body></html>"
    
    full_html = html_header + summary_rows + html_middle + stock_cards + html_footer
    with open("report.html", "w", encoding="utf-8") as f: 
        f.write(full_html)
    print(f"✅ 導航版報告已生成 {'(測試模式)' if TEST_MODE else ''}。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        create_html_report(df_stocks)
