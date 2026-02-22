# 版本號碼：v1.1.4
print(">>> [系統啟動] 正在執行 v1.1.4 版本：強化懸停變色與完整導航功能...")

import os, time, datetime, io, base64, requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from google import genai
except ImportError:
    print("❌ 錯誤：找不到 google-genai 套件。請在 GitHub Action 的 .yml 中加入 google-genai 安裝。")

# ==========================================
# 1. 核心參數與測試開關
# ==========================================
VERSION = "v1.1.4"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"

# TEST_MODE = True: 模擬 AI，測試排版與跳轉功能。
# TEST_MODE = False: 正式發送指令給 Gemini API。
TEST_MODE = True 

# ==========================================
# 2. 數據抓取 (9 大完整欄位)
# ==========================================
def fetch_and_filter_stocks():
    print(f">>> [步驟 1] 抓取數據 ({VERSION})...")
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
# 3. 專業繪圖 (量價疊加與三均線)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 繪製 {ticker} 雙時區圖表...")
    try:
        # --- 日線：抓 2 年數據計算 SMA ---
        df_all = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        df_all['SMA20'] = df_all['Close'].rolling(20).mean()
        df_all['SMA50'] = df_all['Close'].rolling(50).mean()
        df_all['SMA200'] = df_all['Close'].rolling(200).mean()
        df_1y = df_all.tail(252)

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                            row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        # 亮灰色疊加成交量
        fig1.add_trace(go.Bar(x=df_1y.index, y=df_1y['Volume'], marker_color='rgba(220, 220, 220, 0.7)', name="Vol"), row=1, col=1, secondary_y=True)
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close'], name="K"), row=1, col=1, secondary_y=False)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA20'], line=dict(color='cyan', width=1.2), name="SMA20"))
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA50'], line=dict(color='orange', width=1.5), name="SMA50"))
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['SMA200'], line=dict(color='yellow', width=2), name="SMA200"))
        
        fig1.update_yaxes(range=[0, df_1y['Volume'].max()*4], secondary_y=True, showgrid=False)
        fig1.update_layout(height=480, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))

        # --- 1分鐘線：爆量雷達 ---
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            df_1m['Spike'] = df_1m['Volume'] > (df_1m['Vol_Avg'] * 3)
            
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color='rgba(220, 220, 220, 0.7)'), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            
            for idx, row in df_1m[df_1m['Spike']].iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, font=dict(size=10, color=t_color), 
                                    arrowcolor=t_color, bgcolor="black", opacity=0.8, yshift=10)

            fig2.update_yaxes(range=[0, df_1m['Volume'].max()*4], secondary_y=True, showgrid=False)
            fig2.update_layout(height=380, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, bool(df_1y['Close'].iloc[-1] > df_1y['SMA200'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖失敗: {e}"); return None, None, False

# ==========================================
# 4. AI 分析模組
# ==========================================
def get_ai_insight(row, is_above_200):
    status = "站上" if is_above_200 else "低於"
    prompt = f"分析美股 {row['Ticker']}。價格 {row['Price']}, 今日漲幅 {row['Change']}%, 目前{status} SMA200。請以繁體中文提供：1.技術總結 2.日內異動建議 3.贏面分數(1-100)。"

    if TEST_MODE:
        return f"<p style='color:#666; font-size:12px;'>[AI 指令預覽]：<br>{prompt}</p>"

    if not GEMINI_KEY: return "❌ 未設定 API KEY"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(45) 
        return response.text.replace('\n', '<br>')
    except Exception as e: return f"⚠️ AI 分析失敗: {e}"

# ==========================================
# 5. HTML 渲染與導航邏輯
# ==========================================
def create_html_report(df):
    print(f">>> [步驟 3] 整合 HTML 報告與導航跳轉...")
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; background: #f0f2f5; padding: 20px; }}
        .container {{ max-width: 1050px; margin: 0 auto; }}
        .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 50px; font-size: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
        .summary-table th {{ background: #003366; color: white; padding: 12px; }}
        .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; cursor: pointer; }}
        
        /* 懸停效果：變色並顯示手指游標 */
        .summary-table tr:hover {{ background: #d1dce5; cursor: pointer; }}
        
        .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; padding: 25px; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
        .card-header {{ background: #003366; color: white; padding: 15px 20px; border-radius: 8px 8px 0 0; display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .chart-stack {{ display: flex; flex-direction: column; gap: 20px; align-items: center; background: #1a1a1a; padding: 20px; border-radius: 8px; }}
        .chart-stack img {{ width: 100%; max-width: 980px; height: auto; }}
        .analysis-box {{ margin-top: 20px; line-height: 1.8; background: #f8fafc; padding: 20px; border-radius: 6px; font-size: 14px; border-left: 5px solid #003366; }}
        .back-btn {{ background: white; color: #003366; text-decoration: none; padding: 5px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
    </style></head>
    <body><div class="container" id="top">
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 深度研究週報 {VERSION}</h1>
        <table class="summary-table">
            <thead><tr><th>代碼</th><th>公司</th><th>板塊</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead>
            <tbody>
    """
    
    # 生成總表列 (含 onclick 跳轉)
    for _, row in df.iterrows():
        html_header += f"""<tr onclick="window.location='#{row['Ticker']}';">
            <td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Sector']}</td>
            <td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td>
            <td>${row['Price']}</td><td style="color:red;">+{row['Change']}%</td><td>{row['Volume']}</td>
        </tr>"""
    
    html_header += "</tbody></table>"

    # 生成分析卡片 (含返回總表按鈕)
    cards = ""
    for _, row in df.iterrows():
        img1, img2, is_above = generate_stock_images(row['Ticker'])
        if img1:
            ai_text = get_ai_insight(row, is_above)
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header">
                    <span>{row['Ticker']} - {row['Company']}</span>
                    <a href="#top" class="back-btn">⬆ 返回總表</a>
                </div>
                <div class="chart-stack">
                    <img src="data:image/png;base64,{img1}">
                    <img src="data:image/png;base64,{img2}">
                </div>
                <div class="analysis-box"><strong>🛡️ AI 策略師分析：</strong><br>{ai_text}</div>
            </div>"""
    
    with open("report.html", "w", encoding="utf-8") as f: f.write(html_header + cards + "</div></body></html>")
    print(f"✅ 報告已產出 {VERSION}")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty: create_html_report(df_stocks)
