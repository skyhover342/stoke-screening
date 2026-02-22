print(">>> [系統啟動] 正在初始化 1 分鐘線爆量雷達報告環境...")

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
    print(">>> [步驟 1] 抓取 9 大欄位數據...")
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
    except Exception as e:
        print(f"❌ 抓取失敗: {e}"); return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (日線 + 1分線爆量提醒)
# ==========================================
def generate_stock_images(ticker):
    print(f">>> [分析] 處理 {ticker} 1分鐘線雷達...")
    try:
        # --- 1. 一年日線圖 (高度再壓縮) ---
        df_1y = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df_1y.empty: return None, None, 0, False
        if isinstance(df_1y.columns, pd.MultiIndex): df_1y.columns = df_1y.columns.get_level_values(0)
        
        df_1y['200MA'] = df_1y['Close'].rolling(window=200).mean()
        delta = df_1y['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df_1y['RSI'] = 100 - (100 / (1 + gain/loss))

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        fig1.add_trace(go.Candlestick(x=df_1y.index, open=df_1y['Open'], high=df_1y['High'], low=df_1y['Low'], close=df_1y['Close']), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['200MA'], line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig1.add_trace(go.Scatter(x=df_1y.index, y=df_1y['RSI'], line=dict(color='cyan', width=1)), row=2, col=1)
        fig1.update_layout(height=300, width=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        
        # --- 2. 當日 1 分鐘圖 (1m) + 爆量提醒 ---
        # 改為 1m 間隔 
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        fig2_b64 = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            # 爆量判定：當前分鐘量 > 過去 5 根均量 3 倍 
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(5).mean()
            df_1m['Spike'] = df_1m['Volume'] > (df_1m['Vol_Avg'] * 3)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']))
            
            # 加入 1 分鐘爆量箭頭
            for idx, row in df_1m[df_1m['Spike']].iterrows():
                # 買入/拋售方向判定 
                color = "lime" if row['Close'] > row['Open'] else "red"
                symbol = "▲ BUY" if row['Close'] > row['Open'] else "▼ SELL"
                fig2.add_annotation(x=idx, y=row['High'], text=symbol, showarrow=True, arrowhead=1, font=dict(size=9), color=color, bgcolor="black", opacity=0.8)

            fig2.update_layout(height=250, width=800, template="plotly_dark", xaxis_rangeslider_visible=False, title=dict(text=f"{ticker} 1m Spike Radar", font=dict(size=12)), margin=dict(l=10, r=10, t=30, b=10))
            fig2_b64 = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        img1_b64 = base64.b64encode(fig1.to_image(format="png")).decode('utf-8')
        return img1_b64, fig2_b64, float(df_1y['RSI'].iloc[-1]), bool(df_1y['Close'].iloc[-1] > df_1y['200MA'].iloc[-1])
    except Exception as e:
        print(f"⚠️ {ticker} 繪圖異常: {e}"); return None, None, 0, False

def get_ai_insight(row, rsi_val, is_above_200):
    if TEST_MODE: return f"【測試】{row['Ticker']} 1分鐘線顯示成交量異動。長線趨勢{'站上' if is_above_200 else '低於'} MA200。" [cite: 1-905]
    # AI 邏輯維持原樣 

# ==========================================
# 4. HTML 渲染 (美化佈局)
# ==========================================
def create_html_report(df):
    html_header = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 30px; font-size: 11px; }}
            .summary-table th {{ background: #003366; color: white; padding: 8px; position: sticky; top: 0; }}
            .summary-table td {{ border-bottom: 1px solid #ddd; text-align: center; padding: 6px; cursor: pointer; }}
            .summary-table tr:hover {{ background: #eef2f7; }}
            .stock-card {{ background: white; border-radius: 8px; margin-bottom: 40px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); scroll-margin-top: 10px; }}
            .card-header {{ border-bottom: 2px solid #003366; padding-bottom: 5px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
            .chart-container {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; background: #1a1a1a; padding: 8px; border-radius: 4px; }}
            .chart-container img {{ max-width: 480px; height: auto; border: 1px solid #444; }}
            .analysis-text {{ margin-top: 10px; line-height: 1.5; background: #f9f9f9; padding: 12px; border-radius: 4px; font-size: 13px; }}
            .back-btn {{ background: #003366; color: white; text-decoration: none; padding: 4px 8px; border-radius: 3px; font-size: 11px; }}
        </style>
    </head>
    <body>
        <div class="container" id="top">
            <h1 style="color:#003366; text-align:center; font-size:24px;">📈 1分鐘線爆量追蹤報告 (TEST MODE)</h1>
            <table class="summary-table">
                <thead><tr><th>代碼</th><th>公司</th><th>板塊</th><th>產業</th><th>市值</th><th>P/E</th><th>現價</th><th>漲幅</th><th>成交量</th></tr></thead>
                <tbody>
    """
    for _, row in df.iterrows():
        html_header += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Sector']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    html_header += "</tbody></table>"

    cards = ""
    for _, row in df.iterrows():
        img1, img2, rsi, is_above = generate_stock_images(row['Ticker'])
        if img1:
            ai_text = get_ai_insight(row, rsi, is_above)
            cards += f"""
            <div class="stock-card" id="{row['Ticker']}">
                <div class="card-header">
                    <span style="font-size:18px; font-weight:bold;">{row['Ticker']} - {row['Company']}</span>
                    <a href="#top" class="back-btn">⬆ 返回總表</a>
                </div>
                <div class="chart-container">
                    <div><small style="color:white; font-size:10px;">1Y Daily (MA200 & RSI)</small><br><img src="data:image/png;base64,{img1}"></div>
                    <div><small style="color:white; font-size:10px;">Intraday 1m Spike Radar</small><br><img src="data:image/png;base64,{img2}"></div>
                </div>
                <div class="analysis-text">
                    <strong>🛡️ AI 策略師分析：</strong> {ai_text}
                </div>
            </div>
            """
    
    with open("report.html", "w", encoding="utf-8") as f: f.write(html_header + cards + "</div></body></html>")
    print("✅ 1分鐘線雙圖表報告已產出。")

if __name__ == "__main__":
    df = fetch_and_filter_stocks()
    if not df.empty: create_html_report(df)
