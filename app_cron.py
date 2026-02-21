import os
import time
import datetime
import io
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from google import genai
from google.genai import types

# ==========================================
# 1. 核心參數
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']
MAX_CHANGE = 40.0

# ==========================================
# 2. 簡約版爬蟲 (僅抓取清單，不解析 RelVol)
# ==========================================
def fetch_and_filter_stocks():
    print("正在從 Finviz 抓取已篩選標的...")
    # 使用 Overview 視圖 (v=111)
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://finviz.com/'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 抓取資料列 (screener-body-table-nw 是一般資料列的 class)
        rows = soup.find_all('tr', class_='screener-body-table-nw')
        if not rows:
            # 備用方案：尋找表格中的 tr
            table = soup.find('table', class_='table-light')
            rows = table.find_all('tr')[1:] if table else []

        print(f"找到原始資料: {len(rows)} 筆")
        
        data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 10: continue
            
            try:
                # 只抓取我們需要的穩定欄位
                ticker = tds[1].text.strip()
                company = tds[2].text.strip()
                industry = tds[4].text.strip()
                price = tds[8].text.strip()
                change_str = tds[9].text.strip().replace('%', '')
                change_val = float(change_str)
                
                # 執行排除邏輯 (殼公司與漲幅 > 40%)
                if any(x in industry for x in EXCLUDE_INDUSTRIES): continue
                if change_val > MAX_CHANGE: continue
                
                data.append({
                    "Ticker": ticker,
                    "Company": company,
                    "Industry": industry,
                    "Price": price,
                    "Change": change_val
                })
            except:
                continue
                
        df = pd.DataFrame(data)
        print(f"過濾完成，最終入榜數量: {len(df)}")
        return df

    except Exception as e:
        print(f"爬蟲異常: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (200MA + 1分線)
# ==========================================
def generate_charts(ticker):
    print(f"正在分析技術面: {ticker}")
    df_daily = yf.download(ticker, period="2y", interval="1d", progress=False)
    df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
    df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
    
    # 判斷是否在 200MA 之上
    last_price = df_daily['Close'].iloc[-1]
    last_ma200 = df_daily['200MA'].iloc[-1]
    is_above_200 = last_price > last_ma200
    
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, row_heights=[0.7, 0.3],
                        subplot_titles=("1Y Daily + 200MA", "Today 1m Candle", "Daily Volume", "1m Buy/Sell Force"))
    
    # 日線
    fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)
    
    # 1分線
    colors = ['green' if c >= o else 'red' for o, c in zip(df_1m['Open'], df_1m['Close'])]
    fig.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), row=1, col=2)
    fig.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color=colors), row=2, col=2)
    
    fig.update_layout(height=600, width=1100, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return io.BytesIO(img_bytes), is_above_200

# ==========================================
# 4. Gemini 3.1 Thinking AI 分析 (核心)
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "未偵測到 API Key。"
    client = genai.Client(api_key=GEMINI_KEY)
    
    # 指令加入思考型分析
    prompt = f"""
    作為量化分析師，請分析以下爆量股（Relative Volume > 5）：
    股票：{row['Ticker']} ({row['Company']})，價格：{row['Price']}，今日漲幅：{row['Change']}%。
    長線趨勢：{'已站上 200MA (多頭趨勢)' if is_above_200 else '低於 200MA (空頭反彈)'}。
    請利用你的『思考（Thinking）』功能，結合價格位置判斷這是否為具備持續性的真實突破？
    最後給出贏面機率（0-100%）與具體的操作風險提示。
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            )
        )
        time.sleep(35) # 遵守免費版限制
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ==========================================
# 5. PDF 專業報告 (符合 FPDF 2.7.8+ 規範)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text=f'Stock Momentum Report - {datetime.date.today()}', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

def create_report(df):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 第一頁：摘要清單
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, text="Market Scan Summary (High Relative Volume):", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 摘要表格
    pdf.set_font('helvetica', 'B', 10)
    headers = ['Ticker', 'Industry', 'Price', 'Change%']
    widths = [40, 70, 40, 40]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 10, text=h, border=1)
    pdf.ln()

    pdf.set_font('helvetica', '', 10)
    for _, row in df.iterrows():
        pdf.cell(widths[0], 10, text=str(row['Ticker']), border=1)
        pdf.cell(widths[1], 10, text=str(row['Industry'][:20]), border=1)
        pdf.cell(widths[2], 10, text=str(row['Price']), border=1)
        pdf.cell(widths[3], 10, text=f"{row['Change']}%", border=1, new_x="LMARGIN", new_y="NEXT")

    # 分頁詳細分析 (最多前 10 支)
    for _, row in df.head(10).iterrows():
        ticker = row['Ticker']
        try:
            img_buf, is_above_200 = generate_charts(ticker)
            ai_text = get_ai_insight(row, is_above_200)
            
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, text=f"Analysis: {ticker} - {row['Company']}", new_x="LMARGIN", new_y="NEXT")
            
            # 插入圖表
            img_path = f"temp_{ticker}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, x=10, y=30, w=190)
            
            # 插入 AI 分析
            pdf.set_y(150)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, text="Gemini 3.1 Pro Thinking Insight:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 10)
            pdf.multi_cell(0, 6, text=ai_text.replace('\u2022', '-'))
            
            os.remove(img_path)
        except Exception as e:
            print(f"跳過 {ticker} 分析: {e}")

    pdf.output("report.pdf")

# ==========================================
# 6. 主執行流程
# ==========================================
if __name__ == "__main__":
    df = fetch_and_filter_stocks()
    if not df.empty:
        print(f"篩選到 {len(df)} 支標的，開始生成深度報告...")
        create_report(df)
    else:
        print("今日無符合條件標的，生成空白報告。")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text="No stocks matched the criteria today.", align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.output("report.pdf")
