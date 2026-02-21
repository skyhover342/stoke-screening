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
# 1. 核心參數設定
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']
MAX_CHANGE = 40.0  # 排除漲幅過大的標的

# ==========================================
# 2. 資料抓取與過濾 (Finviz)
# ==========================================
def fetch_and_filter_stocks():
    print("正在從 Finviz 抓取符合條件的股票...")
    # 條件：Price > 1, Vol > 500k, Rel Vol > 5, Price Up
    filters = "sh_price_o1,sh_curvol_o500,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', class_='table-light')
        
        data = []
        if table:
            rows = table.find_all('tr')[1:] # 跳過表頭
            for r in rows:
                tds = r.find_all('td')
                if len(tds) < 11: continue
                
                ticker = tds[1].text
                industry = tds[4].text
                change_val = float(tds[9].text.strip('%'))
                
                # 過濾：排除殼公司與漲幅超過 40%
                if any(x in industry for x in EXCLUDE_INDUSTRIES): continue
                if change_val > MAX_CHANGE: continue
                
                data.append({
                    "Ticker": ticker,
                    "Company": tds[2].text,
                    "Sector": tds[3].text,
                    "Industry": industry,
                    "Market Cap": tds[6].text,
                    "Price": float(tds[8].text),
                    "Change": change_val,
                    "Rel_Vol": float(tds[10].text),
                    "Volume": tds[11].text
                })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"抓取失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 繪圖引擎 (1Y Daily + 1m Intraday)
# ==========================================
def generate_charts(ticker):
    print(f"正在為 {ticker} 生成圖表...")
    # 抓取數據
    df_daily = yf.download(ticker, period="2y", interval="1d", progress=False)
    df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
    df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
    
    is_above_200 = df_daily['Close'].iloc[-1] > df_daily['200MA'].iloc[-1]
    
    # 建立雙子圖
    fig = make_subplots(
        rows=2, cols=2, 
        shared_xaxes=False,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker} 1Y Daily (Yellow=200MA)", f"{ticker} 1m Intraday", "Daily Vol", "1m Vol")
    )

    # 日線 K 線 + 200MA
    fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)

    # 1分線 K 線 (買賣力道變色)
    colors_1m = ['green' if c >= o else 'red' for o, c in zip(df_1m['Open'], df_1m['Close'])]
    fig.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), row=1, col=2)
    fig.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color=colors_1m), row=2, col=2)

    fig.update_layout(height=600, width=1100, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return io.BytesIO(img_bytes), is_above_200

# ==========================================
# 4. Gemini 3.1 Thinking AI 分析
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "未設定 API Key，無法生成 AI 分析。"
    
    print(f"正在調用 Gemini 3.1 Thinking 分析 {row['Ticker']}...")
    client = genai.Client(api_key=GEMINI_KEY)
    
    prompt = f"""
    作為專業美股研究專家，分析以下數據：
    股票：{row['Ticker']} ({row['Company']})
    行業：{row['Industry']}
    今日價格：{row['Price']}，漲幅：{row['Change']}%，相對成交量：{row['Rel_Vol']}
    趨勢：{'股價在 200MA 之上（多頭）' if is_above_200 else '股價在 200MA 之下（空頭）'}。
    請利用深度思考（Thinking）判斷：該股的量價共振是否具備持續性？明日贏面機率（0-100%）為何？並給出一個明確的操盤建議。
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            )
        )
        time.sleep(35) # 免費版 2 RPM 限制
        return response.text
    except Exception as e:
        return f"AI 分析發生錯誤: {e}"

# ==========================================
# 5. PDF 報告生成器
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'US Stock Momentum Report - {datetime.date.today()}', ln=True, align='C')
        self.ln(10)

def create_final_report(df):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 第一頁：數據清單
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Summary of Screened Stocks:", ln=True)
    pdf.ln(5)
    
    # 簡化版表格
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(30, 10, 'Ticker', 1)
    pdf.cell(50, 10, 'Industry', 1)
    pdf.cell(30, 10, 'Price', 1)
    pdf.cell(30, 10, 'Change%', 1)
    pdf.cell(30, 10, 'Rel Vol', 1)
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for _, row in df.iterrows():
        pdf.cell(30, 10, row['Ticker'], 1)
        pdf.cell(50, 10, row['Industry'][:20], 1)
        pdf.cell(30, 10, str(row['Price']), 1)
        pdf.cell(30, 10, f"{row['Change']}%", 1)
        pdf.cell(30, 10, str(row['Rel_Vol']), 1)
        pdf.ln()

    # 深度分析頁面
    for _, row in df.head(10).iterrows(): # 每天最多分析前 10 支，避免超時
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        ai_text = get_ai_insight(row, is_above_200)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Analysis: {ticker} - {row['Company']}", ln=True)
        
        # 插入圖表
        with open(f"temp_{ticker}.png", "wb") as f:
            f.write(img_buf.getbuffer())
        pdf.image(f"temp_{ticker}.png", x=10, y=30, w=190)
        
        # 插入 AI 分析文字
        pdf.set_y(150)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Gemini 3.1 Pro Thinking Insight:", ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, ai_text.replace('\u2022', '-')) # 處理特殊字元
        
        os.remove(f"temp_{ticker}.png") # 清理暫存檔

    pdf.output("report.pdf")
    print("PDF 報告生成完成！")

# ==========================================
# 主執行程序
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    stock_df = fetch_and_filter_stocks()
    
    if not stock_df.empty:
        print(f"找到 {len(stock_df)} 支符合條件的股票。開始深度研究...")
        create_final_report(stock_df)
    else:
        print("今日無符合條件的股票。")
    
    print(f"總執行時間: {round((time.time() - start_time)/60, 2)} 分鐘")
