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
# 2. 智慧型爬蟲 (自動偵測欄位)
# ==========================================
def fetch_and_filter_stocks():
    print("正在執行『智慧型』爬蟲邏輯...")
    # 使用你提供的精準篩選 URL
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://finviz.com/'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找所有 tr 標籤
        rows = soup.find_all('tr', class_='screener-body-table-nw')
        if not rows:
            table = soup.find('table', class_='table-light')
            rows = table.find_all('tr')[1:] if table else []

        print(f"成功連線 Finviz，找到原始資料: {len(rows)} 行")
        
        final_data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 10: continue
            
            try:
                # 1. 提取 Ticker (通常在第 2 欄)
                ticker = tds[1].text.strip()
                # 2. 提取產業 (通常在第 5 欄)
                industry = tds[4].text.strip()
                
                # --- 智慧搜尋數值欄位 ---
                price = 0.0
                change_val = 0.0
                
                for td in tds:
                    txt = td.text.strip()
                    # 尋找漲幅 (帶有 % 的)
                    if '%' in txt:
                        change_val = float(txt.replace('%', ''))
                    # 尋找價格 (純數字且點後兩位的)
                    elif '.' in txt and len(txt.split('.')[-1]) >= 2:
                        try:
                            price = float(txt.replace(',', ''))
                        except: pass

                # 排除邏輯
                if any(x in industry for x in EXCLUDE_INDUSTRIES): continue
                if change_val > MAX_CHANGE: continue
                
                # 既然 Finviz 已經過濾過 RelVol > 5，我們就標註為 5+
                final_data.append({
                    "Ticker": ticker,
                    "Company": tds[2].text.strip(),
                    "Industry": industry,
                    "Price": price,
                    "Change": change_val,
                    "Rel_Vol": "5+", # 視圖 v=111 看不到具體數值，但篩選已生效
                    "Sector": tds[3].text.strip()
                })
            except:
                continue
                
        df = pd.DataFrame(final_data)
        print(f"✅ 過濾完成，成功入榜數量: {len(df)}")
        return df

    except Exception as e:
        print(f"爬蟲異常: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成
# ==========================================
def generate_charts(ticker):
    print(f"繪製圖表: {ticker}")
    df_daily = yf.download(ticker, period="2y", interval="1d", progress=False)
    df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
    df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
    
    is_above_200 = df_daily['Close'].iloc[-1] > df_daily['200MA'].iloc[-1]
    
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, row_heights=[0.7, 0.3],
                        subplot_titles=(f"1Y Daily + 200MA", f"Today 1m Candle", "Daily Vol", "1m Vol Force"))
    
    # 日線
    fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)
    
    # 1分線 (顏色力道)
    colors = ['green' if c >= o else 'red' for o, c in zip(df_1m['Open'], df_1m['Close'])]
    fig.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), row=1, col=2)
    fig.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color=colors), row=2, col=2)
    
    fig.update_layout(height=600, width=1100, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return io.BytesIO(img_bytes), is_above_200

# ==========================================
# 4. Gemini 3.1 Thinking
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "API Key Missing."
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"分析 {row['Ticker']}：價格 {row['Price']}，漲幅 {row['Change']}%。趨勢：{'多頭' if is_above_200 else '空頭'}。請思考該股是否為機構掃貨？給出贏面分數。"
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview", contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH"))
        )
        time.sleep(35) # 避開 2 RPM 限制
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# ==========================================
# 5. PDF 報告 (完整 FPDF 2.7.8+ 修復版)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text=f'US Momentum AI Report - {datetime.date.today()}', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

def create_final_report(df):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 摘要頁
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, text="Stock Selection Summary (RelVol > 5):", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 表格
    pdf.set_font('helvetica', 'B', 10)
    col_widths = [30, 60, 30, 30, 30]
    headers = ['Ticker', 'Industry', 'Price', 'Change%', 'RelVol']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, text=h, border=1)
    pdf.ln()

    pdf.set_font('helvetica', '', 9)
    for _, row in df.iterrows():
        pdf.cell(30, 10, text=str(row['Ticker']), border=1)
        pdf.cell(60, 10, text=str(row['Industry'][:20]), border=1)
        pdf.cell(30, 10, text=str(row['Price']), border=1)
        pdf.cell(30, 10, text=f"{row['Change']}%", border=1)
        pdf.cell(30, 10, text=str(row['Rel_Vol']), border=1, new_x="LMARGIN", new_y="NEXT")

    # 每股詳情 (前 10 支)
    for _, row in df.head(10).iterrows():
        ticker = row['Ticker']
        try:
            img_buf, is_above_200 = generate_charts(ticker)
            ai_text = get_ai_insight(row, is_above_200)
            
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, text=f"Deep Dive: {ticker} - {row['Company']}", new_x="LMARGIN", new_y="NEXT")
            
            img_path = f"temp_{ticker}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, x=10, y=30, w=190)
            
            pdf.set_y(150)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, text="Gemini 3.1 Pro Thinking Insight:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('helvetica', '', 10)
            pdf.multi_cell(0, 6, text=ai_text.replace('\u2022', '-'))
            os.remove(img_path)
        except Exception as e:
            print(f"跳過 {ticker}，原因: {e}")

    pdf.output("report.pdf")

# ==========================================
# 6. 主程式執行
# ==========================================
if __name__ == "__main__":
    stock_df = fetch_and_filter_stocks()
    if not stock_df.empty:
        print(f"成功入榜 {len(stock_df)} 支標的，生成 PDF...")
        create_final_report(stock_df)
    else:
        print("今日無標的，生成空白 PDF 以供下載...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text="No matches today based on filters.", align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.output("report.pdf")
