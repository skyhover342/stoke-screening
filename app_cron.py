print(">>> [系統啟動] 正在執行 PDF 排版優化與亂碼修正...")

import os
import time
import datetime
import io
import re # 用於過濾 Markdown
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google import genai
from google.genai import types

# ==========================================
# 1. 核心參數
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash" 
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']

# --- 自我矯正：Markdown 淨化函數 ---
def clean_ai_text(text):
    # 移除 ** (粗體符號)
    text = text.replace('**', '')
    # 移除 ### (標題符號)
    text = re.sub(r'#+', '', text)
    # 移除多餘的 *
    text = text.replace('* ', '- ')
    # 處理特殊點點
    text = text.replace('\u2022', '-')
    return text

# ==========================================
# 2. 爬蟲與技術指標計算 (維持 RSI 邏輯)
# ==========================================
def fetch_and_filter_stocks():
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
                    "Country": tds[5].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        return df[~df['Industry'].isin(EXCLUDE_INDUSTRIES)]
    except: return pd.DataFrame()

def generate_charts(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty or len(df) < 30: return None, 0, False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df['200MA'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))

        last_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        is_above_200 = float(df['Close'].iloc[-1]) > (float(df['200MA'].iloc[-1]) if not pd.isna(df['200MA'].iloc[-1]) else 0)

        # 建立三層圖表
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                            row_heights=[0.5, 0.15, 0.35],
                            subplot_titles=(f"{ticker} Daily", "Vol", "RSI"))
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['200MA'], line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=650, width=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        return io.BytesIO(fig.to_image(format="png")), last_rsi, is_above_200
    except: return None, 0, False

# ==========================================
# 3. AI 分析 (請求英文回覆以解決中文亂碼)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if not GEMINI_KEY: return "No API Key."
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "Above" if is_above_200 else "Below"
        # 矯正：強制 AI 使用英文回覆，徹底解決中文亂碼問題
        prompt = f"""
        Analyze US Stock: {row['Ticker']} ({row['Company']}). 
        Price: {row['Price']}, Change: {row['Change']}%, RSI: {rsi_val:.2f}, {status} 200MA.
        Provide: 1. Technical Summary 2. Winning Score (1-100) 3. Trade Strategy (Buy/Sell/Wait).
        Format the response cleanly. Response must be in ENGLISH.
        """
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(12)
        return clean_ai_text(response.text)
    except Exception as e: return f"AI Error: {e}"

# ==========================================
# 4. PDF 生成 (優化排版)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text=f'Professional Stock AI Report - {datetime.date.today()}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

def create_report(df):
    print(">>> [步驟 3] 正在生成 PDF...")
    pdf = PDFReport(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 總表
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_fill_color(220, 220, 220)
    headers = ['Ticker', 'Company', 'Industry', 'Mkt Cap', 'P/E', 'Price', 'Change%', 'Volume']
    widths = [20, 50, 50, 30, 20, 20, 20, 40]
    for h, w in zip(headers, widths): pdf.cell(w, 10, text=h, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('helvetica', '', 8)
    for _, row in df.iterrows():
        pdf.cell(20, 8, text=str(row['Ticker']), border=1)
        pdf.cell(50, 8, text=str(row['Company'][:28]), border=1)
        pdf.cell(50, 8, text=str(row['Industry'][:32]), border=1)
        pdf.cell(30, 8, text=str(row['MarketCap']), border=1)
        pdf.cell(20, 8, text=str(row['PE']), border=1)
        pdf.cell(20, 8, text=str(row['Price']), border=1)
        pdf.cell(20, 8, text=f"{row['Change']}%", border=1)
        pdf.cell(40, 8, text=str(row['Volume']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 詳情頁
    for i, (index, row) in enumerate(df.head(10).iterrows()):
        img_buf, rsi_val, is_above_200 = generate_charts(row['Ticker'])
        if img_buf:
            ai_text = get_ai_insight(row, rsi_val, is_above_200)
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, text=f"{row['Ticker']} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            img_path = f"tmp_{row['Ticker']}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            # 調整圖表大小：寬度稍微縮小，置中
            pdf.image(img_path, x=58, y=30, w=180) 
            
            # --- 關鍵修正：將文字起始位置下移，避免與三層圖表重疊 ---
            pdf.set_y(165) 
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 8, text="AI Strategist Professional Analysis:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 10)
            
            # 使用拉丁編碼過濾，確保不崩潰
            final_text = ai_text.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 5, text=final_text)
            os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 任務圓滿完成！報告已儲存。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty: create_report(df_stocks)
    else: print("No stocks found.")
