print(">>> [系統啟動] 正在初始化環境...")

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
from fpdf.enums import XPos, YPos
from google import genai
from google.genai import types

# ==========================================
# 1. 核心參數與 API Key 檢查
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-1.5-flash" # 最穩定的配額選擇

# 啟動時的安全檢查
if not GEMINI_KEY:
    print("❌ 關鍵錯誤：找不到 GEMINI_API_KEY 環境變數。")
else:
    print(f"✅ 環境變數檢查通過。Key 預覽: {GEMINI_KEY[:5]}... (長度: {len(GEMINI_KEY)})")

# ==========================================
# 2. 爬蟲與圖表邏輯 (維持穩定版本)
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 正在抓取 Finviz 數據...")
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        target_table = None
        for table in soup.find_all('table'):
            if "Ticker" in table.text[:200]:
                target_table = table
                break
        if not target_table: return pd.DataFrame()
        rows = target_table.find_all('tr', valign="top")
        data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 11: continue
            try:
                data.append({
                    "Ticker": tds[1].text.strip(), "Company": tds[2].text.strip(),
                    "Industry": tds[4].text.strip(), "Country": tds[5].text.strip(),
                    "MarketCap": tds[6].text.strip(), "PE": tds[7].text.strip(),
                    "Price": float(tds[8].text.strip()), "Change": float(tds[9].text.strip('%'))
                })
            except: continue
        df = pd.DataFrame(data)
        print(f"✅ 成功獲取 {len(df)} 支潛力股。")
        return df
    except Exception as e:
        print(f"❌ 爬蟲錯誤: {e}"); return pd.DataFrame()

def generate_charts(ticker):
    print(f">>> [圖表] 正在繪製 {ticker}...")
    try:
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df_daily.empty: return None, False
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)
        df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
        last_close = float(df_daily['Close'].iloc[-1])
        ma_val = df_daily['200MA'].iloc[-1]
        last_ma200 = float(ma_val) if not pd.isna(ma_val) else 0.0
        is_above_200 = last_close > last_ma200

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']))
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=1.5)))
        fig.update_layout(height=450, width=850, template="plotly_dark", xaxis_rangeslider_visible=False)
        return io.BytesIO(fig.to_image(format="png")), is_above_200
    except Exception as e:
        print(f"⚠️ {ticker} 圖表出錯: {e}"); return None, False

# ==========================================
# 3. AI 分析 (強化 429 容錯)
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "無 API KEY。"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "站上" if is_above_200 else "低於"
        prompt = f"分析美股 {row['Ticker']} ({row['Company']})。價格 {row['Price']}, 目前{status} 200日均線。請給出買賣分析與 1-100 的贏面分數。"
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(10) # 1.5 Flash 穩定的冷卻時間
        return response.text
    except Exception as e:
        print(f"⚠️ {row['Ticker']} AI 請求失敗: {e}")
        return "AI 分析額度暫時耗盡，請依據圖表判斷。"

# ==========================================
# 4. PDF 報告 (優先生成總表)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, text=f'Stock Analysis Report - {datetime.date.today()}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

def create_report(df):
    print(">>> [步驟 3] 正在整合 PDF 報告...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 第一頁：總表 (這部分絕對會被執行)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, text="Market Potential Stocks Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font('helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    headers = ['Ticker', 'Price', 'Change%', 'Industry']
    widths = [30, 30, 30, 100]
    for h, w in zip(headers, widths): pdf.cell(w, 10, text=h, border=1, fill=True)
    pdf.ln()
    pdf.set_font('helvetica', '', 9)
    for _, row in df.iterrows():
        pdf.cell(30, 10, text=str(row['Ticker']), border=1)
        pdf.cell(30, 10, text=str(row['Price']), border=1)
        pdf.cell(30, 10, text=f"{row['Change']}%", border=1)
        pdf.cell(100, 10, text=str(row['Industry'][:35]), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # 個股分析 (前 10 支)
    for i, (index, row) in enumerate(df.head(10).iterrows()):
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        if img_buf:
            ai_text = get_ai_insight(row, is_above_200)
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, text=f"{ticker} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            img_path = f"tmp_{ticker}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, w=185)
            pdf.ln(5)
            pdf.set_font('helvetica', 'B', 11); pdf.cell(0, 8, text="Analysis:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 10)
            clean_text = ai_text.replace('\u2022', '-').encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 6, text=clean_text)
            os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 最終報告已生成：report.pdf")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        create_report(df_stocks)
    else:
        print("今日無符合標的。")
