print(">>> [系統檢查] 程式啟動中...") # 這是為了確認程式有沒有跑起來

import os
import time
import datetime
import io
import sys

# 強制排除多線程衝突
os.environ["YF_NO_TICKER_CACHE"] = "1" 

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google import genai
from google.genai import types

print(">>> [系統檢查] 模組導入完成。")

# ==========================================
# 1. 核心參數
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-pro"
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']

# ==========================================
# 2. 強韌版 Finviz 爬蟲
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 正在讀取 Finviz 數據...")
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
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
                    "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%'))
                })
            except: continue
        
        df = pd.DataFrame(data)
        df = df[~df['Industry'].isin(EXCLUDE_INDUSTRIES)]
        print(f"✅ 成功獲取 {len(df)} 筆標的。")
        return df
    except Exception as e:
        print(f"❌ 爬蟲出錯: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (徹底修復 NumPy 警告與 Kaleido 卡死)
# ==========================================
def generate_charts(ticker):
    print(f">>> [步驟 2] 處理 {ticker} 圖表...")
    try:
        # 下載資料 (關閉線程避免卡死)
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False, threads=False)
        if df_daily.empty: return None, False

        # 修正 yfinance 的 MultiIndex 欄位問題
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)

        df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
        
        # --- 核心矯正：使用 .item() 解決 NumPy 警告 ---
        # 將 Series 的最後一筆轉為純 Python float
        last_close = float(df_daily['Close'].iloc[-1])
        last_ma200 = float(df_daily['200MA'].iloc[-1]) if not pd.isna(df_daily['200MA'].iloc[-1]) else 0
        
        is_above_200 = last_close > last_ma200

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], 
                                     low=df_daily['Low'], close=df_daily['Close']))
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)))
        
        fig.update_layout(height=400, width=800, template="plotly_dark", xaxis_rangeslider_visible=False)
        
        # 這裡不使用 engine 參數，讓系統自動處理
        img_bytes = fig.to_image(format="png")
        return io.BytesIO(img_bytes), is_above_200
    except Exception as e:
        print(f"⚠️ {ticker} 圖表失敗: {e}")
        return None, False

# ==========================================
# 4. AI 分析
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "No API Key"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "站上" if is_above_200 else "低於"
        prompt = f"分析美股 {row['Ticker']} ({row['Company']})，價格 {row['Price']}，目前{status} 200MA。請給出贏面分數(0-100)與買賣建議。"
        
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(30) # 降階後的 Pro 版冷卻時間
        return response.text
    except Exception as e:
        return f"AI 分析出錯: {e}"

# ==========================================
# 5. PDF 報告 (新版語法)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text='US Stock AI Report', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def create_report(df):
    print(">>> [步驟 3] 正在生成 PDF 報告...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 摘要表格
    pdf.set_font('helvetica', 'B', 10)
    for col in ['Ticker', 'Price', 'Change%', 'Mkt Cap']:
        pdf.cell(45, 10, text=col, border=1)
    pdf.ln()

    pdf.set_font('helvetica', '', 9)
    for _, row in df.head(10).iterrows(): # 限制數量確保穩定
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        
        ai_text = get_ai_insight(row, is_above_200) if img_buf else "No Data"
        
        # 寫入簡要資訊
        pdf.cell(45, 10, text=str(ticker), border=1)
        pdf.cell(45, 10, text=str(row['Price']), border=1)
        pdf.cell(45, 10, text=f"{row['Change']}%", border=1)
        pdf.cell(45, 10, text=str(row['MarketCap']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        if img_buf:
            img_path = f"tmp_{ticker}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, w=170)
            pdf.ln(2)
            pdf.set_font('helvetica', '', 8)
            pdf.multi_cell(0, 5, text=ai_text.encode('latin-1', 'ignore').decode('latin-1'))
            pdf.ln(5)
            os.remove(img_path)
            pdf.add_page()

    pdf.output("report.pdf")
    print("✅ 報告已儲存。")

if __name__ == "__main__":
    print(">>> [主程式] 啟動成功。")
    stocks_df = fetch_and_filter_stocks()
    if not stocks_df.empty:
        create_report(stocks_df)
    else:
        print("今日無標的。")
