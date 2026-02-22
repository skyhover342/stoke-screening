print(">>> [系統啟動] 正在加載專業技術指標模組...")

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
# 1. 核心參數
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-flash-latest" 
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']

# ==========================================
# 2. 強韌版 Finviz 爬蟲
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 抓取 Finviz 10 大核心欄位...")
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
                    "Sector": tds[3].text.strip(), "Industry": tds[4].text.strip(),
                    "Country": tds[5].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        df = df[~df['Industry'].isin(EXCLUDE_INDUSTRIES)]
        print(f"✅ 成功獲取 {len(df)} 支標的資料。")
        return df
    except Exception as e:
        print(f"❌ 爬蟲出錯: {e}"); return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (新增 RSI 技術指標)
# ==========================================
def generate_charts(ticker):
    print(f">>> [技術分析] 正在運算 {ticker} 的 RSI 與 MA...")
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty or len(df) < 30: return None, 0, False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 計算 200MA
        df['200MA'] = df['Close'].rolling(window=200).mean()

        # --- 自我矯正：計算 RSI ---
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 取得最新標量數值
        last_close = float(df['Close'].iloc[-1])
        last_ma200 = float(df['200MA'].iloc[-1]) if not pd.isna(df['200MA'].iloc[-1]) else 0.0
        last_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        is_above_200 = last_close > last_ma200

        # 建立三層圖表
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3],
                            subplot_titles=(f"{ticker} Daily + 200MA", "Volume", "RSI (14)"))

        # 1. K線 + 200MA
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['200MA'], line=dict(color='yellow', width=2), name="200MA"), row=1, col=1)

        # 2. 成交量
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color='gray'), row=2, col=1)

        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2), name="RSI"), row=3, col=1)
        # 加入 RSI 超買超賣水平線
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        return io.BytesIO(fig.to_image(format="png")), last_rsi, is_above_200
    except Exception as e:
        print(f"⚠️ {ticker} 運算失敗: {e}"); return None, 0, False

# ==========================================
# 4. AI 分析 (納入 RSI 數據)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if not GEMINI_KEY: return "無 API KEY。"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "站上" if is_above_200 else "低於"
        # 這裡將 RSI 告訴 AI，讓它判斷強弱
        prompt = f"""
        請以資深股票研究專家身份分析：
        標的: {row['Ticker']} ({row['Company']})
        數據: 價格 {row['Price']}, 漲幅 {row['Change']}%, RSI(14) 為 {rsi_val:.2f}。
        技術面: 目前價格{status} 200日均線。
        請給出：1. 技術面強弱評語 2. 贏面分數(1-100) 3. 具體的買進、賣出或觀望建議。
        """
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(10)
        return response.text
    except Exception as e:
        return f"AI 請求失敗: {e}"

# ==========================================
# 5. PDF 報告生成
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, text=f'Stock Technical & AI Report - {datetime.date.today()}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

def create_report(df):
    print(">>> [步驟 3] 正在整合 PDF 專業分析報告...")
    pdf = PDFReport(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 總表頁
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 10, text="Potential Stocks Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font('helvetica', 'B', 7)
    pdf.set_fill_color(230, 230, 230)
    col_config = [('Ticker', 20), ('Company', 40), ('Sector', 30), ('Industry', 45), ('Mkt Cap', 25), ('P/E', 15), ('Price', 15), ('Change', 15), ('Volume', 30)]
    for header, width in col_config: pdf.cell(width, 8, text=header, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('helvetica', '', 7)
    for _, row in df.iterrows():
        pdf.cell(20, 7, text=str(row['Ticker']), border=1)
        pdf.cell(40, 7, text=str(row['Company'][:25]), border=1)
        pdf.cell(30, 7, text=str(row['Sector'][:20]), border=1)
        pdf.cell(45, 7, text=str(row['Industry'][:30]), border=1)
        pdf.cell(25, 7, text=str(row['MarketCap']), border=1)
        pdf.cell(15, 7, text=str(row['PE']), border=1)
        pdf.cell(15, 7, text=str(row['Price']), border=1)
        pdf.cell(15, 7, text=f"{row['Change']}%", border=1)
        pdf.cell(30, 7, text=str(row['Volume']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 詳情頁 (前 8 支，因為圖表變高了，每頁放一支)
    for i, (index, row) in enumerate(df.head(8).iterrows()):
        img_buf, rsi_val, is_above_200 = generate_charts(row['Ticker'])
        if img_buf:
            ai_text = get_ai_insight(row, rsi_val, is_above_200)
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, text=f"Analysis: {row['Ticker']} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            img_path = f"tmp_{row['Ticker']}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, x=40, y=30, w=210) # 放大圖表
            pdf.set_y(155) # 調整文字起始高度
            pdf.set_font('helvetica', 'B', 11); pdf.cell(0, 8, text="AI Strategist Professional Insight:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('helvetica', '', 10)
            clean_text = ai_text.replace('\u2022', '-').encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 6, text=clean_text)
            os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 專業版報告已生成。包含 RSI 技術指標與深度 AI 分析。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty: create_report(df_stocks)
    else: print("今日無符合標的。")
