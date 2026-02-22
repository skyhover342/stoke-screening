print(">>> [系統啟動] 正在初始化直式中文排版環境...")

import os
import time
import datetime
import io
import re
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

# 下載中文字體 (確保 PDF 支援中文)
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "noto_sans_tc.otf"

def download_font():
    if not os.path.exists(FONT_PATH):
        print(">>> 正在下載中文字體以支援 PDF...")
        r = requests.get(FONT_URL)
        with open(FONT_PATH, 'wb') as f:
            f.write(r.content)

# 清理 Markdown
def clean_ai_text(text):
    text = text.replace('**', '')
    text = re.sub(r'#+', '', text)
    text = text.replace('* ', '- ')
    return text

# ==========================================
# 2. 爬蟲與技術指標
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
        return pd.DataFrame(data)[~pd.DataFrame(data)['Industry'].isin(EXCLUDE_INDUSTRIES)]
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
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        last_rsi = float(df['RSI'].iloc[-1])
        is_above_200 = float(df['Close'].iloc[-1]) > (float(df['200MA'].iloc[-1]) if not pd.isna(df['200MA'].iloc[-1]) else 0)

        # 建立三層圖表 (直式版面縮減高度)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                            row_heights=[0.5, 0.15, 0.35])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['200MA'], line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=600, width=1100, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        return io.BytesIO(fig.to_image(format="png")), last_rsi, is_above_200
    except: return None, 0, False

# ==========================================
# 3. AI 分析 (繁體中文要求)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if not GEMINI_KEY: return "無 API KEY"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "站上" if is_above_200 else "低於"
        prompt = f"""
        你是專業股票專家，請分析：
        標的: {row['Ticker']} ({row['Company']})
        數據: 價格 {row['Price']}, 漲幅 {row['Change']}%, RSI: {rsi_val:.2f}, 目前{status}200日均線。
        請提供：1. 技術總結 2. 贏面分數(1-100) 3. 交易策略。
        必須使用「繁體中文」回答。
        """
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(12)
        return clean_ai_text(response.text)
    except Exception as e: return f"AI 錯誤: {e}"

# ==========================================
# 4. PDF 生成 (直式優化)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font("Chinese", size=16)
        self.cell(0, 10, text=f'美股 AI 深度研究報告 - {datetime.date.today()}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

def create_report(df):
    download_font()
    pdf = PDFReport(orientation='P', unit='mm', format='A4')
    pdf.add_font("Chinese", "", FONT_PATH)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 總表 (直式需縮減寬度)
    pdf.add_page()
    pdf.set_font("Chinese", size=11)
    pdf.cell(0, 10, text="潛力標的一覽表", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Chinese", size=8)
    pdf.set_fill_color(220, 220, 220)
    # 重新分配直式寬度 (總寬 ~190mm)
    headers = ['Ticker', 'Industry', 'Mkt Cap', 'P/E', 'Price', 'Change%', 'Volume']
    widths = [20, 50, 25, 20, 20, 20, 35]
    for h, w in zip(headers, widths): pdf.cell(w, 10, text=h, border=1, align='C', fill=True)
    pdf.ln()
    
    for _, row in df.iterrows():
        pdf.cell(20, 8, text=str(row['Ticker']), border=1)
        pdf.cell(50, 8, text=str(row['Industry'][:25]), border=1)
        pdf.cell(25, 8, text=str(row['MarketCap']), border=1)
        pdf.cell(20, 8, text=str(row['PE']), border=1)
        pdf.cell(20, 8, text=str(row['Price']), border=1)
        pdf.cell(20, 8, text=f"{row['Change']}%", border=1)
        pdf.cell(35, 8, text=str(row['Volume']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 詳情頁 (一頁一支)
    for i, (index, row) in enumerate(df.head(10).iterrows()):
        img_buf, rsi_val, is_above_200 = generate_charts(row['Ticker'])
        if img_buf:
            ai_text = get_ai_insight(row, rsi_val, is_above_200)
            pdf.add_page()
            pdf.set_font("Chinese", size=14)
            pdf.cell(0, 10, text=f"個股分析: {row['Ticker']} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            img_path = f"tmp_{row['Ticker']}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            # 直式佈局：圖表放上方
            pdf.image(img_path, x=10, y=35, w=190) 
            
            # 調整 Y 座標到圖表下方，字體縮小
            pdf.set_y(145) 
            pdf.set_font("Chinese", size=11)
            pdf.cell(0, 10, text="Gemini 專家分析結論：", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Chinese", size=9)
            pdf.multi_cell(0, 6, text=ai_text)
            os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 直式中文報告已生成。")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty: create_report(df_stocks)
