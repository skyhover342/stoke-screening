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

# ==========================================
# 1. 環境設定與字體準備
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']

# 使用 Noto Sans TC 確保繁體中文不亂碼
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = "noto_sans_tc.otf"

def download_font():
    if not os.path.exists(FONT_PATH):
        print(">>> 正在下載中文字體以支援 PDF...")
        r = requests.get(FONT_URL)
        with open(FONT_PATH, 'wb') as f:
            f.write(r.content)

def clean_ai_text(text):
    # 移除 Markdown 符號以免 PDF 解析錯誤
    text = text.replace('**', '').replace('###', '').replace('#', '').replace('*', '-')
    return text.strip()

# ==========================================
# 2. 數據抓取與技術指標
# ==========================================
def fetch_and_filter_stocks():
    print(">>> 正在抓取 Finviz 數據...")
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
                    "Industry": tds[4].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": float(tds[9].text.strip('%')), "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        return df[~df['Industry'].isin(EXCLUDE_INDUSTRIES)]
    except Exception as e:
        print(f"❌ 爬蟲出錯: {e}")
        return pd.DataFrame()

def generate_charts(ticker):
    print(f">>> 正在繪製 {ticker} 技術圖表 (RSI+200MA)...")
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if df.empty or len(df) < 30: return None, 0, False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['200MA'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))

        last_rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        is_above_200 = float(df['Close'].iloc[-1]) > (float(df['200MA'].iloc[-1]) if not pd.isna(df['200MA'].iloc[-1]) else 0)

        # 建立三層專業圖表 (針對直式高度優化)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.5, 0.1, 0.4])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['200MA'], line=dict(color='yellow', width=1.5)), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=600, width=1000, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        return io.BytesIO(fig.to_image(format="png")), last_rsi, is_above_200
    except: return None, 0, False

# ==========================================
# 3. AI 分析 (繁體中文結構化)
# ==========================================
def get_ai_insight(row, rsi_val, is_above_200):
    if not GEMINI_KEY: return "未偵測到 API Key"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        status = "站上" if is_above_200 else "低於"
        prompt = f"""
        請以資深美股分析師身份，對以下數據進行繁體中文分析：
        標的：{row['Ticker']} ({row['Company']})
        價格：{row['Price']}，漲幅：{row['Change']}%，RSI(14)：{rsi_val:.2f}
        趨勢：目前股價{status} 200日均線。
        請嚴格提供以下三點結論：
        1. 技術面強弱總結
        2. 贏面評分 (1-100)
        3. 具體操盤策略建議 (買進/賣出/觀望)
        """
        response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        time.sleep(12)
        return clean_ai_text(response.text)
    except Exception as e:
        return f"AI 分析暫時不可用: {e}"

# ==========================================
# 4. PDF 生成與排版 (直式 A4)
# ==========================================
class StockPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Chinese", size=8)
        self.cell(0, 10, f"Page {self.page_no()} | 生成日期: {datetime.date.today()}", align='C')

def create_report(df):
    download_font()
    pdf = StockPDF(orientation='P', unit='mm', format='A4')
    pdf.add_font("Chinese", "", FONT_PATH)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 頁面 1: 總表
    pdf.add_page()
    pdf.set_font("Chinese", size=16)
    pdf.cell(0, 15, text="美股潛力標的掃描總表", align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    pdf.set_font("Chinese", size=8)
    pdf.set_fill_color(240, 240, 240)
    # Ticker, Industry, Mkt Cap, P/E, Price, Change%, Volume
    widths = [20, 50, 25, 18, 18, 18, 38]
    headers = ['代碼', '產業', '市值', 'P/E', '現價', '漲幅', '成交量']
    for h, w in zip(headers, widths):
        pdf.cell(w, 8, text=h, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font("Chinese", size=8)
    for _, row in df.iterrows():
        pdf.cell(20, 8, text=str(row['Ticker']), border=1)
        pdf.cell(50, 8, text=str(row['Industry'][:25]), border=1)
        pdf.cell(25, 8, text=str(row['MarketCap']), border=1)
        pdf.cell(18, 8, text=str(row['PE']), border=1)
        pdf.cell(18, 8, text=str(row['Price']), border=1)
        pdf.cell(18, 8, text=f"{row['Change']}%", border=1)
        pdf.cell(38, 8, text=str(row['Volume']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 後續頁面: 個股深度分析
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        img_buf, rsi_val, is_above_200 = generate_charts(row['Ticker'])
        if img_buf:
            ai_text = get_ai_insight(row, rsi_val, is_above_200)
            pdf.add_page()
            
            # 標題區
            pdf.set_font("Chinese", size=14)
            pdf.set_text_color(0, 51, 102) # 深藍色標題
            pdf.cell(0, 10, text=f"【個股深度分析】 {row['Ticker']} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            # 圖表區 (放置於上方，寬度撐滿)
            img_path = f"tmp_{row['Ticker']}.png"
            with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
            pdf.image(img_path, x=10, y=25, w=190) 
            
            # 文字區 (座標定位於圖表下方)
            pdf.set_y(140) 
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Chinese", size=11)
            pdf.cell(0, 10, text="📊 AI 策略師分析結論：", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.set_font("Chinese", size=9)
            pdf.multi_cell(0, 6, text=ai_text)
            
            os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 報告生成成功：直式佈局、中文支持、RSI 指標一應俱全！")

if __name__ == "__main__":
    df_stocks = fetch_and_filter_stocks()
    if not df_stocks.empty:
        create_report(df_stocks)
    else:
        print("今日無符合標的。")
