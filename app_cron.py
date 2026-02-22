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
# 1. 核心參數與環境設定
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']
MAX_CHANGE = 40.0
TARGET_MODEL = "models/gemini-2.5-pro"

# 初始化 AI 客戶端
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# ==========================================
# 2. 強韌版 Finviz 爬蟲
# ==========================================
def fetch_and_filter_stocks():
    print(">>> [步驟 1] 正在連線至 Finviz...")
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
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
        # 排除特定產業
        df = df[~df['Industry'].isin(EXCLUDE_INDUSTRIES)]
        print(f"✅ 解析成功，篩選出 {len(df)} 支標的")
        return df
    except Exception as e:
        print(f"❌ 爬蟲錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (徹底修正 NumPy 警告與 MultiIndex 問題)
# ==========================================
def generate_charts(ticker):
    print(f">>> [步驟 2] 正在生成 {ticker} 的圖表...")
    try:
        # 下載資料
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False, threads=False)
        if df_daily.empty or len(df_daily) < 200: return None, False

        # --- 自我矯正：處理 yfinance 可能的 MultiIndex 欄位 ---
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.get_level_values(0)

        # 計算 200MA
        df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
        
        # --- 自我矯正：使用 .iloc[-1] 與 .item() 確保拿到的是數字而不是陣列 ---
        last_close = df_daily['Close'].iloc[-1]
        last_ma200 = df_daily['200MA'].iloc[-1]
        
        # 如果最後拿到的是 Series (偶發狀況)，則再取一次值
        if hasattr(last_close, 'item'): last_close = last_close.item()
        if hasattr(last_ma200, 'item'): last_ma200 = last_ma200.item()

        is_above_200 = bool(last_close > last_ma200)

        # 下載 1分K
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        if isinstance(df_1m.columns, pd.MultiIndex):
            df_1m.columns = df_1m.columns.get_level_values(0)

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{ticker} Daily", f"{ticker} 1m Intraday", "Daily Vol", "1m Vol"))
        
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)
        
        if not df_1m.empty:
            # 確保使用 1D 陣列計算顏色
            o_val = df_1m['Open'].to_numpy().flatten()
            c_val = df_1m['Close'].to_numpy().flatten()
            colors_1m = ['green' if c >= o else 'red' for o, c in zip(o_val, c_val)]
            fig.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), row=1, col=2)
            fig.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color=colors_1m), row=2, col=2)
        
        fig.update_layout(height=600, width=1100, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        img_bytes = fig.to_image(format="png")
        return io.BytesIO(img_bytes), is_above_200
    except Exception as e:
        print(f"❌ {ticker} 圖表失敗: {e}")
        return None, False

# ==========================================
# 4. AI 分析 (強化頻率限制)
# ==========================================
def get_ai_insight(row, is_above_200):
    if not client: return "未偵測到 API Key。"
    status_200ma = "站上" if is_above_200 else "低於"
    prompt = f"分析美股 {row['Ticker']} ({row['Company']})，產業 {row['Industry']}，市值 {row['MarketCap']}，價格 {row['Price']}，目前{status_200ma} 200MA。請給出贏面分數與具體建議。"

    for attempt in range(2): # 限制重試次數
        try:
            response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
            time.sleep(45) # 免費版 Pro 的安全間隔
            return response.text
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ 頻率限制，等待 60s...")
                time.sleep(60)
            else: break
    return "跳過分析"

# ==========================================
# 5. PDF 報告 (符合最新 fpdf2 規範)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text=f'US Stock AI Report - {datetime.date.today()}', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(10)

def create_report(df):
    print(">>> [步驟 3] 正在製作 PDF...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 摘要頁
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, text="Market Scan Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # 表格標頭
    pdf.set_font('helvetica', 'B', 8)
    cols = ['Ticker', 'Industry', 'Price', 'Change%', 'Mkt Cap']
    for col in cols: pdf.cell(38, 10, text=col, border=1)
    pdf.ln()

    # 表格內容
    pdf.set_font('helvetica', '', 8)
    for _, row in df.iterrows():
        pdf.cell(38, 10, text=str(row['Ticker']), border=1)
        pdf.cell(38, 10, text=str(row['Industry'][:18]), border=1)
        pdf.cell(38, 10, text=str(row['Price']), border=1)
        pdf.cell(38, 10, text=f"{row['Change']}%", border=1)
        pdf.cell(38, 10, text=str(row['MarketCap']), border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 詳細分析 (限制 10 支以防 GitHub 超時)
    for i, (index, row) in enumerate(df.head(10).iterrows()):
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        if img_buf is None: continue
        
        ai_text = get_ai_insight(row, is_above_200)
        
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, text=f"Analysis: {ticker} - {row['Company']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        img_path = f"temp_{ticker}.png"
        with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
        pdf.image(img_path, x=10, y=30, w=190)
        
        pdf.set_y(155)
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 10, text="AI Strategist Insight:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('helvetica', '', 10)
        
        clean_text = ai_text.replace('\u2022', '-').encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 6, text=clean_text)
        os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 任務完成！報告已儲存。")

if __name__ == "__main__":
    df_found = fetch_and_filter_stocks()
    if not df_found.empty:
        create_report(df_found)
    else:
        print("今日無標的。")
