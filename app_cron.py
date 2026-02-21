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
# 1. 核心參數與環境設定
# ==========================================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
EXCLUDE_INDUSTRIES = ['Shell Companies', 'Blank Check', 'SPAC']
MAX_CHANGE = 40.0

# ==========================================
# 2. 修正版 Finviz 爬蟲 (精準對齊 v=111 欄位)
# ==========================================
def fetch_and_filter_stocks():
    print("正在連線至 Finviz 並解析資料...")
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finviz.com/'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        target_table = None
        for table in soup.find_all('table'):
            if "Ticker" in table.text[:200] and "Price" in table.text[:200]:
                target_table = table
                break
        
        if not target_table:
            print("❌ 找不到目標表格")
            return pd.DataFrame()

        rows = target_table.find_all('tr', valign="top")
        data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 11: continue # v=111 至少有 11 欄以上
            
            try:
                # --- 精準對齊 v=111 欄位 ---
                ticker     = tds[1].text.strip()
                company    = tds[2].text.strip()
                sector     = tds[3].text.strip()
                industry   = tds[4].text.strip()
                country    = tds[5].text.strip()
                mkt_cap    = tds[6].text.strip()
                pe_ratio   = tds[7].text.strip()
                price      = float(tds[8].text.strip())
                change_val = float(tds[9].text.strip('%'))
                volume     = tds[10].text.strip()

                # 排除特定產業與漲幅過大的標的
                if any(x in industry for x in EXCLUDE_INDUSTRIES): continue
                if change_val > MAX_CHANGE: continue
                
                data.append({
                    "Ticker": ticker, "Company": company, "Sector": sector, 
                    "Industry": industry, "Country": country, "MarketCap": mkt_cap,
                    "PE": pe_ratio, "Price": price, "Change": change_val, "Volume": volume
                })
            except Exception as e:
                continue
        
        df = pd.DataFrame(data)
        print(f"解析成功，最終篩選出 {len(df)} 支標的")
        return df

    except Exception as e:
        print(f"爬蟲發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (修正 MultiIndex 導致的 Series 錯誤)
# ==========================================
def generate_charts(ticker):
    print(f"正在生成 {ticker} 的圖表...")
    try:
        # 下載資料
        df_daily = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if df_daily.empty or len(df_daily) < 200:
            return None, False

        # 計算 200MA
        df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
        
        # --- 解決 ValueError 的終極寫法 ---
        # 使用 .values[-1] 取得最後一個數值，這會跳過 Pandas 的 Index 比對邏輯
        last_close = df_daily['Close'].values[-1]
        last_ma200 = df_daily['200MA'].values[-1]
        
        # 確保 is_above_200 是一個 Python bool
        is_above_200 = bool(last_close > last_ma200)

        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
        
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, row_heights=[0.7, 0.3],
                            subplot_titles=(f"{ticker} Daily", f"{ticker} 1m Intraday", "Daily Vol", "1m Vol"))
        
        fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], 
                                     low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)
        
        if not df_1m.empty:
            # 確保 1m 的顏色計算也不會出錯
            o_val = df_1m['Open'].values
            c_val = df_1m['Close'].values
            colors_1m = ['green' if c >= o else 'red' for o, c in zip(o_val, c_val)]
            
            fig.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], 
                                         low=df_1m['Low'], close=df_1m['Close']), row=1, col=2)
            fig.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker_color=colors_1m), row=2, col=2)
        
        fig.update_layout(height=600, width=1100, template="plotly_dark", showlegend=False, xaxis_rangeslider_visible=False)
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return io.BytesIO(img_bytes), is_above_200
        
    except Exception as e:
        print(f"❌ {ticker} 圖表生成失敗: {e}")
        return None, False

# ==========================================
# 4. Gemini 3.1 Thinking AI 分析 (強化資訊輸入)
# ==========================================
def get_ai_insight(row, is_above_200):
    if not GEMINI_KEY: return "未偵測到 API Key。"
    
    # 這裡確保傳入 prompt 的字串是乾淨的
    # bool() 轉換是為了解決 ValueError: The truth value of a Series is ambiguous
    status_200ma = "站上" if bool(is_above_200) else "低於"
    
    client = genai.Client(api_key=GEMINI_KEY)
    
    # 回歸你要求的 Gemini 3.1 Pro Preview
    model_id = "gemini-3.1-pro-preview" 
    
    prompt = f"""
    分析美股 {row['Ticker']} ({row['Company']})：
    - 產業: {row['Industry']} (國家: {row['Country']})
    - 財務面: 市值 {row['MarketCap']}, P/E Ratio: {row['PE']}
    - 價格: {row['Price']} (今日漲幅 {row['Change']}%)
    - 技術面: {status_200ma} 200MA。
    
    請身為專業研究員進行思考並給出：
    1. 贏面分數(0-100) 
    2. 具體買進/賣出結論 
    3. 詳細的操作建議與風險提示。
    """
    
    try:
        # 依照 Gemini 3.1 的 Thinking Config 設定
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            )
        )
        # Gemini 3.1 Thinking 比較耗時，保持足夠的 sleep 避免 Rate Limit
        time.sleep(30) 
        return response.text
    except Exception as e:
        print(f"❌ Gemini 3.1 呼叫失敗: {e}")
        return f"AI 分析失敗，錯誤代碼: {e}"

# ==========================================
# 5. PDF 專業報告生成 (更新表格欄位)
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, text=f'US Stock AI Report - {datetime.date.today()}', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

def create_report(df):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 首頁摘要
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, text="Market Scan Summary (Top Potential Movers):", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 表格標頭 (對應你需要的欄位)
    pdf.set_font('helvetica', 'B', 8)
    col_widths = [20, 45, 30, 25, 25, 20, 25]
    headers = ['Ticker', 'Industry', 'Country', 'Mkt Cap', 'P/E', 'Price', 'Change%']
    for i, col in enumerate(headers):
        pdf.cell(col_widths[i], 10, text=col, border=1, align='C')
    pdf.ln()

    # 表格內容
    pdf.set_font('helvetica', '', 8)
    for _, row in df.iterrows():
        pdf.cell(20, 10, text=str(row['Ticker']), border=1)
        pdf.cell(45, 10, text=str(row['Industry'][:20]), border=1)
        pdf.cell(30, 10, text=str(row['Country']), border=1)
        pdf.cell(25, 10, text=str(row['MarketCap']), border=1)
        pdf.cell(25, 10, text=str(row['PE']), border=1)
        pdf.cell(20, 10, text=str(row['Price']), border=1)
        pdf.cell(25, 10, text=f"{row['Change']}%", border=1, new_x="LMARGIN", new_y="NEXT")

    # 每股詳情頁
    for _, row in df.head(10).iterrows():
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        if img_buf is None: continue
        
        ai_text = get_ai_insight(row, is_above_200)
        
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, text=f"Detailed Analysis: {ticker} - {row['Company']}", new_x="LMARGIN", new_y="NEXT")
        
        img_path = f"temp_{ticker}.png"
        with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
        pdf.image(img_path, x=10, y=30, w=190)
        
        pdf.set_y(150)
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, text="AI Strategist Thinking Insight:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('helvetica', '', 10)
        # 清除不支援字符
        clean_text = ai_text.replace('\u2022', '-').encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 6, text=clean_text)
        os.remove(img_path)

    pdf.output("report.pdf")
    print("✅ 報告生成成功：report.pdf")

# ==========================================
# 6. 主執行流程
# ==========================================
if __name__ == "__main__":
    df = fetch_and_filter_stocks()
    if not df.empty:
        create_report(df)
    else:
        print("今日無符合標的。")
