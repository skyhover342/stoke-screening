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
# 2. 強韌版 Finviz 爬蟲 (解決 0 資料問題)
# ==========================================
def fetch_and_filter_stocks():
    print("正在連線至 Finviz 並執行精準篩選...")
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Referer': 'https://finviz.com/'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找資料表格中的所有行
        rows = soup.find_all('tr', class_='screener-body-table-nw')
        if not rows:
            # 備用：若 class 變動，找 table-light
            table = soup.find('table', class_='table-light')
            rows = table.find_all('tr')[1:] if table else []

        print(f"解析成功，發現原始資料列數: {len(rows)}")
        
        data = []
        for r in rows:
            tds = r.find_all('td')
            if len(tds) < 11: continue
            
            try:
                # --- 精準索引校對 (v=111 模式) ---
                ticker = tds[1].text.strip()
                industry = tds[4].text.strip()
                price_str = tds[8].text.strip()
                change_str = tds[9].text.strip('%')
                
                # 在 v=111 下，Rel Vol 通常在最後一欄 (Index 11)，Volume 在 Index 10
                # 我們用 Try-Except 來確保抓到正確的數值
                volume_str = tds[10].text.strip()
                rel_vol_str = tds[11].text.strip() if len(tds) > 11 else "0"
                
                # 轉化數值
                change_val = float(change_str)
                rel_vol = float(rel_vol_str)
                price = float(price_str)

                # DEBUG: 在日誌印出抓到的資料，確保沒抓錯位
                # print(f"檢查中: {ticker} | 漲幅: {change_val}% | 相對量: {rel_vol} | 行業: {industry}")

                # 執行排除邏輯
                if any(x in industry for x in EXCLUDE_INDUSTRIES):
                    # print(f"排除 {ticker}: 屬於殼公司/SPAC")
                    continue
                if change_val > MAX_CHANGE:
                    # print(f"排除 {ticker}: 漲幅過大 ({change_val}%)")
                    continue
                
                data.append({
                    "Ticker": ticker,
                    "Company": tds[2].text.strip(),
                    "Sector": tds[3].text.strip(),
                    "Industry": industry,
                    "Price": price,
                    "Change": change_val,
                    "Rel_Vol": rel_vol,
                    "Volume": volume_str
                })
            except Exception as e:
                # 如果這行印出來，代表索引真的抓錯了
                # print(f"解析股票列失敗: {e}")
                continue
                
        df = pd.DataFrame(data)
        print(f"過濾完成，最終入榜數量: {len(df)}")
        return df

    except Exception as e:
        print(f"連線或解析發生嚴重錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 3. 圖表生成 (日線 + 1分線)
# ==========================================
def generate_charts(ticker):
    print(f"正在生成 {ticker} 的圖表...")
    df_daily = yf.download(ticker, period="2y", interval="1d", progress=False)
    df_daily['200MA'] = df_daily['Close'].rolling(window=200).mean()
    df_1m = yf.download(ticker, period="1d", interval="1m", progress=False)
    
    is_above_200 = df_daily['Close'].iloc[-1] > df_daily['200MA'].iloc[-1]
    
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{ticker} 1Y Daily (Yellow=200MA)", f"{ticker} 1m Intraday", "Daily Vol", "1m Vol"))
    
    fig.add_trace(go.Candlestick(x=df_daily.index, open=df_daily['Open'], high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close']), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_daily.index, y=df_daily['200MA'], line=dict(color='yellow', width=2)), row=1, col=1)
    
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
    if not GEMINI_KEY: return "未偵測到 API Key。"
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"分析美股 {row['Ticker']}：價格 {row['Price']}，今日漲幅 {row['Change']}%，相對成交量 {row['Rel_Vol']}。{'站上' if is_above_200 else '低於'} 200MA。請深度思考並給出贏面分數與操作建議。"
    
    try:
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview", contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="HIGH"))
        )
        time.sleep(35)
        return response.text
    except Exception as e:
        return f"AI 分析失敗: {e}"

# ==========================================
# 5. PDF 專業報告生成 (修復所有語法警告)
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
    pdf.cell(0, 10, text="Market Scan Summary:", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 表格標頭
    pdf.set_font('helvetica', 'B', 10)
    for col in ['Ticker', 'Industry', 'Price', 'Change%', 'RelVol']:
        pdf.cell(38, 10, text=col, border=1)
    pdf.ln()

    # 表格內容
    pdf.set_font('helvetica', '', 10)
    for _, row in df.iterrows():
        pdf.cell(38, 10, text=str(row['Ticker']), border=1)
        pdf.cell(38, 10, text=str(row['Industry'][:15]), border=1)
        pdf.cell(38, 10, text=str(row['Price']), border=1)
        pdf.cell(38, 10, text=f"{row['Change']}%", border=1)
        pdf.cell(38, 10, text=str(row['Rel_Vol']), border=1, new_x="LMARGIN", new_y="NEXT")

    # 每股詳情頁 (最多分析 10 支)
    for _, row in df.head(10).iterrows():
        ticker = row['Ticker']
        img_buf, is_above_200 = generate_charts(ticker)
        ai_text = get_ai_insight(row, is_above_200)
        
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, text=f"Detailed Analysis: {ticker} - {row['Company']}", new_x="LMARGIN", new_y="NEXT")
        
        img_path = f"temp_{ticker}.png"
        with open(img_path, "wb") as f: f.write(img_buf.getbuffer())
        pdf.image(img_path, x=10, y=30, w=190)
        
        pdf.set_y(150)
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, text="Gemini 3.1 Pro Thinking Insight:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('helvetica', '', 10)
        pdf.multi_cell(0, 6, text=ai_text.replace('\u2022', '-'))
        os.remove(img_path)

    pdf.output("report.pdf")

# ==========================================
# 6. 主執行流程
# ==========================================
if __name__ == "__main__":
    df = fetch_and_filter_stocks()
    if not df.empty:
        print(f"篩選到 {len(df)} 支標的，開始生成報告...")
        create_report(df)
    else:
        print("今日無標的，生成測試報告...")
        # 生成一個沒股票時的乾淨報告，避免 GitHub 上傳失敗
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 10, text="No stocks matched the criteria today.", align='C', new_x="LMARGIN", new_y="NEXT")
        pdf.output("report.pdf")
