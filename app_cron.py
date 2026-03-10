# 版本號碼：v1.7.2
print(">>> [系統啟動] v1.7.2 精選績優版：漲幅限制 20%、市值門檻 100M、維持高清五段紅綠視覺...")

import os, time, datetime, io, base64, requests, glob, json
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

try:
    from google import genai
except ImportError:
    print("❌ 錯誤：找不到 google-genai 套件。")

# ==========================================
# 1. 核心參數
# ==========================================
VERSION = "v1.7.2"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.5-flash"
TEST_MODE = False 

# ==========================================
# 2. 數據抓取與精準過濾 (新增市值門檻 100M & 漲幅限制 20%)
# ==========================================
def parse_mkt_cap(val_str):
    """將 Finviz 的市值字串 (如 1.5B, 100M) 轉換為數字以供比較"""
    try:
        val_str = val_str.upper()
        if 'T' in val_str: return float(val_str.replace('T', '')) * 1000000000000
        if 'B' in val_str: return float(val_str.replace('B', '')) * 1000000000
        if 'M' in val_str: return float(val_str.replace('M', '')) * 1000000
        return float(val_str)
    except: return 0.0

def is_market_open_today():
    if TEST_MODE: return True
    try:
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.datetime.now(ny_tz)
        if now_ny.weekday() >= 5: return False
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d")
        return not hist.empty
    except: return True

def fetch_and_filter_stocks():
    print(f">>> [步驟 1] 抓取數據並執行精準篩選 (漲幅<20% & 市值>100M)...")
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
                name = tds[2].text.strip()
                industry = tds[4].text.strip()
                mkt_cap_str = tds[6].text.strip()
                change_val = float(tds[9].text.strip('%'))
                
                # --- 精準過濾邏輯 ---
                # 1. 漲幅限制：原本 40% 改為 20%
                if change_val > 20: continue 
                # 2. 市值限制：必須超過 100M (一億美元)
                if parse_mkt_cap(mkt_cap_str) < 100000000: continue
                # 3. 排除空殼公司
                if "shell" in name.lower() or "shell" in industry.lower(): continue

                data.append({
                    "Ticker": tds[1].text.strip(), "Company": name,
                    "Industry": industry, "MarketCap": mkt_cap_str,
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": change_val, "Volume": tds[10].text.strip()
                })
            except: continue
        df = pd.DataFrame(data)
        if df.empty: return df
        df = df.sort_values(by=['Industry', 'Ticker'], ascending=[True, True])
        return df.head(2) if TEST_MODE else df
    except Exception as e:
        print(f"❌ 抓取失敗: {e}"); return pd.DataFrame()

# ==========================================
# 3. 專業繪圖 (4K渲染、紅綠成交量、五段切換支持)
# ==========================================
def generate_chart(df_plot, is_1m=False):
    rows, height = (3, 800) if not is_1m else (1, 450)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
                        row_heights=[0.5, 0.28, 0.22] if not is_1m else [1.0],
                        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]]*(rows-1))
    
    # 紅綠成交量邏輯
    vol_colors = ['rgba(0,255,0,0.7)' if r['Close'] >= r['Open'] else 'rgba(255,0,0,0.7)' for _, r in df_plot.iterrows()]
    
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker=dict(color=vol_colors, line_width=0), showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1, secondary_y=False)
    
    if not is_1m:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA20'], line=dict(color='cyan', width=1.5), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='orange', width=1.8), name="MA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='yellow', width=2.5), name="MA200"), row=1, col=1)
        fig.update_xaxes(showticklabels=True, row=1, col=1, tickfont=dict(size=10, color='gray'))
        fig.update_xaxes(showticklabels=False, row=2, col=1); fig.update_xaxes(showticklabels=False, row=3, col=1)
        colors = ['rgba(0,255,0,0.9)' if v>=0 else 'rgba(255,0,0,0.9)' for v in df_plot['Hist']]
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Hist'], marker=dict(color=colors, line_width=0)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], line=dict(color='#00FF00', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal'], line=dict(color='#A020F0', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='#E0B0FF', width=2.5)), row=3, col=1)
    else:
        fig.update_xaxes(showticklabels=True, tickfont=dict(size=10, color='gray'))

    fig.update_yaxes(range=[0, df_plot['Volume'].max()*1.8], secondary_y=True, showgrid=False, row=1)
    fig.update_layout(height=height, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))
    return base64.b64encode(fig.to_image(format="png", scale=3)).decode('utf-8')

def generate_stock_images(ticker):
    try:
        df_all = yf.download(ticker, period="4y", interval="1d", progress=False)
        if df_all.empty: return [None]*7
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        df_all = df_all.ffill()
        df_all['SMA20']=df_all['Close'].rolling(20, min_periods=1).mean(); df_all['SMA50']=df_all['Close'].rolling(50, min_periods=1).mean(); df_all['SMA200']=df_all['Close'].rolling(200, min_periods=1).mean()
        exp1=df_all['Close'].ewm(span=12, adjust=False).mean(); exp2=df_all['Close'].ewm(span=26, adjust=False).mean()
        df_all['MACD']=exp1-exp2; df_all['Signal']=df_all['MACD'].ewm(span=9, adjust=False).mean(); df_all['Hist']=df_all['MACD']-df_all['Signal']
        delta=df_all['Close'].diff(); g=delta.where(delta>0,0).rolling(14, min_periods=1).mean(); l=-delta.where(delta<0,0).rolling(14, min_periods=1).mean()
        df_all['RSI']=100-(100/(1+g/l))

        img_1m = generate_chart(df_all.tail(21))
        img_3m = generate_chart(df_all.tail(63))
        img_6m = generate_chart(df_all.tail(126))
        img_1y = generate_chart(df_all.tail(252))
        img_max = generate_chart(df_all.tail(min(len(df_all), 756)))

        df_intra_data = yf.download(ticker, period="1d", interval="1m", progress=False, prepost=True)
        img_intra = ""
        if not df_intra_data.empty:
            if isinstance(df_intra_data.columns, pd.MultiIndex): df_intra_data.columns = df_intra_data.columns.get_level_values(0)
            df_intra_data.index = df_intra_data.index.tz_convert('America/New_York')
            df_intra_data['Vol_Avg'] = df_intra_data['Volume'].rolling(5, min_periods=1).mean()
            intra_vol_colors = ['rgba(0,255,0,0.7)' if r['Close'] >= r['Open'] else 'rgba(255,0,0,0.7)' for _, r in df_intra_data.iterrows()]
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_intra_data.index, y=df_intra_data['Volume'], marker=dict(color=intra_vol_colors, line_width=0), showlegend=False), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_intra_data.index, open=df_intra_data['Open'], high=df_intra_data['High'], low=df_intra_data['Low'], close=df_intra_data['Close']), secondary_y=False)
            reg = df_intra_data[df_intra_data.index.time <= datetime.time(16, 0)]
            if not reg.empty:
                cp=reg.iloc[-1]; ct=reg.index[-1]
                fig2.add_annotation(x=ct, y=cp['Close'], text="🔔 CLOSE (EST)", showarrow=True, arrowhead=2, font=dict(color="white", size=10), bgcolor="#003366", ay=-50)
                fig2.add_shape(type="line", x0=df_intra_data.index[0], y0=cp['Close'], x1=df_intra_data.index[-1], y1=cp['Close'], line=dict(color="red", width=1.5, dash="dot"))
            spikes = df_intra_data[df_intra_data['Volume'] > df_intra_data['Vol_Avg']*3].copy()
            for idx, row in spikes.sort_values(by='Volume', ascending=False).head(10).iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                fig2.add_annotation(x=idx, y=row['High'], text="▲ BUY" if row['Close'] > row['Open'] else "▼ SELL", showarrow=True, arrowhead=1, arrowcolor=t_color, font=dict(size=11, color=t_color, weight='bold'), bgcolor="black", opacity=0.9, ay=-40)
            fig2.update_layout(height=450, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            img_intra = base64.b64encode(fig2.to_image(format="png", scale=3)).decode('utf-8')
            
        return img_1m, img_3m, img_6m, img_1y, img_max, img_intra, bool(df_all['Close'].iloc[-1] > df_all['SMA200'].iloc[-1])
    except: return [None]*7

# ==========================================
# 4. 批量 AI 分析 (Gemini 3 Flash)
# ==========================================
def get_batch_ai_insights(df_subset):
    tickers = df_subset['Ticker'].tolist()
    if TEST_MODE: return {t: f"<b>【測試診斷】</b>：正在測試漲幅過濾(<20%)與市值限制(>100M)。" for t in tickers}
    if not GEMINI_KEY: return {t: "❌ 無 API" for t in tickers}
    summary = "".join([f"- {r['Ticker']}: ${r['Price']} ({r['Change']}%) [{r['Industry']}] MC: {r['MarketCap']}\n" for _, r in df_subset.iterrows()])
    prompt = f"分析美股技術趨勢，提供 150-200 字建議。繁體中文。回傳 JSON：{{\"Ticker\": \"內容\"}} \n數據：\n{summary}"
    try:
        client = genai.Client(api_key=GEMINI_KEY); resp = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        raw = resp.text.strip().replace('```json', '').replace('```', '')
        time.sleep(50); return json.loads(raw)
    except: return {t: "⚠️ 分析產出中..." for t in tickers}

# ==========================================
# 5. HTML 生成 (歷史、導航、五段切換)
# ==========================================
def create_html_report(df):
    ny_tz = pytz.timezone('America/New_York'); today_ny = datetime.datetime.now(ny_tz).strftime("%Y-%m-%d")
    today_str = datetime.date.today().strftime("%Y%m%d")
    os.makedirs("history", exist_ok=True); history_files = sorted(glob.glob("history/report_*.html"), reverse=True)
    
    l_main = "".join([f'<a href="./history/report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])
    l_hist = "".join([f'<a href="./report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])
    
    def get_nav(is_m):
        h = "" if is_m else '<a href="../index.html" class="history-item" style="background:#003366;color:white;font-weight:bold;">🏠 返回最新</a>'
        return f'<div class="history-bar"><div style="font-weight:bold;margin-right:10px;color:#003366;white-space:nowrap;">📅 存檔：</div>{h}{l_main if is_m else l_hist}</div>'

    all_insights = {}
    print(f">>> [步驟 2] 開始深度分析 (共 {len(df)} 支符合門檻之股票)...")
    for i in range(0, len(df), 2): all_insights.update(get_batch_ai_insights(df.iloc[i:i+2]))

    def build_page(is_m):
        return f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link rel="icon" href="https://cdn-icons-png.flaticon.com/512/2422/2422796.png"><title>AI 美股深度掃描</title>
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; margin: 0; }} .container {{ max-width: 1100px; margin: 0 auto; }}
            .history-bar {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; overflow-x: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .history-item {{ text-decoration: none; color: #666; padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; font-size: 12px; white-space: nowrap; }}
            .summary-table-wrapper {{ overflow-x: auto; }} .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 40px; font-size: 12px; min-width: 800px; }}
            .summary-table th {{ background: #003366; color: white; padding: 12px; }} .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; cursor: pointer; }}
            .summary-table tr:hover {{ background-color: #f1f5f9; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
            .card-header-row {{ background: #003366; color: white; padding: 12px; display: grid; grid-template-columns: 80px 200px 100px 80px 80px 80px 1fr; text-align: center; font-size: 13px; font-weight: bold; align-items: center; }}
            @media (max-width: 768px) {{ .card-header-row {{ grid-template-columns: repeat(2, 1fr); font-size: 11px; gap: 8px; }} }}
            .chart-stack {{ display: flex; flex-direction: column; gap: 20px; align-items: center; background: #1a1a1a; padding: 15px; }} .chart-stack img {{ width: 100%; height: auto; border: 1px solid #444; }}
            .toggle-bar {{ background: #333; padding: 10px; width: 100%; display: flex; justify-content: center; gap: 5px; flex-wrap: wrap; border-bottom: 1px solid #444; }}
            .toggle-btn {{ background: #555; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 11px; }} .toggle-btn.active {{ background: #2563eb; font-weight: bold; }}
            .analysis-box {{ padding: 25px; line-height: 1.8; background: #f8fafc; font-size: 14px; border-top: 1px solid #eee; }}
            .btn-group {{ margin-top: 15px; display: flex; justify-content: flex-end; gap: 10px; }}
            .action-btn {{ background: #003366; color: white; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-size: 12px; font-weight: bold; border: none; cursor: pointer; }} .share-btn {{ background: #2563eb; }}
        </style>
        <script>
            function switchPeriod(ticker, period) {{
                const pArr = ['1m', '3m', '6m', '1y', 'max'];
                pArr.forEach(p => {{
                    document.getElementById('img-' + p + '-' + ticker).style.display = (p === period) ? 'block' : 'none';
                    document.getElementById('btn-' + p + '-' + ticker).classList.toggle('active', p === period);
                }});
            }}
            async function shareTicker(t, p) {{
                const s = {{ title: `📈 AI 掃描: ${{t}}`, text: `代碼 ${{t}} 目前 $${{p}}。點擊查看分析。`, url: window.location.origin + window.location.pathname + '?ticker=' + t }};
                try {{ if (navigator.share) {{ await navigator.share(s); }} else {{ alert('網址已複製'); navigator.clipboard.writeText(s.url); }} }} catch (e) {{}}
            }}
            window.onload = function() {{
                const p = new URLSearchParams(window.location.search); const t = p.get('ticker');
                if (t) {{ const e = document.getElementById(t.toUpperCase()); if (e) {{ setTimeout(() => {{ e.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }}, 600); }} }}
            }};
        </script></head><body><div class="container" id="top">{get_nav(is_m)}
        <h1 style="color:#003366; text-align:center; margin-bottom: 5px;">📊 美股 AI 全量深度報告 {VERSION}</h1>
        <h3 style="color:#666; text-align:center; margin-top: 0; font-weight: normal;">🇺🇸 美股交易日：{today_ny}</h3>
        <div class="summary-table-wrapper"><table class="summary-table"><thead><tr><th>代碼</th><th>公司</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead><tbody>"""

    def get_rows(df_in):
        h = ""
        for _, row in df_in.iterrows():
            h += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
        return h

    cards = ""
    for _, row in df.iterrows():
        m1, m3, m6, y1, mmax, m_intra, is_a = generate_stock_images(row['Ticker'])
        ins = all_insights.get(row['Ticker'], "⚠️ 分析產出中...")
        if y1:
            cards += f"""<div class="stock-card" id="{row['Ticker']}"><div class="card-header-row"><div>{row['Ticker']}</div><div>{row['Industry']}</div><div>{row['MarketCap']}</div><div>{row['PE']}</div><div>${row['Price']}</div><div style="color:#ffcccc;">+{row['Change']}%</div><div>{row['Volume']}</div></div>
            <div class="toggle-bar">
                <button id="btn-1m-{row['Ticker']}" class="toggle-btn" onclick="switchPeriod('{row['Ticker']}', '1m')">1M</button>
                <button id="btn-3m-{row['Ticker']}" class="toggle-btn" onclick="switchPeriod('{row['Ticker']}', '3m')">3M</button>
                <button id="btn-6m-{row['Ticker']}" class="toggle-btn" onclick="switchPeriod('{row['Ticker']}', '6m')">6M</button>
                <button id="btn-1y-{row['Ticker']}" class="toggle-btn active" onclick="switchPeriod('{row['Ticker']}', '1y')">1Y</button>
                <button id="btn-max-{row['Ticker']}" class="toggle-btn" onclick="switchPeriod('{row['Ticker']}', 'max')">MAX</button>
            </div>
            <div class="chart-stack">
                <img id="img-1m-{row['Ticker']}" src="data:image/png;base64,{m1}" style="display:none;">
                <img id="img-3m-{row['Ticker']}" src="data:image/png;base64,{m3}" style="display:none;">
                <img id="img-6m-{row['Ticker']}" src="data:image/png;base64,{m6}" style="display:none;">
                <img id="img-1y-{row['Ticker']}" src="data:image/png;base64,{y1}">
                <img id="img-max-{row['Ticker']}" src="data:image/png;base64,{mmax}" style="display:none;">
                <img src="data:image/png;base64,{m_intra}">
            </div>
            <div class="analysis-box"><strong>🛡️ AI 策略師深度診斷：</strong><br>{ins}<div class="btn-group"><button class="action-btn share-btn" onclick="shareTicker('{row['Ticker']}', '{row['Price']}')">📲 分享此股票</button><a href="#top" class="action-btn">⬆ 返回總表</a></div></div></div>"""
    
    rows_h = get_rows(df)
    with open("index.html", "w", encoding="utf-8") as f: 
        f.write(build_page(True) + rows_h + "</tbody></table></div>" + cards + "</div></body></html>")
    with open(f"history/report_{today_str}.html", "w", encoding="utf-8") as f: 
        f.write(build_page(False) + rows_h + "</tbody></table></div>" + cards + "</div></body></html>")

if __name__ == "__main__":
    if is_market_open_today():
        df_res = fetch_and_filter_stocks()
        if not df_res.empty: create_html_report(df_res)
