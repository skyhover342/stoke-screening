# 版本號碼：v1.5.4
print(">>> [系統啟動] 正在執行 v1.5.4 精準版：過濾 >40% 暴衝股、解鎖數量、修復路徑與縮放...")

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
VERSION = "v1.5.4"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TARGET_MODEL = "models/gemini-2.0-flash"
TEST_MODE = True  # 非正式執行

# ==========================================
# 2. 數據抓取與精準過濾 (排除 >40% 暴衝股)
# ==========================================
def is_market_open_today():
    if TEST_MODE: return True
    try:
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.datetime.now(ny_tz)
        if now_ny.weekday() >= 5: 
            print(f"🛑 今日為美東週末 ({now_ny.strftime('%A')})，停止執行。")
            return False
            
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d")
        if hist.empty: 
            print("🛑 無法抓取到最新交易數據，今日可能為市場節日休市。")
            return False
        
        print(f"⏰ 當前美東時間: {now_ny.strftime('%Y-%m-%d %H:%M')}")
        print(f"📊 最新交易日數據日期: {hist.index[-1].date()}")
        return True 
    except Exception as e:
        print(f"⚠️ 休市檢查異常 (預設繼續執行): {e}"); return True

def fetch_and_filter_stocks():
    print(f">>> [步驟 1] 正在抓取數據並過濾漲幅 > 40% 的投機股...")
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
                change_val = float(tds[9].text.strip('%'))
                # --- 核心邏輯：過濾漲幅大於 40% 的股票 ---
                if change_val > 40: continue 

                data.append({
                    "Ticker": tds[1].text.strip(), "Company": tds[2].text.strip(),
                    "Industry": tds[4].text.strip(), "MarketCap": tds[6].text.strip(),
                    "PE": tds[7].text.strip(), "Price": float(tds[8].text.strip()), 
                    "Change": change_val, "Volume": tds[10].text.strip()
                })
            except: continue
        
        df = pd.DataFrame(data)
        print(f"✅ 完成過濾，共計分析 {len(df)} 支符合條件的股票。")
        return df.head(2) if TEST_MODE else df
    except Exception as e:
        print(f"❌ 抓取失敗: {e}"); return pd.DataFrame()

# ==========================================
# 3. 專業繪圖 (日期移至第一層 + 視覺一致化)
# ==========================================
def generate_chart(df_plot, height=800):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
                        row_heights=[0.5, 0.28, 0.22], 
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
    
    # 日線成交量 (rgba 210, 210, 210) 與 K 線
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker=dict(color='rgba(210, 210, 210, 0.8)', line_width=0), showlegend=False), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA20'], line=dict(color='cyan', width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='orange', width=1.5), name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='yellow', width=2.2), name="MA200"), row=1, col=1)
    
    # 核心需求：日期放在第一層
    fig.update_xaxes(showticklabels=True, row=1, col=1, tickfont=dict(size=10, color='gray'))
    fig.update_xaxes(showticklabels=False, row=2, col=1); fig.update_xaxes(showticklabels=False, row=3, col=1)

    # MACD 顯色強化 (無邊框)
    colors = ['rgba(0,255,0,0.9)' if v>=0 else 'rgba(255,0,0,0.9)' for v in df_plot['Hist']]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Hist'], marker=dict(color=colors, line_width=0)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], line=dict(color='#00FF00', width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal'], line=dict(color='#A020F0', width=1.8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='#E0B0FF', width=2.2)), row=3, col=1)
    
    fig.update_yaxes(range=[0, df_plot['Volume'].max()*1.8], secondary_y=True, showgrid=False, row=1)
    fig.update_layout(height=height, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, barmode='overlay', margin=dict(l=10, r=10, t=30, b=10))
    return base64.b64encode(fig.to_image(format="png")).decode('utf-8')

def generate_stock_images(ticker):
    try:
        df_all = yf.download(ticker, period="4y", interval="1d", progress=False)
        if df_all.empty: return None, None, None, False
        if isinstance(df_all.columns, pd.MultiIndex): df_all.columns = df_all.columns.get_level_values(0)
        df_all = df_all.ffill()
        df_all['SMA20'] = df_all['Close'].rolling(window=20, min_periods=1).mean()
        df_all['SMA50'] = df_all['Close'].rolling(window=50, min_periods=1).mean()
        df_all['SMA200'] = df_all['Close'].rolling(window=200, min_periods=1).mean()
        exp1 = df_all['Close'].ewm(span=12, adjust=False).mean(); exp2 = df_all['Close'].ewm(span=26, adjust=False).mean()
        df_all['MACD'] = exp1 - exp2; df_all['Signal'] = df_all['MACD'].ewm(span=9, adjust=False).mean(); df_all['Hist'] = df_all['MACD'] - df_all['Signal']
        delta = df_all['Close'].diff(); gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean(); loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        df_all['RSI'] = 100 - (100 / (1 + gain/loss))

        img_1y = generate_chart(df_all.tail(252))
        img_max = generate_chart(df_all.tail(min(len(df_all), 756)))

        # 1分鐘圖：時區校準與懸浮長標籤
        df_1m = yf.download(ticker, period="1d", interval="1m", progress=False, prepost=True)
        img_1m = ""
        if not df_1m.empty:
            if isinstance(df_1m.columns, pd.MultiIndex): df_1m.columns = df_1m.columns.get_level_values(0)
            df_1m.index = df_1m.index.tz_convert('America/New_York')
            df_1m['Vol_Avg'] = df_1m['Volume'].rolling(window=5, min_periods=1).mean()
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Bar(x=df_1m.index, y=df_1m['Volume'], marker=dict(color='rgba(210, 210, 210, 0.8)', line_width=0)), secondary_y=True)
            fig2.add_trace(go.Candlestick(x=df_1m.index, open=df_1m['Open'], high=df_1m['High'], low=df_1m['Low'], close=df_1m['Close']), secondary_y=False)
            
            # 收盤標註 (長指引線)
            reg_session = df_1m[df_1m.index.time <= datetime.time(16, 0)]
            if not reg_session.empty:
                cp = reg_session.iloc[-1]; ct = reg_session.index[-1]; cpr = cp['Close']
                fig2.add_annotation(x=ct, y=cpr, text="🔔 CLOSE (EST)", showarrow=True, arrowhead=2, font=dict(color="white", size=10), bgcolor="#003366", ay=-50, yshift=10)
                fig2.add_shape(type="line", x0=df_1m.index[0], y0=cpr, x1=df_1m.index[-1], y1=cpr, line=dict(color="red", width=1.5, dash="dot"))

            # Top 10 爆量 (懸浮標籤)
            spikes = df_1m[df_1m['Volume'] > df_1m['Vol_Avg']*3].copy()
            for idx, row in spikes.sort_values(by='Volume', ascending=False).head(10).iterrows():
                t_color = "lime" if row['Close'] > row['Open'] else "red"
                fig2.add_annotation(x=idx, y=row['High'], text="▲ BUY" if row['Close'] > row['Open'] else "▼ SELL", showarrow=True, arrowhead=1, arrowcolor=t_color, font=dict(size=11, color=t_color, weight='bold'), bgcolor="black", opacity=0.9, ay=-40, yshift=5)
            fig2.update_layout(height=450, width=1050, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            img_1m = base64.b64encode(fig2.to_image(format="png")).decode('utf-8')

        return img_1y, img_max, img_1m, bool(df_all['Close'].iloc[-1] > df_all['SMA200'].iloc[-1])
    except: return None, None, None, False

# ==========================================
# 4. 批量 AI 分析 (深度與穩定性強化)
# ==========================================
def get_batch_ai_insights(df_subset):
    tickers = df_subset['Ticker'].tolist()
    if TEST_MODE: return {t: f"測試診斷 - {t}" for t in tickers}
    if not GEMINI_KEY: return {t: "❌ 無 API KEY" for t in tickers}

    summary = "".join([f"- {r['Ticker']}: 價格 ${r['Price']}, 漲幅 {r['Change']}%, 產業: {r['Industry']}\n" for _, r in df_subset.iterrows()])
    prompt = f"""
    你是一位華爾街專業分析師。請對以下股票進行【深度技術診斷】：
    {summary}
    要求內容涵蓋：
    1. 【趨勢】：分析 SMA20/50/200 的多空排列與支撐壓力位。
    2. 【動能】：解釋 MACD 柱狀圖變化與 RSI 是否過熱或背離。
    3. 【籌碼】：結合 1 分鐘圖 Top 10 爆量點說明主力意圖。
    
    每支股票字數必須在 150-200 字，口吻專業。使用繁體中文。
    必須回傳 JSON：{{"Ticker1": "深度內容", "Ticker2": "深度內容"}}。
    """
    try:
        client = genai.Client(api_key=GEMINI_KEY); response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        raw_text = response.text.strip().replace('```json', '').replace('```', '')
        time.sleep(50) # 確保穩定不遺漏
        return json.loads(raw_text)
    except: return {t: "⚠️ 由於分析請求密集，AI 分析暫時無法產出。" for t in tickers}

# ==========================================
# 5. HTML 生成 (智慧路徑與縮放啟用)
# ==========================================
def create_html_report(df):
    today_str = datetime.date.today().strftime("%Y%m%d")
    os.makedirs("history", exist_ok=True)
    
    history_files = sorted(glob.glob("history/report_*.html"), reverse=True)
    links_main = "".join([f'<a href="./history/report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])
    links_hist = "".join([f'<a href="./report_{f.split("_")[1][:8]}.html" class="history-item">{f.split("_")[1][:4]}-{f.split("_")[1][4:6]}-{f.split("_")[1][6:8]}</a>' for f in history_files])

    all_insights = {}
    print(f">>> [步驟 2] 開始全量分析 (共 {len(df)} 支符合漲幅條件之股票)...")
    for i in range(0, len(df), 2): all_insights.update(get_batch_ai_insights(df.iloc[i:i+2]))

    ICON_URL = "https://cdn-icons-png.flaticon.com/512/2422/2422796.png"
    def build_page(lks):
        return f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><link rel="icon" href="{ICON_URL}"><title>AI 美股深度掃描</title>
        <style>
            body {{ font-family: sans-serif; background: #f0f2f5; padding: 10px; margin: 0; }} .container {{ max-width: 1100px; margin: 0 auto; }}
            .history-bar {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; overflow-x: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .history-item {{ text-decoration: none; color: #666; padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px; margin-right: 10px; font-size: 12px; white-space: nowrap; }}
            .summary-table-wrapper {{ overflow-x: auto; }} .summary-table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 40px; font-size: 12px; min-width: 800px; }}
            .summary-table th {{ background: #003366; color: white; padding: 12px; }} .summary-table td {{ border-bottom: 1px solid #eee; text-align: center; padding: 10px; cursor: pointer; }}
            .stock-card {{ background: white; border-radius: 12px; margin-bottom: 60px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.15); scroll-margin-top: 20px; }}
            .card-header-row {{ background: #003366; color: white; padding: 12px; display: grid; grid-template-columns: 80px 200px 100px 80px 80px 80px 1fr; text-align: center; font-size: 13px; font-weight: bold; align-items: center; }}
            @media (max-width: 768px) {{ .card-header-row {{ grid-template-columns: repeat(2, 1fr); font-size: 11px; gap: 8px; }} }}
            .chart-stack {{ display: flex; flex-direction: column; gap: 20px; align-items: center; background: #1a1a1a; padding: 15px; }} .chart-stack img {{ width: 100%; height: auto; border: 1px solid #444; }}
            .toggle-bar {{ background: #333; padding: 10px; width: 100%; display: flex; justify-content: center; gap: 10px; border-bottom: 1px solid #444; }}
            .toggle-btn {{ background: #555; color: white; border: none; padding: 5px 20px; border-radius: 4px; cursor: pointer; font-size: 12px; }} .toggle-btn.active {{ background: #2563eb; font-weight: bold; }}
            .analysis-box {{ padding: 25px; line-height: 1.8; background: #f8fafc; font-size: 14px; border-top: 1px solid #eee; }}
            .btn-group {{ margin-top: 15px; display: flex; justify-content: flex-end; gap: 10px; }}
            .action-btn {{ background: #003366; color: white; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-size: 12px; font-weight: bold; border: none; cursor: pointer; }} .share-btn {{ background: #2563eb; }}
        </style>
        <script>
            function switchPeriod(ticker, period) {{
                const i1y = document.getElementById('img-1y-'+ticker); const im = document.getElementById('img-max-'+ticker);
                const b1y = document.getElementById('btn-1y-'+ticker); const bm = document.getElementById('btn-max-'+ticker);
                if(period === 'max') {{ i1y.style.display='none'; im.style.display='block'; b1y.classList.remove('active'); bm.classList.add('active'); }}
                else {{ i1y.style.display='block'; im.style.display='none'; b1y.classList.add('active'); bm.classList.remove('active'); }}
            }}
            async function shareTicker(t, p) {{
                const s = {{ title: `📈 AI 掃描: ${{t}}`, text: `代碼 ${{t}} 目前 $${{p}}。點擊查看詳細分析報告。`, url: window.location.origin + window.location.pathname + '?ticker=' + t }};
                try {{ if (navigator.share) {{ await navigator.share(s); }} else {{ alert('網址已複製'); navigator.clipboard.writeText(s.url); }} }} catch (e) {{}}
            }}
            window.onload = function() {{
                const p = new URLSearchParams(window.location.search); const t = p.get('ticker');
                if (t) {{ const e = document.getElementById(t.toUpperCase()); if (e) {{ setTimeout(() => {{ e.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }}, 600); }} }}
            }};
        </script></head><body><div class="container" id="top"><div class="history-bar"><div style="font-weight:bold;margin-right:10px;color:#003366;white-space:nowrap;">📅 歷史存檔：</div>{lks}</div>
        <h1 style="color:#003366; text-align:center;">📊 美股 AI 全量深度報告 {VERSION}</h1><div class="summary-table-wrapper"><table class="summary-table"><thead><tr><th>代碼</th><th>公司</th><th>產業</th><th>市值</th><th>P/E</th><th>價格</th><th>漲幅</th><th>成交量</th></tr></thead><tbody>"""

    html_mid = ""
    for _, row in df.iterrows():
        html_mid += f"<tr onclick=\"window.location='#{row['Ticker']}';\"><td><b>{row['Ticker']}</b></td><td>{row['Company']}</td><td>{row['Industry']}</td><td>{row['MarketCap']}</td><td>{row['PE']}</td><td>${row['Price']}</td><td style='color:red;'>+{row['Change']}%</td><td>{row['Volume']}</td></tr>"
    
    cards = ""
    for _, row in df.iterrows():
        i1, im, i1m, is_a = generate_stock_images(row['Ticker'])
        ins = all_insights.get(row['Ticker'], "⚠️ 分析產出中，請參考指標自行判斷。")
        if i1:
            cards += f"""<div class="stock-card" id="{row['Ticker']}"><div class="card-header-row"><div>{row['Ticker']}</div><div>{row['Industry']}</div><div>{row['MarketCap']}</div><div>{row['PE']}</div><div>${row['Price']}</div><div style="color:#ffcccc;">+{row['Change']}%</div><div>{row['Volume']}</div></div>
            <div class="toggle-bar"><button id="btn-1y-{row['Ticker']}" class="toggle-btn active" onclick="switchPeriod('{row['Ticker']}', '1y')">📅 1Y 日線</button><button id="btn-max-{row['Ticker']}" class="toggle-btn" onclick="switchPeriod('{row['Ticker']}', 'max')">♾️ MAX (Up to 3Y)</button></div>
            <div class="chart-stack"><img id="img-1y-{row['Ticker']}" src="data:image/png;base64,{i1}"><img id="img-max-{row['Ticker']}" src="data:image/png;base64,{im}" style="display:none;"><img src="data:image/png;base64,{i1m}"></div>
            <div class="analysis-box"><strong>🛡️ AI 策略師深度診斷：</strong><br>{ins}<div class="btn-group"><button class="action-btn share-btn" onclick="shareTicker('{row['Ticker']}', '{row['Price']}')">📲 分享此股票</button><a href="#top" class="action-btn">⬆ 返回總表</a></div></div></div>"""
    
    with open("index.html", "w", encoding="utf-8") as f: 
        f.write(build_page(links_main) + html_mid + "</tbody></table></div>" + cards + "</div></body></html>")
    with open(f"history/report_{today_str}.html", "w", encoding="utf-8") as f: 
        f.write(build_page(links_hist) + html_mid + "</tbody></table></div>" + cards + "</div></body></html>")
    print(f"✅ v1.5.4 報告產出完成。已過濾 >40% 暴漲股。")

if __name__ == "__main__":
    if is_market_open_today():
        df = fetch_and_filter_stocks()
        if not df.empty: create_html_report(df)
