import os
import time
import datetime
import io
import tempfile
import logging
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from google import genai

# ==========================================
# 1. 核心參數與環境設定
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GEMINI_KEY           = os.getenv("GEMINI_API_KEY")
EXCLUDE_INDUSTRIES   = {"Shell Companies", "Blank Check", "SPAC"}
MAX_CHANGE           = 40.0          # 漲跌幅上限（%）
MAX_STOCKS_TO_ANALYZE = 10           # 詳細分析的標的上限
AI_RATE_LIMIT_SLEEP  = 45            # Gemini 免費版安全間隔（秒）
AI_RETRY_SLEEP       = 60            # 429 後等待時間（秒）
AI_MAX_RETRIES       = 2
TARGET_MODEL         = "models/gemini-2.5-pro"

FINVIZ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}

# 初始化 AI 客戶端（僅在 key 存在時）
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
if not client:
    log.warning("GEMINI_API_KEY not set — AI insights will be skipped.")


# ==========================================
# 2. 工具：建立帶自動重試的 requests Session
# ==========================================
def _build_session() -> requests.Session:
    """Return a requests Session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ==========================================
# 3. Finviz 爬蟲（強韌版 + 動態欄位對應）
# ==========================================
def fetch_and_filter_stocks() -> pd.DataFrame:
    log.info("[Step 1] Connecting to Finviz...")
    filters = "ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    url = f"https://finviz.com/screener.ashx?v=111&f={filters}"

    session = _build_session()
    try:
        resp = session.get(url, headers=FINVIZ_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Network error fetching Finviz: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 找含 "Ticker" 的目標表格
    target_table = next(
        (t for t in soup.find_all("table") if "Ticker" in t.text[:200]),
        None,
    )
    if target_table is None:
        log.error("Could not locate screener table in Finviz HTML.")
        return pd.DataFrame()

    # --- 動態讀取表頭，建立欄位 index 對應 ---
    header_row = target_table.find("tr")
    if header_row is None:
        log.error("No header row found in table.")
        return pd.DataFrame()

    headers = [th.text.strip() for th in header_row.find_all("td")]
    col_map = {name: idx for idx, name in enumerate(headers)}

    required = {"No.", "Ticker", "Company", "Sector", "Industry",
                "Country", "Market Cap", "P/E", "Price", "Change", "Volume"}
    missing = required - set(col_map)
    if missing:
        # Fallback: use positional indices (original behaviour)
        log.warning(f"Header mismatch, falling back to positional parsing. Missing: {missing}")
        col_map = {
            "Ticker": 1, "Company": 2, "Sector": 3, "Industry": 4,
            "Country": 5, "Market Cap": 6, "P/E": 7, "Price": 8,
            "Change": 9, "Volume": 10,
        }

    def _safe_float(text: str) -> float | None:
        try:
            return float(text.strip().strip("%").replace(",", ""))
        except ValueError:
            return None

    rows = target_table.find_all("tr", valign="top")
    data = []
    skipped = 0
    for r in rows:
        tds = r.find_all("td")
        if len(tds) < max(col_map.values()) + 1:
            continue
        try:
            price  = _safe_float(tds[col_map["Price"]].text)
            change = _safe_float(tds[col_map["Change"]].text)
            if price is None or change is None:
                raise ValueError("Non-numeric price or change")
            data.append({
                "Ticker":    tds[col_map["Ticker"]].text.strip(),
                "Company":   tds[col_map["Company"]].text.strip(),
                "Sector":    tds[col_map["Sector"]].text.strip(),
                "Industry":  tds[col_map["Industry"]].text.strip(),
                "Country":   tds[col_map["Country"]].text.strip(),
                "MarketCap": tds[col_map["Market Cap"]].text.strip(),
                "PE":        tds[col_map["P/E"]].text.strip(),
                "Price":     price,
                "Change":    change,
                "Volume":    tds[col_map["Volume"]].text.strip(),
            })
        except (ValueError, IndexError) as e:
            skipped += 1
            log.debug(f"Skipping malformed row: {e}")

    if skipped:
        log.warning(f"Skipped {skipped} malformed row(s) during parsing.")

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # 套用過濾條件
    df = df[~df["Industry"].isin(EXCLUDE_INDUSTRIES)]
    df = df[df["Change"].abs() <= MAX_CHANGE]
    log.info(f"Screener returned {len(df)} qualifying ticker(s).")
    return df.reset_index(drop=True)


# ==========================================
# 4. 圖表生成（修正 MultiIndex、NumPy 警告）
# ==========================================
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance MultiIndex columns to a flat level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _scalar(value) -> float:
    """Safely extract a Python scalar from a possibly array-like value."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__len__") and len(value) == 1:
        return float(value.iloc[0] if hasattr(value, "iloc") else value[0])
    return float(value)


def generate_charts(ticker: str) -> tuple[io.BytesIO | None, bool | None]:
    """
    Returns (image_buffer, is_above_200ma).
    Returns (None, None) on failure — callers must handle None explicitly.
    """
    log.info(f"[Step 2] Generating charts for {ticker}...")
    try:
        df_daily = yf.download(ticker, period="2y", interval="1d",
                               progress=False, threads=False)
        df_daily = _flatten_columns(df_daily)

        if df_daily.empty or len(df_daily) < 200:
            log.warning(f"{ticker}: insufficient daily data ({len(df_daily)} rows).")
            return None, None

        df_daily["200MA"] = df_daily["Close"].rolling(window=200).mean()

        last_close = _scalar(df_daily["Close"].iloc[-1])
        last_ma200 = _scalar(df_daily["200MA"].iloc[-1])
        is_above_200 = last_close > last_ma200

        df_1m = yf.download(ticker, period="1d", interval="1m",
                            progress=False, threads=False)
        df_1m = _flatten_columns(df_1m)

        fig = make_subplots(
            rows=2, cols=2,
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f"{ticker} Daily (2Y)", f"{ticker} 1m Intraday",
                "Daily Volume", "1m Volume"
            ),
        )

        # Daily candlestick
        fig.add_trace(go.Candlestick(
            x=df_daily.index,
            open=df_daily["Open"], high=df_daily["High"],
            low=df_daily["Low"],  close=df_daily["Close"],
            name="Price",
        ), row=1, col=1)

        # 200MA line
        fig.add_trace(go.Scatter(
            x=df_daily.index, y=df_daily["200MA"],
            line=dict(color="yellow", width=2), name="200MA",
        ), row=1, col=1)

        # Daily volume
        fig.add_trace(go.Bar(
            x=df_daily.index, y=df_daily["Volume"],
            marker_color="steelblue", name="Vol",
        ), row=2, col=1)

        if not df_1m.empty:
            o_arr = df_1m["Open"].to_numpy().flatten()
            c_arr = df_1m["Close"].to_numpy().flatten()
            colors_1m = ["green" if c >= o else "red" for o, c in zip(o_arr, c_arr)]

            fig.add_trace(go.Candlestick(
                x=df_1m.index,
                open=df_1m["Open"], high=df_1m["High"],
                low=df_1m["Low"],  close=df_1m["Close"],
                name="1m Price",
            ), row=1, col=2)

            fig.add_trace(go.Bar(
                x=df_1m.index, y=df_1m["Volume"],
                marker_color=colors_1m, name="1m Vol",
            ), row=2, col=2)

        fig.update_layout(
            height=600, width=1100,
            template="plotly_dark",
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )

        img_bytes = fig.to_image(format="png")
        return io.BytesIO(img_bytes), is_above_200

    except Exception as e:
        log.error(f"{ticker} chart generation failed: {e}", exc_info=True)
        return None, None


# ==========================================
# 5. AI 分析（改進重試 + 指數退避）
# ==========================================
def get_ai_insight(row: pd.Series, is_above_200: bool) -> str:
    """Call Gemini for a trading insight. Returns a string (may be a fallback message)."""
    if not client:
        return "AI analysis skipped: GEMINI_API_KEY not configured."

    status_200ma = "above" if is_above_200 else "below"
    prompt = (
        f"Analyze US stock {row['Ticker']} ({row['Company']}), "
        f"industry: {row['Industry']}, market cap: {row['MarketCap']}, "
        f"price: {row['Price']}, currently {status_200ma} its 200-day MA. "
        f"Today's change: {row['Change']}%. "
        "Provide a win-probability score (0–100) and specific actionable advice "
        "including entry, target, and stop-loss levels."
    )

    wait = AI_RETRY_SLEEP
    for attempt in range(1, AI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=TARGET_MODEL, contents=prompt)
            time.sleep(AI_RATE_LIMIT_SLEEP)
            return response.text
        except Exception as e:
            err = str(e)
            if "429" in err:
                log.warning(f"Rate limit hit (attempt {attempt}/{AI_MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 300)   # 指數退避，上限 5 分鐘
            else:
                log.error(f"Gemini error for {row['Ticker']}: {e}")
                break

    return "AI analysis skipped after retries."


# ==========================================
# 6. PDF 報告（Unicode 支援 + 安全的臨時檔案處理）
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(
            0, 10,
            text=f"US Stock AI Report — {datetime.date.today()}",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.ln(5)

    def _safe_text(self, text: str) -> str:
        """Replace characters that fpdf2 latin-1 mode cannot render."""
        replacements = {
            "\u2022": "-", "\u2013": "-", "\u2014": "--",
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        }
        for char, sub in replacements.items():
            text = text.replace(char, sub)
        # Encode to latin-1 dropping anything still unmappable
        return text.encode("latin-1", errors="ignore").decode("latin-1")


def create_report(df: pd.DataFrame, output_path: str = "report.pdf") -> None:
    log.info("[Step 3] Building PDF report...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Summary page ---
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, text="Market Scan Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    col_widths = [22, 55, 22, 25, 35]
    col_labels = ["Ticker", "Industry", "Price", "Change %", "Mkt Cap"]

    pdf.set_font("helvetica", "B", 8)
    for label, w in zip(col_labels, col_widths):
        pdf.cell(w, 8, text=label, border=1, align="C")
    pdf.ln()

    pdf.set_font("helvetica", "", 8)
    for _, row in df.iterrows():
        pdf.cell(col_widths[0], 8, text=str(row["Ticker"]), border=1)
        pdf.cell(col_widths[1], 8, text=str(row["Industry"])[:24], border=1)
        pdf.cell(col_widths[2], 8, text=str(row["Price"]), border=1, align="R")
        pdf.cell(col_widths[3], 8, text=f"{row['Change']:.2f}%", border=1, align="R")
        pdf.cell(col_widths[4], 8, text=str(row["MarketCap"]), border=1,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # --- Per-stock detail pages ---
    for _, row in df.head(MAX_STOCKS_TO_ANALYZE).iterrows():
        ticker = row["Ticker"]
        img_buf, is_above_200 = generate_charts(ticker)

        if img_buf is None:
            log.warning(f"Skipping {ticker}: chart generation failed.")
            continue

        # is_above_200 is None only when chart failed — already handled above
        ai_text = get_ai_insight(row, bool(is_above_200))

        pdf.add_page()
        pdf.set_font("helvetica", "B", 13)
        pdf.cell(
            0, 10,
            text=f"{ticker} — {row['Company']}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        # Use a proper temp file — deleted automatically even if an error occurs
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_buf.getbuffer())
            tmp_path = tmp.name

        try:
            pdf.image(tmp_path, x=10, y=30, w=190)
        finally:
            os.remove(tmp_path)

        pdf.set_y(155)
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 8, text="AI Strategist Insight:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 6, text=pdf._safe_text(ai_text))

    pdf.output(output_path)
    log.info(f"Report saved to: {output_path}")


# ==========================================
# 7. 主程式
# ==========================================
if __name__ == "__main__":
    df_found = fetch_and_filter_stocks()

    if df_found.empty:
        log.info("No qualifying stocks found today.")
    else:
        log.info(f"Found {len(df_found)} stock(s). Analyzing top {MAX_STOCKS_TO_ANALYZE}...")
        create_report(df_found)
