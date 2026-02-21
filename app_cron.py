import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_and_probe():
    print("=== [Step 1] 開始連線至 Finviz 探測原始數據 ===")
    
    # 使用你要求的 v=111 視圖與精準參數
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Referer': 'https://finviz.com/',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        print(f"連線狀態碼: {resp.status_code}")
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 1. 檢查網頁標題 (這能判斷是否被 Cloudflare 擋住)
        title = soup.title.text if soup.title else "No Title"
        print(f"網頁標題: {title}")
        
        # 2. 印出前 500 個字元的 HTML，確認內容是否為正常的股票頁面
        print("\n--- 網頁內容片段 (前500字) ---")
        print(resp.text[:500])
        print("----------------------------\n")

        # 3. 找出所有 tr (資料列)
        # Finviz 的資料列通常帶有這個特定的 class
        rows = soup.find_all('tr', class_='screener-body-table-nw')
        
        # 如果沒找到，改找一般的 tr
        if not rows:
            print("找不到特定 class 的行，嘗試抓取所有 <tr>...")
            rows = soup.find_all('tr')

        print(f"全網頁共找到 {len(rows)} 個 <tr> 標籤")

        # 4. 開始列印每一欄的內容 (取前 3 支股票作為樣本)
        count = 0
        for i, r in enumerate(rows):
            tds = r.find_all('td')
            # 過濾掉欄位太少的行（廣告或表頭）
            if len(tds) < 10:
                continue
            
            ticker = tds[1].text.strip()
            # 排除表頭行 (第一列通常是 "Ticker")
            if ticker == "Ticker" or not ticker:
                continue
                
            count += 1
            print(f"\n[樣本 {count}] 股票代號: {ticker}")
            for idx, td in enumerate(tds):
                print(f"  索引 [{idx}]: {td.text.strip()}")
            
            if count >= 3: # 只看前 3 支就夠了
                break
        
        if count == 0:
            print("❌ 警告：雖然找到標籤，但無法解析出任何有效的股票資料列。")

    except Exception as e:
        print(f"❌ 執行發生錯誤: {e}")

if __name__ == "__main__":
    fetch_and_probe()
    # 建立一個簡單的 report.pdf 避免 GitHub Action 上傳步驟報錯
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(200, 10, text="Debug run completed. Check Logs for data.", align='C')
    pdf.output("report.pdf")
