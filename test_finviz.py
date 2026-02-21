import requests
from bs4 import BeautifulSoup
import pandas as pd

def probe_finviz(test_name, url):
    print(f"\n=== 測試項目: {test_name} ===")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://finviz.com/'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        print(f"狀態碼: {resp.status_code}")
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 尋找所有可能的表格列
        rows = soup.find_all('tr', class_='screener-body-table-nw')
        if not rows:
            table = soup.find('table', class_='table-light')
            rows = table.find_all('tr')[1:] if table else []

        print(f"找到原始資料列數: {len(rows)}")

        # 只探測前 2 支股票，印出所有欄位內容
        for i, r in enumerate(rows[:2]):
            tds = r.find_all('td')
            ticker = tds[1].text.strip() if len(tds) > 1 else "Unknown"
            print(f"\n[股票 {i+1}: {ticker}] 欄位探測:")
            for idx, td in enumerate(tds):
                print(f"  索引 [{idx}]: {td.text.strip()}")
                
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    # 你提供的原始網址 (Overview 視圖 v=111)
    url_v111 = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o500,sh_price_o1,sh_relvol_o5,ta_change_u"
    
    # 測試 A: 你提供的網址
    probe_finviz("原始 v=111 視圖", url_v111)
    
    # 測試 B: 技術分析視圖 v=171 (通常這會包含更多技術指標)
    url_v171 = url_v111.replace("v=111", "v=171")
    probe_finviz("切換 v=171 視圖", url_v171)
