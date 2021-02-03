import re
import requests
from bs4 import BeautifulSoup
import pickle
import time
import pandas as pd

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
ARCHIVE_URL = 'https://www.lawinenwarndienst-bayern.de/res/archiv/lageberichte/'


# ARCHIVE

r = requests.get(ARCHIVE_URL, headers = HEADERS)

soup = BeautifulSoup(r.text).find_all(class_ = "list-group-item")
lageberichte_urls = []
for url in soup:
    lageberichte_urls.append(ARCHIVE_URL + url.get("href"))
    
pickle.dump(lageberichte_urls, open("lageberichte_urls.p", "wb"))  

lageberichte_urls = lageberichte_urls[2:] # exclude from 2019 on, as they switched to different layout

# YEARLY "LAGEBERICHTE"

daily_urls = []

for url in lageberichte_urls:
    r = requests.get(url, headers = HEADERS)
    soup = BeautifulSoup(r.text).find_all(class_ = "list-group-item")
    for url in soup:
        daily_urls.append(ARCHIVE_URL + url.get("href"))
    time.sleep(3)
    
pickle.dump(daily_urls, open("daily_urls.p", "wb"))  
daily_urls = pickle.load(open("daily_urls.p", "rb"))
    
# DAILY "LAGEBERICHT"

warning_levels = pd.DataFrame([])

for i, url in enumerate(daily_urls[0:1]):
    print(f"Processing url {i+1} out of {len(daily_urls)}")
    r = requests.get(url, headers = HEADERS)
    soup = BeautifulSoup(r.text)
    date = "".join(re.findall(r"[0-9].+[0-9]{4}", soup.h4.text))
    try:
        danger_per_zone = {}
        for zone, html_code in enumerate(soup.find(class_ = "panel-body").find_all("img")):
            danger_code = "".join(re.findall(r"[0-9]", str(html_code.get("src"))))
            danger_per_zone.setdefault(zone, danger_code)
        warning_levels = warning_levels.append(pd.DataFrame(danger_per_zone, index = [date]))
        print("Found values!")
        
    except AttributeError:
        print("Skipping...")
        continue
    finally:
        time.sleep(2)
        
        
url = "http://asdasdasd.de"

daily_urls[0:10]

url = "https://www.lawinenwarndienst-bayern.de/res/archiv/lageberichte/lagebericht.php?id=2301&lb_zur=1367359200"


for zone, html in enumerate(soup.find(class_ = "panel-body").find_all("img")):
    print(html)
    
soup.find(class_ = "panel-body").find_all("img")