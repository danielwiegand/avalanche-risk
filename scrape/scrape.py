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

# TODO: Make the danger code integers when scraping

warning_levels = pd.DataFrame([])

for i, url in enumerate(daily_urls):
    print(f"Processing url {i+1} out of {len(daily_urls)}")
    r = requests.get(url, headers = HEADERS)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text)
        images = soup.find(class_ = "panel-body").find_all("img")
        if len(images) > 0:
            date = "".join(re.findall(r"[0-9].+[0-9]{4}", soup.h4.text))
            danger_per_zone = {}
            for zone, html_code in enumerate(images):
                danger_code = "".join(re.findall(r"[0-9]", str(html_code.get("src"))))
                danger_per_zone.setdefault(zone+1, danger_code)
            warning_levels = warning_levels.append(pd.DataFrame(danger_per_zone, index = [date]))
            print("Found values!")
            time.sleep(2)
        else:
            print("Skipping...")
            time.sleep(2)
            continue

# Remove some days without danger levels
warning_levels = warning_levels[warning_levels[9].isna()].loc[:,0:6]

pickle.dump(warning_levels, open("../data/warning_levels.p", "wb")) 

warning_levels.to_csv("../data/warnstufen_archiv.csv")