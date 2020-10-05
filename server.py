import time
from selenium import webdriver
from bs4 import BeautifulSoup

from config import Env

ScrapingTime = 3
driver_path = Env.ChromeDriver

options = webdriver.chrome.options.Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--user-agent=hogehoge')

#特定のWEBページからURLリストを取得する
def getScrapingImageURLs(url):
    time.sleep(ScrapingTime)
    driver = webdriver.Chrome(chrome_options=options)
    driver.get(url)
    html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.find_all('img')
    image_urls = []
    for img in imgs:
        image_urls.append(img["src"])
    return image_urls

if __name__ == "__main__":
    save_dir = "./test"

    #画像の取得チェック
    # url = 'https://www.pinterest.jp/search/pins/?q=%E6%B2%B9%E7%B5%B5&rs=rs&eq=&etslf=1971&term_meta[]=%E6%B2%B9%E7%B5%B5%7Crecentsearch%7C2'
    # image_urls = getScrapingImageURLs(url)
    # save_urls = saveImageFromURLs(image_urls, save_dir)