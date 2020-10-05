import os
import time
from io import BytesIO, StringIO
from PIL import Image
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib
import ulid
from config import Env

ScrapingTime = Env.ScrapingCoolTime
driver_path = Env.ChromeDriver

options = webdriver.chrome.options.Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--user-agent=hogehoge')

#特定のWEBページからURLリストを取得する
def getScrapingImageURLs(url):
    time.sleep(ScrapingTime)
    driver = webdriver.Chrome(executable_path=driver_path, chrome_options=options)
    #driver = webdriver.Chrome(chrome_options=options)
    driver.get(url)
    html = driver.page_source.encode('utf-8')
    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.find_all('img')
    image_urls = []
    for img in imgs:
        image_urls.append(img["src"])
    return image_urls

#画像のURLリストから指定ディレクトリに保存する
def saveImageFromURLs(image_urls, save_dir, save_type="local"):
    save_urls = []
    for index, image_url in enumerate(image_urls):
        filename = getULIDStr() + ".jpg"
        save_path = convDirAddName(save_dir, filename)
        try:
            pil_image = urlToPILImage(image_url)
            save_url = savePILImageInLocal(save_path, pil_image)
        except Exception as e:
            print("saveImageFromURLs Error")
            print(e)
            os.remove(save_path)
    return save_urls

def convDirAddName(dirname, filename):
    if dirname[-1] == "/" or dirname[-1] == "\\":
        return dirname+filename
    return dirname + "/" + filename

def urlToPILImage(url):
    f = BytesIO(urllib.request.urlopen(url).read())
    return Image.open(f)

def savePILImageInLocal(save_path, pil_image):
    pil_image.save(save_path)
    return save_path

def getULIDStr():
    id1 = ulid.new()
    return id1.str

if __name__ == "__main__":
    save_dir = "images/color"
    url = 'https://500px.com/search?q=%E9%9B%B2&type=photos&sort=relevance'#スクレイピングしたいURL
    image_urls = getScrapingImageURLs(url)
    print(image_urls)
    save_urls = saveImageFromURLs(image_urls, save_dir)
    print("save complete")