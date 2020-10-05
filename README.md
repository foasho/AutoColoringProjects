#『自動着彩AI』アプリを作ってみよう
##はじめに
###最終ゴールを決める
最終ゴールは、簡単な線を描くアプリと
その線から自動で絵を着彩するようなサービスを作りたいと思います。
今回の開発環境について
```commandline
ライブラリ等：Python3.6＋Keras2.2.4(Tensorflow-gpu1.14.0)
Windows10
開発の流れ
1.フォルダ構成と設計
2.スクレイピングによる画像収集
3.線画データの抽出
4.モデルの学習と評価
5.サーバー化してサービスにする
```
以上の5部構成にしたいと思います。

##1.フォルダ構成と設計
```commandline
-projects
 |-images
   |-example(評価用画像格納フォルダ)
   |-generated_images(学習中の経過画像格納フォルダ)
   |-edge(線画画像の格納フォルダ)
   |-color(元画像の格納フォルダ)
 |-templates
   -index.html
 |-static
 |-model(作成したモデルを保存するフォルダ)
 |-config.py
 |-scraping.py
 |-edge_transration.py
 |-train.py
 |-server.py
```

##2.スクレイピングによる画像収集
スクレイピングには以下のライブラリを利用します。
```commandline
pip install Pillow==7.1.1
pip install selenium==3.141.0
pip install bs4==0.0.1
pip install urllib3==1.25.8
pip install ulid-py==1.1.0
```

次に、Chromeのドライバーをダウンロードします。
[chromedriver_version = 84.0.4147.30のダウンロード](http://chromedriver.chromium.org/downloads)
![download](images/example/cdriver.jpg "サンプル")
windowsなので、chromedriver_win32.zipを選択しダウンロード、
staticの下にdriverとディレクトリを作成し、そこに解凍。
/static/driver/chromedriver.exeとなるようにしてください。

次はようやくコーディングに入ります。
まずは,config.pyの編集から
```python
import os
abs_path = os.getcwd()

class Env:
    ChromeDriver = abs_path + "\static\driver\chromedriver.exe"
    ScrapingCoolTime = 3
```

scraping.pyの編集
```python
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
    url = ''#スクレイピングしたいURL
    image_urls = getScrapingImageURLs(url)
    print(image_urls)
    save_urls = saveImageFromURLs(image_urls, save_dir)
    print("save complete")
```

urlのところにスクレイピングしたいURLを入れることでダウンロードができます。
高品質な写真が欲しいため、今回のスクレイピング先は"500px"というサービスから取得することにしました。
取得する画像は[青空](https://500px.com/search?submit=%E9%80%81%E4%BF%A1&q=%E9%9D%92%E7%A9%BA&type=photos)

```commandline
python scraping.py
```
これで実行すると、青空をテーマにした画像が50枚程度ダウンロードできました。
ただこれでは少ないので、空と雲の検索URLもスクレイピングし、150枚の画像を収集しました。
当然画像は少ないですが、とりあえずこれで進めていきます。

