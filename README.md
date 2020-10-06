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


##2.画像データの線画化
こんにちは、前回に引き続き、自動着彩AIアプリを作ろう第3回目です。
読むのが面倒なときは、
```commandline
git clone https://github.com/foasho/AutoColoringProjects.git
pip install -r requirements.txt
python edge_transration.py
```
この3行で終わります。

まず必要なライブラリのインストールから
```commandline
pip install opencv-contrib-python==4.2.0.34
pip install opencv-python==4.2.0.34
pip install numpy==1.16.4
```

今日編集するのは、edge_transration.py
早速コードを書いていきましょう。
```python
import cv2
import numpy as np
import os

#線画変換
def edge_detect(cv2img):
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    cv2.merge((gray, gray, gray), cv2img)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(cv2img, kernel, iterations=1)
    diff = cv2.subtract(dilation, cv2img)
    negaposi = 255 - diff
    return negaposi

if __name__=='__main__':
    target_dir = "./images/color/"
    save_dir_edge = "./images/edge/"
    if not os.path.exists(save_dir_edge):
        os.makedirs(save_dir_edge)
    image_names = os.listdir(target_dir)
    for image_name in image_names:
        read_path = target_dir + image_name
        try:
            color_image = cv2.imread(read_path)
            edge_image = edge_detect(color_image)
            cv2.imwrite(save_dir_edge + image_name + ".jpg", edge_image)
        except Exception as e:
            print(read_path)
            print("error")
            print(e)
```

たったこれだけです。
あとは、実行
```commandline
python edge_transration.py
```
これでimages/edge内に線画の画像が追加されたのを確認できればOKです。

お疲れ様でした。
次はとうとう用意した線画とカラー画像を使ってAIモデル学習をします。
PCはGPUでの計算推奨です。
されていない場合は、[Qiitaで設定の仕方の記事](https://qiita.com/osakasho/items/e3b0b14bd26ae1060413)
を書いているので、そちらを行ってから、次のセクションを行ってください。

ではまた。

##4.AIモデルの作成と評価

