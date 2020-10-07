from flask import Flask, request, render_template, abort
import base64
from PIL import Image
from io import BytesIO
from auto_color import autoColor

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/auto-color", methods=['POST'])
def auto_color():
    paint_image = request.form['paint_image']
    img_binary = base64.b64decode(paint_image)
    pil_image = Image.open(BytesIO(img_binary))
    resultImage = autoColor(pil_image)
    buffered = BytesIO()
    resultImage.save(buffered, format="JPEG")
    img_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")
    return img_str

if __name__ == '__main__':
    app.run()
