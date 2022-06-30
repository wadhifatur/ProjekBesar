from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0: 'tomat 1', 1: 'tomat 2'}

model = load_model('tomat_detection.h5')

model.make_predict_function()


def predict_label(img_path):  # buat 2 pengkonsian 0 fake 1 real
    i = image.load_img(img_path, target_size=(200, 200))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 200, 200, 3)
    p = np.argmax(model.predict(i), axis=1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please upload image file!!!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
