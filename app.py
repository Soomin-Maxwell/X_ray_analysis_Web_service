from flask import Flask, render_template, request, jsonify,redirect

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from PIL import Image
from io import BytesIO


model = None
app = Flask(__name__)


def load_model():
    json_file = open('model/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model
    model = model_from_json(loaded_model_json)

    model.load_weights("model/model.h5")
    print(model.summary())

def process_image(image):
    #read image
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((96, 96))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["POST","GET"])
def index():
    predictions = {}
    if request.method == "POST":
        # only make predictions after sucessfully receiving the file
        if request.files:
            try:
                image = request.files["image"].read()
                image = process_image(image)
                out = model.predict(image)
                # send the predictions to index page
                predictions = {"Cardiomegaly":str(np.round(out[0][0],2)),
                                "Emphysema":str(np.round(out[0][1],2)),
                                "Effusion":str(np.round(out[0][2],2)),
                                "Hernia":str(np.round(out[0][3],2)),
                                "Infiltration":str(np.round(out[0][4],2)),
                                "Mass":str(np.round(out[0][5],2)),
                                "Nodule":str(np.round(out[0][6],2)),
                                "Atelectasis":str(np.round(out[0][7],2)),
                                "Pneumothorax":str(np.round(out[0][8],2)),
                                "Pleural_Thickening":str(np.round(out[0][9],2)),
                                "Pneumonia":str(np.round(out[0][10],2)),
                                "Fibrosis":str(np.round(out[0][11],2)),
                                "Edema":str(np.round(out[0][12],2)),
                                "Consolidation":str(np.round(out[0][13],2))
                                }
            except:
                predictions ={}
                redirect('/')   
    return render_template("index.html",predictions=predictions)

if __name__ == "__main__":
    load_model()
    app.run(debug = True, threaded = False)

if __name__ == "app":
    load_model()
