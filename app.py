from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model(
    "model/model.h5"
)

classes = [
    "Healthy",
    "Rust",
    "LeafSpot",
    "Blight"
]

treatments = {
    "Healthy":"No treatment required",
    "Rust":"Apply fungicide spray",
    "LeafSpot":"Remove infected leaves",
    "Blight":"Use copper fungicide"
}

def prepare_image(path):

    img = Image.open(path)

    img = img.resize((128,128))

    img = np.array(img)/255.0

    img = np.expand_dims(img, axis=0)

    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = prepare_image(filepath)

    prediction = model.predict(image)[0]

    index = np.argmax(prediction)

    disease = classes[index]

    confidence = round(
        float(prediction[index])*100,
        2
    )

    treatment = treatments[disease]

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        treatment=treatment
    )

if __name__ == "__main__":
    app.run(debug=True)