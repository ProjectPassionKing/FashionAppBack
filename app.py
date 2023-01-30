from __future__ import division, print_function
from keras.applications.resnet import ResNet50
# coding=utf-8
import numpy as np
from keras.utils import load_img, img_to_array
import base64

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Check https://keras.io/applications/
model = ResNet50(weights='imagenet')
model.save('')


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    # image_b64 = request.form.get("file")
    image_b64 = request.get_json()["file"]
    image_bytes = base64.b64decode(image_b64)
    with open("./uploads/29.png", "wb") as f:
        f.write(image_bytes)

    # Make prediction
    preds = model_predict("./uploads/29.png", model)

    # Process your result for human
    # pred_class = preds.argmax(axis=-1)            # Simple argmax
    pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    result = str(pred_class[0][0][1])               # Convert to string
    return jsonify({'prediction': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
