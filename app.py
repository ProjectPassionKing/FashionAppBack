from __future__ import division, print_function
import os
from keras.applications.resnet import ResNet50
import numpy as np
from keras.utils import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from glob import glob
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64

# Define a flask app
app = Flask(__name__)

# Check https://keras.io/applications/
# model = ResNet50(weights='imagenet')


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


# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return "Hello world"


@app.route('/', methods=['POST'])
def predict():
    f = request.files["file"]

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    preds = model_predict(file_path, model)

    # Process your result for human
    # pred_class = preds.argmax(axis=-1)            # Simple argmax
    pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    result = str(pred_class[0][0][1])               # Convert to string
    return jsonify({'prediction': str(result)})


@app.route('/sim', methods=['POST'])
def simulate():
    # load a pretrained model (recommended for training)
    model = YOLO('./simulate/top_pants.pt')

    f = request.files["file"]
    f.save("uploads\person\person_1.jpg")
    human_path = "uploads\person\person_1.jpg"

    clothes_dir = glob("uploads\clothes\*.jpg")

    human_img = cv2.imread(human_path)
    # human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
    human_result = model.predict(human_path)[0].boxes

    boxes = {}
    for box in human_result:
        boxes[box.cls[0].item()] = box.xyxy[0].numpy().astype('int')
    print(boxes)

    for clothes_path in clothes_dir:
        clothes_img = cv2.imread(clothes_path)
        # clothes_img = cv2.cvtColor(clothes_img, cv2.COLOR_BGR2RGB)
        clothes_result = model.predict(clothes_path)[0].boxes
        clothes_label = clothes_result[0].cls[0].item()
        clothes_box = clothes_result[0].xyxy[0].numpy().astype('int')

        if clothes_label not in boxes.keys():
            continue

        xy = boxes[clothes_label]

        target = clothes_img[clothes_box[1]:clothes_box[3], clothes_box[0]:clothes_box[2]]
        h, w, c = human_img[xy[1]:xy[3], xy[0]:xy[2]].shape
        target = cv2.resize(target, (w, h))

        alpha = 0.35
        human_img[xy[1]: xy[3], xy[0]:xy[2]] = cv2.addWeighted(
            human_img[xy[1]: xy[3], xy[0]:xy[2]], alpha, target, 1-alpha, 0)
    plt.imshow(human_img)

    result_path = r"uploads\result\result_1.jpg"

    cv2.imwrite(result_path, human_img)

    with open(result_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())

    return b64_string


if __name__ == '__main__':
    app.run(debug=True)

# py -m flask run --host=0.0.0.0 --port=5000
