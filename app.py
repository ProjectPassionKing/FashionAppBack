from __future__ import division, print_function
import json
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import base64
from recommend import get_models, recommend
from view import view

sim_model = YOLO('models/top_pants.pt')
models = get_models()

# Define a flask app
app = Flask(__name__)

# Check https://keras.io/applications/


@app.route('/', methods=['GET'])
def index():
    # Main page
    return "Hello world"


@app.route('/pred', methods=['POST'])
def predict():
    f = request.files["file"]

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # Make prediction
    paths, keywords, color = recommend(sim_model, models, file_path,
                                       gender='여자', weight='straight')

    if paths == None:
        return Response(json.dumps({"response": "error"}), mimetype="application/json")

    outer = keywords[0]
    top = keywords[1]
    bottom = keywords[2]
    img = view(paths[0], paths[1], paths[2], color)
    cv2.imwrite('result.jpg', img)

    with open("result.jpg", "rb") as f1:
        file1_data = f1.read()

    file1_encoded = base64.b64encode(file1_data).decode("utf-8")

    files = {
        "file1": file1_encoded
    }

    return Response(json.dumps(files), mimetype="application/json")


@app.route('/sim', methods=['POST'])
def simulate():
    f = request.files["file"]
    f.save("uploads\person\person_1.jpg")
    human_path = "uploads\person\person_1.jpg"

    clothes_dir = random.choice(glob("uploads\clothes\*"))
    clothes_dir = glob(clothes_dir + "/*")

    human_img = cv2.imread(human_path)
    # human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
    human_result = sim_model.predict(human_path)[0]

    human_boxes = {}
    for i in range(len(human_result)):
        box = human_result.boxes[i]
        human_boxes[box.cls[0].item()] = box.xyxy[0].cpu(
        ).numpy().astype('int')

    for clothes_path in clothes_dir:
        clothes_img = cv2.imread(clothes_path)
        # clothes_img = cv2.cvtColor(clothes_img, cv2.COLOR_BGR2RGB)
        clothes_result = sim_model.predict(clothes_path)[0]
        clothes_label = clothes_result.boxes[0].cls[0].item()
        clothes_box = clothes_result.boxes[0].xyxy[0].cpu(
        ).numpy().astype('int')

        if clothes_label not in human_boxes:
            continue

        xy = human_boxes[clothes_label]

        clothes_mask = np.where(
            clothes_result.masks.data[0].cpu().numpy() != 0, 255, 0)
        clothes_mask = cv2.resize(clothes_mask.astype(
            np.float32), (clothes_img.shape[1], clothes_img.shape[0]))
        clothes_img[clothes_mask == 0] = 0

        target = clothes_img[clothes_box[1]
            :clothes_box[3], clothes_box[0]:clothes_box[2]]
        h, w, c = human_img[xy[1]:xy[3], xy[0]:xy[2]].shape
        target = cv2.resize(target, (w, h))
        clothes_mask = cv2.resize(clothes_mask, (w, h))

        alpha = 0.20
        human_img[xy[1]: xy[3], xy[0]:xy[2]][target != (0, 0, 0)] = cv2.addWeighted(human_img[xy[1]: xy[3], xy[0]:xy[2]][target != (0, 0, 0)], alpha,
                                                                                    target[target != (0, 0, 0)], 1-alpha, 0)[:, 0]

    result_path = r"uploads\result\result_1.jpg"

    cv2.imwrite(result_path, human_img)

    with open(result_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read())

    return b64_string


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

# py -m flask run --host=0.0.0.0 --port=5000
