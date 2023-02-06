from __future__ import division, print_function
import os
import random
from keras.applications.resnet import ResNet50
import numpy as np
from keras.utils import load_img, img_to_array
import cv2
from ultralytics import YOLO
import cv2
from glob import glob
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64

sim_model = YOLO('top_pants.pt')

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

    alpha = 0.15
    human_img[xy[1]: xy[3], xy[0]:xy[2]][target != (0, 0, 0)] = cv2.addWeighted(human_img[xy[1]: xy[3], xy[0]:xy[2]][target != (0, 0, 0)], alpha,
                                                                                target[target != (0, 0, 0)], 1-alpha, 0)[:, 0]

result_path = r"uploads\result\result_1.jpg"

cv2.imwrite(result_path, human_img)
