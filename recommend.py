import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import timm

import warnings
warnings.filterwarnings('ignore')

category_map = {0: '가디건',
                1: '니트웨어',
                2: '래깅스',
                3: '베스트',
                4: '브라탑',
                5: '블라우스',
                6: '셔츠',
                7: '스커트',
                8: '재킷',
                9: '점퍼',
                10: '조거팬츠',
                11: '짚업',
                12: '청바지',
                13: '코트',
                14: '탑',
                15: '티셔츠',
                16: '패딩',
                17: '팬츠',
                18: '후드티'}

size_map = {0: '노말', 1: '롱', 2: '크롭'}

outfit_map = {0: '노멀', 1: '루즈', 2: '스키니'}

color_map = {0: '골드',
                1: '그레이',
                2: '그린',
                3: '네온',
                4: '네이비',
                5: '라벤더',
                6: '레드',
                7: '민트',
                8: '베이지',
                9: '브라운',
                10: '블랙',
                11: '블루',
                12: '스카이블루',
                13: '실버',
                14: '옐로우',
                15: '오렌지',
                16: '와인',
                17: '카키',
                18: '퍼플',
                19: '핑크',
                20: '화이트'}

neck_map = {0: '노카라',
            1: '라운드넥',
            2: '보트넥',
            3: '브이넥',
            4: '스위트하트',
            5: '스퀘어넥',
            6: '오프숄더',
            7: '원숄더',
            8: '유넥',
            9: '터틀넥',
            10: '홀터넥',
            11: '후드'}

kara_map = {0: '밴드칼라',
            1: '보우칼라',
            2: '세일러칼라',
            3: '셔츠칼라',
            4: '숄칼라',
            5: '차이나칼라',
            6: '테일러드칼라',
            7: '폴로칼라',
            8: '피터팬칼라'}

label_map = [category_map, size_map, outfit_map, color_map, neck_map, kara_map]


class BaseModel(nn.Module):
    def __init__(self, num_classes=12):
        super(BaseModel, self).__init__()
#         self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        # self.backbone = timm.create_model('convnext_base', pretrained=True)
#         self.backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True)
#         self.backbone = models.vit_l_16(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.softmax(self.classifier(x))
        return x


def get_models():
    model1 = BaseModel(num_classes=len(category_map))
    model1.load_state_dict(torch.load('models/model_category.pt'))
    model1.eval()

    model2 = BaseModel(num_classes=len(size_map))
    model2.load_state_dict(torch.load('models/model_size.pt'))
    model2.eval()

    model3 = BaseModel(num_classes=len(outfit_map))
    model3.load_state_dict(torch.load('models/model_outfit.pt'))
    model3.eval()

    model4 = BaseModel(num_classes=len(color_map))
    model4.load_state_dict(torch.load('models/model_color.pt'))
    model4.eval()

    model5 = BaseModel(num_classes=len(neck_map))
    model5.load_state_dict(torch.load('models/model_neck.pt'))
    model5.eval()

    model6 = BaseModel(num_classes=len(kara_map))
    model6.load_state_dict(torch.load('models/model_kara.pt'))
    model6.eval()
    return [model1, model2, model3, model4, model5, model6]


def recommend(yolo, models, path, weight='straight'):
    print('start')

    human_img, human_boxes = crop_clothes(yolo, path)

    results = {}

    for box in human_boxes:
        xy = human_boxes[box]
        img = human_img[xy[1]:xy[3], xy[0]:xy[2]]
        img = cv2.resize(img, (224, 224))
        img = img / 127.5 - 1
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        preds = []
        for model, label in zip(models, label_map):
            pred = model(torch.tensor(img, dtype=torch.float32)
                         ).argmax(1).cpu().numpy()
            preds.append(label[pred[0]])

        results[box] = preds

    paths = []
    keywords = []

    for clothes in ['top', 'bottom']:
        result_path, keyword = check_sim(results, clothes, weight)
        paths.append(result_path)
        keywords.append(keyword)

    return paths, keywords


def crop_clothes(yolo, path):
    seg_map = {0: 'top', 1: 'top', 2: 'bottom'}

    human_img = cv2.imread(path)
    human_result = yolo.predict(path)[0]

    human_boxes = {}
    for i in range(len(human_result)):
        box = human_result.boxes[i]
        human_boxes[seg_map[box.cls[0].item()]] = box.xyxy[0].cpu(
        ).numpy().astype('int')

    return human_img, human_boxes


def check_sim(results, clothes, weight='straight'):
    name = weight + ' ' + clothes + '.csv'
    csv_name = f'datas/{name}'
    df = pd.read_csv(csv_name)

    if clothes not in results:
        result = df.sample(1)
        path = 'Result/' + weight + ' ' + \
            clothes + '/' + result['경로'].values[0]
        cat = ['카테고리', '기장', '핏', '색', '넥라인'] if clothes == '상의' else [
            '카테고리', '기장', '핏', '색']
        keyword = result[cat].values[0].tolist()
        return path, keyword
    else:
        array = results[clothes]
        data = df.iloc[:, 2:]
        result = data[data.apply(lambda x: len(set(x) & set(array)), axis=1) == data.apply(
            lambda x: len(set(x) & set(array)), axis=1).max()]
        result = df.iloc[result.sample(1).index]
        path = 'Result/' + weight + ' ' + \
            clothes + '/' + result['경로'].values[0]
        cat = ['카테고리', '기장', '핏', '색', '넥라인'] if clothes == '상의' else [
            '카테고리', '기장', '핏', '색']
        keyword = result[cat].values[0].tolist()

    return path, keyword
