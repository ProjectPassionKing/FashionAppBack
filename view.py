import numpy as np
import cv2
import random

clothes_h = 220
clothes_w = 220

colors = {
    '화이트': [190, 190, 190], '핑크': [230, 167, 178], '옐로우': [192, 125, 59], 
    '블랙': [50, 50, 50], '레드': [178, 34, 34], '그레이': [100, 100, 100], 
    '와인': [140, 30, 30], '브라운': [71, 33, 20], '베이지': [200, 160, 125],
    '카키': [56, 56, 32], '실버': [152, 152, 154], '스카이블루': [145, 161, 176],
    '블루': [35, 66, 156], '네이비': [19, 32, 66], '그린': [17, 88, 54]
}

def ton_on_ton(color):
    color2 = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_RGB2HSV)
    weight = np.random.randint(80, 200)

    backcolor = color2.copy()
    if color2[:, :, 1].mean() < 70:
        backcolor[:, :, 1] += 50
        backcolor[:, :, 2] += np.random.randint(0, 30)
    else:
        backcolor[:, :, 1] -= 50
        backcolor[:, :, 2] += np.random.randint(0, 30)

    if np.random.random() > 0.5:
        color2[:, :, 2] += weight
        color3 = color2.copy()
        color3[:, :, 2] += weight
    else:
        color2[:, :, 2] += weight
        color3 = color2.copy()
        color3[:, :, 2] += weight

    color2 = cv2.cvtColor(color2.astype(np.uint8), cv2.COLOR_HSV2RGB)
    color2 = np.clip(color2, 35, 200)
    color3 = cv2.cvtColor(color3.astype(np.uint8), cv2.COLOR_HSV2RGB)
    color3 = np.clip(color3, 35, 200)
    
    backcolor = cv2.cvtColor(backcolor.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return color, color2, color3, backcolor


def ton_in_ton(color):
    color2 = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_RGB2HSV)
    weight = np.random.randint(80, 200)

    backcolor = color2.copy()
    if color2[:, :, 2].mean() < 70:
        backcolor[:, :, 2] += 50
    else:
        backcolor[:, :, 2] -= 50
    
    if np.random.random() > 0.5:
        color2[:, :, 1] += weight
        color3 = color2.copy()
        color3[:, :, 1] += weight
    else:
        color2[:, :, 1] += weight
        color3 = color2.copy()
        color3[:, :, 1] += weight
    color2 = cv2.cvtColor(color2.astype(np.uint8), cv2.COLOR_HSV2RGB)
    color2 = np.clip(color2, 35, 200)
    color3 = cv2.cvtColor(color3.astype(np.uint8), cv2.COLOR_HSV2RGB)
    color3 = np.clip(color3, 35, 200)
    
    backcolor = cv2.cvtColor(backcolor.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return color, color2, color3, backcolor

def change_color(img, color):
    color = np.append(color, 0)
    color = np.full(img.shape, color)
    color = color.astype(np.uint8)
    mask = np.where(img[:, :, 3] != 0, 255, 0)
    img = cv2.add(img, color) - img[mask == 255].mean()*0.4
    img[mask != 255] = 0
    return img


def view(outer_path, top_path, bottom_path, color):

    color = colors[color]
    color = np.full((256, 256, 3), color)
    if np.random.random() > 0.5:
        color, color2, color3, backcolor = ton_on_ton(color)
    else:
        color, color2, color3, backcolor = ton_in_ton(color)
    
    fashion_color = [color, color2, color3]
    np.random.shuffle(fashion_color)
    color, color2, color3 = fashion_color

    outer = cv2.imread(outer_path, cv2.IMREAD_UNCHANGED)
    top = cv2.imread(top_path, cv2.IMREAD_UNCHANGED)
    bottom = cv2.imread(bottom_path, cv2.IMREAD_UNCHANGED)
    outer = cv2.cvtColor(outer, cv2.COLOR_BGRA2RGBA)
    top = cv2.cvtColor(top, cv2.COLOR_BGRA2RGBA)
    bottom = cv2.cvtColor(bottom, cv2.COLOR_BGRA2RGBA)
    
    total_h = 550
    total_w = 440

    back = np.full((total_h, total_w, 3), backcolor[0][0])

    outer = change_color(outer, color[0][0])
    top = change_color(top, color2[0][0])
    bottom = change_color(bottom, color3[0][0])

    o_h, o_w = outer.shape[:2]
    t_h, t_w = top.shape[:2]
    b_h, b_w = bottom.shape[:2]


    ash= clothes_h / t_h
    asw= clothes_w / t_w
    sizeas=(int(t_w*asw),int(t_h*asw))

    top = cv2.resize(top, sizeas)

    back[70:70+sizeas[1], 10:10+sizeas[0]][top[:,:,3] != 0] = top[:, :, :3][top[:,:,3] != 0]

    ash= clothes_h / b_h
    asw= clothes_w / b_w
    sizeas=(int(b_w*asw),int(b_h*asw))

    bottom = cv2.resize(bottom, sizeas)

    back[70:70+sizeas[1], 210:210+sizeas[0]][bottom[:,:,3] != 0] = bottom[:, :, :3][bottom[:,:,3] != 0]

    ash= clothes_h / o_h
    asw= clothes_w / o_w

    sizeas=(int(o_w*asw),int(o_h*asw))
    outer = cv2.resize(outer, sizeas)

    back[120:120+sizeas[1], 110:110+sizeas[0]][outer[:,:,3] != 0] = outer[:, :, :3][outer[:,:,3] != 0]
    back = cv2.cvtColor(back.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return back