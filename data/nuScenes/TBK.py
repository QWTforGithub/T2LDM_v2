# coding=UTF-8
import numpy as np
from num2words import num2words
from tqdm import tqdm
import pickle
from data.nuScenes.descriptor2 import relative_position, orientation_text_rad, count_num, NUM_THRESHOD

CLASS_NAMES = [
    'car',      'truck',      'construction_vehicle', 'bus',         'trailer',
    'barrier',  'motorcycle', 'bicycle',              'pedestrian',  'traffic_cone'
]

def read_pkl(file_path):

    with open(file=file_path, mode="rb") as f:
        data = pickle.load(f)

    return data

def check_object(text, box, name, object, target):
    if (text.__contains__(object)):
        if (text.__contains__('is the')):
            for i,n in enumerate(name):
                if(n == object):
                    object_box = box[i]
                    for j,n in enumerate(name):
                        if (n == target):
                            target_box = box[j]
                            x, y = relative_position(object_box, target_box)
                            text_pre = f"One {target} is the {y} of one {object}."
                            if (text == text_pre):
                                return 1
                            else:
                                return 0
        else:
            if (object in name and target in name):
                return 1
            else:
                return 0
    return 0

def TBK(texts, boxes, names, TARGET_CLASS='car'):
    '''
        texts: ["","",...]
        boxes: [[],[],...]
        names: [[],[],...]
    '''
    total_num = len(texts)
    right_num = 0

    for i, (text, box, name) in enumerate(tqdm(zip(texts, boxes, names), total=total_num)):

        # ---- No cars. ----
        if(text.__contains__('No cars') or text.__contains__('no cars')):
            if('car' not in name):
                right_num += 1
            continue
        # ---- No cars. ----

        # ---- facing ----
        if (text.__contains__('facing')):
            for i, n in enumerate(name):
                if(n == TARGET_CLASS):
                    target_boxes_index = box[i]
                    text_pre = f"One {TARGET_CLASS} is {orientation_text_rad(target_boxes_index)}."
                    if(text == text_pre):
                        right_num += 1
                        break
            continue
        # ---- facing ----

        # ---- car and barrier ----
        if(text.__contains__('barrier')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="barrier",
                target=TARGET_CLASS
            )
            continue
        # ---- car and barrier ----

        # ---- car and bicycle ----
        if(text.__contains__('bicycle')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="bicycle",
                target=TARGET_CLASS
            )
            continue
        # ---- car and bicycle ----

        # ---- car and bus ----
        if(text.__contains__('bus')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="bus",
                target=TARGET_CLASS
            )
            continue
        # ---- car and bus ----

        # ---- car and vehicle ----
        if(text.__contains__('vehicle')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="vehicle",
                target=TARGET_CLASS
            )
            continue
        # ---- car and vehicle ----

        # ---- car and motorcycle ----
        if(text.__contains__('motorcycle')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="motorcycle",
                target=TARGET_CLASS
            )
            continue
        # ---- car and motorcycle ----

        # ---- car and trailer ----
        if(text.__contains__('trailer')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="trailer",
                target=TARGET_CLASS
            )
            continue
        # ---- car and trailer ----

        # ---- car and truck ----
        if(text.__contains__('truck')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="truck",
                target=TARGET_CLASS
            )
            continue
        # ---- car and truck ----

        # ---- car and pedestrian ----
        if(text.__contains__('pedestrian')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="pedestrian",
                target=TARGET_CLASS
            )
            continue
        # ---- car and pedestrian ----

        # ---- car and traffic cone ----
        if(text.__contains__('cone')):
            right_num += check_object(
                text=text,
                box=box,
                name=name,
                object="cone",
                target=TARGET_CLASS
            )
            continue
        # ---- car and traffic cone ----

        # ---- car num ----
        counter = count_num(name)
        car_num = counter[TARGET_CLASS]
        car_num_word = num2words(car_num)
        suffix = "s"
        if (car_num == 0):
            text_pre = f"No {TARGET_CLASS}{suffix}."
        elif (car_num == 1):
            text_pre = f"One {TARGET_CLASS}."
        elif (car_num <= NUM_THRESHOD):
            text_pre = f"{car_num_word} {TARGET_CLASS}{suffix}."
            text_pre = text_pre.capitalize()
        else:
            text_pre = f"More than {num2words(NUM_THRESHOD)} {TARGET_CLASS}{suffix}."
        if(text == text_pre):
            right_num += 1
        # ---- car num ----

    right_rate = right_num / total_num
    print(f"Total num : {total_num}, Right num : {right_num}, TBK : {right_rate}")

if __name__ == '__main__':
    '''
        text_root_path : a text list
            ["xxx1", "xxx2", ...]
        boxes_pkl_path : a pkl file producted by a detector (FSHNet).
    '''
    text_root_path = "/ihoment/youjie10/qwt/res/text"
    boxes_pkl_path = "/ihoment/youjie10/qwt/model/FSHNet-TBK/tools/results.pkl"
    infos = read_pkl(boxes_pkl_path)

    texts = []
    boxes = []
    names = []
    for info in infos:
        name = info["name"]
        box = info["boxes_lidar"]
        label = np.expand_dims(info["pred_labels"]-1, axis=1)
        box = np.concatenate([box, label], axis=-1)

        boxes.append(box)
        names.append(name)

        id = info["frame_id"]
        text_name = id.replace(".ply", ".pkl.txt").replace("generation_", "text_")
        text_path = f"{text_root_path}/{text_name}"
        with open(text_path, "r") as f:
            text = f.readline()
            text = text.replace("Rainy. ", "")
            texts.append(text)
    TBK(
        texts=texts,
        boxes=boxes,
        names=names
    )
    pass
