# coding=utf-8

import numpy as np
from num2words import num2words
from tqdm import tqdm
import pickle

CLASS_NAMES = [
    'ignore',                   # 0
    'car',                      # 1
    'bicycle',                  # 2
    'motorcycle',               # 3
    'truck',                    # 4
    'other-vehicle',            # 5
    'person',                   # 6
    'bicyclist',                # 7
    'motorcyclist',             # 8
    'road',                     # 9
    'parking',                  # 10
    'sidewalk',                 # 11
    'ground',                   # 12 'other-ground'
    'building',                 # 13
    'fence',                    # 14
    'vegetation',               # 15
    'trunk',                    # 16
    'terrain',                  # 17
    'pole',                     # 18
    'sign',                     # 19 'traffic-sign'
]

def read_pkl(file_path):

    with open(file=file_path, mode="rb") as f:
        data = pickle.load(f)

    return data

def contains_all_elements(list1, list2):
    """
    判断 list1 是否包含 list2 的所有元素
    """
    return all(x in list1 for x in list2)

def TSK(texts, labels, target="car"):

    total_num = 0
    right_num = 0

    for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=total_num)):
        text = text.lower()

        if(not text.__contains__(target)):
            continue

        names_id = []
        for i, name in enumerate(CLASS_NAMES):
            if(text.__contains__(name)):
                names_id.append(i)

        if(contains_all_elements(label, names_id)):
            right_num += 1

        total_num += 1

    right_rate = right_num / total_num
    print(f"Total num : {total_num}, Right num : {right_num}, TSK : {right_rate}")


if __name__ == '__main__':

    text_root_path = "/root/res/text"
    boxes_pkl_path = "/root/models/LSK3DNet-main/results.pkl"
    infos = read_pkl(boxes_pkl_path)

    ids = infos["id"]
    pred_labels = infos["label"]

    texts = []
    labels = []

    for id, label in zip(ids, pred_labels):

        labels.append(label)

        id = id.split("/")[-1]
        text_name = id.replace(".ply", ".pkl.txt").replace("generation_", "text_")
        text_path = f"{text_root_path}/{text_name}"
        with open(text_path, "r") as f:
            text = f.readline()
            text = text.replace("Rainy. ", "")
            texts.append(text)
    TSK(
        texts=texts,
        labels=labels,
    )