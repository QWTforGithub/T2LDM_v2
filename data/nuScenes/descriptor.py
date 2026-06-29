# coding=utf-8
import glob
import shutil
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import os
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
import numpy as np
import open3d
import pickle
from collections import Counter
from num2words import num2words
from word2number import w2n
import copy
import math
import torch
import re
from tqdm import tqdm
from utils import common
from models.T5.T5 import t5

ROOT_PATH = "/root/dataset/nuScenes"
DESCRIPTION = "nuscenes.pkl"
NUM = 34149
SEMANTIC_CLASS_NUM = 16
NUM_THRESHOD = 5

OBJECT_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
OBJECT_NAME = ["barrier","bicycle","bus",
               "car","construction_vehicle",
               "motorcycle", "pedestrian",
               "traffic_cone","trailer","truck",]
SCENE_INDEX = [10, 11, 12, 13, 14, 15]
SCENE_NAME= ["driveable_surface","otherflat",
             "sidewalk","terrain",
             "manmade", "vegetation",]

INDEX_NAME = {
    0: "barrier",
    1: "bicycle",
    2: "bus",
    3: "car",
    4: "construction_vehicle",
    5: "motorcycle",
    6: "pedestrian",
    7: "traffic_cone",
    8: "trailer",
    9: "truck",
    10: "driveable_surface",
    11: "otherflat",
    12: "sidewalk",
    13: "terrain",
    14: "manmade",
    15: "vegetation",
    16: "ignore",
}

NAME_INDEX = {
    "barrier":              0,
    "bicycle":              1,
    "bus":                  2,
    "car":                  3,
    "construction_vehicle": 4,
    "motorcycle":           5,
    "pedestrian":           6,
    "traffic_cone":         7,
    "trailer":              8,
    "truck":                9,
    "driveable_surface":    10,
    "otherflat":            11,
    "sidewalk":             12,
    "terrain":              13,
    "manmade":              14,
    "vegetation":           15,
    "ignore":               16,
}

TARGET_CLASS = "car"
CLASS_NUM = [
    12320,  7474,   10986,  33266,  9495,   7518,   27862,  14853,  9432,   24118,
    34149,  15462,  33549,  29291,  34149,  33425
]

CLASS_CAR_NUM = [
    10709,  7017,   10533,  33266,  7861,   7266,   26397,  13294,  8380,   23563,
    34149,  15462,  33549,  29291,  34149,  33425
]

def class_num(index):
    names = {
        0:  "barrier",                  # 12320
        1:  "bicycle",                  # 7474
        2:  "bus",                      # 10986
        3:  "car",                      # 33266
        4:  "constructionvehicle",      # 9495
        5:  "motorcycle",               # 7518
        6:  "pedestrian",               # 27862
        7:  "trafficcone",              # 14853
        8:  "trailer",                  # 9432
        9:  "truck",                    # 24118

        10: "driveablesurface",         # 34149
        11: "otherflat",                # 15462
        12: "sidewalk",                 # 33549
        13: "terrain",                  # 29291
        14: "manmade",                  # 34149
        15: "vegetation",               # 33425
        16: "ignore",                   #
    }

    return names[index]

# ---- function ----
def get_nusc():
    return NuScenes(
                version='v1.0-trainval',
                dataroot=f"{ROOT_PATH}/v1.0-trainval",
                verbose=True
            )

# def get_cam(
#     token,
#     nusc=None,
#     root_path=f"{ROOT_PATH}/v1.0-trainval",
#     version='v1.0-trainval',
#     cam_names=None,
# ):
#     """
#     根据 nuScenes token 获取同一个 sample 对应的 6 个 camera 图像路径。
#
#     token 可以是：
#     1. sample token
#     2. LIDAR_TOP sample_data token
#
#     return:
#         cam_paths: dict, {cam_name: image_path}
#         cam_tokens: dict, {cam_name: sample_data_token}
#         lidar_path: str
#         lidar_token: str
#         sample_token: str
#     """
#
#     if nusc is None:
#         assert root_path is not None, "nusc is None 时，必须传入 root_path"
#         nusc = NuScenes(
#             version=version,
#             dataroot=root_path,
#             verbose=True
#         )
#
#     if cam_names is None:
#         cam_names = [
#             'CAM_FRONT',
#             'CAM_FRONT_LEFT',
#             'CAM_FRONT_RIGHT',
#             'CAM_BACK',
#             'CAM_BACK_LEFT',
#             'CAM_BACK_RIGHT'
#         ]
#
#     # ------------------------------------------------
#     # 1. 判断 token 类型：sample token 还是 sample_data token
#     # ------------------------------------------------
#     sample = None
#     sample_token = None
#     lidar_sd = None
#     lidar_token = None
#
#     # 情况 A：token 是 sample token
#     try:
#         sample = nusc.get('sample', token)
#         sample_token = token
#         lidar_token = sample['data']['LIDAR_TOP']
#         lidar_sd = nusc.get('sample_data', lidar_token)
#
#     except KeyError:
#         # 情况 B：token 是 sample_data token
#         try:
#             lidar_sd = nusc.get('sample_data', token)
#             lidar_token = token
#             sample_token = lidar_sd['sample_token']
#             sample = nusc.get('sample', sample_token)
#
#         except KeyError:
#             raise KeyError(
#                 f"Token not found in both sample and sample_data tables: {token}"
#             )
#
#     # ------------------------------------------------
#     # 2. 获取 LiDAR path
#     # ------------------------------------------------
#     lidar_path = nusc.get_sample_data_path(lidar_token).split("/v1.0-trainval/")[-1]
#
#     # ------------------------------------------------
#     # 3. 获取所有 camera path/token
#     # ------------------------------------------------
#     cam_paths = {}
#     cam_tokens = {}
#
#     for cam in cam_names:
#         if cam not in sample['data']:
#             raise KeyError(f"{cam} not found in sample['data'].")
#
#         cam_token = sample['data'][cam]
#         cam_tokens[cam] = cam_token
#         cam_paths[cam] = nusc.get_sample_data_path(cam_token).split("/v1.0-trainval/")[-1]
#
#     return cam_paths, cam_tokens, lidar_path, lidar_token, sample_token

def make_transform_matrix(translation, rotation):
    """
    Args:
        translation: [3]
        rotation: quaternion [w, x, y, z]

    Returns:
        T: [4,4], sensor -> ego
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.asarray(translation, dtype=np.float32)
    return T


def get_cam(
    token,
    nusc=None,
    root_path=f"{ROOT_PATH}/v1.0-trainval",
    version='v1.0-trainval',
    cam_names=None,
):
    """
    根据 nuScenes token 获取同一个 sample 对应的 6 个 camera 图像路径、
    camera token，以及对应相机内外参。

    token 可以是：
    1. sample token
    2. LIDAR_TOP sample_data token

    return:
        cam_paths: dict, {cam_name: image_path}
        cam_tokens: dict, {cam_name: sample_data_token}
        cam_infos: dict, {cam_name: calib info}
        lidar_path: str
        lidar_token: str
        sample_token: str
    """

    if nusc is None:
        assert root_path is not None, "nusc is None 时，必须传入 root_path"
        nusc = NuScenes(
            version=version,
            dataroot=root_path,
            verbose=True
        )

    if cam_names is None:
        cam_names = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]

    # ------------------------------------------------
    # 1. 判断 token 类型：sample token 还是 sample_data token
    # ------------------------------------------------
    try:
        sample = nusc.get('sample', token)
        sample_token = token
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_token)

    except KeyError:
        try:
            lidar_sd = nusc.get('sample_data', token)
            lidar_token = token
            sample_token = lidar_sd['sample_token']
            sample = nusc.get('sample', sample_token)

        except KeyError:
            raise KeyError(
                f"Token not found in both sample and sample_data tables: {token}"
            )

    # ------------------------------------------------
    # 2. 获取 LiDAR path 和 LiDAR calibration
    # ------------------------------------------------
    lidar_path = nusc.get_sample_data_path(lidar_token).split("/v1.0-trainval/")[-1]

    lidar_calib = nusc.get(
        'calibrated_sensor',
        lidar_sd['calibrated_sensor_token']
    )

    T_lidar_to_ego = make_transform_matrix(
        lidar_calib['translation'],
        lidar_calib['rotation']
    )

    T_ego_to_lidar = np.linalg.inv(T_lidar_to_ego)

    # ------------------------------------------------
    # 3. 获取所有 camera path/token/calibration
    # ------------------------------------------------
    cam_paths = {}
    cam_tokens = {}
    cam_infos = {}

    for cam in cam_names:
        if cam not in sample['data']:
            raise KeyError(f"{cam} not found in sample['data'].")

        cam_token = sample['data'][cam]
        cam_sd = nusc.get('sample_data', cam_token)

        cam_calib = nusc.get(
            'calibrated_sensor',
            cam_sd['calibrated_sensor_token']
        )

        cam_path = nusc.get_sample_data_path(cam_token).split("/v1.0-trainval/")[-1]

        K = np.asarray(cam_calib['camera_intrinsic'], dtype=np.float32)

        R_cam_to_ego = Quaternion(cam_calib['rotation']).rotation_matrix.astype(np.float32)
        t_cam_to_ego = np.asarray(cam_calib['translation'], dtype=np.float32)

        T_cam_to_ego = make_transform_matrix(
            cam_calib['translation'],
            cam_calib['rotation']
        )

        # camera -> lidar
        T_cam_to_lidar = T_ego_to_lidar @ T_cam_to_ego

        cam_paths[cam] = cam_path
        cam_tokens[cam] = cam_token

        cam_infos[cam] = {
            "sample_data_token": cam_token,
            "path": cam_path,

            # intrinsic
            "intrinsic": K,

            # camera -> ego
            "rotation": R_cam_to_ego,
            "translation": t_cam_to_ego,
            "T_cam_to_ego": T_cam_to_ego,

            # camera -> lidar
            "T_cam_to_lidar": T_cam_to_lidar,

            # raw records, optional
            "sample_data": cam_sd,
            "calibrated_sensor": cam_calib,
        }

    return cam_paths, cam_tokens, cam_infos, lidar_path, lidar_token, sample_token

def get_camera_yaw_range(cam_info, image_size=(1600, 900)):
    """
    Args:
        cam_info:
            cam_infos[cam_name]
            must contain:
                intrinsic: [3,3]
                T_cam_to_lidar: [4,4]

        image_size: (W, H)

    Returns:
        yaw_left: degree
        yaw_right: degree
        yaw_center: degree
    """

    W, H = image_size
    K = cam_info["intrinsic"]
    T_cam_to_lidar = cam_info["T_cam_to_lidar"]

    fx = K[0, 0]
    cx = K[0, 2]

    # nuScenes camera coordinate:
    # x right, y down, z forward
    rays_cam = np.array([
        [(0.0 - cx) / fx,      0.0, 1.0],      # left image boundary
        [((W - 1) - cx) / fx,  0.0, 1.0],      # right image boundary
        [((W / 2) - cx) / fx,  0.0, 1.0],      # center ray
    ], dtype=np.float32)

    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    R_cam_to_lidar = T_cam_to_lidar[:3, :3]
    rays_lidar = (R_cam_to_lidar @ rays_cam.T).T

    yaw = np.rad2deg(np.arctan2(rays_lidar[:, 1], rays_lidar[:, 0]))

    yaw_left = yaw[0]
    yaw_right = yaw[1]
    yaw_center = yaw[2]

    return yaw_left, yaw_right, yaw_center

def normalize_angle_deg(angle):
    """Normalize angle to [-180, 180)."""
    return (angle + 180) % 360 - 180


def get_camera_yaw_range_from_info(
    cam_info,
    image_size=(1600, 900),
):
    """
    Args:
        cam_info:
            {
                "intrinsic": K,              # [3,3]
                "T_cam_to_lidar": T          # [4,4]
            }

        image_size: (W, H)

    Returns:
        {
            "yaw_left": float,
            "yaw_right": float,
            "yaw_center": float,
            "hfov": float,
            "wrap": bool
        }
    """

    W, H = image_size

    K = np.asarray(cam_info["intrinsic"], dtype=np.float32)
    T_cam_to_lidar = np.asarray(cam_info["T_cam_to_lidar"], dtype=np.float32)

    fx = K[0, 0]
    cx = K[0, 2]

    # nuScenes camera coordinate:
    # x: right, y: down, z: forward
    rays_cam = np.array([
        [(0.0 - cx) / fx,      0.0, 1.0],      # left boundary ray
        [((W - 1) - cx) / fx,  0.0, 1.0],      # right boundary ray
        [((W / 2.0) - cx) / fx, 0.0, 1.0],     # center ray
    ], dtype=np.float32)

    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    R_cam_to_lidar = T_cam_to_lidar[:3, :3]

    rays_lidar = (R_cam_to_lidar @ rays_cam.T).T  # [3,3]

    yaw = np.rad2deg(np.arctan2(rays_lidar[:, 1], rays_lidar[:, 0]))
    yaw = normalize_angle_deg(yaw)

    yaw_left = float(yaw[0])
    yaw_right = float(yaw[1])
    yaw_center = float(yaw[2])

    # horizontal FoV
    hfov = abs((yaw_left - yaw_right + 180) % 360 - 180)

    # 是否跨越 -180/180 seam
    wrap = abs(yaw_left - yaw_right) > 180

    return {
        "yaw_left": yaw_left,
        "yaw_right": yaw_right,
        "yaw_center": yaw_center,
        "hfov": float(hfov),
        "wrap": bool(wrap),
    }


def get_all_camera_yaw_ranges(
    cam_infos,
    image_size=(1600, 900),
):
    """
    Args:
        cam_infos: dict, from your get_cam()
            {
                "CAM_FRONT": cam_info,
                ...
            }

    Returns:
        cam_yaw_ranges: dict
    """
    cam_yaw_ranges = {}

    for cam_name, cam_info in cam_infos.items():
        cam_yaw_ranges[cam_name] = get_camera_yaw_range_from_info(
            cam_info,
            image_size=image_size
        )

    return cam_yaw_ranges

def count_num(gt_names):
    counter = Counter(gt_names)
    return counter

def relative_position(boxA, boxB, threshold=0.0):
    '''
        获取简单的boxA在boxB的所处位置
        boxA : 源盒子， (N,)
        boxB : 目标盒子，源盒子的参考系，(N,)
    '''

    dx = boxA[0] - boxB[0]
    dy = boxA[1] - boxB[1]

    # 前后
    if dx > threshold:
        pos_x = "ahead"
    elif dx < -threshold:
        pos_x = "behind"
    else:
        pos_x = "aligned"

    # 左右
    if dy > threshold:
        pos_y = "left"
    elif dy < -threshold:
        pos_y = "right"
    else:
        pos_y = "center"

    return pos_x, pos_y

def get_boxes_to_boxes(item, object="car", only_object=True):
    gt_names = item["gt_names"]
    if (len(gt_names) > 0):
        gt_names = gt_names[gt_names != "ignore"]
    gt_boxes = item["gt_boxes"]

    if(only_object):
        gt_names_new = []
        gt_boxes_new = []
        for i in range(len(gt_names)):
            if(OBJECT_NAME.__contains__(gt_names[i])):
                gt_names_new.append(gt_names[i])
                gt_boxes_new.append(gt_boxes[i])
        gt_names = np.asarray(gt_names_new)
        gt_boxes = np.asarray(gt_boxes_new)

    # ---- 将目标盒子和其他类别盒子分离开 ----
    object_boxes = []
    other_boxes = []
    other_names = []
    for i in range(len(gt_names)):
        if (gt_names[i] == object):
            object_boxes.append(gt_boxes[i])
        else:
            other_boxes.append(gt_boxes[i])
            other_names.append(gt_names[i])
    # ---- 将目标盒子和其他类别盒子分离开 ----

    # ---- all_descs的首顺序是以：其他类别开始的，每个盒子，盒子位置 -----
    all_descs = []
    for i in range(len(other_boxes)):
        descs = []
        for j in range(len(object_boxes)):
            desc = relative_position(object_boxes[j], other_boxes[i])
            descs.append(desc)
        all_descs.append(descs)
    # ---- all_descs的首顺序是以：其他类别开始的，每个盒子，盒子位置 -----

    return all_descs, other_boxes, other_names, object_boxes

def orientation_text_rad(boxes, two_ori=False):
    """
    将车辆 yaw（弧度制）转换为前/后/左/右朝向描述
    """
    yaw_rad = boxes[6]
    yaw_deg = math.degrees(yaw_rad) % 360  # 转为 0~360 度

    if(two_ori):
        if 315 <= yaw_deg or yaw_deg < 135:
            return "facing forward"
        else:
            return "facing backward"
    else:
        if 315 <= yaw_deg or yaw_deg < 45:
            return "facing forward"
        elif 45 <= yaw_deg < 135:
            return "facing left"
        elif 135 <= yaw_deg < 225:
            return "facing backward"
        else:
            return "facing right"
# ---- function ----

def class_name(index):
    names = {
        0:  "barrier",
        1:  "bicycle",
        2:  "bus",
        3:  "car",
        4:  "constructionvehicle",
        5:  "motorcycle",
        6:  "pedestrian",
        7:  "trafficcone",
        8:  "trailer",
        9:  "truck",
        10: "driveablesurface",
        11: "otherflat",
        12: "sidewalk",
        13: "terrain",
        14: "manmade",
        15: "vegetation",
        16: "ignore",
    }

    return names[index]

def save_pkl(
        root_path=None,
        description=None,
        infos=None
):
    if (root_path is None):
        root_path = ROOT_PATH

    if (description is None):
        description = DESCRIPTION

    save_path = f"{root_path}/v1.0-trainval/{description}"
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"---- Saving : {save_path} ----")

def read_pkl(
    root_path = None,
    description = None,
):
    if(root_path is None):
        root_path = ROOT_PATH

    if(description is None):
        description = DESCRIPTION

    file_path = f"{root_path}/v1.0-trainval/{description}"

    with open(file_path, 'rb') as f:
        infos = pickle.load(f)

    return infos

def count_class(infos):

    '''

        car : 33266
        pedestrian : 27862
        trafficcone : 14853
        truck : 24118
        driveablesurface : 34149
        otherflat : 15462
        sidewalk : 33549
        terrain : 29291
        manmade : 34149
        vegetation : 33425
        ignore : 34149
        constructionvehicle : 9495
        barrier : 12320
        motorcycle : 7518
        bicycle : 7474
        bus : 10986
        trailer : 9432
    '''

    class_num = {}
    for info in infos:
        semantic = info["semantic"]

        for i in range(SEMANTIC_CLASS_NUM+1):
            if(semantic.__contains__(i)):
                name = class_name(i)
                if(class_num.keys().__contains__(name)):
                    class_num[name] += 1
                else:
                    class_num[name] = 1

    for key in class_num.keys():
        print(f"{key} : {class_num[key]}")

def check_text(infos=None,description="text.pkl",text_name="text"):
    if(infos is None):
        infos = read_pkl(description=description)
    print(len(infos))
    text_num = {}
    for info in infos:
        text = info[text_name]

        if(text_num.keys().__contains__(text)):
            text_num[text] += 1
        else:
            text_num[text] = 1

    for key in text_num.keys():
        print(f"{key} : {text_num[key]}")


    return text_num

def get_everyclassnum_for_text(description="text.pkl"):

    infos = read_pkl(description=description)
    class_dict = {}
    # del NAME_INDEX['car']
    name_keys = NAME_INDEX.keys()

    for info in infos:
        text = info["text"]
        for name_key in name_keys:
            if(text.__contains__(name_key)):
                if(class_dict.__contains__(name_key)):
                    class_dict[name_key] += 1
                else:
                    class_dict[name_key] = 1

    for key in class_dict.keys():
        print(f"{key} : {class_dict[key]}")

# ---- Text ----
def text_quantity_l1(item, object=TARGET_CLASS, threshold=NUM_THRESHOD):
    '''
        Key: text_quantity_l1
        Using The Training and Combination.
        Exmaple : NUM_THRESHOLD = 2
            No cars.
            One car.
            Two cars.
            More than two cars.
    '''

    counter = count_num(item["gt_names"])
    car_num = counter[object]
    car_num_word = num2words(car_num)

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (car_num == 0):
        text = f"No {object}{suffix}."
    elif (car_num == 1):
        text = f"One {object}."
    elif (car_num <= threshold):
        text = f"{car_num_word} {object}{suffix}."
        text = text.capitalize()
    else:
        text = f"More than {num2words(threshold)} {object}{suffix}."
    return text

def text_orientation_l1c(item, object=TARGET_CLASS):
    '''
        Key: text_orientation_l1
        Using The Training and Combination.
        Exmaple :

            One car is facing forward.
            One car is facing right.
            One car is facing backward.
            One car is facing left.
    '''

    gt_names = item["gt_names"]
    if (len(gt_names) > 0):
        gt_names = gt_names[gt_names != "ignore"]
    gt_boxes = item["gt_boxes"]

    suffix = "s"
    if (object == "bus"):
        suffix = "es"

    if (not object in gt_names.tolist()):
        return f"No {object}{suffix}."

    target_class_index = np.where(gt_names == TARGET_CLASS)[0][0]
    target_boxes_index = gt_boxes[target_class_index]

    text = f"One {object} is {orientation_text_rad(target_boxes_index)}."

    return text

def text_weather(info):
    '''
        Key: text_weather
        Using The Training and Combination.
        Exmaple :
            Rainy.
            Sunny.
    '''

    text = info["description"]
    if(text.__contains__("rain") or text.__contains__("Rain")):
        return "Rainy. "
    else:
        return None

# ---- Text ----

def get_text(infos):

    new_infos = []
    class_num_dict = {
        "barrier": 0,
        "bicycle": 0,
        "bus": 0,
        "car": 0,
        "vehicle": 0,  # "construction vehicle"
        "motorcycle": 0,
        "pedestrian": 0,
        "traffic cone": 0,  # "traffic cone"
        "trailer": 0,
        "truck": 0,
        "driveable_surface": 0,
        "otherflat": 0,
        "sidewalk": 0,
        "terrain": 0,
        "manmade": 0,
        "vegetation": 0,
        "ignore": 0,
    }

    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Text Decription: ")):

        gt_names = info["gt_names"]
        gt_names_unique = np.unique(gt_names)
        gt_boxes = info["gt_boxes"]

        weather = text_weather(info)
        token = info["token"]
        lidar_path = info["lidar_path"]
        semantic = lidar_path.replace("/LIDAR_TOP/", "/SEMANTIC/").replace("bin", "npy")

        gt_names = info["gt_names"]
        gt_boxes = info["gt_boxes"]

        # ---- No cars. ----
        '''
            "No cars."
            "There are no cars."
            "A scene with no cars."
        '''
        if(not gt_names_unique.__contains__(TARGET_CLASS)):
            new_info = {}
            text = "No cars."
            new_info["text"] =  text #f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            new_info = {}
            text = "There are no cars."
            new_info["text"] = text#f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            new_info = {}
            text = "A scene with no cars."
            new_info["text"] = text#f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            continue
        # ---- No cars. ----

        # ---- car num ----
        '''
            There are cars.
            The scene contains one car.
            One Car.
            Two Cars.
            Three Cars.
            Four Cars.
            Five cars.
            More than five cars.
            One car is facing forward.
            One car is facing right.
            One car is facing backward.
            One car is facing left.
        '''
        if (gt_names_unique.__contains__(TARGET_CLASS)):
            new_info = {}
            text = text_quantity_l1(info)

            text_temp = f"More than five {TARGET_CLASS}s."
            if(text == text_temp):
                if(not class_num_dict.keys().__contains__(text_temp)):
                    class_num_dict[text_temp] = 0
                else:
                    text_temp_num = class_num_dict[text_temp]
                    if(text_temp_num >= 3457):
                        text = text_orientation_l1c(info)
                        text = f"{weather}{text}" if weather is not None else text
                    else:
                        class_num_dict[text_temp] = text_temp_num + 1

            new_info["text"] = text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)
        # ---- car num ----

        # ---- car and barrier ----
        '''
            Cars and barriers.
            The scene contains a car and a barrier.
            One car is to the left of one barrier.
            One car is to the right of one barrier.
        '''
        if (gt_names_unique.__contains__("barrier") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "barrier"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and barrier ----

        # ---- car and bicycle ----
        '''
            Cars and bicycles.
            The scene contains a car and a bicycle.
            One car is to the left of one bicycle.
            One car is to the right of one bicycle.
        '''
        if (gt_names_unique.__contains__("bicycle") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "bicycle"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and bicycle ----

        # ---- car and bus ----
        '''
            Cars and buses.
            The scene contains a car and a bus.
            One car is to the left of one bus.
            One car is to the right of one bus.
        '''
        if (gt_names_unique.__contains__("bus") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "bus"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}es."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and bus ----

        # ---- car and vehicle ----
        '''
            Cars and vehicles.
            The scene contains a car and a vehicle.
            One car is to the left of one vehicle.
            One car is to the right of one vehicle.
        '''
        if (gt_names_unique.__contains__("construction_vehicle") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "vehicle"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == "construction_vehicle")[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and vehicle ----

        # ---- car and motorcycle ----
        '''
            Cars and motorcycles.
            The scene contains a car and a motorcycle.
            One car is to the left of one motorcycle.
            One car is to the right of one motorcycle.
        '''
        if (gt_names_unique.__contains__("motorcycle") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "motorcycle"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and motorcycle ----

        # ---- car and motorcycle ----
        '''
            Cars and pedestrians.
            The scene contains a car and a pedestrian.
            One car is to the left of one pedestrian.
            One car is to the right of one pedestrian.
        '''
        if (gt_names_unique.__contains__("pedestrian") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "pedestrian"
            num = class_num_dict[name]
            if(num < 4000):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 8000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and motorcycle ----

        # ---- car and motorcycle ----
        '''
            Cars and traffic cones.
            The scene contains a car and a traffic cone.
            One car is to the left of one traffic cone.
            One car is to the right of one traffic cone.
        '''
        if (gt_names_unique.__contains__("traffic_cone") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "traffic cone"
            num = class_num_dict[name]
            if(num < 2000):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 4000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == "traffic_cone")[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and motorcycle ----

        # ---- car and trailer ----
        '''
            Cars and trailers.
            The scene contains a car and a trailer.
            One car is to the left of one trailer.
            One car is to the right of one trailer.
        '''
        if (gt_names_unique.__contains__("trailer") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "trailer"
            num = class_num_dict[name]
            if(num < 1500):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 3000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and trailer ----

        # ---- car and truck ----
        '''
            Cars and trucks.
            The scene contains a car and a truck.
            One car is to the left of one truck.
            One car is to the right of one truck.
        '''
        if (gt_names_unique.__contains__("truck") and gt_names_unique.__contains__(TARGET_CLASS)):
            name = "truck"
            num = class_num_dict[name]
            if(num < 4000):
                text = f"{TARGET_CLASS}s and {name}s."
                text = text.capitalize()
            elif(num < 8000):
                text = f"The scene contains a {TARGET_CLASS} and a {name}."
            else:
                object_index = np.where(gt_names == name)[0][0]
                object_box = gt_boxes[object_index]
                target_boxes = np.where(gt_names == TARGET_CLASS)[0][0]
                target_box = gt_boxes[target_boxes]
                x,y = relative_position(object_box,target_box)
                text = f"One {TARGET_CLASS} is the {y} of one {name}."

            new_info = {}
            new_info["text"] = f"{weather}{text}" if weather is not None else text
            new_info["lidar_path"] = lidar_path
            new_info["token"] = token
            new_info["semantic"] = semantic
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_infos.append(new_info)

            class_num_dict[name] = num + 1
            pass
        # ---- car and truck ----

    return new_infos

def get_brief_nusc(infos):

    new_infos = []

    nusc = get_nusc()
    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Text Decription: ")):

        token = info["token"]
        cam_paths, cam_tokens, cam_infos, lidar_path, lidar_token, sample_token = get_cam(
            token,
            nusc=nusc
        )

        description = info["description"]
        text = info["text_aim"]
        gt_names = info["gt_names"]
        gt_boxes = info["gt_boxes"]
        gt_class = np.asarray([NAME_INDEX[name] for name in gt_names]).reshape(-1, 1)
        gt_boxes = np.concatenate([gt_boxes, gt_class], axis=-1)
        lidar_path = info["lidar_path"]
        semantic = lidar_path.replace("/LIDAR_TOP/", "/SEMANTIC/").replace("bin", "npy")

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["CAM_FRONT"] = cam_paths['CAM_FRONT']
        new_info["CAM_FRONT_LEFT"] = cam_paths['CAM_FRONT_LEFT']
        new_info["CAM_FRONT_RIGHT"] = cam_paths['CAM_FRONT_RIGHT']
        new_info["CAM_BACK"] = cam_paths['CAM_BACK']
        new_info["CAM_BACK_LEFT"] = cam_paths['CAM_BACK_LEFT']
        new_info["CAM_BACK_RIGHT"] = cam_paths['CAM_BACK_RIGHT']

        new_infos.append(new_info)

    return new_infos

def get_camera_nusc(infos):

    new_infos = []

    nusc = get_nusc()
    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Text Decription: ")):

        token = info["token"]
        cam_paths, cam_tokens, cam_infos, lidar_path, lidar_token, sample_token = get_cam(
            token,
            nusc=nusc
        )

        cam_yaw_ranges = get_all_camera_yaw_ranges(
            cam_infos,
            image_size=(1600, 900)
        )

        # for cam, info in cam_yaw_ranges.items():
        #     print(cam, info)

        description = info["description"]
        text = info["text"]
        gt_names = info["gt_names"]
        gt_boxes = info["gt_boxes"]
        gt_class = np.asarray([NAME_INDEX[name] for name in gt_names]).reshape(-1, 1)
        gt_boxes = np.concatenate([gt_boxes, gt_class], axis=-1)
        lidar_path = info["lidar_path"]
        semantic = lidar_path.replace("/LIDAR_TOP/", "/SEMANTIC/").replace("bin", "npy")

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_FRONT']
        new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT']
        new_infos.append(new_info)

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_FRONT_LEFT']
        new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT_LEFT']
        new_infos.append(new_info)

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_FRONT_RIGHT']
        new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT_RIGHT']
        new_infos.append(new_info)

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_BACK']
        new_info["camera_info"] = cam_yaw_ranges['CAM_BACK']
        new_infos.append(new_info)

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_BACK_LEFT']
        new_info["camera_info"] = cam_yaw_ranges['CAM_BACK_LEFT']
        new_infos.append(new_info)

        new_info = {}
        new_info["lidar_path"] = lidar_path
        new_info["semantic"] = semantic
        new_info["text"] = text
        new_info["description"] = description
        new_info["gt_names"] = gt_names
        new_info["gt_boxes"] = gt_boxes
        new_info["token"] = token
        new_info["camera"] = cam_paths['CAM_BACK_RIGHT']
        new_info["camera_info"] = cam_yaw_ranges['CAM_BACK_RIGHT']
        new_infos.append(new_info)

    return new_infos

def get_camera_nusc_nowarp(infos):

    new_infos = []

    nusc = get_nusc()
    for i, info in enumerate(tqdm(infos, total=NUM, desc="Getting Text Decription: ")):

        token = info["token"]
        cam_paths, cam_tokens, cam_infos, lidar_path, lidar_token, sample_token = get_cam(
            token,
            nusc=nusc
        )

        cam_yaw_ranges = get_all_camera_yaw_ranges(
            cam_infos,
            image_size=(1600, 900)
        )

        # for cam, info in cam_yaw_ranges.items():
        #     print(cam, info)

        description = info["description"]
        text = info["text"]
        gt_names = info["gt_names"]
        gt_boxes = info["gt_boxes"]
        gt_class = np.asarray([NAME_INDEX[name] for name in gt_names]).reshape(-1, 1)
        gt_boxes = np.concatenate([gt_boxes, gt_class], axis=-1)
        lidar_path = info["lidar_path"]
        semantic = lidar_path.replace("/LIDAR_TOP/", "/SEMANTIC/").replace("bin", "npy")

        if(not cam_yaw_ranges['CAM_FRONT']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_FRONT']
            new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT']
            new_infos.append(new_info)

        if (not cam_yaw_ranges['CAM_FRONT_LEFT']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_FRONT_LEFT']
            new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT_LEFT']
            new_infos.append(new_info)

        if (not cam_yaw_ranges['CAM_FRONT_RIGHT']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_FRONT_RIGHT']
            new_info["camera_info"] = cam_yaw_ranges['CAM_FRONT_RIGHT']
            new_infos.append(new_info)

        if (not cam_yaw_ranges['CAM_BACK']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_BACK']
            new_info["camera_info"] = cam_yaw_ranges['CAM_BACK']
            new_infos.append(new_info)

        if (not cam_yaw_ranges['CAM_BACK_LEFT']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_BACK_LEFT']
            new_info["camera_info"] = cam_yaw_ranges['CAM_BACK_LEFT']
            new_infos.append(new_info)

        if (not cam_yaw_ranges['CAM_BACK_RIGHT']['wrap']):
            new_info = {}
            new_info["lidar_path"] = lidar_path
            new_info["semantic"] = semantic
            new_info["text"] = text
            new_info["description"] = description
            new_info["gt_names"] = gt_names
            new_info["gt_boxes"] = gt_boxes
            new_info["token"] = token
            new_info["camera"] = cam_paths['CAM_BACK_RIGHT']
            new_info["camera_info"] = cam_yaw_ranges['CAM_BACK_RIGHT']
            new_infos.append(new_info)

    return new_infos

def save_text_features():

    save_path = "/ihoment/youjie10//qwt/dataset/nuscenes/v1.0-trainval/nuscenes_text_features.npy"

    text_dict = check_text(description="nuscenes_description_plus_plus.pkl")
    texts = list(text_dict.keys())

    text_encoder = t5("large").to("cuda")
    text_emb = text_encoder.tokenize(texts, device="cuda")

    text_features = text_encoder.encode_text(text_emb, pool_features=False)  # B, 512

    mask = text_emb["attention_mask"].unsqueeze(-1)  # [B, L, 1]
    features = (text_features * mask).sum(dim=1) / mask.sum(dim=1)

    features = features.cpu().detach().numpy()

    dict_text = {}
    dict_text["features"] = features
    dict_text["texts"] = texts

    save_pkl(description="nuscenes_text_features.pkl", infos=dict_text)

    return features

def find_most_similar_gpu(x1, x2):
    """
    x1: [1, 1024] tensor
    x2: [N, 1024] tensor

    return:
        index: 最相似下标
        distance: 最小L2距离
    """

    # 计算 L2 距离
    distances = torch.norm(x2 - x1, dim=1)  # [N]

    # 找最小距离
    index = torch.argmin(distances)

    return index.item(), distances[index].item()

def find_most_similar_cpu(x1, x2):
    """
    x1: [1, 1024] ndarray
    x2: [N, 1024] ndarray

    return:
        index: 最相似下标
        distance: 最小L2距离
    """

    # 计算 L2 距离
    distances = np.linalg.norm(x2 - x1, axis=1)  # [N]

    # 找最小距离
    index = np.argmin(distances)

    return int(index), float(distances[index])

def text_features():

    dict_text = read_pkl(description="nuscenes_text_features.pkl")
    texts = dict_text["texts"]

    text_features = dict_text["features"]
    text_features = torch.from_numpy(text_features).cuda()

    text = ["Raining. No cars."]
    text_encoder = t5("large").to("cuda")
    text_emb = text_encoder.tokenize(text, device="cuda")
    text_feature = text_encoder.encode_text(text_emb, pool_features=False)  # B, 512
    mask = text_emb["attention_mask"].unsqueeze(-1)  # [B, L, 1]
    text_feature = (text_feature * mask).sum(dim=1) / mask.sum(dim=1)

    idx,_ = find_most_similar_gpu(x1=text_feature, x2=text_features)
    text = texts[idx]

    print(idx)
    print(text)

def downsampling_LIDAR(
        root_path = "/root/dataset/nuScenes/v1.0-trainval/samples/LIDAR_TOP",
        dest_path = "/root/dataset/nuScenes/v1.0-trainval/samples/LIDAR_TOP_DOWNSAMPLING",
        up_rate=0.25
):
    dest_path = f"{dest_path}{up_rate}"
    if(not os.path.exists(dest_path)):
        os.makedirs(dest_path)

    bins = sorted(glob.glob(f"{root_path}/*.bin"))
    for i, bin in enumerate(bins):
        points = common.get_lidar_sweep(bin, return_intensity=False, return_time=False, dim=5)

        if(up_rate != 1.0):
            points = torch.from_numpy(points).float().cuda()
            sampling_points = points.unsqueeze(0).permute(0, 2, 1)  # [1,C,N]
            sampling_points = common.midpoint_interpolate(sampling_points, up_rate=up_rate, only_FPS=True)  # [1,C,N/2]
            sampling_points = sampling_points.permute(0, 2, 1).squeeze()  # [1,N/2,C] -> [N/2,C]
            sampling_points = sampling_points.cpu().numpy()
        else:
            sampling_points = points[:, :3]

        bin_name = bin.split("/")[-1]
        dest_bin_path = f"{dest_path}/{bin_name}"
        sampling_points.tofile(dest_bin_path)

        print(f"---- {i}/{len(bins)} {dest_bin_path} ----")

if __name__ == '__main__':

    infos = read_pkl()[:NUM]
    infos = get_text(infos)
    check_text(infos)
    save_pkl(description="nuscenes_description_plus_plus.pkl", infos=infos)

    pass

