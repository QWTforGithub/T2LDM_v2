# coding=utf-8
import os
import open3d
import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional
from itertools import filterfalse
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from pathlib import Path
from typing import List
import torch
import torch.distributed as dist
import random
import numpy as np
import glob
import shutil
import joblib
import cv2
import matplotlib.cm as cm
import pickle
import argparse
from einops import rearrange, repeat
# from pointops.functions import pointops
import matplotlib.pyplot as plt
import subprocess

def lidar_to_bev(
    points: np.ndarray,
    H: int = 64,
    W: int = 1024,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
):
    """
    Convert LiDAR point cloud [N,3] to binary occupancy BEV [H,W]

    Args:
        points: [N,3] numpy array (x,y,z)
        H,W: BEV size
        min_depth: same as RM min depth
        max_depth: same as RM max depth

    Returns:
        bev: [H,W] binary occupancy map
    """

    assert points.ndim == 2
    assert points.shape[1] >= 3

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # spherical depth
    depth = np.sqrt(x**2 + y**2 + z**2)

    # keep points within RM range
    mask = (
        (depth >= min_depth) &
        (depth <= max_depth)
    )

    x = x[mask]
    y = y[mask]

    # BEV spatial range
    x_min, x_max = -max_depth, max_depth
    y_min, y_max = -max_depth, max_depth

    # metric -> pixel
    x_idx = ((x - x_min) / (x_max - x_min) * W).astype(np.int32)
    y_idx = ((y - y_min) / (y_max - y_min) * H).astype(np.int32)

    # clamp
    x_idx = np.clip(x_idx, 0, W - 1)
    y_idx = np.clip(y_idx, 0, H - 1)

    # occupancy BEV
    bev = np.zeros((H, W), dtype=np.float32)

    bev[y_idx, x_idx] = 1.0

    # flip y-axis so forward direction is upward
    bev = np.flip(bev, axis=0)

    return bev

def lidar_to_bev_torch(
    points: torch.Tensor,
    H: int = 32,
    W: int = 1024,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
):
    """
    Convert LiDAR point cloud [N,3] to binary occupancy BEV [H,W]

    Args:
        points: [N,3]  (x,y,z)
        H,W: BEV size
        min_depth: same as RM min depth
        max_depth: same as RM max depth

    Returns:
        bev: [H,W] binary occupancy map
    """

    assert points.ndim == 2
    assert points.shape[1] >= 3

    device = points.device

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # spherical depth (same as RM)
    depth = torch.sqrt(x**2 + y**2 + z**2)

    # keep points within RM range
    mask = (
        (depth >= min_depth) &
        (depth <= max_depth)
    )

    x = x[mask]
    y = y[mask]

    # BEV spatial range
    x_min, x_max = -max_depth, max_depth
    y_min, y_max = -max_depth, max_depth

    # metric -> pixel
    x_idx = ((x - x_min) / (x_max - x_min) * W).long()
    y_idx = ((y - y_min) / (y_max - y_min) * H).long()

    # clamp
    x_idx = torch.clamp(x_idx, 0, W - 1)
    y_idx = torch.clamp(y_idx, 0, H - 1)

    # occupancy bev
    bev = torch.zeros((H, W), dtype=torch.float32, device=device)

    bev[y_idx, x_idx] = 1.0

    # optional:
    # flip y-axis so that forward direction is upward
    bev = torch.flip(bev, dims=[0])

    return bev

def sort_with_index(arr):
    """
    对 ndarray 从小到大排序，并返回排序后的数组以及对应索引

    Args:
        arr: np.ndarray 或 list

    Returns:
        sorted_arr: 排序后的数组
        indices: 排序索引
    """
    arr = np.array(arr)

    # 返回排序索引
    indices = np.argsort(arr)

    # 根据索引排序
    sorted_arr = arr[indices]

    return sorted_arr, indices

def get_batch(
    batch=None,
    points=None,
    xyz=None,
    reflectance=None,
    time=None,
    semantic=None,
    depth=None,
    mask=None,
    text=None,
    semantic_org=None,
    sampling_points=None,
    sampling_depth=None,
    sampling_batch=None,
    box=None,
    camera=None,
    camera_info=None,
    bev=None,
):
    batchs = {}
    batchs['batch'] = batch
    batchs['points'] = points
    batchs['xyz'] = xyz
    batchs["reflectance"] = reflectance
    batchs['time'] = time
    batchs['semantic'] = semantic
    batchs['depth'] = depth
    batchs['mask'] = mask
    batchs['text'] = text
    batchs['semantic_org'] = semantic_org
    batchs['sampling_points'] = sampling_points
    batchs['sampling_depth'] = sampling_depth
    batchs['sampling_batch'] = sampling_batch
    batchs['box'] = box
    batchs['camera'] = camera
    batchs['camera_info'] = camera_info
    batchs['bev'] = bev

    return batchs

def get_inputs():
    inputs = {}
    inputs['x'] = None
    inputs['texts'] = None
    inputs['texts_null'] = None
    inputs["cfg_texts"] = None
    inputs['text_features'] = None
    inputs['text_null_features'] = None
    inputs['semantic'] = None
    inputs['semantic_org'] = None
    inputs['gl'] = None
    inputs['xyz'] = None
    inputs['points'] = None
    inputs['batch'] = None
    inputs['sparse_dense'] = None
    inputs['sampling_points'] = None
    inputs['sampling_depth'] = None
    inputs['sampling_batch'] = None
    inputs['sparse_dense'] = None
    inputs['box'] = None
    inputs['camera'] = None
    inputs['camera_info'] = None
    inputs['bev'] = None

    return inputs

def get_text_features(
        text_encoder,
        text,
        text_null,
        clip_pool_features,
        device
):
    text_emb = text_encoder.tokenize(text, device=device)
    text_null_emb = text_encoder.tokenize(text_null, device=device)
    with torch.no_grad():
        text_features = text_encoder.encode_text(text_emb, pool_features=clip_pool_features)
        text_null_features = text_encoder.encode_text(text_null_emb, pool_features=clip_pool_features)
    return text_features, text_null_features

def split_channels(
        image: torch.Tensor,
        train_depth=True,
        train_reflectance=True,
):
    channels = [
        1 if train_depth else 0,
        1 if train_reflectance else 0,
    ]

    depth, rflct = torch.split(image, channels, dim=1)
    return depth, rflct

def preprocess(
        inputs,
        batch,

        classifier_dropout=0.1,
        use_text=False,
        use_seg=False,
        train_depth=True,
        train_reflectance=True,
        upsampling=False,
        downsampling=False,

        resolution=(32,1024),
        lidar_utils=None,
        device=None,
):
    if("depth" in batch.keys() and batch["depth"] is not None):
        x = []
        if train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=resolution,
            mode="nearest-exact",
        )
        inputs["x"] = x

    cfg_texts = None
    texts = None
    if (use_text and "text" in batch.keys()):
        texts = batch["text"]
        cfg_texts = []
        for text in texts:
            if random.random() < classifier_dropout:
                cfg_texts.append("")
            else:
                cfg_texts.append(text)
    inputs["cfg_texts"] = cfg_texts
    inputs["texts"] = texts

    semantic = None
    semantic_org = None
    if (use_seg and "semantic" in batch.keys() and "semantic_org" in batch.keys()):
        semantic = batch["semantic"]
        if (not semantic.is_cuda):
            semantic = semantic.cuda()
        semantic_org = batch["semantic_org"]
        if (not semantic_org.is_cuda):
            semantic_org = semantic_org.cuda()
    inputs["semantic"] = semantic
    inputs["semantic_org"] = semantic_org

    points = None
    if("points" in batch.keys()):
        points = batch["points"]
        if (not points.is_cuda):
            points = points.cuda()
    inputs["points"] = points

    batches = None
    if ("batch" in batch.keys()):
        batches = batch["batch"]
        if (not batches.is_cuda):
            batches = batches.cuda()
    inputs["batch"] = batches

    xyz = None
    if("xyz" in batch.keys()):
        xyz = batch["xyz"]
        if(not xyz.is_cuda):
            xyz = xyz.cuda()
    inputs["xyz"] = xyz

    sampling_points = None
    sampling_depth = None
    sampling_batch = None
    if ((upsampling or downsampling) and "sampling_points" in batch.keys() and "sampling_depth" in batch.keys() and "sampling_batch" in batch.keys()):
        sampling_points = batch["sampling_points"]
        if (not sampling_points.is_cuda):
            sampling_points = sampling_points.cuda()

        sampling_depth = batch["sampling_depth"]
        sampling_depth = lidar_utils.convert_depth(sampling_depth)
        sampling_depth = lidar_utils.normalize(sampling_depth)
        sampling_depth = F.interpolate(
            sampling_depth.to(device),
            size=resolution,
            mode="nearest-exact",
        )

        sampling_batch = batch["sampling_batch"]
        if (not sampling_batch.is_cuda):
            sampling_batch = sampling_batch.cuda()
    inputs["sampling_points"] = sampling_points
    inputs["sampling_depth"] = sampling_depth
    inputs["sampling_batch"] = sampling_batch

    box = None
    if("box" in batch.keys()):
        box = batch["box"]
    inputs["box"] = box

    camera = None
    if("camera" in batch.keys()):
        camera = batch["camera"]
    inputs["camera"] = camera

    camera_info = None
    if("camera_info" in batch.keys()):
        camera_info = batch["camera_info"]
    inputs["camera_info"] = camera_info

    bev = None
    if("bev" in batch.keys()):
        bev = batch["bev"]
        inputs["semantic_org"] = None
    inputs["bev"] = bev

    inputs["text_features"] = None

    return inputs

def normalize_image(img, mean=None, std=None, to_float=True, channel_first=False):
    """
    img: np.ndarray, [H,W,C] or [C,H,W]
    mean/std: list or tuple, e.g. [0.485, 0.456, 0.406]
    channel_first: True if img is [C,H,W]
    """
    img = img.astype(np.float32) if to_float else img

    # 1. 如果是 uint8 图像，先归一化到 [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    # 2. 如果没有指定 mean/std，只做 [0,1] 归一化
    if mean is None or std is None:
        return img

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    if channel_first:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, None, :]
        std = std[None, None, :]

    img = (img - mean) / std
    return img

def boxes_to_corners_3d(boxes, box_dim_order="lwh"):
    """
    Args:
        boxes: [M, 7]
            [x, y, z, d1, d2, d3, yaw]

        box_dim_order:
            "lwh": d1=length, d2=width,  d3=height
            "wlh": d1=width,  d2=length, d3=height

    Returns:
        corners: [M, 8, 3]
    """
    boxes = np.asarray(boxes, dtype=np.float32)

    centers = boxes[:, 0:3]
    d1 = boxes[:, 3]
    d2 = boxes[:, 4]
    h = boxes[:, 5]
    yaw = boxes[:, 6]

    if box_dim_order == "lwh":
        l = d1
        w = d2
    elif box_dim_order == "wlh":
        w = d1
        l = d2
    else:
        raise ValueError("box_dim_order should be 'lwh' or 'wlh'.")

    # local corners: x is length direction, y is width direction
    x_corners = np.stack([
        l / 2,  l / 2, -l / 2, -l / 2,
        l / 2,  l / 2, -l / 2, -l / 2
    ], axis=1)

    y_corners = np.stack([
        w / 2, -w / 2, -w / 2,  w / 2,
        w / 2, -w / 2, -w / 2,  w / 2
    ], axis=1)

    z_corners = np.stack([
        h / 2, h / 2, h / 2, h / 2,
       -h / 2, -h / 2, -h / 2, -h / 2
    ], axis=1)

    corners_local = np.stack([x_corners, y_corners, z_corners], axis=-1)

    #yaw = -boxes[:, 6]  # 关键！！！

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    R = np.zeros((boxes.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = cos_yaw
    R[:, 0, 1] = -sin_yaw
    R[:, 1, 0] = sin_yaw
    R[:, 1, 1] = cos_yaw
    R[:, 2, 2] = 1.0

    corners = corners_local @ np.transpose(R, (0, 2, 1))
    corners = corners + centers[:, None, :]

    return corners

def get_semantic_from_corners(
    points,
    corners,
    classes,
    background_id=16,
):
    """
    Args:
        points:  [N, 3]
        corners: [M, 8, 3]
        classes: [M]
        background_id: background label

    Returns:
        labels: [N]
    """
    points = np.asarray(points, dtype=np.float32)
    corners = np.asarray(corners, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int64)

    N = points.shape[0]
    M = corners.shape[0]

    labels = np.full((N,), background_id, dtype=np.int64)

    for i in range(M):
        c = corners[i]  # [8, 3]

        # box center
        center = c.mean(axis=0)

        # 根据你之前的 corner 顺序：
        # 0,1,2,3 top; 4,5,6,7 bottom
        # length direction: 3 -> 0
        # width  direction: 1 -> 0
        # height direction: 4 -> 0
        axis_l = c[0] - c[3]
        axis_w = c[0] - c[1]
        axis_h = c[0] - c[4]

        l = np.linalg.norm(axis_l)
        w = np.linalg.norm(axis_w)
        h = np.linalg.norm(axis_h)

        if l < 1e-6 or w < 1e-6 or h < 1e-6:
            continue

        axis_l = axis_l / l
        axis_w = axis_w / w
        axis_h = axis_h / h

        # points -> local coordinates
        pts = points - center[None, :]

        local_l = pts @ axis_l
        local_w = pts @ axis_w
        local_h = pts @ axis_h

        mask = (
            (np.abs(local_l) <= l / 2) &
            (np.abs(local_w) <= w / 2) &
            (np.abs(local_h) <= h / 2)
        )

        labels[mask] = classes[i]

    return labels

def folder_files(
        root_path="/ihoment/youjie10/qwt/res",
        suffix="ply"
):
    files = sorted(glob.glob(f"{root_path}/*.{suffix}"))

    print(len(files))
    return files

def colorize_points_by_semantic(
        points=None,
        labels=None,
        dest_path=None
):
    colors = get_color()
    gt_color = []
    for label in labels:
        gt_color.append(colors[label])
    gt_color = np.asarray(gt_color)

    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    if (colors is not None):
        pc.colors = open3d.utility.Vector3dVector(gt_color)
    open3d.io.write_point_cloud(dest_path, pc)



def downsampling_LIDAR(
        root_path = "/ihoment/youjie10/qwt/dataset/nuscenes/v1.0-trainval/samples/LIDAR_TOP",
        dest_path = "/ihoment/youjie10/qwt/dataset/nuscenes/v1.0-trainval/samples/LIDAR_TOP_DOWNSAMPLING",
        up_rate=0.25
):
    bins = sorted(glob.glob(f"{root_path}/*.bin"))
    for i, bin in enumerate(bins):
        points = get_lidar_sweep(bin, return_intensity=False, return_time=False, dim=5)

        if(up_rate != 1.0):
            points = torch.from_numpy(points).float().cuda()
            sampling_points = points.unsqueeze(0).permute(0, 2, 1)  # [1,C,N]
            sampling_points = midpoint_interpolate(sampling_points, up_rate=up_rate)  # [1,C,N/2]
            sampling_points = sampling_points.permute(0, 2, 1).squeeze()  # [1,N/2,C] -> [N/2,C]
            sampling_points = sampling_points.cpu().numpy()
        else:
            sampling_points = points[:, :3]

        bin_name = bin.split("/")[-1]
        dest_bin_path = f"{dest_path}/{bin_name}"
        sampling_points.tofile(dest_bin_path)

        print(f"---- {i}/{len(bins)} {dest_bin_path} ----")


def excute_linux_command():

    zips = sorted(glob.glob(f"/zhangq0619/KITTI360/*.zip"))

    for zip in zips:
        subprocess.run(["unzip", zip])

def copy_folder_to_folder_semantic(
        src="/zhangq0619/qwt/models/T2LDM-final/test_nuScenes_0417",
        dst="/zhangq0619/qwt/models/results",
        generation=True,
        xyz=False,
        semantic=False,
        dense=False,
        sparse=False,
        text=False,
        box=False
):

    folders = sorted(glob.glob(f"{src}/*_ddpm1024_*"))
    folder_num = len(folders)
    files = []
    for j,folder in enumerate(folders):

        if(generation):
            # ---- generation folder ----
            generation_folder = f"{folder}/plys/generation"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/generation"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                files.append(dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- generation folder ----

        if(xyz):
            # ---- xyz folder ----
            generation_folder = f"{folder}/plys/xyz"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/xyz"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- xyz folder ----

        if(semantic):
            # ---- semantic/generationcolor folder ----
            generation_folder = f"{folder}/plys/semantic/generationcolor"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/semantic/generationcolor"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- semantic/generationcolor folder ----

            # ---- semantic/xyzcolor folder ----
            generation_folder = f"{folder}/plys/semantic/xyzcolor"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/semantic/xyzcolor"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying:---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- semantic/xyzcolor folder ----

            # ---- semantic/semantic_data folder ----
            generation_folder = f"{folder}/plys/semantic/semantic_data"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.npy"))
            dst_generation_folder = f"{dst}/semantic/semantic_data"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- semantic/xyzcolor folder ----

        if(dense):
            # ---- dense folder ----
            generation_folder = f"{folder}/plys/dense"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/dense"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- generation folder ----

        if(sparse):
            # ---- sparse folder ----
            generation_folder = f"{folder}/plys/sparse"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.ply"))
            dst_generation_folder = f"{dst}/sparse"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- generation folder ----

        if(text):
            # ---- text folder ----
            generation_folder = f"{folder}/plys/text"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.npy"))
            dst_generation_folder = f"{dst}/text"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                ply_name = ply.split("/")[-1].replace(".npy", ".txt")
                ply = ply.replace(".pkl.npy", ".txt")
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- text folder ----

        if(box):
            # ---- box folder ----
            generation_folder = f"{folder}/plys/box"
            generation_plys = sorted(glob.glob(f"{generation_folder}/*.npy"))
            dst_generation_folder = f"{dst}/box"
            if not os.path.exists(dst_generation_folder):
                os.makedirs(dst_generation_folder)
            for k,ply in enumerate(generation_plys):
                ply_name = ply.split("/")[-1]
                dst_generation_ply = f"{dst_generation_folder}/{ply_name}"
                shutil.copy(ply, dst_generation_ply)
                print(f"Copying: ---- {j}/{folder_num} ---- {dst_generation_ply}")
            # ---- box folder ----

    print(len(files))

def copy_folders_to_folder_unconditional(
        src="/root/models/T2LDM-plus/test_nuScenes_0510_10240",
        dst="/root/models/res"
):
    if(not os.path.exists(dst)):
        os.makedirs(dst)
    folders = sorted(glob.glob(f"{src}/*_ddpm1024_unconditional_*"))
    files = []
    for i,folder in enumerate(folders):
        plys_path = f"{folder}/plys/generation/*.ply"
        plys = sorted(glob.glob(plys_path))
        for j, ply in enumerate(plys):
            ply_name = ply.split("/")[-1]
            dst_ply = f"{dst}/{ply_name}"
            files.append(dst_ply)
            shutil.copy(ply, dst_ply)
            print(f"Copying: ---- {i}/{len(folders)} {dst_ply}")

    print(len(files))

def copy_folders_to_folder(srcs, dst):

    for src in srcs:
        plys = sorted(glob.glob(f"{src}/*.ply"))
        for ply in plys:
            ply_name = ply.split("/")[-1]
            shutil.copy(ply, f"{dst}/{ply_name}")
            print(f"Copied {dst}/{ply_name}")

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def save_pkl(save_path=None, infos=None, create_folder=True):

    if(create_folder):
        parent = os.path.dirname(save_path)
        os.makedirs(parent, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)
        print(f"\n---- Saving : {save_path} ----")

def del_objs(
        names_to_clear= [
        'sample', 'pred_x_0', 'noise', 'conditional_x_0',
        'xyz', 'normal', 'metric', 'mask', 'bev',
        'points', 'batches'
    ]
):
    for name in names_to_clear:
        if name in locals():
            obj = locals()[name]
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj

    # 强制释放缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_semantickitti_learning_map(ignore_index=19):
    learning_map = {
        0: ignore_index,  # "unlabeled"
        1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
        10: 0,  # "car"
        11: 1,  # "bicycle"
        13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
        15: 2,  # "motorcycle"
        16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
        18: 3,  # "truck"
        20: 4,  # "other-vehicle"
        30: 5,  # "person"
        31: 6,  # "bicyclist"
        32: 7,  # "motorcyclist"
        40: 8,  # "road"
        44: 9,  # "parking"
        48: 10,  # "sidewalk"
        49: 11,  # "other-ground"
        50: 12,  # "building"
        51: 13,  # "fence"
        52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
        60: 8,  # "lane-marking" to "road" ---------------------------------mapped
        70: 14,  # "vegetation"
        71: 15,  # "trunk"
        72: 16,  # "terrain"
        80: 17,  # "pole"
        81: 18,  # "traffic-sign"
        99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
        252: 0,  # "moving-car" to "car" ------------------------------------mapped
        253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
        254: 5,  # "moving-person" to "person" ------------------------------mapped
        255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
        256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
        257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
        258: 3,  # "moving-truck" to "truck" --------------------------------mapped
        259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    }
    return learning_map

def get_color():
    colors=[
        [229, 25,  74 ], # 0 ，#E5194A，红色，"barrier"，障碍物
        [60,  179, 77 ], # 1 ，#3CB34D，绿色，"bicycle"，自行车
        [255, 224, 25 ], # 2 ，#FFE019，黄色，"bus"，公共汽车
        [68 , 99 , 216], # 3 ，#4463D8，蓝色，"car"，小车辆
        [245, 130, 49 ], # 4 ，#F58231，橙色，"construction_vehicle"，建筑车辆
        [144, 30 , 180], # 5 ，#901EB4，紫色，"motorcycle"，摩托车
        [68 , 211, 245], # 6 ，#44D3F5，青色， "pedestrian"，行人
        [239, 50 , 232], # 7 ，#EF32E8，深粉色，"traffic_cone"，交通锥
        [191, 238, 70 ], # 8 ，#BFEE46，金绿色，"trailer"，拖车
        [251, 189, 212], # 9 ，#FBBDD4，粉色，"truck"，卡车
        [82 , 146, 141], # 10，#52928D，深绿色， "driveable_surface"，可行驶路面
        [220, 190, 254], # 11，#DCBEFF，浅紫色，"other_flat"，其他平坦路面
        [154, 98 , 37 ], # 12，#9A6225，咖啡色， "sidewalk"，人行道
        [255, 249, 199], # 13，#FFF9C7，浅黄色，"terrain"，地形
        [128, 0  , 0  ], # 14，#800000，深红色，"manmade"，人造对象
        [170, 255, 195], # 15，#AAFFC3，浅绿色，"vegetation"，植被
        [127, 128, 0  ], # 16，#7F8000，深黄色
        [255, 215, 177], # 17，#FFD7B1，浅红色
        [1  , 0  , 119], # 18，#010077，浅蓝色
        [169, 169, 169], # 19，#A9A9A9，灰色

        [255, 255, 255], # #000000，黑色
    ]

    colors = np.asarray(colors) / 255

    return colors
#
# def preprocess(
#         batch,
#         classifier_dropout=0.1,
#         use_text=False,
#         use_semantic=False,
#         train_depth=True,
#         train_reflectance=True,
#         lidar_utils=None,
#         text_name="text_aim"
# ):
#     x = []
#     if train_depth:
#         x += [lidar_utils.convert_depth(batch["depth"])]
#     if train_reflectance:
#         x += [batch["reflectance"]]
#     x = torch.cat(x, dim=1)
#     x = lidar_utils.normalize(x)
#     x = F.interpolate(
#         x.to("cuda"),
#         size=lidar_utils.resolution,
#         mode="nearest-exact",
#     )
#
#     new_texts = None
#     texts = None
#     if (use_text):
#         texts = batch[text_name]
#         new_texts = []
#         for text in texts:
#             if random.random() < classifier_dropout:
#                 new_texts.append("")
#             else:
#                 new_texts.append(text)
#
#     semantic = None
#     if (use_semantic):
#         semantic = batch["semantic"]
#         if(not semantic.is_cuda):
#             semantic = semantic.cuda()
#
#     xyz = None
#     if("xyz" in batch.keys()):
#         xyz = batch["xyz"]
#         if(not xyz.is_cuda):
#             xyz = xyz.cuda()
#
#     points = None
#     if("points" in batch.keys()):
#         points = batch["points"]
#         if(not points.is_cuda):
#             points = points.cuda()
#
#     batches = None
#     if("batch" in batch.keys()):
#         batches = batch["batch"]
#         if(not batches.is_cuda):
#             batches = batches.cuda()
#
#     semantic_org = None
#     if("semantic_org" in batch.keys()):
#         semantic_org = batch["semantic_org"]
#         if(not semantic_org.is_cuda):
#             semantic_org = semantic_org.cuda()
#
#     if(points is not None and batches is not None):
#         return x, new_texts, texts, semantic, points, batches, semantic_org, xyz
#     else:
#         return x, new_texts, texts, semantic

def lrepa_cosine_single(
        feat_s: torch.Tensor,
        feat_t: torch.Tensor,
        eps: float = 1e-6,
        reduction: str = "mean"
) -> torch.Tensor:
    """
    feat_s, feat_t: [B,C,H,W]（先确保空间大小一致；若不一致先插值）
    sem_w:         [B,1,H,W] 可选语义/空间权重（道路、边缘等），范围建议 [0,1]
    return:        标量或逐像素损失
    """
    # 归一化到单位向量（沿通道维）
    fs = F.normalize(feat_s, dim=1, eps=eps)
    ft = F.normalize(feat_t, dim=1, eps=eps)

    # 逐像素 cos：sum_c(fs * ft)
    cos = (fs * ft).sum(dim=1, keepdim=True)  # [B,1,H,W]
    loss_pix = 1.0 - cos                      # [B,1,H,W]

    if reduction == "mean":
        return loss_pix.mean()
    elif reduction == "sum":
        return loss_pix.sum()
    else:
        return loss_pix.squeeze(1)  # [B,H,W]



class SCRGLossWeight():
    def __init__(
            self,
            weights=[0.001,0.01,0.1,1.0],
            step_interval=25000
    ):
        self.weights = weights
        self.step_interval = step_interval

    def get_loss_weight(self, current_step):
        x = current_step // self.step_interval
        if(x < len(self.weights)):
            weight =  self.weights[x]
        else:
            weight =  self.weights[-1]
        return weight

def build_semantic_weights(
    labels,
    road_ids = [10, 11],
    weights = [0.2, 0.8],
):
    mask = torch.isin(labels, torch.tensor(road_ids, device=labels.device))
    weights = torch.where(mask, weights[1], weights[0])

    return weights

def remove_empty_dirs(
        root: str | Path, *,
        remove_root: bool = False
) -> List[Path]:
    """
    递归删除 `root` 下的所有空文件夹（自底向上）。
    Args:
        root: 根目录路径
        remove_root: 若最终 root 也为空，是否一并删除
    Returns:
        deleted: 实际删除的目录 Path 列表（按删除顺序）
    """
    root = Path(root).resolve()
    deleted: List[Path] = []
    if not root.exists() or not root.is_dir():
        return deleted

    # 多轮自底向上：先清理叶子，再尝试其父级
    # Path.rglob('*') 会列出所有层级，按长度倒序确保“先子后父”
    for p in sorted(root.rglob('*'), key=lambda x: len(x.parts), reverse=True):
        # 只处理“真实目录”（跳过符号链接目录）
        if p.is_dir() and not p.is_symlink():
            try:
                # 目录为空才能 rmdir
                if not any(p.iterdir()):
                    p.rmdir()
                    deleted.append(p)
            except PermissionError:
                pass  # 无权限时跳过
            except OSError:
                pass  # 并发/瞬时写入导致非空等，忽略

    # 最后处理根目录
    if remove_root and not any(root.iterdir()):
        try:
            root.rmdir()
            deleted.append(root)
        except Exception:
            pass

    return deleted

def w_smooth(
        depth_pred,
        depth_gt,
        threshold=0.9,
        mask=None
):
    mask = torch.ones_like(depth_pred) if mask is None else mask

    gx = grad_w(depth_pred)
    # 用 GT 的边缘抑制正则（或用你生成的 edge 权重图）
    edge = (sobel_edge_mag(depth_gt) > threshold).float()
    w = torch.exp(-5.0 * edge) * mask  # 非边缘处 w≈1，边缘处 w≈~0
    L_smoothW = (w * gx ** 2).mean()

    return L_smoothW

def grad_w(img):  # [B,1,H,W]
    g = img[..., :, 1:] - img[..., :, :-1]
    return F.pad(g, (1,0,0,0))

def sobel_edge_mag(x, circular_w=True, eps=1e-12):
    """
    x: [B,1,H,W]   （建议先把 depth 正则化到 [-1,1] 或 z-score）
    return: edge magnitude [B,1,H,W]
    """
    B, C, H, W = x.shape
    assert C == 1

    # Sobel 核
    kx = torch.tensor([[1., 0., -1.],
                       [2., 0., -2.],
                       [1., 0., -1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[ 1.,  2.,  1.],
                       [ 0.,  0.,  0.],
                       [-1., -2., -1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)

    if circular_w:
        # 垂直方向普通 pad，水平方向“循环 pad”
        # 先在列方向做 roll 拼接，等价于 circular padding
        left  = x[..., :, -1:].clone()
        right = x[..., :, :1].clone()
        x_pad = torch.cat([left, x, right], dim=-1)        # W+2
        # 再在行方向做普通 pad
        x_pad = F.pad(x_pad, (0,0,1,1), mode='replicate')  # H+2
        gx = F.conv2d(x_pad, kx)
        gy = F.conv2d(x_pad, ky)
    else:
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)

    mag = torch.sqrt(gx*gx + gy*gy + eps)  # [B,1,H,W]
    return mag

def ignore_label(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        pred="n_pred",
        target="n_target",
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()

        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.pred = pred
        self.target = target
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):

        if(pred.dim() == 4):
            B,C,H,W = pred.size()
            pred = pred.permute(0, 2, 3, 1).reshape(-1, C)

        if target.dim() == 4:
            B,C,H,W = target.size()
            target = target.permute(0, 2, 3, 1).reshape(-1)

        if(self.ignore_index != -1):
            pred, target = ignore_label(pred,target,self.ignore_index)
        loss = self.loss(pred, target) * self.loss_weight
        # validate_data(loss ,"Cross Entropy Loss")
        return loss


BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_softmax(
    probas, labels, classes="present", class_seen=None, per_image=False, ignore=None
):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probas(probas, labels, ignore),
            classes=classes,
            class_seen=class_seen
        )
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present", class_seen=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    # for c in class_to_sum:
    for c in labels.unique():
        if class_seen is None:
            fg = (labels == c).type_as(probas)  # foreground for class c
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
        else:
            if c in class_seen:
                fg = (labels == c).type_as(probas)  # foreground for class c
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, c]
                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                perm = perm.data
                fg_sorted = fg[perm]
                losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nan-mean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = filterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str="multiclass",
        pred="n_pred",
        target="n_target",
        class_seen: Optional[int] = None,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        """Lovasz loss for segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.pred = pred
        self.target = target
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.class_seen = class_seen
        self.loss_weight = loss_weight

    def forward(self, pred, target):

        if(pred.dim() == 4):
            B,C,H,W = pred.size()
            pred = pred.permute(0, 2, 3, 1).reshape(-1, C)

        if target.dim() == 4:
            B,C,H,W = target.size()
            target = target.permute(0, 2, 3, 1).reshape(-1)

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(
                pred, target, per_image=self.per_image, ignore=self.ignore_index
            )
        elif self.mode == MULTICLASS_MODE:
            pred = pred.softmax(dim=1)
            loss = _lovasz_softmax(
                pred,
                target,
                class_seen=self.class_seen,
                per_image=self.per_image,
                ignore=self.ignore_index,
            )
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss * self.loss_weight

def reflectance_norm(refl: np.ndarray) -> np.ndarray:
    x = refl.astype(np.float32, copy=False)
    if np.nanmax(x) > 2:     # 认为是 0..255
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def trans_mlp(x, mlp):
    if(len(x.shape) == 3):
        B,C,M = x.shape
        x = mlp(x.view(B,M,C)).view(B,-1,M)
    else:
        B, C, H, W = x.shape
        x = mlp(x.view(B,H,W,C)).view(B,-1,H,W)
    return x

def focal_dice_loss(
    logits,             # [B,1,H,W]  —— 分割头输出（raw）
    target01,           # [B,1,H,W]  —— 车=1/非车=0
    valid=None,         # [B,1,H,W]  —— 忽略区=0（可选）
    weight=None,        # [B,1,H,W]  —— 像素权（如盒内>1，可选）
    alpha=0.9,
    gamma=2.0,
    eps=1e-6
):
    p = torch.sigmoid(logits)
    if valid  is None: valid  = torch.ones_like(p)
    if weight is None: weight = torch.ones_like(p)

    # —— Focal BCE（带权+忽略）——
    pt = p*target01 + (1-p)*(1-target01)
    w  = (alpha*target01 + (1-alpha)*(1-target01)) * weight * valid
    focal = -(w * (1-pt).clamp_min(1e-6).pow(gamma) * torch.log(pt.clamp_min(1e-6))).sum() \
            / valid.sum().clamp_min(1)

    # —— Soft Dice（在 valid 内）——
    p, t = p*valid, target01*valid
    dice = 1.0 - (2*(p*t).sum()+eps)/((p+t).sum()+eps)

    return focal + dice

def get_hdl64e_linear_ray_angles(
        resolution: [int, int] = (64, 1024),
        fov: [float, float] = (3,-25),
        start_a: int = -180,
        end_a: int = 180,
        device: torch.device = "cpu"
):
    h_up, h_down = fov[0], fov[1]
    w_left, w_right = end_a, start_a
    H, W = resolution[0], resolution[1]
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

class RandomFlip(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
                if coord1 is not None:
                    coord1[:, 0] = -coord1[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
                if coord1 is not None:
                    coord1[:, 1] = -coord1[:, 1]
        return coord, coord1

class RandomRotateAligned(object):
    def __init__(self, rot=np.pi / 4, p=1.):
        self.rot = rot
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            angle_z = np.random.uniform(-self.rot, self.rot)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
            if coord1 is not None:
                coord1 = np.dot(coord1, R)
        return coord, coord1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd, pcd1=None):
        for t in self.transforms:
            pcd, pcd1 = t(pcd, pcd1)
        return pcd, pcd1

def get_lidar_transform(aug, training):
    transform_list = []
    if "rotation" in aug:
        transform_list.append(RandomRotateAligned())
    if "flip" in aug:
        transform_list.append(RandomFlip())
    return Compose(transform_list) if len(transform_list) > 0 and training else None

def colorize(tensor, cmap_fn=cm.turbo):
    colors = cmap_fn(np.linspace(0, 1, 256))[:, :3]
    colors = torch.from_numpy(colors).to(tensor)
    tensor = tensor.squeeze(1) if tensor.ndim == 4 else tensor
    ids = (tensor * 256).clamp(0, 255).long()
    tensor = F.embedding(ids, colors).permute(0, 3, 1, 2)
    tensor = tensor.mul(255).clamp(0, 255).byte()
    return tensor

def save_depth_vis(depth, save_path="depth_vis.png", cmap="turbo", gamma=0.5, percentile=95):
    """
    可视化并保存深度图为彩色图像
    支持 torch.Tensor 或 numpy.ndarray 输入

    Args:
        depth: (H, W) 或 (1, H, W) 的深度图
        save_path: 保存路径
        cmap: 颜色映射 ('turbo', 'plasma', 'jet', 'inferno' 等)
        gamma: 伽马增强系数 (0.4~0.6 比较自然)
        percentile: 百分位裁剪，去掉远处异常深度值
    """
    # --- 转 numpy ---
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth.squeeze()

    # --- 去除异常值 ---
    depth = np.nan_to_num(depth)
    max_val = np.percentile(depth, percentile)
    depth = np.clip(depth, 0, max_val)

    # --- 归一化到 [0,1] ---
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # --- gamma 校正（增强近处细节） ---
    depth_gamma = np.power(depth_norm, gamma)

    # --- 颜色映射 ---
    depth_color = plt.get_cmap(cmap)(depth_gamma)[:, :, :3]  # 去掉 alpha 通道
    depth_color = (depth_color * 255).astype(np.uint8)

    # --- 保存 ---
    plt.imsave(save_path, depth_color)
    print(f"✅ Depth visualization saved to: {save_path}")

def save_img(path, img, depth_color=False):
    cv2.imwrite(path, img)
    if(depth_color):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        save_depth_vis(depth=depth, save_path=path)

    print(f"---- Saving image: {path}")

def get_lidar_sweep(path, return_intensity=False, return_time=False, dim=4):

    if(str(path).endswith(".ply")):
        pc = open3d.io.read_point_cloud(path)
        scan = np.asarray(pc.points)

    else:
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, dim))

        if(return_intensity and return_time):
            scan = scan[:,:5]
        elif(return_intensity):
            scan = scan[:, :4]
        else:
            scan = scan[:, :3]

    return scan

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_multiplier: float = 0.01
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_multiplier, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def total_count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def collate_fn(batch):

    collated = {}

    if("id" in batch[0].keys() and batch[0]["id"] is not None):
        collated["id"] = [item["id"] for item in batch]

    if("batch" in batch[0].keys() and batch[0]["batch"] is not None):
        collated["batch"] = torch.cat([torch.tensor(item["batch"]) for item in batch], dim=-1)

    if("sampling_batch" in batch[0].keys() and batch[0]["sampling_batch"] is not None):
        collated["sampling_batch"] = torch.cat([torch.tensor(item["sampling_batch"]) for item in batch], dim=-1)

    if("points" in batch[0].keys() and batch[0]["points"] is not None):
        collated["points"] = torch.cat([torch.tensor(item["points"]) for item in batch], dim=0)

    if("sampling_points" in batch[0].keys() and batch[0]["sampling_points"] is not None):
        collated["sampling_points"] = torch.cat([torch.tensor(item["sampling_points"]) for item in batch], dim=0)

    if("semantic_org" in batch[0].keys() and batch[0]["semantic_org"] is not None):
        #collated["semantic_org"] = torch.cat([torch.tensor(item["semantic_org"]) for item in batch], dim=0)
        collated["semantic_org"] = torch.stack([torch.tensor(item["semantic_org"]) for item in batch], dim=0)

    if("xyz" in batch[0].keys() and batch[0]["xyz"] is not None):
        collated["xyz"] = torch.stack([torch.tensor(item["xyz"]) for item in batch], dim=0)  # (B, 3, H, W)

    if("reflectance" in batch[0].keys() and batch[0]["reflectance"] is not None):
        collated["reflectance"] = torch.stack([torch.tensor(item["reflectance"]) for item in batch],dim=0)  # (B, 1, H, W)

    if("time" in batch[0].keys() and batch[0]["time"] is not None):
        collated["time"] = torch.stack([torch.tensor(item["time"]) for item in batch], dim=0)  # (B, 1, H, W)

    if("semantic" in batch[0].keys() and batch[0]["semantic"] is not None):
        collated["semantic"] = torch.stack([torch.tensor(item["semantic"]) for item in batch], dim=0)  # (B, H, W)

    if("depth" in batch[0].keys() and batch[0]["depth"] is not None):
        collated["depth"] = torch.stack([torch.tensor(item["depth"]) for item in batch], dim=0)  # (B, 1, H, W)

    if("sampling_depth" in batch[0].keys() and batch[0]["sampling_depth"] is not None):
        collated["sampling_depth"] = torch.stack([torch.tensor(item["sampling_depth"]) for item in batch], dim=0)  # (B, 1, H, W)

    if("mask" in batch[0].keys() and batch[0]["mask"] is not None):
        collated["mask"] = torch.stack([torch.tensor(item["mask"]) for item in batch], dim=0)  # (B, 1, H, W)

    if("text" in batch[0].keys() and batch[0]["text"] is not None):
        collated["text"] = [item["text"] for item in batch]

    if("box" in batch[0].keys() and batch[0]["box"] is not None):
        collated["box"] = [item["box"] for item in batch]

    if("camera" in batch[0].keys() and batch[0]["camera"] is not None):
        collated["camera"] = [item["camera"] for item in batch]

    if("camera_info" in batch[0].keys() and batch[0]["camera_info"] is not None):
        collated["camera_info"] = [item["camera_info"] for item in batch]

    if("bev" in batch[0].keys() and batch[0]["bev"] is not None):
        collated["bev"] = [item["bev"] for item in batch]


    return collated

def pcd2range(
        pcd,
        size=(64, 1024),
        fov=(3, -25),
        depth_range=(1.0, 56.0),
        remission=None,
        labels=None,
        **kwargs
):
    # pcd : [N,3]

    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    range_img = np.full(size, -1, dtype=np.float32)
    range_img[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    # proj_range : [H,W]
    return range_img, proj_feature

def range2pcd(
        range_img,
        fov=(3, -25),
        depth_range=(1.0, 56.0),
        log_scale=True,
        label=None,
        color=None,
        use_mask=True,
        **kwargs
):
    if log_scale:
        depth_scale = np.log2(depth_range[1] + 1)
        depth_thresh = (np.log2(1. / 255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        depth_scale = depth_range[1]
        depth_thresh = (1. / 255. / depth_scale) * 2. - 1 + 1e-6

    # laser parameters
    H, W = 64, 1024
    if(len(range_img.shape) == 2):
        H,W = range_img.shape
    elif(len(range_img.shape) == 3):
        C,H,W = range_img.shape
    elif(len(range_img.shape) == 4):
        B,C,H,W = range_img.shape
    size = [H, W]

    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1] # W
    scan_y = scan_y.astype(np.float64) / size[0] # H

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    if(use_mask):
        pcd = pcd[mask, :]

    # label
    if label is not None:
        label = label.flatten()[mask]

    # default point color
    if color is not None:
        color = color.reshape(-1, 3)[mask, :]
    else:
        color = np.ones((pcd.shape[0], 3)) * [0.7, 0.7, 1]

    return pcd, color, label


def process_scan(
        range_img,
        log_scale=True,
        depth_range=(1.0, 56.0),
):

    if log_scale:
        depth_scale = np.log2(depth_range[1] + 1)
        depth_thresh = (np.log2(1. / 255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        depth_scale = depth_range[1]
        depth_thresh = (1. / 255. / depth_scale) * 2. - 1 + 1e-6

    # range_img : [H,W]
    range_img = np.where(range_img < 0, 0, range_img)

    if log_scale:
        # log scale
        range_img = np.log2(range_img + 0.0001 + 1)

    range_img = range_img / depth_scale
    range_img = range_img * 2. - 1.

    range_img = np.clip(range_img, -1, 1)
    range_img = np.expand_dims(range_img, axis=0)

    # mask
    range_mask = np.ones_like(range_img)
    range_mask[range_img < depth_thresh] = -1

    # range_img : [1,H,W], range_mask : [1,H,W]
    return range_img, range_mask

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if(isinstance(val, np.ndarray)):
            batch_dict[key] = torch.from_numpy(val)
        if(isinstance(val, torch.Tensor)):
            batch_dict[key] = val.cuda()

def save_points(points, colors=None, name="pc.ply"):

    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    if(colors is not None):
        pc.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(name, pc)
    print(f"----- Saving : {name} -----")

def get_ray_angles(
        size=(64,1024),
        fov=(3, -25),
        device: torch.device = "cpu"
):

    h_up, h_down = fov[0], fov[1]
    H, W = size[0], size[1]
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(H, device=device) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(W, device=device) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

def to_xyz(
        range_img,
        ray_angles,
        depth_range=(1.45, 80.0),
):
    assert len(range_img.shape) == 4

    is_ndarray = False
    if(isinstance(range_img, np.ndarray)):
        range_img = torch.from_numpy(range_img)
        is_ndarray = True

    min_depth = depth_range[0]
    max_depth = depth_range[1]

    mask = (range_img > min_depth) & (range_img < max_depth)
    phi = ray_angles[:, [0]]
    theta = ray_angles[:, [1]]
    grid_x = range_img * phi.cos() * theta.cos()
    grid_y = range_img * phi.cos() * theta.sin()
    grid_z = range_img * phi.sin()
    xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
    xyz = xyz * mask.float()
    return xyz.numpy() if is_ndarray else xyz

def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array

# def points_as_images(
#     points,
#     scan_unfolding: bool = False,
#     size=(64, 1024),
#     fov = (3, -25),
#     depth_range=(1.45,80.0),
#     return_all=False,
# ):
#     # load xyz & intensity and add depth & mask
#     # points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
#
#     # points : [N,3]
#
#     H, W = size[0], size[1]
#     min_depth, max_depth = depth_range[0], depth_range[1]
#
#     xyz = points[:, :3]  # xyz
#     x = xyz[:, [0]]
#     y = xyz[:, [1]]
#     z = xyz[:, [2]]
#     depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
#
#     mask = (depth >= min_depth) & (depth <= max_depth)
#     points = np.concatenate([points, depth, mask], axis=1)
#     dim = points.shape[-1]
#
#     if scan_unfolding:
#         # the i-th quadrant
#         # suppose the points are ordered counterclockwise
#         quads = np.zeros_like(x, dtype=np.int32)
#         quads[(x >= 0) & (y >= 0)] = 0  # 1st
#         quads[(x < 0) & (y >= 0)] = 1  # 2nd
#         quads[(x < 0) & (y < 0)] = 2  # 3rd
#         quads[(x >= 0) & (y < 0)] = 3  # 4th
#
#         # split between the 3rd and 1st quadrants
#         diff = np.roll(quads, shift=1, axis=0) - quads
#         delim_inds, _ = np.where(diff == 3)  # number of lines
#         inds = list(delim_inds) + [len(points)]  # add the last index
#
#         # vertical grid
#         grid_h = np.zeros_like(x, dtype=np.int32)
#         cur_ring_idx = H - 1  # ...0
#         for i in reversed(range(len(delim_inds))):
#             grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
#             if cur_ring_idx >= 0:
#                 cur_ring_idx -= 1
#             else:
#                 break
#     else:
#         h_up, h_down = np.deg2rad(fov[0]), np.deg2rad(fov[1])
#         elevation = np.arcsin(z / depth) + abs(h_down)
#         grid_h = 1 - elevation / (h_up - h_down)
#         grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)
#
#     # horizontal grid
#     grid_w = 1/2 * (1 - np.arctan2(y, x) / np.pi) % 1  # [0,1]
#     grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)
#
#     grid = np.concatenate((grid_h, grid_w), axis=1)
#
#     # projection
#     order = np.argsort(-depth.squeeze(1))
#
#     # 数组, 索引，值
#
#     # [H,W,C]
#     if(return_all):
#         proj_points = np.zeros((H, W, dim), dtype=points.dtype)
#         proj_points = scatter(proj_points, grid[order], points[order])
#     else:
#         proj_points = np.zeros((H, W, 1), dtype=points.dtype)
#         proj_points = scatter(proj_points, grid[order], depth[order])
#
#     return proj_points.astype(np.float32)


def points_as_images(
    points,
    size=(64, 1024),
    fov = (3, -25),
    depth_range=(1.45,80.0),
    return_all=False,
):
    # load xyz & intensity and add depth & mask
    # points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))

    # points : [N,3]

    H, W = size[0], size[1]
    min_depth, max_depth = depth_range[0], depth_range[1]

    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)

    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)
    dim = points.shape[-1]

    h_up, h_down = np.deg2rad(fov[0]), np.deg2rad(fov[1])
    elevation = np.arcsin(z / depth) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    grid_w = 1/2 * (1 - np.arctan2(y, x) / np.pi) % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))

    # 数组, 索引，值

    # [H,W,C]
    if(return_all):
        proj_points = np.zeros((H, W, dim), dtype=points.dtype)
        proj_points = scatter(proj_points, grid[order], points[order])
    else:
        proj_points = np.zeros((H, W, 1), dtype=points.dtype)
        proj_points = scatter(proj_points, grid[order], depth[order])

    return proj_points.astype(np.float32)


# def points_as_images_angle(
#     points,
#     start_a=-180,
#     end_a=180,
#     start_a2=None,
#     end_a2=None,
#     wrap=False,
#     size=(64, 1024),
#     fov=(3, -25),
#     depth_range=(1.45, 80.0),
#     return_all=False,
# ):
#     """
#     Args:
#         points: [N, 3] or [N, C]
#         start_a, end_a: horizontal angle range in degrees
#         size: (H, W)
#         fov: vertical FoV, (up, down), degrees
#         depth_range: valid depth range
#         return_all:
#             False -> return depth RM [H, W, 1]
#             True  -> return point/depth/mask RM [H, W, C+2]
#
#     Returns:
#         proj_points: [H, W, 1] or [H, W, C+2]
#     """
#
#     H, W = size
#     min_depth, max_depth = depth_range
#
#     points = np.asarray(points, dtype=np.float32)
#
#     xyz = points[:, :3]
#     x = xyz[:, [0]]
#     y = xyz[:, [1]]
#     z = xyz[:, [2]]
#
#     depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)  # [N,1]
#
#     # depth valid mask
#     mask_depth = (depth >= min_depth) & (depth <= max_depth)
#
#     # vertical projection
#     h_up, h_down = np.deg2rad(fov[0]), np.deg2rad(fov[1])
#     elevation = np.arcsin(z / np.maximum(depth, 1e-6)) + abs(h_down)
#
#     grid_h = 1.0 - elevation / (h_up - h_down)
#     grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)  # [N,1]
#
#     # horizontal angle
#     yaw = np.arctan2(y, x)  # [N,1], range [-pi, pi]
#
#     yaw_min = np.deg2rad(start_a)
#     yaw_max = np.deg2rad(end_a)
#
#     # angle valid mask
#     mask_fov = (yaw >= yaw_min) & (yaw <= yaw_max)
#
#     # final valid mask
#     valid = (mask_depth & mask_fov).squeeze()  # [N]
#
#     # filter all point-wise arrays
#     points_valid = points[valid]
#     depth_valid = depth[valid]
#     grid_h_valid = grid_h[valid]
#     yaw_valid = yaw[valid]
#
#     # append depth and valid mask
#     valid_mask = np.ones_like(depth_valid, dtype=np.float32)
#     points_valid = np.concatenate(
#         [points_valid, depth_valid, valid_mask],
#         axis=1
#     )
#
#     dim = points_valid.shape[-1]
#
#     grid_w = (yaw_max - yaw_valid) / (yaw_max - yaw_min) # 顺时针投影
#     grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)  # [Nv,1]
#
#     grid = np.concatenate((grid_h_valid, grid_w), axis=1)  # [Nv,2], (h,w)
#
#     # z-buffer: far first, near later overwrite
#     order = np.argsort(-depth_valid.squeeze(1))
#
#     if return_all:
#         proj_points = np.zeros((H, W, dim), dtype=np.float32)
#         proj_points = scatter(proj_points, grid[order], points_valid[order])
#     else:
#         proj_points = np.zeros((H, W, 1), dtype=np.float32)
#         proj_points = scatter(proj_points, grid[order], depth_valid[order])
#
#     return proj_points.astype(np.float32)


def points_as_images_angle(
    points,
    start_a=-180,
    end_a=180,
    start_a2=None,
    end_a2=None,
    wrap=False,
    size=(64, 1024),
    fov=(3, -25),
    depth_range=(1.45, 80.0),
    return_all=False,
):
    """
    Args:
        points: [N, 3] or [N, C]
        start_a, end_a: horizontal angle range in degrees
        size: (H, W)
        fov: vertical FoV, (up, down), degrees
        depth_range: valid depth range
        return_all:
            False -> return depth RM [H, W, 1]
            True  -> return point/depth/mask RM [H, W, C+2]

    Returns:
        proj_points: [H, W, 1] or [H, W, C+2]
    """

    H, W = size
    min_depth, max_depth = depth_range

    points = np.asarray(points, dtype=np.float32)

    xyz = points[:, :3]
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]

    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)  # [N,1]

    # depth valid mask
    mask_depth = (depth >= min_depth) & (depth <= max_depth)

    # vertical projection
    h_up, h_down = np.deg2rad(fov[0]), np.deg2rad(fov[1])
    elevation = np.arcsin(z / np.maximum(depth, 1e-6)) + abs(h_down)

    grid_h = 1.0 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)  # [N,1]

    # horizontal angle
    yaw = np.arctan2(y, x)  # [N,1], range [-pi, pi]

    yaw_min = np.deg2rad(start_a)
    yaw_max = np.deg2rad(end_a)

    # angle valid mask
    mask_fov = (yaw >= yaw_min) & (yaw <= yaw_max)

    if(wrap and start_a2 is not None and end_a2 is not None):
        yaw_min2 = np.deg2rad(start_a2)
        yaw_max2 = np.deg2rad(end_a2)
        mask_fov2 = (yaw >= yaw_min2) & (yaw <= yaw_max2)
        mask_fov = mask_fov | mask_fov2

    # final valid mask
    valid = (mask_depth & mask_fov).squeeze()  # [N]

    # filter all point-wise arrays
    points_valid = points[valid]
    depth_valid = depth[valid]
    grid_h_valid = grid_h[valid]
    yaw_valid = yaw[valid]

    # append depth and valid mask
    valid_mask = np.ones_like(depth_valid, dtype=np.float32)
    points_valid = np.concatenate(
        [points_valid, depth_valid, valid_mask],
        axis=1
    )

    dim = points_valid.shape[-1]

    if wrap:
        yaw_valid = yaw_valid.copy()
        yaw_valid[yaw_valid > 0] -= 2 * np.pi

        yaw_max = np.deg2rad(end_a2)  # -129.31
        yaw_min = np.deg2rad(start_a - 360.0)  # 165.75 - 360 = -194.25

    grid_w = (yaw_max - yaw_valid) / (yaw_max - yaw_min) # 顺时针投影
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)  # [Nv,1]

    grid = np.concatenate((grid_h_valid, grid_w), axis=1)  # [Nv,2], (h,w)

    # z-buffer: far first, near later overwrite
    order = np.argsort(-depth_valid.squeeze(1))

    if return_all:
        proj_points = np.zeros((H, W, dim), dtype=np.float32)
        proj_points = scatter(proj_points, grid[order], points_valid[order])
    else:
        proj_points = np.zeros((H, W, 1), dtype=np.float32)
        proj_points = scatter(proj_points, grid[order], depth_valid[order])

    return proj_points.astype(np.float32)


def range_img_normalize(range_img, depth_range):
    return  range_img / depth_range[-1]

def range_img_denormalize(range_img, depth_range):
    return  range_img * depth_range[-1]

def points_4dim_t0_2dim(points):
    B, C, H, W = points.shape
    points = points.permute(0, 2, 3, 1).reshape(B, H * W, C).cpu().numpy()[0]
    return points

def pc2range_range2pc_1():
    log_scale = True
    depth_range = (1.0, 56.0)
    size = (64, 1024)
    fov = (3, -25)

    lidar_filename = '/data/qwt/models/T2L-baseline/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000049.bin'
    points = get_lidar_sweep(lidar_filename)

    save_points(points, name="r_pc.ply")

    if log_scale:
        depth_scale = np.log2(depth_range[1] + 1)
        depth_thresh = (np.log2(1. / 255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        depth_scale = depth_range[1]
        depth_thresh = (1. / 255. / depth_scale) * 2. - 1 + 1e-6

    range_img,_ = pcd2range(
        pcd=points,
        size=size,
        fov=fov,
        depth_range=depth_range
    )
    range_img,_ = process_scan(
        range_img=range_img,
        log_scale=True,
        depth_range=depth_range
    )

    # [1,1,H,W]
    range_img = np.expand_dims(range_img, axis=0)

    points,colors,_ = range2pcd(
        range_img,
        fov=fov,
        depth_range=depth_range,
        log_scale=True,
        use_mask=True
    )

    save_points(points, name="t_pc.ply")


def pc2range_range2pc_2():
    log_scale = True
    depth_range = (1.45, 80.0)
    size = (64, 1024)
    fov = (3, -25)

    lidar_filename = '/data/qwt/models/T2L-baseline/dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000049.bin'
    points = get_lidar_sweep(lidar_filename)

    save_points(points, name="r_pc.ply")

    if log_scale:
        depth_scale = np.log2(depth_range[1] + 1)
        depth_thresh = (np.log2(1. / 255. + 1) / depth_scale) * 2. - 1 + 1e-6
    else:
        depth_scale = depth_range[1]
        depth_thresh = (1. / 255. / depth_scale) * 2. - 1 + 1e-6

    # [1,H,W]
    range_img = points_as_images(
        points,
        size=size,
        depth_range=depth_range,
    ).transpose(2, 0, 1)

    range_img = range_img_normalize(range_img=range_img, depth_range=depth_range)

    # [1,1,H,W]
    range_img = np.expand_dims(range_img, axis=0)

    range_img = range_img_denormalize(range_img=range_img, depth_range=depth_range)

    points = to_xyz(
        range_img=range_img,
        ray_angles=get_ray_angles(
            size=size,
            fov=fov,
        ),
        depth_range=depth_range,
    )

    B, C, H, W = points.shape
    points = points.transpose(0, 2, 3, 1).reshape(B, H * W, C)[0]

    save_points(points, name="t_pc.ply")

def filer_name_keys(name=None, keys=None):

    if(name is None or keys is None):
        return True

    for key in keys:
        if(name.__contains__(key)):
            return False

    return True

def set_param_grad_by_prefix(model: nn.Module, train_prefixes=None, freeze_prefixes=None, print_info=True):
    """
    train_prefixes: 以这些前缀开头的参数名将被设置为 requires_grad = True，接受为list
    freeze_prefixes: 以这些前缀开头的参数名将被设置为 requires_grad = False，接受为list
    规则：如果同时匹配 train_prefixes 和 freeze_prefixes，以 freeze 为准（安全优先）。
    """

    if(train_prefixes is not None):
        if(not isinstance(train_prefixes, list)):
            print("train_prefixes must be a list !")
            exit(0)

    if(freeze_prefixes is not None):
        if(not isinstance(freeze_prefixes, list)):
            print("freeze_prefixes must be a list !")
            exit(0)

    train_params = []
    freeze_params = []

    for name, p in model.named_parameters():

        p.requires_grad = True

        if(train_prefixes is not None):
            fige = True
            for prefix in train_prefixes:
                if(name.__contains__(prefix)):
                    fige = False
                    break
            if(fige):
                p.requires_grad = False

        elif(freeze_prefixes is not None):
            fige = False
            for prefix in freeze_prefixes:
                if(name.__contains__(prefix)):
                    fige = True
                    break
            if(fige):
                p.requires_grad = False

        if(p.requires_grad):
            train_params.append(name)
        else:
            freeze_params.append(name)

    if(print_info):
        print("Train params: ", train_params)
        print("Freeze params: ", freeze_params)

    return model, train_params, freeze_params


# def load_optimizer_state_filtered(optimizer, checkpoint_state, model):
#     """
#     加载 optimizer state_dict，只加载那些 requires_grad=True 的参数。
#     其他参数的 state 会被跳过。
#     """
#
#     old_state = checkpoint_state['state']
#     old_groups = checkpoint_state['param_groups']
#
#     # 建立一个映射：旧参数索引 -> 新参数对象
#     new_params = list(optimizer.param_groups[0]['params'])
#     mapping = {}  # old_param -> new_param
#     for new_p in new_params:
#         for old_p in old_state.keys():
#             # 按形状和 requires_grad 匹配（也可按 name）
#             if new_p.shape == old_p.shape and new_p.requires_grad:
#                 mapping[old_p] = new_p
#                 break
#
#     # 构造新 state，只保留 requires_grad=True 的参数
#     new_state = {}
#     for old_p, state_val in old_state.items():
#         if old_p in mapping and mapping[old_p].requires_grad:
#             new_state[mapping[old_p]] = state_val  # 迁移 state
#
#     # 构造新 param_groups
#     new_param_groups = []
#     for group in old_groups:
#         new_group = {k: v for k, v in group.items() if k != 'params'}
#         new_group['params'] = [p for p in new_params if p.requires_grad]
#         new_param_groups.append(new_group)
#
#     # 组装新 state_dict
#     new_optim_state = {
#         'state': new_state,
#         'param_groups': new_param_groups
#     }
#
#     optimizer.load_state_dict(new_optim_state)
#     print(f"✅ 已加载 optimizer 状态，仅保留了 {len(new_state)} 个可训练参数。")


def load_optimizer_state_filtered_safe(optimizer, loaded_state_dict):
    """
    加载优化器状态，仅为 requires_grad=True 的参数加载 state。
    保留 param_groups 结构，避免 group size mismatch。
    """
    current_state = optimizer.state_dict()

    # 获取当前需要梯度的参数 id
    grad_param_ids = {id(p) for group in optimizer.param_groups for p in group['params'] if p.requires_grad}

    # 保留 optimizer 自己的 param_groups，不修改
    new_state_dict = {
        'param_groups': current_state['param_groups'],  # 保持结构
        'state': {}
    }

    # 只加载匹配的 state
    for pid, state in loaded_state_dict['state'].items():
        if pid in grad_param_ids:
            new_state_dict['state'][pid] = state

    return new_state_dict

def print_load_report(load_info, model_name, weight_num, print_info=True):
    missing = load_info.missing_keys
    unexpected = load_info.unexpected_keys
    print(f"📦 [{model_name}] 权重加载报告：")
    print(f"   ✔️ 成功加载参数数量: {len(list(load_info.keys_loaded)) if hasattr(load_info, 'keys_loaded') else 'N/A'}")
    print(f"   ✔️ 成功加载参数数: {weight_num}")
    print(f"   ⚠️ 缺失参数数: {len(missing)}")
    # print(f"   ⚠️ 未预期参数数: {len(unexpected)}")

    if(print_info):
        if missing:
            print("\n   🔴 缺失参数 (在模型中有, 但 checkpoint 中没有):")
            for k in missing:
                print(f"      • {k}")

        if unexpected:
            print("\n   🟠 未预期参数 (在 checkpoint 中有, 但模型中无对应):")
            for k in unexpected:
                print(f"      • {k}")
    print()

def weights_num(weights):
    total_params = sum(v.numel() for v in weights.values())
    return total_params

def load_checkpoint(
        checkpoint_path,
        ema_model,
        optimizer=None,
        lr_scheduler=None,
        strict=True,
        print_info=True,
        map_location="cuda"
):
    """
    恢复训练的checkpoint加载函数

    Args:
        checkpoint_path (str): checkpoint文件路径
        model (torch.nn.Module): 主模型
        ema_model (torch.nn.Module): EMA模型
        optimizer (torch.optim.Optimizer, optional): 优化器
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
        map_location (str, optional): 加载设备

    Returns:
        cfg (dict): 保存的配置字典
        global_step (int): 当前训练步数
    """
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 恢复配置
    cfg = checkpoint["cfg"]

    # 加载模型参数
    weights = checkpoint["weights"]
    ema_weights = checkpoint["ema_weights"]
    load_info_online = ema_model.online_model.load_state_dict(weights, strict=strict)
    load_info_ema = ema_model.ema_model.load_state_dict(ema_weights, strict=strict)
    print(f"✅ EMA weights loaded : online_weights {weights_num(weights)}, ema_weights : {weights_num(ema_weights)}")

    print_load_report(load_info_online, "online_model", len(weights), print_info=print_info)
    print_load_report(load_info_ema, "ema_model", len(ema_weights), print_info=print_info)

    # 加载优化器和学习率调度器
    if optimizer is not None and "optimizer" in checkpoint:

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"optimizer参数load之前：Param group {i}:")
            for key, value in param_group.items():
                if key != "params":  # params 太长
                    print(f"  {key}: {value}")

        optimizer.load_state_dict(checkpoint["optimizer"])

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"optimizer参数load之后：Param group {i}:")
            for key, value in param_group.items():
                if key != "params":  # params 太长
                    print(f"  {key}: {value}")

        removed_cnt = 0
        kept_cnt = 0
        for group in optimizer.param_groups:
            new_params = []
            for p in group["params"]:
                if getattr(p, "requires_grad", True):
                    new_params.append(p)
                    kept_cnt += 1
                else:
                    # 删除该参数的状态，节省内存
                    if p in optimizer.state:
                        del optimizer.state[p]
                    removed_cnt += 1
            group["params"] = new_params

        print(
                f"[load_and_prune_optimizer] 加载完成：保留 {kept_cnt} 个可训练参数的历史状态；"
                f"移除 {removed_cnt} 个冻结参数及其 state。"
            )

        print("✅ Optimizer state loaded.")
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        print(f"lr_scheduler参数load之前：{lr_scheduler.state_dict()}:")
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print(f"lr_scheduler参数load之后：{lr_scheduler.state_dict()}:")
        print("✅ LR scheduler state loaded.")

    # 恢复 global_step
    global_step = checkpoint.get("global_step", 0)
    print(f"✅ Global step restored: {global_step}")

    return cfg, global_step

def load_conditional_checkpoint(checkpoint_path, conditional_model, optimizer=None, lr_scheduler=None, map_location="cuda"):
    """
    恢复训练的checkpoint加载函数

    Args:
        checkpoint_path (str): checkpoint文件路径
        model (torch.nn.Module): 主模型
        conditional_model (torch.nn.Module): conditional model模型
        optimizer (torch.optim.Optimizer, optional): 优化器
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器
        map_location (str, optional): 加载设备

    Returns:
        cfg (dict): 保存的配置字典
        global_step (int): 当前训练步数
    """
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 恢复配置
    cfg = checkpoint["cfg"]

    # 加载模型参数
    conditional_model.load_state_dict(checkpoint["weights"])
    print("✅ EMA weights loaded.")

    # 加载优化器和学习率调度器
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("✅ Optimizer state loaded.")
    if lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("✅ LR scheduler state loaded.")

    # 恢复 global_step
    global_step = checkpoint.get("global_step", 0)
    print(f"✅ Global step restored: {global_step}")

    return cfg, global_step


def bin_to_ply(lidar_path):
    points = get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)
    points = points[:, :3]
    name = lidar_path.split("/")[-1].split(".")[0]
    name =f"{name}.ply"
    save_points(points=points,name=name)

def get_lidar_names():
    names = [
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-13-41-50-0400__LIDAR_TOP__1537292788698591.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-07-27-11-24-31+0800__LIDAR_TOP__1532662225898263.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-14-54-39-0400__LIDAR_TOP__1537297623899491.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-09-25-13-17-43+0800__LIDAR_TOP__1537852761697118.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-06-15-06-32-0400__LIDAR_TOP__1533583330697082.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-10-08-16-03-24+0800__LIDAR_TOP__1538986135897755.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385264199335.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-06-15-06-32-0400__LIDAR_TOP__1533583206147176.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-31-11-19-57-0400__LIDAR_TOP__1535728864947467.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-08-01-15-10-21+0800__LIDAR_TOP__1533107762946873.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-08-21-11-53-44-0400__LIDAR_TOP__1534867307949278.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-09-25-11-10-38+0800__LIDAR_TOP__1537845402697878.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-08-02-17-28-51+0800__LIDAR_TOP__1533202213700346.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-13-10-39-0400__LIDAR_TOP__1537290675798905.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-15-26-58-0400__LIDAR_TOP__1537299502450667.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281449949036.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n008-2018-09-18-15-26-58-0400__LIDAR_TOP__1537298911950371.pcd.bin",
        "/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-09-25-11-10-38+0800__LIDAR_TOP__1537845632197967.pcd.bin",
    ]
    return names

def get_conditional_x_0(lidar_utils, batch_size):

    names = get_lidar_names()
    names = names[:batch_size]
    x = []
    for name in names:
        points = get_lidar_sweep(name, return_intensity=True, return_time=True, dim=5)
        range_image = points_as_images(
            points,
            size = lidar_utils.resolution,
            fov = lidar_utils.fov,
            depth_range = [lidar_utils.min_depth, lidar_utils.max_depth],
            return_all=True,
        ).transpose(2, 0, 1)
        x.append(range_image)

def setup_seed(seed: int):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # 为每个rank生成不同的seed
    final_seed = seed + rank * 1000

    # Python随机数
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)

    # cuDNN设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # 若追求可复现，可设True，但会略慢

    # if rank == 0:
    #     print(f"[Seed Setup] base_seed={seed}, rank={rank}, final_seed={final_seed}")

    return final_seed


def copy_files(source, dest, print_info=True):

    plys = [ply.split("/")[-1] for ply in glob.glob(os.path.join(source, "*.ply"))]
    plys = sorted(plys)
    for i, ply in enumerate(plys):
        source_path = os.path.join(source, ply)
        dest_path = os.path.join(dest, ply)
        shutil.copy(source_path, dest_path)
        if(print_info): print(f"---- {i}/{len(plys)} Copying : {dest_path} ----")

def load_pcd_ascii(path):
    with open(path, 'r') as f:
        # 读到 DATA 行
        line = f.readline().strip()
        header = [line]
        while not line.startswith('DATA'):
            line = f.readline().strip()
            header.append(line)
        data = np.loadtxt(f)  # 之后就是数值

    return data

def cat_descriptor(info, keys):

    if(keys is None):
        return info["text_aim"]

    text_l0 = info["text_l0"]
    keys = keys.split(" ")
    if(text_l0.__contains__("No")):
        return info[keys[0]]

    cat_text = ""
    for i, key in enumerate(keys):
        text = info[key]
        if(i>0):
            text = " " + text
        cat_text += text

    return cat_text

def get_sample_by_text(infos, text_keys="Less than,More than", num=50, name="text_l0"):

    keys = text_keys.split(",")
    new_infos = []
    for info in infos:
        text = info[name]
        for keyword in keys:
            if keyword in text:
                new_infos.append(info)

                if(len(new_infos) >= num):
                    return new_infos
    return new_infos

def read_pkl(file_path):

    with open(file=file_path, mode="rb") as f:
        data = pickle.load(f)

    return data

def encode_strings(str_list, max_len=64):
    # 把字符串编码为ASCII序列并padding
    encoded = []
    for s in str_list:
        arr = [ord(c) for c in s[:max_len]]
        arr += [0] * (max_len - len(arr))
        encoded.append(arr)
    return torch.tensor(encoded, dtype=torch.long)

def decode_tensor(tensor):
    return [''.join(chr(i) for i in row if i > 0) for row in tensor.tolist()]

def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # 1，GT，sample
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance

def midpoint_interpolate(sparse_pts, up_rate=4, normal=False, only_FPS=False):
    # sparse_pts: (b, 3, 256)
    sparse_pts = sparse_pts.float().contiguous()



    if (normal):
        sparse_pts, centroid, furthest_distance = normalize_point_cloud(sparse_pts)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)

    if(only_FPS):
        return FPS(sparse_pts, up_pts_num)

    if(up_rate >= 1):
        k = int(2 * up_rate)
        # (b, 3, n, k)
        knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
        # (b, 3, n, k)
        repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
        # (b, 3, n, k)
        mid_pts = (knn_pts + repeat_pts) / 2.0
        # (b, 3, (n k))
        mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
        # note that interpolated_pts already contain sparse_pts
        interpolated_pts = mid_pts
        # fps: (b, 3, up_pts_num)
        interpolated_pts = FPS(interpolated_pts, up_pts_num)
    elif(up_rate < 1):
        interpolated_pts = FPS(sparse_pts, up_pts_num)


    if (normal):
        interpolated_pts = centroid + interpolated_pts * furthest_distance

    return interpolated_pts

def points_as_images_torch(
    points: torch.Tensor,
    size=(32, 1024),
    fov=(3.0, -25.0),
    depth_range=(0.01, 50.0),
    return_all=False,
):
    """
    Project LiDAR points to range image using pure PyTorch (GPU-compatible).

    Args:
        points: Tensor [N,3] or [N,C] (xyz [+ features])
        scan_unfolding: unused placeholder (keep for compatibility)
        size: (H, W)
        fov: (up, down) in degrees
        depth_range: (min_depth, max_depth)
        return_all: if True, return all channels else only depth

    Returns:
        proj_points: [H, W, C] or [H, W, 1]
    """

    assert points.ndim == 2, f"Expected [N,C], got {points.shape}"
    device = points.device
    dtype = points.dtype

    H, W = size
    min_depth, max_depth = depth_range
    h_up, h_down = torch.deg2rad(torch.tensor(fov[0], device=device)), torch.deg2rad(torch.tensor(fov[1], device=device))

    # --- xyz and depth ---
    xyz = points[:, :3]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    depth = torch.linalg.norm(xyz, dim=1, keepdim=True)

    # --- mask ---
    mask = (depth >= min_depth) & (depth <= max_depth)
    mask = mask.float()

    # concat
    points_aug = torch.cat([points, depth, mask], dim=1)
    dim = points_aug.shape[-1]

    # --- compute elevation (vertical) ---
    elevation = torch.arcsin(z / depth.squeeze(1)) + abs(h_down)
    grid_h = 1.0 - elevation / (h_up - h_down)
    grid_h = torch.clamp((grid_h * H).floor(), 0, H - 1).long().unsqueeze(1)  # [N,1]

    # --- compute azimuth (horizontal) ---
    grid_w = 0.5 * (1 - torch.atan2(y, x) / torch.pi)
    grid_w = torch.remainder(grid_w, 1.0)  # ensure [0,1)
    grid_w = torch.clamp((grid_w * W).floor(), 0, W - 1).long().unsqueeze(1)  # [N,1]

    grid = torch.cat((grid_h, grid_w), dim=1)  # [N,2]
    order = torch.argsort(-depth.squeeze(1))  # sort by descending depth

    # --- scatter operation ---
    proj_shape = (H, W, dim if return_all else 1)
    proj_points = torch.zeros(proj_shape, dtype=dtype, device=device)

    grid_sorted = grid[order]
    if return_all:
        vals = points_aug[order]
    else:
        vals = depth[order]

    # 将2D索引展平后scatter
    flat_index = grid_sorted[:, 0] * W + grid_sorted[:, 1]
    proj_points_flat = proj_points.view(-1, proj_points.shape[-1])
    proj_points_flat.index_copy_(0, flat_index, vals)
    proj_points = proj_points_flat.view(proj_shape)

    return proj_points


def normalize(x):
    """Scale from [0, 1] to [-1, +1]"""
    return x * 2 - 1

def get_mask(metric, min_depth=0.01, max_depth=50.0):
    mask = (metric > min_depth) & (metric < max_depth)
    return mask.float()

def convert_depth(
    metric: torch.Tensor,
    mask: torch.Tensor | None = None,
    image_format="log_depth",
    min_depth=0.01,
    max_depth=50.0
) -> torch.Tensor:
    """
    Convert metric depth in [0, `max_depth`] to normalized depth in [0, 1].
    """

    if mask is None:
        mask = get_mask(metric)
    if image_format == "log_depth":
        normalized = torch.log2(metric + 1 + 0.0001) / np.log2(max_depth + 1 + 0.0001)
    elif image_format == "inverse_depth":
        normalized = min_depth / metric.add(1e-8)
    elif image_format == "depth":
        normalized = metric.div(max_depth)
    else:
        raise ValueError
    normalized = normalized.clamp(0, 1) * mask
    return normalized

def revert_depth(
    normalized: torch.Tensor,
    image_format="log_depth",
    max_depth=50.0,
    min_depth=0.01,
) -> torch.Tensor:
    """
    Revert normalized depth in [0, 1] back to metric depth in [0, `max_depth`].
    """

    if image_format == "log_depth":
        metric = torch.exp2(normalized * np.log2(max_depth + 1 + 0.0001)) - 1 - 0.0001
    elif image_format == "inverse_depth":
        metric = min_depth / normalized.add(1e-8)
    elif image_format == "depth":
        metric = normalized.mul(max_depth)
    else:
        raise ValueError
    return metric * get_mask(metric)

def denormalize(x):
    """Scale from [-1, +1] to [0, 1]"""
    return (x + 1) / 2

def read_ply(
    file_path
):
    pc = open3d.io.read_point_cloud(file_path)
    points = np.asarray(pc.points)

    return points

def test_mid(
        target_path = "",
        save_path = "",
        rate=4
):
    target = read_ply(target_path)
    target = torch.from_numpy(target).unsqueeze(0).permute(0,2,1).cuda()

    points = midpoint_interpolate(target, up_rate=rate)

    points = points.cpu().permute(0,2,1).squeeze()
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)

    open3d.io.write_point_cloud(filename=save_path, pointcloud=pc)
    print(f"Saved : {save_path}")

def upsampling_single_point_cloud():
    root_path = "/ihoment/youjie10/qwt/model/T2LDM-plus/examples/example_files/xyz_process_0_rank_22_dataset_nuScenes_step_750000_sample_1024_batch_5.ply"
    points = read_ply(root_path)
    points = torch.from_numpy(points).float().cuda()
    sampling_points = points.unsqueeze(0).permute(0, 2, 1)  # [1,C,N]
    sampling_points = midpoint_interpolate(sampling_points, up_rate=0.25)  # [1,C,N/2]
    sampling_points = sampling_points.permute(0, 2, 1).squeeze()  # [1,N/2,C] -> [N/2,C]
    sampling_points = sampling_points.cpu().numpy()

    dest_path = "/ihoment/youjie10/qwt/model/T2LDM-plus/examples/example_files/xyz_process_0_rank_22_dataset_nuScenes_step_750000_sample_1024_batch_5.025.ply"
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(sampling_points)
    open3d.io.write_point_cloud(dest_path, pc)

if __name__ == '__main__':

    # root_path = "/ihoment/youjie10/qwt/model/T2LDM-final/test_semantickitti_0330"
    # folders = [f"{floder}/plys/generation" for floder in sorted(glob.glob(f"{root_path}/*"))]
    # print(folders)
    # dest_path = "/ihoment/youjie10/qwt/model/results/SemanticKITTI"
    # copy_folders_to_folder(folders, dest_path)

    # downsampling_LIDAR()
    # copy_folders_to_folder_unconditional(
    #     src="/ihoment/youjie10/qwt/model/T2LDM-plus/test_SemanticKITTI10240_0502",
    #     dst="/ihoment/youjie10/qwt/res",
    # )

    # folder_files()

    # copy_folders_to_folder_unconditional()
    # copy_folder_to_folder_semantic(
    #     src="/ihoment/youjie10/qwt/model/T2LDM-plus/test_densetosparse05_0525",
    #     dst="/ihoment/youjie10/qwt/dense_to_sparse_0.125",
    #     generation=True,
    #     xyz=True,
    #     dense=True
    # )


    copy_folder_to_folder_semantic(
        src="/root/models/T2LDM-plus_0521/test_semantic_0626",
        dst="/root/res",
        generation=True,
        semantic=True,
        xyz=True
    )



    pass






