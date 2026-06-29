"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""
import torch
import glob
import multiprocessing
import os.path
from functools import partial
import random
from eval.metric_utils import compute_logits, compute_pairwise_cd, \
    compute_pairwise_emd, pcd2bev_sum, compute_pairwise_cd_batch, pcd2bev_bin
from eval.fid_score import calculate_frechet_distance

from utils import common
import pickle
import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from pathlib import Path
from eval import OUTPUT_TEMPLATE
import open3d


KITTI360_SCENES = [0, 2, 3, 4, 5, 6, 7, 9, 10]
SEMANTICKITTI_SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

KITTT_360_PATH = "/root/dataset/KITTI360"
SEMANTICKITTI_PATH = "/root/dataset/SemanticKITTI/sequences"
NUSCENES_PATH = "/root/dataset/nuScenes"

os.chdir(os.path.dirname(__file__))


def compute_cd(reference, samples):
    """
    Calculate score of Chamfer Distance (CD)

    """
    print('Evaluating (CD) ...')
    results = []
    for x, y in zip(reference, samples):
        d = compute_pairwise_cd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('CD  ', score))

    return score


def compute_emd(reference, samples):
    """
    Calculate score of Earth Mover's Distance (EMD)

    """
    print('Evaluating (EMD) ...')
    results = []
    for x, y in zip(reference, samples):
        d = compute_pairwise_emd(x, y)
        results.append(d)
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('EMD ', score))

    return score


def compute_mmd(reference, samples, dataset, dist='cd', verbose=True):
    """
    Calculate the score of Minimum Matching Distance (MMD)

    """
    print('Evaluating (MMD) ...')
    assert dist in ['cd', 'emd']
    reference, samples = pcd2bev_bin(dataset, reference, samples)
    compute_dist_func = compute_pairwise_cd_batch if dist == 'cd' else compute_pairwise_emd
    results = []
    for r in tqdm(reference, disable=not verbose):
        dists = compute_dist_func(r, samples)
        results.append(min(dists))
    score = sum(results) / len(results)
    print(OUTPUT_TEMPLATE.format('MMD ', score))

    return score


def compute_jsd(reference, samples, dataset):
    """
    Calculate the score of Jensen-Shannon Divergence (JSD)

    """
    print('Evaluating (JSD) ...')
    reference, samples = pcd2bev_sum(dataset, reference, samples)
    reference = (reference / np.sum(reference)).flatten()
    samples = (samples / np.sum(samples)).flatten()
    score = jensenshannon(reference, samples)
    print(OUTPUT_TEMPLATE.format('JSD ', score))

    return score


def compute_fd(reference, samples):
    mu1, mu2 = np.mean(reference, axis=0), np.mean(samples, axis=0)
    sigma1, sigma2 = np.cov(reference, rowvar=False), np.cov(samples, rowvar=False)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return distance


def compute_frid(reference, samples, dataset, results_path):
    """
    Calculate the score of Fréchet Range Image Distance (FRID)

    """
    print('Evaluating (FRID) ...')
    gt_logits, samples_logits = compute_logits(dataset=dataset, modality='range', reference=reference, sample=samples,
                                               results_path=results_path)
    score = compute_fd(gt_logits, samples_logits)  # 计算两个特征的均值和方差
    print(OUTPUT_TEMPLATE.format('FRID', score))

    return score


def compute_fsvd(reference, samples, dataset, results_path):
    """
    Calculate the score of Fréchet Sparse Volume Distance (FSVD)

    """
    print('Evaluating (FSVD) ...')
    gt_logits, samples_logits = compute_logits(dataset=dataset, modality='voxel', reference=reference, sample=samples,
                                               results_path=results_path)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FSVD', score))

    return score


def compute_fpvd(reference, samples, dataset, results_path):
    """
    Calculate the score of Fréchet Point-based Volume Distance (FPVD)

    """
    print('Evaluating (FPVD) ...')
    gt_logits, samples_logits = compute_logits(dataset=dataset, modality='point_voxel', reference=reference,
                                               sample=samples, results_path=results_path)
    score = compute_fd(gt_logits, samples_logits)
    print(OUTPUT_TEMPLATE.format('FPVD', score))

    return score


def evaluate(
        reference,
        samples,
        metrics,
        dataset,
        results_path="results",
        mmd_kitti=None
):
    results = {}

    # perceptual
    if 'frid' in metrics and (dataset == "kitti360" or dataset == "semantickitti"):
        frid = compute_frid(reference, samples, dataset, results_path)
        results["FRID"] = frid

    if 'fsvd' in metrics:
        fsvd = compute_fsvd(reference, samples, dataset, results_path)
        results["FSVD"] = fsvd

    if 'fpvd' in metrics:
        fpvd = compute_fpvd(reference, samples, dataset, results_path)
        results["FPVD"] = fpvd

    # reconstruction
    if 'cd' in metrics:
        cd = compute_cd(reference, samples)
        results["CD"] = cd

    if 'emd' in metrics:
        emd = compute_emd(reference, samples)
        results["EMD"] = emd

    # statistical
    if 'jsd' in metrics:
        jsd = compute_jsd(mmd_kitti, samples, dataset)
        results["JSD"] = jsd

    if 'mmd' in metrics:
        mmd = compute_mmd(mmd_kitti, samples, dataset)
        results["MMD"] = mmd

    return results


def normalize_point_cloud(pc, method='sphere'):
    """
    Normalize point cloud coordinates.

    Args:
        pc (ndarray or Tensor): [N, 3] 点云坐标 (x, y, z)
        method (str): 'sphere' 将点云归一化到单位球;
                      'cube' 将点云归一化到[-1, 1]立方体

    Returns:
        pc_normalized (same type): 归一化后的点云
        centroid (ndarray): 原始质心
        scale (float): 缩放系数（半径或最大边长）
    """
    if isinstance(pc, torch.Tensor):
        is_tensor = True
        pc_np = pc.detach().cpu().numpy()
    else:
        is_tensor = False
        pc_np = pc

    # 1️⃣ 去中心化
    centroid = np.mean(pc_np, axis=0)
    pc_np = pc_np - centroid

    # 2️⃣ 缩放到单位球 or 立方体
    if method == 'sphere':
        scale = np.max(np.linalg.norm(pc_np, axis=1))
    elif method == 'cube':
        scale = np.max(np.abs(pc_np))
    else:
        raise ValueError("method must be 'sphere' or 'cube'")

    pc_np = pc_np / (scale + 1e-6)

    if is_tensor:
        return torch.from_numpy(pc_np).to(pc.device).type(pc.dtype), torch.from_numpy(centroid).to(pc.device), scale
    else:
        return pc_np, centroid, scale


def get_generated_points(
        root_path=None
):
    pointss = []
    plys = glob.glob(os.path.join(f"{root_path}/*.ply"))

    progress_bar = tqdm(
        range(len(plys)),
        dynamic_ncols=True,
    )

    for ply in plys:
        points = common.read_ply(ply)
        # points = normalize_point_cloud(points)
        pointss.append(points)
        progress_bar.update(1)

    print(f"---- Sample : {len(pointss)} ----")
    return pointss


def get_nuScenes_points(
        data_root=NUSCENES_PATH,
        version='v1.0-trainval',
        pkl="nuscenes.pkl",
):

    pkl_path = f"{data_root}/{version}/{pkl}"
    print(pkl_path)
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    progress_bar = tqdm(
        range(len(infos)),
        dynamic_ncols=True,
    )

    pointss = []
    for info in infos:
        if (not info.keys().__contains__("lidar_path")):
            continue
        lidar_path = info["lidar_path"]
        lidar_path = f"{data_root}/{version}/{lidar_path}"
        points = common.get_lidar_sweep(lidar_path, return_intensity=False, return_time=False, dim=5)
        # points = normalize_point_cloud(points)
        pointss.append(points)
        progress_bar.update(1)
    print(f"---- Reference : {len(pointss)} ----")

    return pointss


def get_kitti360_points(
        data_root=KITTT_360_PATH,
):
    file_paths = []
    for scene in KITTI360_SCENES:
        wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
        file_paths += sorted(Path(data_root).glob(wildcard))
    file_paths = file_paths
    progress_bar = tqdm(
        range(len(file_paths)),
        dynamic_ncols=True,
    )

    pointss = []
    for file_path in file_paths:
        points = common.get_lidar_sweep(file_path, return_intensity=False, dim=4)
        pointss.append(points)
        progress_bar.update(1)
    print(f"---- Reference : {len(pointss)} ----")

    return pointss


def get_semantickitti_points(
        data_root=SEMANTICKITTI_PATH,
):
    file_paths = []
    for scene in SEMANTICKITTI_SCENES:
        wildcard = f"{scene:02d}/velodyne/*.bin"
        file_paths += sorted(Path(data_root).glob(wildcard))
    file_paths = file_paths
    progress_bar = tqdm(
        range(len(file_paths)),
        dynamic_ncols=True,
    )

    pointss = []
    for file_path in file_paths:
        points = common.get_lidar_sweep(file_path, return_intensity=False, dim=4)
        pointss.append(points)
        progress_bar.update(1)
    print(f"---- Reference : {len(pointss)} ----")

    return pointss


def eval_results(
        results_path="results",
        sample_folder=None,
        dataset='nuscenes',  # nuscenes kitti360
        metrics=['frid', 'fsvd', 'fpvd'],  # specify metrics to evaluate, ['mmd', 'jsd', 'frid', 'fsvd', 'fpvd']
        mmd_kitti_path=None,
        use_reference=False
):
    reference = []
    mmd_kitti = []  # GT的点云
    if use_reference:
        if (dataset == "nuscenes"):
            reference = get_nuScenes_points()  # GT的点云
        elif (dataset == "kitti360"):
            reference = get_kitti360_points()
        elif (dataset == "semantickitti"):
            reference = get_semantickitti_points()

            # if (os.path.exists(mmd_kitti_path)):
            #     index = common.read_pkl(mmd_kitti_path)
            # else:
            #     index = random.sample(range(len(reference)), 2000)
            #     common.save_pkl(save_path=mmd_kitti_path, infos=index)
            #
            # mmd_kitti = [reference[i] for i in index]
    samples = get_generated_points(sample_folder)  # 生成的点云

    results = evaluate(
        reference=reference,
        mmd_kitti=mmd_kitti,
        samples=samples,
        metrics=metrics,
        dataset=dataset,
        results_path=results_path
    )

    return results


def single_result(
        results_path="./results",
        sample_folder="",
        dataset='nuscenes',  # nuscenes kitti360
        use_reference=True,
        metrics=['fsvd', 'fpvd']  # ['mmd', 'jsd', 'frid', 'fsvd', 'fpvd']
):
    os.makedirs(results_path, exist_ok=True)  # ✅ 保证文件夹存在

    import datetime
    project_name = datetime.datetime.now().strftime("%Y%m]%dT%H%M%S")
    log_file = f"{results_path}/{project_name}_log.txt"

    with open(file=log_file, mode="w") as f:
        results = eval_results(
            results_path=results_path,
            sample_folder=sample_folder,
            dataset=dataset,
            metrics=metrics,
            use_reference=use_reference
        )
        f.write(f"{sample_folder}:\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

        f.write("\n\n")
        f.flush()


if __name__ == '__main__':

    '''
        For 'mmd', 'jsd', please select 2000 generated samples and 2000 real samples for the evaluation.
        Please finish the data root path:
            KITTT_360_PATH = "/root/dataset/KITTI360"
            SEMANTICKITTI_PATH = "/root/dataset/SemanticKITTI/sequences"
            NUSCENES_PATH = "/root/dataset/nuScenes"

    '''
    use_reference = False
    sample_folder = "/root/res/generation"
    dataset = "nuscenes"# "semantickitti" # kitti360 nuscenes
    single_result(
        sample_folder=sample_folder,
        dataset=dataset,
        use_reference=use_reference
    )



