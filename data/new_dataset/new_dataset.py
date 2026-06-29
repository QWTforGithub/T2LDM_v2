from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np
import torch

class NewDataset(Dataset):
    def __init__(
            self,
            data_root='/root/dataset/rsd_data/nuscenes',
            pkl='nuscenes_infos_10sweeps_description.pkl',
            version='v1.0-trainval',
            training=True,
            aug=["rotation", "flip"],

            resolution=(32,1024),
            depth_range=[0.01,50.0],
            fov=[3,-25],

            only_class=-1,
            text_keys="text_aim",
            semantic_class_num=17.0, # 共有16类别+1忽略类别

            print_info = True
    ):

        self.data_root = data_root
        self.pkl = pkl
        self.training = training
        self.aug = aug

        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov

        self.semantic_class_num = semantic_class_num - 1

        self.transform = common.get_lidar_transform(self.aug, self.training)

        self.lidar_path = []
        self.lidar_description = []
        self.lidar_semantic = []

        pkl_path = f"{data_root}/{version}/{pkl}"
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)

        # ---- Reading LiDAR Information ----
        for info in infos:

            if(not info.keys().__contains__("lidar_path")):
                continue

            lidar_path = info["lidar_path"]
            self.lidar_path.append(f"{data_root}/{version}/{lidar_path}")

            '''
                if there is not any description for lidar:
                    self.lidar_description.append("")
            '''
            description = info[text_keys]
            self.lidar_description.append(description)

            '''
                if there is not any semantic for lidar:
                    self.lidar_semantic.append(None)
            '''
            semantic = info["semantic"]
            if(only_class >= 0):
                semantic[...] = (semantic == only_class)
            self.lidar_semantic.append(semantic)
        # ---- Reading LiDAR Information ----

        if(print_info):
            print(f" ---- New Dataset with {len(self.lidar_path)} ---- ")

        # ---- ConditionalX0 Dataset for DDPM Validation -----
        self.conditionalx0_lidar_path = []
        self.conditionalx0_lidar_description = []
        self.conditionalx0_lidar_semantic = []

        lists = [
            8145, 9136, 10245, 11234, 13478,
            15423, 16789, 17789, 18788, 19745,
            0, 174, 356, 1024, 4132,
            5124, 6154, 6657, 7145, 7542,
        ]

        for l in lists:
            self.conditionalx0_lidar_path.append(self.lidar_path[l])
            self.conditionalx0_lidar_description.append(self.lidar_description[l])
            self.conditionalx0_lidar_semantic.append(self.lidar_semantic[l])
        # ---- ConditionalX0 Dataset for DDPM Validation -----

        return

    def __len__(self):
        return len(self.lidar_path)

    def __getitem__(self, idx):

        # ---- Reading LiDAR Information ----
        lidar_path = self.lidar_path[idx]
        lidar_description = self.lidar_description[idx]
        lidar_semantic = self.lidar_semantic[idx]

        points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

        semantic = np.expand_dims(lidar_semantic, axis=1)
        points = np.concatenate([points, semantic], axis=-1)
        # ---- Reading LiDAR Information ----

        if self.transform:
            points[:,:3],_ = self.transform(points[:,:3])

        # ---- LiDAR -> Range Map ----
        range_image = common.points_as_images(
            points,
            size = self.resolution,
            fov = self.fov,
            depth_range = self.depth_range,
            return_all=True,
        ).transpose(2, 0, 1)
        # ---- LiDAR -> Range Map ----

        sample = {
            "id": lidar_path,
            "batch": [len(points),],
            "points": points[:, :3],                                        # (N,3)
            "xyz": range_image[:3],                                         # (3 H, W)
            "reflectance": common.reflectance_norm(range_image[[3]]),       # (1, H, W)
            "time": range_image[[4]],                                       # (1, H, W)
            "semantic": range_image[[5]] / self.semantic_class_num,         # (1, H, W)
            "depth": range_image[[6]],                                      # (1, H, W)
            "mask": range_image[[7]],                                       # (1, H, W)
            "text": lidar_description,                                      # String
            "semantic_org": self.lidar_semantic[idx],
        }

        return sample


if __name__ == '__main__':
    data_root = "/root/dataset/rsd_data/nuscenes"
    version = "v1.0-trainval"
    pkl = "nuscenes_infos_10sweeps_description.pkl"
    dataset = NuScenesDataset(
        data_root=data_root,
        version=version,
        pkl=pkl,
        text_keys=("text_aim")
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )

    for batch in dataloader:
        common.load_data_to_gpu(batch)
        print(batch)

    pass