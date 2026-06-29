from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np
import pickle
from data.kitti_semantic.descriptor import read_pkl

SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class KITTISemanticDataset(Dataset):

    def __init__(
            self,
            data_root="/root/dataset/SemanticKITTI/sequences",
            training=True,
            aug=["rotation", "flip"],

            resolution=(64, 1024),
            depth_range=[1.45, 80.0],
            fov=[3, -25],

            semantic_class_num=20.0, # 共有19个类别+1个忽略类别
            pkl=None,

            print_info=True
    ):
        super().__init__()
        self.data_root = data_root
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
        self.lidar_info = []

        self.learning_map = common.get_semantickitti_learning_map(semantic_class_num)

        pkl_path = f"{data_root}/{pkl}"
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)

        for info in infos:
            lidar_path = f"{data_root}/{info['lidar_path']}"
            self.lidar_path.append(lidar_path)

            semantic = f"{data_root}/{info['semantic']}"
            self.lidar_semantic.append(semantic)

            self.lidar_description.append(info["text"])

            self.lidar_info.append(None)


        if(print_info):
            print(f" ---- SemanticKITTI Dataset with {len(self.lidar_path)} ---- ")


        self.conditionalx0_lidar_path = []
        self.conditionalx0_lidar_description = []
        self.conditionalx0_lidar_semantic = []
        self.conditionalx0_lidar_info = []

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
            self.conditionalx0_lidar_info.append(self.lidar_info[l])


    def __len__(self):
        return len(self.lidar_path)

    def __getitem__(self, idx):
        lidar_path = self.lidar_path[idx]
        lidar_description = self.lidar_description[idx]
        lidar_semantic = self.lidar_semantic[idx]

        points = common.get_lidar_sweep(lidar_path, return_intensity=True)
        if self.transform:
            points[:,:3],_ = self.transform(points[:,:3])

        with open(lidar_semantic, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(self.learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        semantic = np.expand_dims(semantic, axis=1)
        points = np.concatenate([points, semantic], axis=-1)

        range_image = common.points_as_images(
            points,
            size = self.resolution,
            fov = self.fov,
            depth_range = self.depth_range,
            return_all=True,
        ).transpose(2, 0, 1)

        sample = {
            "points": points[:, :3],  # (N,3)
            "batch": [len(points), ],
            "xyz": range_image[:3],                                      # (3  H, W)
            "reflectance": range_image[[3]],                             # (1, H, W)
            "semantic": range_image[[4]] / self.semantic_class_num,      # (1, H, W)
            "depth": range_image[[5]],                                   # (1, H, W)
            "mask": range_image[[6]],                                    # (1, H, W)
            "text": lidar_description,
            "semantic_org": range_image[[4]],
        }

        return sample

if __name__ == '__main__':

    dataset = KITTISemanticDataset(
        pkl="semantic_kitti_description_.pkl"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        collate_fn=common.collate_fn
    )

    for batch in dataloader:
        common.load_data_to_gpu(batch)
        xyz = batch["xyz"]
        reflectance = batch["reflectance"]
        semantic = batch["semantic"]
        depth = batch["depth"]
        mask = batch["mask"]
        text = batch["text"]

        print(f"xyz : {xyz.shape}, max : {xyz.max()}, min : {xyz.min()}")
        print(f"reflectance : {reflectance.shape}, max : {reflectance.max()}, min : {reflectance.min()}")
        print(f"semantic : {semantic.shape}, max : {semantic.max()}, min : {semantic.min()}")
        print(f"depth : {depth.shape}, max : {depth.max()}, min : {depth.min()}")
        print(f"mask : {mask.shape}")
        print(f"text : {text}")

        break

    pass