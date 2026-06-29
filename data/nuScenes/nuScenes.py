from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset,DataLoader
from utils import common
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import shutil

class NuScenesDataset(Dataset):
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

            text_keys="text",
            semantic_class_num=17.0, # 共有16类别+1忽略类别

            sampling=False,
            sample_type="all", # "train", "val"

            use_3dbox=False,

            use_camera=False,
            resize_scale=0.4,

            use_bev=False,
            bev_size=(256,256),

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

        self.sampling = sampling

        self.use_3dbox = use_3dbox
        self.use_camera = use_camera
        self.resize_scale = resize_scale
        self.use_bev = use_bev
        self.bev_size = bev_size

        self.lidar_path = []
        self.lidar_description = []
        self.lidar_semantic = []
        self.lidar_info = []

        pkl_path = f"{data_root}/{version}/{pkl}"
        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)

        if(sample_type == "train"):
            infos = infos[:28130]
        elif(sample_type == "val"):
            infos= infos[28130:]

        for info in infos:

            if(not info.keys().__contains__("lidar_path")):
                continue

            lidar_path = info["lidar_path"]
            self.lidar_path.append(f"{data_root}/{version}/{lidar_path}")

            description = info[text_keys]
            self.lidar_description.append(description)

            if(self.use_3dbox):
                box = info["gt_boxes"]
                name = info["gt_names"]
                self.lidar_semantic.append([box,name])
                self.lidar_info.append(None)
            elif(self.use_camera):
                camera = info["camera"]
                self.lidar_semantic.append(f"{data_root}/{version}/{camera}")
                camere_info = info["camera_info"]
                self.lidar_info.append(camere_info)
            else:
                semantic = info["semantic"]
                self.lidar_semantic.append(f"{data_root}/{version}/{semantic}")
                self.lidar_info.append(None)

        if(print_info):
            print(f" ---- nuScenes Dataset with {len(self.lidar_path)} ---- ")

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

        lists = [
            0, 1, 4, 3, 4,
            15423, 16789, 17789, 18788, 19745,
            0, 174, 356, 1024, 4132,
            5124, 6154, 6657, 7145, 7542,
        ]

        if(sample_type != "val"):
            for l in lists:
                self.conditionalx0_lidar_path.append(self.lidar_path[l])
                self.conditionalx0_lidar_description.append(self.lidar_description[l])
                self.conditionalx0_lidar_semantic.append(self.lidar_semantic[l])
                self.conditionalx0_lidar_info.append(self.lidar_info[l])

        return

    def __len__(self):
        return len(self.lidar_path)

    def __getitem__(self, idx):
        lidar_path = self.lidar_path[idx]
        lidar_description = self.lidar_description[idx]
        lidar_semantic = self.lidar_semantic[idx]
        lidar_info = self.lidar_info[idx]

        points = common.get_lidar_sweep(lidar_path, return_intensity=True, return_time=True, dim=5)

        start_a = -180
        end_a = 180
        camera_info = None
        wrap = False
        if(self.use_3dbox):
            gt_class = lidar_semantic[0][:, -1]
            corners = common.boxes_to_corners_3d(lidar_semantic[0])
            semantic = common.get_semantic_from_corners(points[:, :3],corners,gt_class,background_id=self.semantic_class_num)
        elif(self.use_camera):
            semantic = plt.imread(lidar_semantic)  # [H, W, C] [900, 1600, 3]
            H,W,C = semantic.shape
            # semantic = cv2.resize(
            #     semantic,
            #     (self.resolution[1], self.resolution[0]),
            #     interpolation=cv2.INTER_LINEAR
            # )

            semantic = cv2.resize(
                semantic,
                (int(W * self.resize_scale), int(H * self.resize_scale)),
                interpolation=cv2.INTER_LINEAR
            )
            semantic = semantic.transpose((2, 0, 1))
            semantic = common.normalize_image(semantic)
            start_a = lidar_info["yaw_right"]
            end_a = lidar_info["yaw_left"]
            wrap = lidar_info["wrap"]

            camera_info = {}
            if(wrap):
                camera_info["start_a"] = start_a - 360
            else:
                camera_info["start_a"] = start_a
            camera_info["end_a"] = end_a
            camera_info["wrap"] = wrap
        elif(self.use_bev):
            semantic = common.lidar_to_bev(
                points=points[:,:3],
                H=self.bev_size[0],
                W=self.bev_size[1],
                min_depth=self.depth_range[0],
                max_depth=self.depth_range[1]
            ) # BEV only includes 0 or 1 in the pixel space.

            semantic = np.expand_dims(semantic, axis=0)
            semantic = np.repeat(semantic, 3, axis=0)
        else:
            semantic = np.load(lidar_semantic)

        if(self.use_camera or self.use_bev or self.sampling):
            semantic_ = np.ones((len(points), 1), dtype=np.float32)
            points = np.concatenate([points, semantic_], axis=-1)
        else:
            semantic = np.expand_dims(semantic, axis=1)
            points = np.concatenate([points, semantic], axis=-1)

        if self.transform:
            points[:,:3],_ = self.transform(points[:,:3])

        if(not wrap):
            range_image = common.points_as_images_angle(
                points,
                size = self.resolution,
                fov = self.fov,
                start_a=start_a,
                end_a=end_a,
                depth_range = self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)
        else:
            range_image = common.points_as_images_angle(
                points,
                size = self.resolution,
                fov = self.fov,
                start_a=start_a,
                end_a=180,
                start_a2=-180,
                end_a2=end_a,
                wrap=wrap,
                depth_range = self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)

        sampling_points = None
        sampling_range_image = None
        sampling_batch = None
        if(self.sampling):
            sampling_lidar_path = lidar_path.replace("/LIDAR_TOP/", "/LIDAR_TOP_DOWNSAMPLING0.25/")
            sampling_points = common.get_lidar_sweep(sampling_lidar_path, dim=3)
            sampling_range_image = common.points_as_images(
                sampling_points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=False,
            ).transpose(2, 0, 1)
            sampling_batch = [len(sampling_points),]

        if(not self.use_camera and not self.use_bev):
            semantic = range_image[[5]] / self.semantic_class_num

        sample = {
            "id": lidar_path,
            "batch": [len(points),],
            "points": points[:, :3],                                        # (N,3)
            "xyz": range_image[:3],                                         # (3 H, W)
            "reflectance": common.reflectance_norm(range_image[[3]]),       # (1, H, W)
            "time": range_image[[4]],                                       # (1, H, W)
            "semantic": semantic,                                           # (1, H, W)
            "depth": range_image[[6]],                                      # (1, H, W)
            "mask": range_image[[7]],                                       # (1, H, W)
            "text": lidar_description,                                      # String
            "semantic_org": range_image[[5]],
            "sampling_points": sampling_points,                             # (N/2, 3)
            "sampling_depth": sampling_range_image,                         # (1, H, W)
            "sampling_batch": sampling_batch,                               # (1, H, W)
            "box": lidar_semantic if self.use_3dbox else None,              # (M, 10)
            "camera": lidar_semantic,                                       # image path
            "camera_info": camera_info,
            "bev": semantic[0,:,:] if self.use_bev else None,   # (1,H,W)
        }

        return sample



if __name__ == '__main__':
    data_root = "/root/dataset/nuScenes"
    version = "v1.0-trainval"
    pkl = "nuscenes_description_plus_plus.pkl"
    # pkl = "nuscenes_camera.pkl"
    resolution = (32, 1024)
    dataset = NuScenesDataset(
        data_root=data_root,
        version=version,
        resolution=resolution,
        pkl=pkl,
        text_keys="text",

        sampling=False,
        use_3dbox=True,
        use_camera=False,
        use_bev=False,
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

    for i,batch in enumerate(dataloader):
        common.load_data_to_gpu(batch)
        print(f"---- {i}/{len(dataloader)} ----")
        #print(batch)

        #break

    # xx = '/data/qwt/dataset/nuscenes/raw/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin'
    # points = common.get_lidar_sweep(xx, return_intensity=True, return_time=True, dim=5)
    #
    # range_image = common.points_as_images(
    #     points,
    #     size=(32, 1024),
    #     fov=(3, -25),
    #     depth_range=(0.01,50.0),
    #     return_all=True,
    # ).transpose(2, 0, 1)
    pass