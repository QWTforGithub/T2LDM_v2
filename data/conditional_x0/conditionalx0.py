import random
from nuscenes import NuScenes
import pickle
from torch.utils.data import Dataset, DataLoader
from utils import common
import numpy as np
from pathlib import Path
from data.kitti_semantic.descriptor import read_pkl
import matplotlib.pyplot as plt
import cv2

KITTI360_SCENES = [0, 2, 3, 4, 5, 6, 7, 9, 10]
SEMANTICKITTI_SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class ConditionalX0(Dataset):
    def __init__(
            self,
            conditionalx0_lidar_path=None,
            conditionalx0_lidar_description=None,
            conditionalx0_lidar_semantic=None,
            conditionalx0_lidar_info=None,

            data_root='/root/dataset/rsd_data/nuscenes',
            pkl='nuscenes_infos_10sweeps_description.pkl',
            version='v1.0-trainval',
            text_keys="text",

            semantic_class_num=17.0,

            training=True,
            aug=["rotation", "flip"],

            resolution=(32, 1024),
            depth_range=[1.45, 80.0],
            fov=[3, -25],

            sampling=False,
            up_rate=0.25,
            down_rate=1.0,

            use_seg=True,
            use_3dbox=False,
            use_camera=False,
            use_bev=False,

            bev_size=(256, 256),
            resize_scale=0.4,

            type="nuScenes",
            random_num=256, # 256
            print_info=True
    ):

        self.aug = aug
        self.resolution = resolution
        self.depth_range = depth_range
        self.fov = fov
        self.type = type
        self.training = training
        self.sampling = sampling
        self.up_rate = up_rate
        self.down_rate = down_rate
        self.semantic_class_num = semantic_class_num - 1
        self.use_seg = use_seg
        self.use_3dbox = use_3dbox
        self.use_camera = use_camera
        self.resize_scale = resize_scale
        self.use_bev = use_bev
        self.bev_size = bev_size

        self.transform = common.get_lidar_transform(self.aug, self.training)
        self.learning_map = common.get_semantickitti_learning_map(20)

        self.conditionalx0_lidar_path = conditionalx0_lidar_path
        self.conditionalx0_lidar_description = conditionalx0_lidar_description
        self.conditionalx0_lidar_semantic = conditionalx0_lidar_semantic
        self.conditionalx0_lidar_info = conditionalx0_lidar_info

        if (self.conditionalx0_lidar_path is None):
            self.lidar_path = []
            self.lidar_description = []
            self.lidar_semantic = []
            self.lidar_info = []

            lists = [
                8145, 9136, 10245, 11234, 13478,
                15423, 16789, 17789, 18788, 19745,
                0, 174, 356, 1024, 4132,
                5124, 6154, 6657, 7145, 7542,
            ]

            lists = [lists] * 10
            lists = [x for sub in lists for x in sub]

            self.conditionalx0_lidar_path = []
            self.conditionalx0_lidar_description = []
            self.conditionalx0_lidar_semantic = []
            self.conditionalx0_lidar_info = []

            if (type == "nuScenes"):

                pkl_path = f"{data_root}/{version}/{pkl}"
                with open(pkl_path, 'rb') as f:
                    infos = pickle.load(f)

                for info in infos:

                    if (not info.keys().__contains__("lidar_path")):
                        continue

                    lidar_path = info["lidar_path"]
                    self.lidar_path.append(f"{data_root}/{version}/{lidar_path}")

                    description = info[text_keys]
                    self.lidar_description.append(description)

                    if (self.use_3dbox):
                        box = info["gt_boxes"]
                        name = info["gt_names"]
                        self.lidar_semantic.append([box, name])
                        self.lidar_info.append(None)
                    elif (self.use_camera):
                        camera = info["camera"]
                        self.lidar_semantic.append(f"{data_root}/{version}/{camera}")
                        camere_info = info["camera_info"]
                        self.lidar_info.append(camere_info)
                    else:
                        semantic = info["semantic"]
                        self.lidar_semantic.append(f"{data_root}/{version}/{semantic}")
                        self.lidar_info.append(None)

            elif (type == "kitti_360"):
                for scene in KITTI360_SCENES:
                    wildcard = f"*_{scene:04d}_sync/velodyne_points/data/*.bin"
                    self.lidar_path += sorted(Path(data_root).glob(wildcard))

                for _ in self.lidar_path:
                    self.lidar_description.append(None)
                    self.lidar_semantic.append(None)
                    self.lidar_info.append(None)

            elif (type == "kitti_semantic"):
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


            if(random_num > 0):
                lists = random.sample(range(len(self.lidar_path)), random_num)
                print(f"Random list for {type}: {lists}")

            for l in lists:
                self.conditionalx0_lidar_path.append(self.lidar_path[l])
                self.conditionalx0_lidar_description.append(self.lidar_description[l])
                self.conditionalx0_lidar_semantic.append(self.lidar_semantic[l])
                self.conditionalx0_lidar_info.append(self.lidar_info[l])

        if (print_info):
            print(f" ---- ConditionalX0 Dataset with {len(self.conditionalx0_lidar_path)} ---- ")

        return

    def __len__(self):
        return len(self.conditionalx0_lidar_path)

    def __getitem__(self, idx):

        sample = None

        lidar_path = self.conditionalx0_lidar_path[idx]
        lidar_description = self.conditionalx0_lidar_description[idx]
        lidar_semantic = self.conditionalx0_lidar_semantic[idx]
        lidar_info = self.conditionalx0_lidar_info[idx]

        if (self.type == "nuScenes"):
            if(self.sampling):
                sampling_lidar_path = lidar_path.replace("/LIDAR_TOP/", f"/LIDAR_TOP_DOWNSAMPLING{self.up_rate}/")
                points = common.get_lidar_sweep(sampling_lidar_path, dim=3)
                others_ = np.ones((len(points), 2), dtype=np.float32)
                points = np.concatenate([points, others_],axis=-1)
            else:
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
                    points=points[:, :3],
                    H=self.bev_size[0],
                    W=self.bev_size[1],
                    min_depth=self.depth_range[0],
                    max_depth=self.depth_range[1]
                )  # BEV only includes 0 or 1 in the pixel space.
                semantic = np.expand_dims(semantic, axis=0)
                semantic = np.repeat(semantic, 3, axis=0)
            else:
                semantic = np.load(lidar_semantic)

            if(self.use_camera or self.use_bev or self.sampling):
                semantic_ = np.ones((len(points),1), dtype=np.float32)
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
                sampling_lidar_path = lidar_path.replace("/LIDAR_TOP/", f"/LIDAR_TOP_DOWNSAMPLING{self.down_rate}/")
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

        elif (self.type == "kitti_360"):

            points = common.get_lidar_sweep(lidar_path, return_intensity=True)

            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            range_image = common.points_as_images(
                points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=True
            )

            range_image = range_image.transpose(2, 0, 1)

            sample = {
                "batch": [len(points), ],
                "points": points[:, :3],  # (N,3)
                "xyz": range_image[:3],  # (3 H, W)
                "reflectance": range_image[[3]],  # (1, H, W)
                "depth": range_image[[4]],  # (1, H, W)
                "mask": range_image[[5]],  # (1, H, W)
            }

        elif (self.type == "kitti_semantic"):

            points = common.get_lidar_sweep(lidar_path, return_intensity=True)
            if self.transform:
                points[:, :3], _ = self.transform(points[:, :3])

            with open(lidar_semantic, "rb") as a:
                semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
                semantic = np.vectorize(self.learning_map.__getitem__)(
                    semantic & 0xFFFF
                ).astype(np.int32)

            semantic = np.expand_dims(semantic, axis=1)
            points = np.concatenate([points, semantic], axis=-1)

            range_image = common.points_as_images(
                points,
                size=self.resolution,
                fov=self.fov,
                depth_range=self.depth_range,
                return_all=True,
            ).transpose(2, 0, 1)

            sample = {
                "points": points[:, :3],  # (N,3)
                "batch": [len(points), ],
                "xyz": range_image[:3],  # (3  H, W)
                "reflectance": range_image[[3]],  # (1, H, W)
                "semantic": range_image[[4]] / self.semantic_class_num,  # (1, H, W)
                "depth": range_image[[5]],  # (1, H, W)
                "mask": range_image[[6]],  # (1, H, W)
                "text": lidar_description,
                "semantic_org": range_image[[4]],
            }

        return sample


if __name__ == '__main__':

    from utils import lidar

    # resolution = (32, 1024)  # (32,1024) (64, 1024)
    resolution = (32, 256)  # (32,1024) (64, 1024)
    depth_range = (0.01, 50.0)  # (0.0001,50.0) (1.45, 80.0)
    fov = (3, -25)

    li = lidar.LiDARUtility(
        resolution=resolution,
        depth_range=depth_range,
        fov=fov,
        project_dir="/root/models/temp"
    )
    li = li.cuda()

    data_root = "/ihoment/youjie10/qwt/dataset/nuscenes"
    version = "v1.0-trainval"
    pkl = "nuscenes_camera.pkl"
    dataset = ConditionalX0(
        data_root=data_root,
        version=version,
        pkl=pkl,

        resolution=resolution,

        use_camera=False,
        use_seg=False,

        sampling=True,
        up_rate=0.25,
        down_rate=1.0
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
        depth = batch["depth"]
        depth = li.convert_depth(depth)
        depth = li.normalize(depth)
        # li.sample_to_lidar(generation=depth)
        break
        # print(batch)

    pass