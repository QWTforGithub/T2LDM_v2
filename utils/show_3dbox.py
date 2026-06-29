"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
import pickle

BIN_PATH = "G:/dataset/nuScenes/nuscenes/v1.0-trainval"
ROOT_PATH = "sample_data"
DESCRIPTION = "nuscenes_description_plus_plus.pkl"

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

def get_color():
    colors=[
        [229, 25,  74 ], # 0 ，#E5194A， 红色，"barrier"，障碍物
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
        [0  , 0,   0  ], # #FFFFFF，白色
    ]

    colors = np.asarray(colors) / 255

    return colors

def read_pkl(
    root_path = None,
    description = None,
):
    if(root_path is None):
        root_path = ROOT_PATH

    if(description is None):
        description = DESCRIPTION

    file_path = f"{root_path}/{description}"

    with open(file_path, 'rb') as f:
        infos = pickle.load(f)

    return infos

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

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0


    # 背景
    # vis.get_render_option().background_color = np.zeros(3)
    vis.get_render_option().background_color = np.asarray([1,1,1])
    # vis.get_render_option().background_color = np.asarray([0.85,0.85,0.85])

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        #pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        colors = np.full(shape=(points.shape[0], 3), fill_value=[1,0,0])
        #pts.colors = open3d.utility.Vector3dVector(colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(len(gt_boxes)):
        gt_box = gt_boxes[i]
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.01
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.015
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.02
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.025
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.03
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.035
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.04
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.045
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.05
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.055
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.06
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.065
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.07
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)

        gt_box_1 = gt_boxes[i]
        gt_box_1[3:6] = gt_box_1[3:6] + 0.075
        line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)


        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_scenes_many_box(
        points,
        gt_boxes=None,
        ref_boxes=None,
        ref_labels=None,
        ref_scores=None,
        point_colors=None,
        draw_origin=False,
        box_colors=[(0,1,0),(1,0,0),(1,1,0),(1,0,1),(0,1,1)]
):

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if isinstance(gt_boxes, torch.Tensor):
        for i, gt_box in enumerate(gt_boxes):
            gt_boxes[i] = gt_box.cpu().numpy()

    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0

    # 背景
    # vis.get_render_option().background_color = np.zeros(3)
    vis.get_render_option().background_color = np.asarray([1, 1, 1])
    # vis.get_render_option().background_color = np.asarray([0.85,0.85,0.85])

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        # pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        colors = np.full(shape=(points.shape[0], 3), fill_value=[1, 0, 0])
        # pts.colors = open3d.utility.Vector3dVector(colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_many_box(vis, gt_boxes, box_colors)

    if ref_boxes is not None:
        vis = draw_many_box(vis, ref_boxes, box_colors, ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def draw_many_box(vis, gt_many_boxes, colors=(0, 1, 0), ref_labels=None, score=None):

    for j in range(len(colors)):
        color = colors[j]
        gt_boxes = gt_many_boxes[j]

        for i in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[i]
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box)

            if ref_labels is None:
                line_set.paint_uniform_color(color)
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])

            vis.add_geometry(line_set)

            gt_box_1 = gt_boxes[i]
            gt_box_1[3:6] = gt_box_1[3:6] + 0.01
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

            gt_box_1 = gt_boxes[i]
            gt_box_1[3:6] = gt_box_1[3:6] + 0.015
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

            gt_box_1 = gt_boxes[i]
            gt_box_1[3:6] = gt_box_1[3:6] + 0.02
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

            gt_box_1 = gt_boxes[i]
            gt_box_1[3:6] = gt_box_1[3:6] + 0.025
            line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color(color)
            vis.add_geometry(line_set)

            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.03
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.035
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.04
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.045
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.05
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.055
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.06
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.065
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.07
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)
            #
            # gt_box_1 = gt_boxes[i]
            # gt_box_1[3:6] = gt_box_1[3:6] + 0.075
            # line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
            # line_set.paint_uniform_color(color)
            # vis.add_geometry(line_set)


        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

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

def corners_to_boxes_3d(corners):
    """
    Convert 8 corners to 3D boxes.

    Args:
        corners: np.ndarray, shape [M, 8, 3]
            Corner order:
                0,1,2,3: top face
                4,5,6,7: bottom face

    Returns:
        boxes: np.ndarray, shape [M, 7]
            [x, y, z, l, w, h, yaw]
    """
    corners = np.asarray(corners, dtype=np.float32)

    # center
    center = corners.mean(axis=1)  # [M, 3]

    # height: average vertical edge length
    h = np.mean([
        np.linalg.norm(corners[:, 0] - corners[:, 4], axis=1),
        np.linalg.norm(corners[:, 1] - corners[:, 5], axis=1),
        np.linalg.norm(corners[:, 2] - corners[:, 6], axis=1),
        np.linalg.norm(corners[:, 3] - corners[:, 7], axis=1),
    ], axis=0)

    # length: along corner 0 -> 1 or 3 -> 2
    # width:  along corner 1 -> 2 or 0 -> 3
    # 如果你之前 corners 定义是 x方向为 length, y方向为 width，
    # 更准确的是：
    l = np.mean([
        np.linalg.norm(corners[:, 0] - corners[:, 3], axis=1),
        np.linalg.norm(corners[:, 1] - corners[:, 2], axis=1),
        np.linalg.norm(corners[:, 4] - corners[:, 7], axis=1),
        np.linalg.norm(corners[:, 5] - corners[:, 6], axis=1),
    ], axis=0)

    w = np.mean([
        np.linalg.norm(corners[:, 0] - corners[:, 1], axis=1),
        np.linalg.norm(corners[:, 3] - corners[:, 2], axis=1),
        np.linalg.norm(corners[:, 4] - corners[:, 5], axis=1),
        np.linalg.norm(corners[:, 7] - corners[:, 6], axis=1),
    ], axis=0)

    # yaw: length direction
    # use vector from corner 3 -> 0
    direction = corners[:, 0] - corners[:, 3]  # [M, 3]
    yaw = np.arctan2(direction[:, 1], direction[:, 0])

    boxes = np.stack([
        center[:, 0],
        center[:, 1],
        center[:, 2],
        l,
        w,
        h,
        yaw
    ], axis=1)

    return boxes

def get_semantic_from_boxes(
        points: torch.Tensor,  # [N,3], float32/float16, LiDAR坐标系
        boxes: torch.Tensor,  # [M,10], (cx,cy,cz, dx,dy,dz, yaw, ..., class_id)
        background_id: int = 10,
        ignore_ids=None,  # e.g. {255}；None 表示不过滤
        prefer: str = "smallest_box", # "nearest_center",  # or "smallest_box"
        box_chunk: int = None  # e.g. 256/512；None=不分块
):
    """
    返回:
      sem_labels: [N] (torch.long)  每个点的语义类别（未命中=background_id）
    说明:
      - yaw 视为绕 +Z 轴的右手旋转（弧度）
      - 多盒命中用 prefer 决策："nearest_center" 或 "smallest_box"
      - 可选 box_chunk 对 M 分块，降低显存
    """
    if(not isinstance(points, torch.Tensor)):
        points = torch.from_numpy(points).float()

    if(not isinstance(boxes, torch.Tensor)):
        boxes = torch.from_numpy(boxes).float()

    device, dtype = points.device, points.dtype
    N = points.shape[0]
    M = boxes.shape[0]

    # 默认结果：全背景
    sem_labels = torch.full((N,), background_id, dtype=torch.long, device=device)
    if N == 0 or M == 0:
        return sem_labels

    centers_all = boxes[:, 0:3].to(device=device, dtype=dtype)  # [M,3]
    sizes_all = boxes[:, 3:6].to(device=device, dtype=dtype)  # [M,3] (dx,dy,dz)
    yaw_all = boxes[:, 6].to(device=device, dtype=dtype)  # [M]
    cls_all = boxes[:, 9].long().to(device)  # [M]

    # 过滤 ignore 类（可选）
    if ignore_ids is not None and len(ignore_ids) > 0:
        keep = ~torch.isin(cls_all, torch.as_tensor(list(ignore_ids), device=device))
        centers_all, sizes_all, yaw_all, cls_all = centers_all[keep], sizes_all[keep], yaw_all[keep], cls_all[keep]
        M = centers_all.shape[0]
        if M == 0:
            return sem_labels

    # 维护全局最优（每个点）
    if prefer == "nearest_center":
        best_score = torch.full((N,), float("inf"), dtype=dtype, device=device)  # 最小中心距离^2
    elif prefer == "smallest_box":
        best_score = torch.full((N,), float("inf"), dtype=dtype, device=device)  # 最小体积
    else:
        raise ValueError(f"Unknown prefer={prefer}")

    best_cls = torch.full((N,), background_id, dtype=torch.long, device=device)

    # 分块遍历盒子，避免一次性 M×N 太大
    if box_chunk is None or box_chunk <= 0:
        box_chunk = M

    for s in range(0, M, box_chunk):
        e = min(s + box_chunk, M)
        centers = centers_all[s:e]  # [m,3]
        sizes = sizes_all[s:e]  # [m,3]
        yaw = yaw_all[s:e]  # [m]
        cls_id = cls_all[s:e]  # [m]
        m = centers.shape[0]
        if m == 0:
            continue

        # Rz(yaw)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)  # [m]
        R = torch.zeros((m, 3, 3), dtype=dtype, device=device)
        R[:, 0, 0] = cos_y;
        R[:, 0, 1] = -sin_y
        R[:, 1, 0] = sin_y;
        R[:, 1, 1] = cos_y
        R[:, 2, 2] = 1.0

        # p_local = R^T * (p - c) ；广播到 [m,N,3]
        p_rel = points.unsqueeze(0) - centers.unsqueeze(1)  # [m,N,3]
        p_local = torch.einsum('mij,mnj->mni', R.transpose(1, 2), p_rel)  # [m,N,3]

        half = sizes / 2.0
        inside = (p_local[:, :, 0].abs() <= half[:, 0, None]) \
                 & (p_local[:, :, 1].abs() <= half[:, 1, None]) \
                 & (p_local[:, :, 2].abs() <= half[:, 2, None])  # [m,N]

        if prefer == "nearest_center":
            # 分数=到盒子中心的距离^2，inside 之外设为 +inf
            dist2 = (p_rel ** 2).sum(dim=2)  # [m,N]
            score = torch.where(inside, dist2, torch.full_like(dist2, float("inf")))
        else:  # "smallest_box"
            vol = (sizes[:, 0] * sizes[:, 1] * sizes[:, 2]).unsqueeze(1)  # [m,1]
            score = torch.where(inside, vol, torch.full_like(vol, float("inf")))  # [m,N]

        # 在当前块内选最佳盒子
        cur_best_score, cur_best_idx = score.min(dim=0)  # [N], [N]
        improve = cur_best_score < best_score  # [N]

        # 仅在有命中(非 inf) 且 更好 时更新
        hit_and_better = improve & torch.isfinite(cur_best_score)

        best_score[hit_and_better] = cur_best_score[hit_and_better]
        best_cls[hit_and_better] = cls_id[cur_best_idx[hit_and_better]]

    sem_labels = best_cls  # [N]
    return sem_labels.cpu().numpy()

def colorize_point_by_label(points, labels, name="sample_data/color_point_cloud.ply"):
    color_setting = get_color()

    colors = []
    for i in range(len(points)):
        color = color_setting[labels[i]]
        colors.append(color)

    colors = np.asarray(colors)
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.colors = open3d.utility.Vector3dVector(colors)

    open3d.io.write_point_cloud(filename=name, pointcloud=pc)

    print(f"Saving :{name}")

def get_semantic_from_corners(
    points,
    corners,
    classes,
    background_id=20,
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

if __name__ == '__main__':
    import numpy as np

    pc_path = "xxx.ply" # [N, 3]
    gt_boxes_path = "xxx.npy" # [M, 7] (x,y,z, length,width,high, yaw)

    points = open3d.io.read_point_cloud(pc_path)
    points = np.asarray(points.points)
    gt_boxes = np.load(gt_boxes_path)

    # ---- 画3D检测框 ----
    gt_many_boxes=[
        gt_boxes,
    ]

    box_colors = [
        (1, 0, 0), # 红
        # (0, 1, 0), # 黄
        # (1, 1, 0), # 绿色
        # (1, 0, 1), # 粉色
        # (0, 0, 0) # 橙色
    ]

    draw_scenes_many_box(points, gt_boxes=gt_many_boxes, box_colors=box_colors)
    # ---- 画3D检测框 ----


    pass