# coding=utf-8
from pathlib import Path
from utils import common
import numpy as np
from tqdm import tqdm
import random
import pickle

ROOT_PATH = "/root/dataset/SemanticKITTI/sequences"
DESCRIPTION = "semantic_kitti_description.pkl"
SCENES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEMANTIC_CLASS_NUM = 19
OBJECT = "car"
SAMPLE_NUM = 121066

names = {
    0: "car",
    1: "bicycle",
    2: "motorcycle",
    3: "truck",
    4: "other-vehicle",
    5: "person",
    6: "bicyclist",
    7: "motorcyclist",
    8: "road",
    9: "parking",
    10: "sidewalk",
    11: "other-ground",
    12: "building",
    13: "fence",
    14: "vegetation",
    15: "trunk",
    16: "terrain",
    17: "pole",
    18: "traffic-sign",
    19: "ignore",
}

LOW_OBJECTS = [
    "motorcycle",                                   # 3820
    "bicyclist",                                    # 2230
    "motorcyclist",                                 # 719
]

LOW_INDEXES = [
    2,
    6,
    7
]

MIDDLE_OBJECTS = [
    "person",                                       # 6851
    "bicycle",                                      # 5379
    "other-vehicle",                                # 7322
]

MIDDLE_INDEXES = [
    5,
    1,
    4,
]


HIGH_OBJECTS = [
    "car",                                          # 21855
    "trunk",                                        # 21062
    "traffic-sign",                                 # 16112
    "pole",                                         # 22688
]

HIGH_INDEXES = [
    0,
    15,
    18,
    17,
]


W1 = [
    "The scene contains motorcyclists and {CLASS}",
    "Motorcyclists and {CLASS}.",
]

W2 =[
    "A road scene with motorcyclists.",
    "Motorcyclists in a scene with vegetation.",
    "There are motorcyclists in the scene.",
    "Motorcyclists.",
]

def save_pkl(infos=None, save_path=None):
    if (save_path is None):
        save_path = f"{ROOT_PATH}/{DESCRIPTION}"
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

    file_path = f"{root_path}/{description}"

    with open(file_path, 'rb') as f:
        infos = pickle.load(f)

    return infos

def check_semantic_high_objects(semantic):
    indexes = []
    for high_index in HIGH_INDEXES:
        if(semantic.__contains__(high_index)):
            indexes.append(high_index)
    return indexes

def check_semantic_high_objects_and_object(semantic, object_i):
    indexes = []
    for high_index in HIGH_INDEXES:
        if(semantic.__contains__(high_index)):
            indexes.append(high_index)
    return indexes

def expand_list(li, N):
    """
    Expand lst to length N by randomly sampling from original elements (with replacement).
    """
    if N <= len(li):
        return li[:N]

    result = li.copy()
    while len(result) < N:
        result.append(random.choice(li))

    return result

def w1(class_name1, class_name2, index):

    if(index == 0):
        w = f"The scene contains {class_name1}s and {class_name2}s."
    else:
        w = f"{class_name1}s and {class_name2}s."
        w = w.capitalize()
    return w

def w2(class_name, index):


    if(index == 0):
        w =  f"A road scene with {class_name}s."
    elif(index == 1):
        w = f"{class_name}s in a scene with vegetation."
        w = w.capitalize()
    elif(index == 2):
        w = f"There are {class_name}s in the scene."
    else:
        w = f"{class_name}s."
        w = w.capitalize()

    return w

def get_w1_dict(
        bin_path,
        semantic_path,
        high_objects,
        object_class_name,
        use_random=False,
        use_change=False
):
    w1_i = True
    if(use_random):
        w1_i = random.choice([True, False])
    dict_list = []
    for high_object_i in high_objects:
        class_name1 = object_class_name
        class_name2 = class_name(high_object_i)
        if(use_change):
            change_class_name = random.choice([True, False])
            if(change_class_name):
                temp = class_name1
                class_name1 = class_name2
                class_name2 = temp
        text = w1(class_name1=class_name1, class_name2=class_name2, index=0 if w1_i else 1)
        w1_i = not w1_i

        item = {
            "lidar_path": bin_path,
            "semantic": semantic_path,
            "text": text
        }

        dict_list.append(item)
    return dict_list

def get_w2_dict(
        bin_path,
        semantic_path,
        object_class_name,
        w2_num,
        use_random=False,
):
    dict_list = []

    for w2_i in range(w2_num):
        if(use_random):
            w2_i = random.choice(range(len(W2)))
        text = w2(class_name=object_class_name, index=w2_i)

        item = {
            "lidar_path": bin_path,
            "semantic": semantic_path,
            "text": text
        }

        dict_list.append(item)

    return dict_list

# ---- Low Objects ----
def motorcyclist(bin_paths, semantic_paths):
    '''
        W1 X 4
        W2 X 4
        719 -> 4492
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(7)):
            continue

        # 组合扩充
        indexes = check_semantic_high_objects(semantic)
        index_num = len(indexes)
        if(index_num > 0):
            if(index_num == 1):
                indexes = expand_list(indexes, 2)
            else:
                indexes = expand_list(indexes, 4)
            item_list += get_w1_dict(bin_path, semantic_path, indexes, "motorcyclist")

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"motorcyclist",4)
    print(len(item_list))
    return item_list

def bicyclist(bin_paths, semantic_paths):
    '''
        W1 X 1
        W2 X 2
        6690
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(6)):
            continue

        # 组合扩充
        indexes = check_semantic_high_objects(semantic)
        index_num = len(indexes)
        if(index_num > 0):
            indexes = random.sample(indexes, 1)
            item_list += get_w1_dict(
                bin_path,
                semantic_path,
                indexes,
                "bicyclist",
                use_random=True
            )

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"bicyclist",2, use_random=True)

    print(len(item_list))
    return item_list

def motorcycle(bin_paths, semantic_paths):
    '''
        W1 X 1
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(2)):
            continue

        # 组合扩充
        indexes = check_semantic_high_objects(semantic)
        index_num = len(indexes)
        if(index_num > 0):
            indexes = random.sample(indexes, 1)
            item_list += get_w1_dict(
                bin_path,
                semantic_path,
                indexes,
                "motorcycle",
                use_random=True
            )

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"motorcycle",1, use_random=True)

    print(len(item_list))
    return item_list
# ---- Low Objects ----

# ---- Middle Objects ----
def person(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(5)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"person",1, use_random=True)

    print(len(item_list))
    return item_list

def bicycle(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(1)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"bicycle",1, use_random=True)

    print(len(item_list))
    return item_list

def othervehicle(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(4)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"othervehicle",1, use_random=True)

    print(len(item_list))
    return item_list
# ---- Middle Objects ----

# ---- High Objects ----
def car(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(0)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"car",1, use_random=True)

    print(len(item_list))
    return item_list

def trunk(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(15)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"trunk",1, use_random=True)

    print(len(item_list))
    return item_list

def trafficsign(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(18)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"trafficsign",1, use_random=True)

    print(len(item_list))
    return item_list

def pole(bin_paths, semantic_paths):
    '''
        W1 X 0
        W2 X 1
    '''

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)

    item_list = []
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        if(not semantic.__contains__(17)):
            continue

        # 单锚点
        item_list += get_w2_dict(bin_path, semantic_path,"pole",1, use_random=True)

    print(len(item_list))
    return item_list
# ---- High Objects ----


def class_name(index):
    names = {
        0:  "car",
        1:  "bicycle",
        2:  "motorcycle",
        3:  "truck",
        4:  "othervehicle",
        5:  "person",
        6:  "bicyclist",
        7:  "motorcyclist",
        8:  "road",
        9:  "parking",
        10: "sidewalk",
        11: "other-ground",
        12: "building",
        13: "fence",
        14: "vegetation",
        15: "trunk",
        16: "terrain",
        17: "pole",
        18: "trafficsign",
        19: "ignore",
    }

    return names[index]

def get_num_everyclass():
    class_num = {}
    bin_paths, semantic_paths = get_bin_semantic_paths()
    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)
    for bin_path, semantic_path in tqdm(zip(bin_paths,semantic_paths), total=len(semantic_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        for i in range(SEMANTIC_CLASS_NUM+1):
            if(semantic.__contains__(i)):
                name = class_name(i)
                if(class_num.keys().__contains__(name)):
                    class_num[name] += 1
                else:
                    class_num[name] = 1

    for key in class_num.keys():
        print(f"{key} : {class_num[key]}")

def get_bin_semantic_paths():
    bin_paths = []
    semantic_paths = []
    for scene in SCENES:
        wildcard = f"{scene:02d}/velodyne/*.bin"
        bin_paths += sorted(Path(ROOT_PATH).glob(wildcard))

        wildcard = f"{scene:02d}/labels/*.label"
        semantic_paths += sorted(Path(ROOT_PATH).glob(wildcard))
    return bin_paths, semantic_paths

def get_dict(
        bin_path,
        semantic_path,
        semantic,
        object_i,
        w1_num=0,
        use_random_w1=False,
        w2_num=1,
        use_random_w2=False,

):
    item_list = []
    if(not semantic.__contains__(object_i)):
        return item_list

    object_name = class_name(object_i)

    # 组合扩充
    if(w1_num > 0):
        indexes = check_semantic_high_objects(semantic)
        index_num = len(indexes)
        if(index_num > 0):
            if(index_num == 1):
                indexes = expand_list(indexes, 2)
            else:
                indexes = expand_list(indexes, w1_num)
            item_list += get_w1_dict(
                bin_path,
                semantic_path,
                indexes,
                object_name,
                use_random_w1
            )

    # 单锚点
    item_list += get_w2_dict(
        bin_path,
        semantic_path,
        object_name,
        w2_num,
        use_random_w2
    )
    return item_list

def downsample_text(infos, text, save_num=4000):
    li = []
    for i, info in enumerate(infos):
        info_text = info["text"]
        if(info_text == text):
            li.append(i)
    if(len(li) == 0):
        return infos
    x = len(li) - save_num
    if(x <= 0):
        x = len(li)
    li = random.sample(li, x)

    infos = np.delete(infos, li)
    return infos

def generation_text():

    bin_paths, semantic_paths = get_bin_semantic_paths()
    item_list = []

    item_motorcyclist = []
    item_bicyclist = []
    item_motorcycle = []

    item_othervehicle = []
    item_bicycle = []
    item_person = []

    item_trafficsign = []
    item_trunk = []
    item_pole = []
    item_car = []

    learning_map = common.get_semantickitti_learning_map(SEMANTIC_CLASS_NUM)
    for bin_path, semantic_path in tqdm(zip(bin_paths, semantic_paths), total=len(bin_paths)):
        with open(semantic_path, "rb") as a:
            semantic = np.fromfile(a, dtype=np.int32).reshape(-1)
            semantic = np.vectorize(learning_map.__getitem__)(
                semantic & 0xFFFF
            ).astype(np.int32)

        # ---- Low Objects ----
        # # ---- motorcyclist ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=7,
            w1_num=4,
            use_random_w1=False,
            w2_num=4,
            use_random_w2=False
        )
        item_motorcyclist += items
        item_list += items
        # # ---- motorcyclist ----

        # ---- bicyclist ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=6,
            w1_num=1,
            use_random_w1=True,
            w2_num=2,
            use_random_w2=True
        )
        item_bicyclist += items
        item_list += items
        # ---- bicyclist ----

        # ---- motorcycle ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=2,
            w1_num=1,
            use_random_w1=True,
            w2_num=1,
            use_random_w2=True
        )
        item_motorcycle += items
        item_list += items
        # ---- motorcycle ----
        # ---- Low Objects ----

        # ---- Middle Objects ----
        # ---- othervehicle ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=4,
            w1_num=1,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_othervehicle += items
        item_list += items
        # ---- othervehicle ----

        # ---- bicycle ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=1,
            w1_num=1,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_bicycle += items
        item_list += items
        # ---- bicycle ----

        # ---- person ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=5,
            w1_num=1,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_person += items
        item_list += items
        # ---- person ----
        # ---- Middle Objects ----

        # ---- High Objects ----
        # ---- pole Objects ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=17,
            w1_num=0,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_pole += items
        item_list += items
        # ---- pole Objects ----

        # ---- trafficsign Objects ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=18,
            w1_num=0,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_trafficsign += items
        item_list += items
        # ---- trafficsign Objects ----

        # ---- trunk Objects ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=15,
            w1_num=0,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_trunk += items
        item_list += items
        # ---- trunk Objects ----

        # ---- car Objects ----
        items = get_dict(
            bin_path=bin_path,
            semantic_path=semantic_path,
            semantic=semantic,
            object_i=0,
            w1_num=0,
            use_random_w1=False,
            w2_num=1,
            use_random_w2=True
        )
        item_car += items
        item_list += items
        # ---- car Objects ----
        # ---- High Objects ----

    save_pkl(infos=item_list)

    count_everyclass_text(item_list)

    print(f"totol num : {len(item_list)}")

def print_info():
    infos = read_pkl()
    if(len(infos) > SAMPLE_NUM):
        infos = infos[:-1]
    for info in infos:
        print(info["text"])

def count_everyclass_text(infos):
    motorcyclist = 0
    bicyclist = 0
    motorcycle = 0

    other_vehicle = 0
    bicycle = 0
    person = 0

    traffic_sign = 0
    trunk = 0
    pole = 0
    car = 0

    for info in infos:
        info_text = info["text"]
        if(info_text.__contains__("motorcyclist")):
            motorcyclist += 1
        if(info_text.__contains__("bicyclist")):
            bicyclist += 1
        if(info_text.__contains__("motorcycle")):
            motorcycle += 1

        if(info_text.__contains__("vehicle")):
            other_vehicle += 1
        if(info_text.__contains__("bicycle")):
            bicycle += 1
        if(info_text.__contains__("person")):
            person += 1

        if(info_text.__contains__("sign")):
            traffic_sign += 1
        if(info_text.__contains__("trunk")):
            trunk += 1
        if(info_text.__contains__("pole")):
            pole += 1
        if(info_text.__contains__("car")):
            car += 1

    print(f"motorcyclist : {motorcyclist}")
    print(f"bicyclist : {bicyclist}")
    print(f"motorcycle : {motorcycle}")

    print(f"othervehicle : {other_vehicle}")
    print(f"bicycle : {bicycle}")
    print(f"person : {person}")

    print(f"pole : {pole}")
    print(f"traffic_sign : {traffic_sign}")
    print(f"trunk : {trunk}")
    print(f"car : {car}")

def count_text(infos=None):
    if(infos is None):
        infos = read_pkl()
    # if(len(infos) > SAMPLE_NUM):

    count_num = {}
    for info in infos:
        try:
            text = info["text"]
        except Exception as e:
            print(info)
            exit(1)

        if(count_num.keys().__contains__(text)):
            count_num[text] += 1
        else:
            count_num[text] = 1

    total_num = 0
    for key in count_num.keys():
        print(f"{key}: {count_num[key]}")
        total_num += count_num[key]

    print(total_num)


def delete_text(infos, text):
    new_info = []
    for info in infos:
        info_text = info["text"]
        if(info_text != text):
            new_info.append(info)

    return new_info

def check_path(infos):
    new_infos = []
    for info in tqdm(infos, total=len(infos)):
        bin_path = str(info["lidar_path"])
        bin_path = bin_path.replace(f"{ROOT_PATH}/","")
        info["lidar_path"] = bin_path

        semantic_path = str(info["semantic"])
        info["semantic"] = semantic_path.replace(f"{ROOT_PATH}/","")

        new_infos.append(info)

    return new_infos

if __name__ == '__main__':

    '''
        car : 21855
        motorcyclist : 719
        road : 23201
        parking : 9097
        sidewalk : 22123
        building : 21187
        fence : 22591
        vegetation : 23201
        trunk : 21062
        terrain : 22596
        pole : 22688
        traffic-sign : 16112
        ignore : 23201
        person : 6851
        other-vehicle : 7322
        bicyclist : 2230
        bicycle : 5379
        other-ground : 6158
        motorcycle : 3820
        truck : 2527
        
        car : 21855
        sidewalk : 22123
        building : 21187
        fence : 22591
        vegetation : 23201
        trunk : 21062
        terrain : 22596
        pole : 22688
        
        
        
        car : 21855
        trunk : 21062
        traffic-sign : 16112
        pole : 22688
        
        person : 6851
        bicycle : 5379
        other-vehicle : 7322
        
        motorcycle : 3820
        bicyclist : 2230
        motorcyclist : 719
        
        
        1.  The scene contains motorcyclists and {CLASS}. X 1
        2.  Motorcyclists and {CLASS}. X 1 
        3.  Motorcyclists are around {CLASS}. X 1 
        4.  A road scene with motorcyclists. X 1
        5.  There are motorcyclists in the scene. X 1
        6.  Motorcyclists. X 1
        
        motorcyclist : 4492 
        bicyclist : 7622 
        motorcycle : 7683 
        
        othervehicle : 7322 
        bicycle : 5379 
        person : 6851 
        
        pole : 22688 
        trafficsign : 16112 
        trunk : 21062 
        car : 21855 
        
        
        totol num : 121067
        
    '''


    # generation_text()

    item_list = read_pkl(description="semantic_kitti_description_all.pkl")
    infos = check_path(item_list)
    save_pkl(infos)

    count_text(item_list)
    print()

    item_list = delete_text(item_list, "Othervehicles and poles.")
    item_list = delete_text(item_list, "The scene contains othervehicles and poles.")
    item_list = delete_text(item_list, "The scene contains motorcycles and poles.")
    item_list = delete_text(item_list, "Motorcycles and poles.")
    item_list = delete_text(item_list, "Motorcycles and trunks.")
    item_list = delete_text(item_list, "The scene contains persons and trunks.")
    item_list = delete_text(item_list, "The scene contains motorcyclists and trunks.")
    item_list = delete_text(item_list, "The scene contains motorcycles and trunks.")
    item_list = delete_text(item_list, "Bicyclists and trunks.")
    item_list = delete_text(item_list, "The scene contains bicyclists and trunks.")
    item_list = delete_text(item_list, "The scene contains othervehicles and trunks.")
    item_list = delete_text(item_list, "Motorcyclists and cars.")
    item_list = delete_text(item_list, "Motorcyclists and trafficsigns.")
    item_list = delete_text(item_list, "The scene contains motorcyclists and poles.")
    item_list = delete_text(item_list, "The scene contains bicycles and trunks.")

    item_list = downsample_text(item_list, "The scene contains othervehicles and cars.", save_num=2000)
    item_list = downsample_text(item_list, "Othervehicles in a scene with vegetation.", save_num=1500)
    item_list = downsample_text(item_list, "There are othervehicles in the scene.", save_num=1500)
    item_list = downsample_text(item_list, "A road scene with othervehicles.", save_num=1500)

    item_list = downsample_text(item_list, "Trafficsigns in a scene with vegetation.", save_num=2500)
    item_list = downsample_text(item_list, "There are trafficsigns in the scene.", save_num=2500)
    item_list = downsample_text(item_list, "A road scene with trafficsigns.", save_num=2500)

    item_list = downsample_text(item_list, "A road scene with othervehicles.", save_num=1500)
    item_list = downsample_text(item_list, "There are othervehicles in the scene.", save_num=1500)
    item_list = downsample_text(item_list, "Othervehicles.", save_num=1500)
    item_list = downsample_text(item_list, "Othervehicles in a scene with vegetation.", save_num=1500)

    item_list = downsample_text(item_list, "A road scene with trafficsigns.", save_num=2500)
    item_list = downsample_text(item_list, "Trafficsigns.", save_num=2500)
    item_list = downsample_text(item_list, "There are trafficsigns in the scene.", save_num=2500)
    item_list = downsample_text(item_list, "Trafficsigns in a scene with vegetation.", save_num=2500)

    item_list = downsample_text(item_list, "The scene contains persons and cars.", save_num=4812)

    count_everyclass_text(item_list)
    count_text(item_list)
    save_pkl(infos=item_list)
    pass