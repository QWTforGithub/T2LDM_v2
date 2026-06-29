# coding=utf-8
from torchsparse import SparseTensor
from torch import nn
from torchsparse import nn as spnn
import torch


if __name__ == '__main__':
    model = nn.Sequential(
        spnn.Conv3d(3, 10, 3),
        spnn.BatchNorm(10),
        spnn.ReLU(True),
    )

    coords = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.int32)

    # 2️⃣ 每个点的特征（例如通道数 = 3）
    feats = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)

    # 3️⃣ 创建 SparseTensor
    x = SparseTensor(feats, coords)

    # 4️⃣ 查看属性
    print('SparseTensor:', x)
    print('coords:', x.C.shape, x.C[:5])
    print('feats:', x.F.shape, x.F[:5])