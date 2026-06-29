from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
from utils import common

chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load

    chamfer_3D = load(name="chamfer_3D",
                      sources=[
                          "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
                          "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
                      ])
    print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class chamfer_3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        chamfer_3D.backward(
            xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
        )
        return gradxyz1, gradxyz2


class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return chamfer_3DFunction.apply(input1, input2)


def hausdorff_distance(X,Y):
    '''
     the HD is from MPU
    Parameters
    ----------
    X
    Y

    Returns
    -------

    '''
    B,N,C = X.shape
    dist1, dist2 ,_ ,_ = chamfer_3DFunction.apply(X, Y)
    h1 = torch.amax(dist1, dim=1).view(B,1)
    h2 = torch.amax(dist2, dim=1).view(B,1)
    hd_loss = torch.cat([h1,h2],dim=-1)
    return hd_loss

if __name__ == '__main__':
    import torch.nn.functional as F

    torch.set_printoptions(precision=10)

    pc_path1 = "/data/qwt/models/ControlLidar_aimax/upsampling/baseline.ply"
    pc_path2 = "/data/qwt/models/ControlLidar_aimax/upsampling/pudm.xyz"
    # pc_path2 = "/data/qwt/models/ControlLidar_aimax/upsampling/gradpu.ply"
    pc_path2 = "/data/qwt/models/ControlLidar_aimax/upsampling/1.0_80_20.ply"

    x = common.read_ply(pc_path1)
    x = torch.from_numpy(x).unsqueeze(0).float().cuda()
    x, _, _ = common.normalize_point_cloud(x)

    y = common.read_ply(pc_path2)
    y = torch.from_numpy(y).unsqueeze(0).float().cuda()
    y, _, _ = common.normalize_point_cloud(y)

    # x = torch.zeros(size=(2,1024,3)).cuda()
    # y = torch.zeros(size=(2,1024,3)).cuda()

    cd = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cd(x,y)

    dist1 = dist1.mean(-1)
    dist2 = dist2.mean(-1)

    dist = (dist1 + dist2) / 2

    print(dist1)
    print(dist2)

    print(dist)

    xx = hausdorff_distance(x,y).mean(-1)
    print(xx)

    mse = F.mse_loss(x, y)  # 默认是 mean
    print(mse)

    # from utils import lidar
    #
    # lidar.li_rm_li(
    #     li_path="/data/qwt/models/ControlLidar_aimax/logs/diffusion/nuScenes/spherical-1024/20251107T170443/plys/rank_0_num_step_1_sample_50_batch_0_reference.ply",
    #     rm_path="/data/qwt/models/ControlLidar_aimax/upsampling",
    #     new_li_path="/data/qwt/models/ControlLidar_aimax/upsampling"
    # )