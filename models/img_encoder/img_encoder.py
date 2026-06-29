import torch
import torch.nn as nn
import numpy as np
import models.img_encoder.resnet as resnet
import torch.nn.functional as F



# C:\Users\lenovo/.cache\torch\hub\checkpoints\resnet34-333f7ec4.pth
class ImageEncoder(nn.Module):
    def __init__(
            self,
            resolution=[32, 1024]
    ):
        super(ImageEncoder, self).__init__()
        self.resolution = resolution

        self.backbone = resnet.resnet34(in_channels=3, pretrained=True, progress=True)

    def forward(self, x):
        resnet_out = self.backbone(x)
        resnet_out = F.interpolate(
            resnet_out,
            size=self.resolution,
            mode='bilinear',
            align_corners=False
        )
        # return resnet_out[2],resnet_out[3],resnet_out[4], resnet_out[5]
        return resnet_out
if __name__ == '__main__':
    # /ihoment/youjie10/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth
    data = torch.ones(size=(2,3,256,256))
    ie = ImageEncoder()
    result = ie(data)
    I1 = result
    print(I1.size())

