from numpy import pad
import torch
import torch.nn as nn


class CADET(nn.Module):
    def __init__(self, in_channels=1, wf=63, depth=20):
        super(CADET, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        for i in range(self.depth - 1):
            if i == 0:
                self.down_path.append(VGG(insize=in_channels, outsize=wf))
            else:
                self.down_path.append(VGG(insize=wf, outsize=wf))
        self.last = nn.Conv2d(in_channels=wf, out_channels=in_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # 存储特征层
        featureMap = []
        tmp = x
        for i, down in enumerate(self.down_path):
            out1, out2 = down(tmp)
            featureMap.append(out2)
            tmp = out1
        featureMap.append(self.last(out1))
        # 不确定是否需要将input图像也加入
        out = x
        for item in featureMap:
            out = out + item
        return out


class VGG(nn.Module):
    def __init__(self, insize, outsize):
        super(VGG, self).__init__()
        block = [
            nn.Conv2d(in_channels=insize, out_channels=outsize, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ]
        self.block = nn.Sequential(*block)
        self.conv = nn.Conv2d(in_channels=insize, out_channels=1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv(x)
        return out1, out2
