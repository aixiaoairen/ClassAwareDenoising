import torch
import torch.nn as nn

class  CADET(nn.Module):
    def __init__(self, in_channels=1, wf=63):
        super(CADET, self).__init__()
        self.oconv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=wf, kernel_size=(3, 3), padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=wf, out_channels=wf, kernel_size=(3, 3), padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.oconv2 = nn.Conv2d(in_channels=wf, out_channels=in_channels, kernel_size=(3, 3), padding=1, bias=True)

    def forward(self, x, depth=20):
        # 存储特征层
        featureMap = []
        # 生成63×h×w的特征层
        x11 = self.conv1(x)
        x11 = self.relu(x11)
        # 存储第一层生成的特征图
        featureMap.append(self.oconv1(x))
        for i in range(depth - 2):
            featureMap.append(self.oconv2(x11))
            x11 = self.vgg(x11)
        featureMap.append(self.oconv2(x11))

        # 不确定是否需要将input图像也加入
        out = x
        for item in featureMap:
            out = out + item
        return out
