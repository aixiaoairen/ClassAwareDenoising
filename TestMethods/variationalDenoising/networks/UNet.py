#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F
from .SubBlocks import conv3x3

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, depth=4, wf=64, slope=0.2):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597
        Raisess
        s

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        # ModuleList 可以存储多个 model，传统的方法，一个model 就要写一个 forward ，但是如果将它们存到一个 ModuleList 的话，就可以使用一个 forward。
        # 因为，UNetConvBlock是一个网络，可以看作Block。所以，用ModuleList来创建一个新的神经网络
        self.down_path = nn.ModuleList()
        # 根据设定的深度，来增加 UNetConvBlock
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i) * wf
        # ModuleList 可以存储多个 model，传统的方法，一个model 就要写一个 forward ，但是如果将它们存到一个 ModuleList 的话，就可以使用一个 forward。
        self.up_path = nn.ModuleList()
        # 倒序添加网络blocks
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_channels, bias=True)

    def forward(self, x):
        blocks = []
        # 编码区
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        # 解码区
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)
# UNet卷积块
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetConvBlock, self).__init__()
        block = []
        # Conv
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        # ReLu
        block.append(nn.LeakyReLU(slope, inplace=True))
        # Conv
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True))
        # ReLu
        block.append(nn.LeakyReLU(slope, inplace=True))
        # 通过nn.Sequential()将网络层和激活函数结合起来，输出激活后的网络节点
        self.block = nn.Sequential(*block)

    def forward(self, x):
        # 在UNetConvBlock中传播
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        # 反卷积
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        # UNet卷积块
        self.conv_block = UNetConvBlock(in_size, out_size, slope)
    # 适度裁剪
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

