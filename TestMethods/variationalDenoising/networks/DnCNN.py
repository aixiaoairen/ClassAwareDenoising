#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:27:05

import torch.nn as nn
from .SubBlocks import conv3x3
# 相关介绍：https://lianghao.work/96-2/
class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dep=20, num_filters=64, slope=0.2):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 20
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN, self).__init__()
        # 卷积 conv1
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        # 激活函数
        self.relu = nn.LeakyReLU(slope, inplace=True)
        mid_layer = []
        # 批量添加 conv 和 LeakyReLU
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(slope, inplace=True))
        # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
        self.mid_layer = nn.Sequential(*mid_layer)
        # 最后输出前进行以此卷积
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)
    def forward(self, x):
        # 输入1
        x = self.conv1(x)
        x = self.relu(x)
        # 输入中间
        x = self.mid_layer(x)
        # 输出
        out = self.conv_last(x)
        return out

