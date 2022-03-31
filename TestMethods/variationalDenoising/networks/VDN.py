#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06
'''
将两个已设计好的 S-Net和D-Net神经网络组合在一起
'''
import torch.nn as nn
from .DnCNN import DnCNN
from .UNet import UNet
# 网络权重初始化：根据网络的不同定义来设置不同的初始化方法
def weight_init_kaiming(net):
    '''
    根据网络层的不同定义不同的初始化方式
    '''
    for m in net.modules():
        # 判断当前网络是否是 conv2d，使用相应的初始化方式
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        # 是否为归一层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        # D-Net: the U-Net of 4 depth
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        # U-Net: the DnCNN of 5 depth
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)
    # 前向传播函数：根据不同的mode进行选择
    def forward(self, x, mode='train'):
        # 用于“训练”的前向传播函数
        # string.lower()：将字符串的全部字母转为小写
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        # 用于“测试”的前向传播函数
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        # “sigma”
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma