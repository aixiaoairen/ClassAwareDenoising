import torch
import cv2 as cv
import numpy as np
import torch.nn as nn

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