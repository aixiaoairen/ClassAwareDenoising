import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, img_as_float32

import matplotlib
matplotlib.use("TkAgg")

def GenerateNoisy(img, peak):
    '''
    以无噪图像img为基准，生成含噪图像
    :param img: RGB 图像，dtype = Uint8
    :param peak: 设定的峰值
    :return: Noisy_Img, dtype=float32, [0, 1.0]
    '''
    # 归一化
    img = img_as_float32(img)
    # 变为实数
    peak = float(peak)
    # 模拟生成被泊松噪声破坏的图像
    noisy = np.random.poisson(img * peak) / peak
    return noisy.astype(np.float32)

if __name__ == '__main__':
    pass