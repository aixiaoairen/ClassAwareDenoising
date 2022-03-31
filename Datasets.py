import os
import torch
import random
import cv2 as cv
import h5py as h5
import numpy as np
import DataTools as Tool
from torch.utils import data
from skimage import img_as_float32, img_as_ubyte
from BaseDatasets import BaseDataSetH5, BaseDataSetImg

class SimulateTrain(BaseDataSetImg):
    def __init__(self, im_list, length, patch_size=128, peak=4):
        super(SimulateTrain, self).__init__(img_list=im_list, length=length, patch_size=patch_size)
        self.num_images = len(im_list)
        self.peak = peak
    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        patch_size = self.patch_size
        ind_img = random.randint(0, self.num_images - 1)
        # 读取RGB彩色图像
        gt = cv.imread(self.img_list[ind_img], 1)[:, :, ::-1]
        gt = gt.astype(np.float32)
        # 转为灰度图像
        gt = Tool.rgb2ycbcr(gt)
        gt = gt[:, :, 0]
        # 裁剪为patch
        im_gt = Tool.crop_patch(gt, self.patch_size)
        # 归一化

        # [-0.5, 0.5]
        # 使用系统提供的归一化方式
        # im_gt = img_as_float32(gt)
        # 作者的归一化方式
        maxval = np.amax(np.amax(im_gt))
        im_gt = im_gt.astype(np.float32) * (1.0 / float(maxval)) - 0.5
        img_peak = (0.5 + im_gt ) * float(self.peak)
        im_noisy = Tool.add_poisson_noise_image(img_peak).astype(np.float32)
        im_noisy = (im_noisy / float(self.peak)) - 0.5
        # 数据增强
        im_gt, im_noisy = Tool.rand_batch_augmentation(im_gt, im_noisy)
        # 增加维度 (h, w) ----> (h, w, 1)
        im_gt = im_gt[:, :, np.newaxis]
        im_noisy = im_noisy[:, :, np.newaxis]
        # 变为tensor类型: numpy(h, w, c) ---> tensor(c, h, w)
        im_gt = torch.from_numpy(im_gt.copy().transpose(2, 0, 1))
        im_noisy = torch.from_numpy(im_noisy.copy().transpose(2, 0, 1))
        return im_gt, im_noisy

