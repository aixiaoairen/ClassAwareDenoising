import random
import cv2 as cv
import h5py as h5
import numpy as np
import DataTools as Tool
from torch.utils import data


class BaseDataSetImg(data.Dataset):
    def __init__(self, img_list, length, patch_size=128):
        """
        基本图像数据类
        :param img_list: paths of images
        :param length: length of Datasets
        :param patch_size: patch size of the cropped patch from each image
        """
        super(BaseDataSetImg, self).__init__()
        self.img_list = img_list
        self.length = length
        self.patch_size = patch_size
        self.num_images = len(img_list)

    def __len__(self):
        return self.length

    def crop_path(self, img):
        h, w = img.shape[0:2]
        h = max(h, self.patch_size)
        w = max(w, self.patch_size)
        img = cv.resize(src=img, dsize=(w, h))
        ind_h = random.randint(0, h - self.patch_size)
        ind_w = random.randint(0, w - self.patch_size)
        patch = img[ind_h: ind_h + self.patch_size, ind_w: ind_w + self.patch_size]
        return patch


class BaseDataSetH5(data.Dataset):
    def __init__(self, h5_path, length=None, patch_size=128):
        """
        :param h5_path: path of h5py file
        :param length: the length of DataSets
        """
        super(BaseDataSetH5, self).__init__()
        self.h5path = h5_path
        self.length = length
        self.patch_size = patch_size
        # 打开h5py文件
        with open(self.h5path, 'r') as fp:
            self.keys = list(fp.keys())
            self.num_images = len(self.keys)
            fp.close()

    def __len__(self):
        if self.length is None:
            return self.num_images
        else:
            return self.length

    def crop_pacth(self, imgs_sets):
        """
        imgs_sets [h, w, 2c]: 包含了含噪图像和无噪图像
        imgs_sets[h, w, :c] - Noisy Image
        imgs_sets[h, w, c:] - Clean Image
        """
        h, w, c2 = imgs_sets.shape
        c = c2 // 2
        ind_h = random.randint(0, h - self.patch_size)
        ind_w = random.randint(0, w - self.patch_size)
        im_noisy = np.array(imgs_sets[ind_h: ind_h + self.patch_size, ind_w: ind_w + self.patch_size, : c])
        im_gt = np.array(imgs_sets[ind_h: ind_h + self.patch_size, ind_w: ind_w + self.patch_size, c:])
        return im_gt, im_noisy
