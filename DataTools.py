import os
import glob
import random
import cv2 as cv
import numpy as np


def img_augmentation(image, mode):
    """
    Performs dat augmentation of the input image
    :param image: input image
    :param mode: choosen augmentation mode
    :return:
    """
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(m=image, k=1)
    elif mode == 3:
        out = np.rot90(m=image, k=2)
    elif mode == 4:
        out = np.rot90(m=image, k=3)
    elif mode == 5:
        out = np.rot90(m=image, k=1)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(m=image, k=2)
        out = np.flipud(out)
    elif mode == 7:
        out = np.rot90(m=image, k=3)
        out = np.flipud(out)

    return out


def rand_batch_augmentation(*args):
    """
    对批量图像进行数据增强
    :param args: [image1, image2, image3, ....., ]
    :return: [img_augmentation1, img_augmentation2, img_augmentation3, ...]
    """
    out = []
    flags = random.randint(0, 7)
    for data in args:
        out.append(img_augmentation(data, flags))
    return out


def add_poisson_noise_image(img):
    """
    以灰度图像img为基本，模拟生成被泊松噪声污染的图像.
    注意：该灰度图像已经归一化并乘以设置的peak value
    :param img: 灰度图像， dtype=float32, [0.0 * peak, 1.0 * peak]
    :return: 被泊松噪声破坏的灰度图像
    """
    sy, sx  = img.shape
    # 每个像素点对应的lambda
    lambda_flat = np.reshape(a=img, newshape=[1, -1]).astype(np.float32)
    #
    noisy_flat = np.random.poisson(lam=lambda_flat)
    noisy = np.reshape(a=noisy_flat, newshape=[sy, sx]).astype(np.float32)

    return noisy


def crop_patch(image, patch_size):
    """
    从图像中裁剪出一块patch
    :param image: original image
    :param patch_size: the size of patch
    :return: patch, and its size is patch_size × patch_size
    """
    h, w = image.shape[0], image.shape[1]
    if h < patch_size or w < patch_size:
        # 如果真实图像大小的height or width 小于 patchsize, 那么就resize,使得h,w ≥ patch size
        h = max(patch_size, h)
        w = max(patch_size, w)
        image = cv.resize(src=image, dsize=(w, h))
    # 随机选择patch左上角的点
    ind_h = random.randint(0, h - patch_size)
    ind_w = random.randint(0, w - patch_size)
    # crop patch from img
    patch = image[ind_h: ind_h + patch_size, ind_w: ind_w + patch_size]

    return patch

def rgb2ycbcr(rgb_img):
    """
    将RGB图像转为灰度图像
    gray = r * 0.299 + g * 0.587 + b * 0.114
    :param rgb_img: RGB img, dtype=float32, [0.0, 1.0]
    :return: gray 灰度图像， dtype=float32, [0.0, 1.0]
    """
    r, g, b = rgb_img[:, :, 0 : 1], rgb_img[:, :, 1 : 2], rgb_img[:, :, 2 : 3]
    gray = r * 0.299 + g * 0.587 + b * 0.114
    return gray

def get_gt_image(folder_path, trainsets):
    """
    @Brief: combine images(path) in different floders together
    args:
        folder_path: common floder. Example: "/home/liang/Public/MyWork/data/train"
        trainsets: subset folder. Example : { "BSDS200" : "*.png","T91" : "*.png"}
    return: [
                folder_path/trainsets[0].key()/*.png,
                folder_path/trainsets[1].ket()/*.png,
                ....
                folder_path/trainsets[n].key()/*.png
            ]
    """
    try:
        train_images_path = []
        for key, val in trainsets.items():
            pathName = os.path.join(folder_path, key, val)
            for img in glob.glob(pathname=pathName, recursive=True):
                train_images_path.append(img)
    except:
        print("Erroe in path given")

    return train_images_path