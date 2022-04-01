# @Time : 2022/4/1 11:04
# @Author : LiangHao
# @File : TestSimulate.py
import sys
import torch
import matplotlib
import numpy as np
import torch.nn as nn
import DataTools as Tools
import matplotlib.pyplot as plt
from network import CADET
from Datasets import SimulateTrain
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

matplotlib.use("TkAgg")

def TestOne(SimulateData, Model):
    # 从数据集中随机选取一堆 无噪-含噪 图像对
    item = np.random.randint(0, SimulateData.num_images - 1)
    gt, noisy = SimulateData.__getitem__(item)
    # 转为numpy类型
    np_gt, np_noisy = gt.cpu().numpy(), noisy.cpu().numpy()
    # 调整(c, h, w) --- > (h, w, c)
    np_gt = np_gt.transpose((1, 2, 0)).squeeze()
    np_noisy = np_noisy.transpose((1, 2, 0)).squeeze()
    # 去噪
    denoise = Model(noisy)
    np_denoise = denoise.detach().cpu().numpy()
    np_denoise = np_denoise.transpose((1, 2, 0)).squeeze()
    # 因为原来的区间是[-0.5, 0.5]，现在调整为[0, 1]
    np_gt, np_noisy, np_denoise = np_gt + 0.5, np_noisy + 0.5, np_denoise + 0.5
    # 计算PSNR

    psnr_noisy = psnr(image_true=(np_gt * 255).astype(np.uint8), image_test=(np_noisy * 255).astype(np.uint8),
                      data_range=255)
    psnr_denoise = psnr(image_true=(np_gt * 255).astype(np.uint8), image_test=(np_denoise * 255).astype(np.uint8),
                        data_range=255)

    ssim_noisy = ssim(im1=(np_gt * 255).astype(np.uint8), im2=(np_noisy * 255).astype(np.uint8),
                      data_range=255, multichannel=False)
    ssim_denoise = ssim(im1=(np_gt * 255).astype(np.uint8), im2=(np_denoise * 255).astype(np.uint8),
                        data_range=255, multichannel=False)
    mse = nn.MSELoss()
    mse_denoise = mse(gt.cpu(), denoise.cpu())
    mse_noisy = mse(gt.cpu(), noisy.cpu())

    print("Noisy and Gound truth: psnr:  {0}  ,  ssim:  {1}, mseloss: {2}".format(psnr_noisy, ssim_noisy, mse_noisy))
    print("Denoise and Gound truth: psnr:  {0}  ,  ssim:  {1}, mseloss: {2}".format(psnr_denoise, ssim_denoise, mse_denoise))

    plt.subplot(131), plt.imshow(np_gt, 'gray'), plt.title("Original Image")
    plt.subplot(132), plt.imshow(np_noisy, 'gray'), plt.title("Noisy Image")
    plt.subplot(133), plt.imshow(np_denoise, 'gray'), plt.title("denoise")
    plt.show()

def main():
    datasets = {
        "JPEGImages": "*.jpg"
    }
    dirpath = "F:\\Program File\\DataSpell\\Projetcs\\ImageDenoising\\ConvolutionalPoissonNoise\\dsProject\\TestMethods\\data"
    train_paths = np.array(sorted(Tools.get_gt_image(dirpath, datasets)))
    SimulateData = SimulateTrain(im_list=train_paths, length=64, patch_size=128, peak=4.0)
    Model = CADET(in_channels=1, wf=63)
    modelpath = "F:\\Program File\\DataSpell\\Projetcs\\ImageDenoising\\ConvolutionalPoissonNoise\\dsProject\\Model\\model_resume2\\model020.pth"
    checkpoint = torch.load(modelpath)
    state_dict_one = Model.state_dict()
    for name, value in checkpoint["model_state_dict"].items():
        state_dict_one[name] = value
    Model.load_state_dict(state_dict_one)
    TestOne(SimulateData, Model)

if __name__ == '__main__':
    main()



