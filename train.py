import os
import torch
import numpy as np
import DataTools as Tool
import torch.optim as optim
from network import CADET
from Datasets import SimulateTrain
from torch.utils.data import DataLoader
from utils import weight_init_kaiming
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def Train(net, datasets, optimizer, criterion, model_dir, batch_size=64, epochs=100):
    """
    训练函数
    :param net: 已经初始化的神经网络，已经进行了gpu加速
    :param datasets: 自定的数据集
    :param optimizer: 优化器Adam
    :param lr: 学习率
    :param criterion: 损失函数，这里使用 MSE
    :param batch_size: batch size
    :param epochs: epoch size
    :return:
    """
    train_loss = []
    train_epochs_loss = []
    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True, num_workers=1)
    for epoch in range(epochs):
        net.train()
        train_epoch_loss = []
        for idx, data in enumerate(dataloader):
            # gpu accelerate
            gt, nosiy = [x.cuda() for x in data]
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            out = net(nosiy)
            # 计算损失值
            loss = criterion(out, gt)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 记录loss
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(dataloader) //2 ) == 0:
                print("epoch= {0}/{1}, {2}/{3} of train, loss = {4}".format(epoch,
                                                            epochs, idx, len(dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        # 没一个epoch完成后就保存模型
        torch.save({
            'model_state_dict': net.state_dict(),
            'loss': train_epochs_loss
        }, os.path.join(model_dir, 'model{}.pth'.format(str(epoch+1).zfill(2))))

def main(model_dir):
    lr = 1e-4
    batch_size = 64
    epochs = 60
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    model = CADET(in_channels=1, wf=63)
    # 初始化权重参数
    model = weight_init_kaiming(model)
    # 加速
    model = model.cuda()
    # 准备数据集
    datasets = {
        "JPEGImages" : "*.jpg"
    }
    dirpath = "./TestMethods/data"
    train_paths = np.array(sorted(Tool.get_gt_image(dirpath, datasets)))
    traindatasets = SimulateTrain(im_list=train_paths, length=64, patch_size=128, peak=20)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print("开始训练")
    Train(model, traindatasets, optimizer, criterion, model_dir, batch_size, epochs)
    print("结束训练")

if __name__ == '__main__':
    model_dir = "F:\\Program File\\DataSpell\\Projetcs\\ImageDenoising\\ConvolutionalPoissonNoise\\dsProject\\Models"
    main(model_dir)
