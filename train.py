import os
import torch
import numpy as np
import DataTools as Tool
import option as op
import torch.optim as optim
from network import CADET
from Datasets import SimulateTrain
from torch.utils.data import DataLoader
from utils import weight_init_kaiming
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 加载配置文件
args = op.option()

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

def main():
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = CADET(in_channels=args.in_channel, wf=args.wf)
    if args.resume == False:
        # 初始化权重参数
        model = weight_init_kaiming(model)
        model_dir = args.model_dir
    else:
        path = os.path.join(args.model, args.checkpoint)
        checkpoint = torch.load(path)
        state_dict_one = model.state_dict()
        for name, value in checkpoint["model_state_dict"].items():
            state_dict_one[name] = value
        model.load_state_dict(state_dict_one)
        model_dir = args.model_resume_dir
    # 检查存储路径是否存在
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    # 加速
    if torch.cuda.is_available():
        model = model.cuda()
    # 准备数据集
    datasets = args.datasets
    dirpath = args.data_dir
    train_paths = np.array(sorted(Tool.get_gt_image(dirpath, datasets)))
    # print(len(train_paths))
    traindatasets = SimulateTrain(im_list=train_paths, length=args.batch_size, patch_size=args.patch_size, peak=args.peak)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print("开始训练")
    Train(model, traindatasets, optimizer, criterion, model_dir, batch_size, epochs)
    print("结束训练")

if __name__ == '__main__':
    main()
