
# 1. ClassAwareDenoising
[Class-Aware Fully Convolutional Gaussian and Poisson Denoising](https://ieeexplore.ieee.org/abstract/document/8418389)

The authors say that this work can denoise **Gaussian noise** and **Poisson Noise**,
Because I am concentrating on Poisson Denoising, *I only train this neural network in Image Sets which include Poisson Noise.*

**I set the peak value is 4.0.**

The denoising effect is barely satisfactory.

![](https://picgo-bed-1307807721.cos.ap-nanjing.myqcloud.com/markdown/20220401114724.png)
# *2.How to run it*
## 2.1 Edit the file named "option.py"
## 2.2 python train.py
## 2.3 python TestSimulate.py

# Note
This should be the first formal reproduction of my entry into the field of deep learning, in which the code structure is chaotic, and I will gradually optimize my programming style in the future learning process.

# 2. [New Version (pytorch)](https://github.com/aixiaoairen/ClassicalPoissonDenoising/tree/master/ClassAware)
