
# Imporve-lenet
1.基于openpose骨架图的分类算法模型是基于lenet做的改进优化，命名为Bnnet，具体改进包括：

（1）数据集扩增：transforms.RandomHorizontalFlip()

（2）求输入图像的均值、方差做白化处理：transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649), std=(0.15804708, 0.16410254, 0.0643605)

（3）增加了BatchNorm层，极大加速模型收敛，减少训练时间，同时有抑制过拟合的作用[不要使用dropout]

（4）使用目前最新最好的优化算法：Adabound，加速收敛的同时，模型更加稳定；

(5)学习率衰减机制：CosineAnnealingLR，可以有效提升模型精度。

(6)图像resize(64)，扩大图像，可以防止过拟合，同时特征更清晰；

(7)理解数据很重要！理解数据很重要！！理解数据很重要！！！

2.除了改进lenet之外，还重构了Mobilenet和resnet网络模型（在Bnnet满足性能要求的条件下，不建议使用这两个模型，相比较Bnnet而言速度较慢）

3.代码说明：
（1）Bnnet.py,mobilenet.py,resnet.py是三个不同的网络模型，可自行切换使用：

if torch.cuda.is_available()==True:

    model=Net(num_classes=9).to('cuda')
    #model=MobileNetV2(n_class=9).to('cuda')
    #model=resnet50(num_class=9,pretrained=False).to('cuda')
    print(model)
    print("cuda:0")
else:

    model=Net(num_classes=9).to('cpu')
    #model = MobileNetV2(n_class=9).to("cpu")
    #model=resnet50(num_class=9).to("cpu")
    print("cpu")

不同模型需修改resize()和batch_size(),其他都不需要修改。

(2)train.py是训练代码，直接运行，不需要其他参数。

(3)rename.py和resize.py是前期对图像的处理

(4)utils.py主要使用了混淆矩阵的计算和绘制混淆矩阵图

4.可视化工具：

进入终端命令：cd 到目录下，然后tesnorboard --logdir=Result

5.由于涉及公司机密，数据暂不提供。

可加我我微信：

