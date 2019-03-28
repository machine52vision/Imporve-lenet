import torch,adabound
from tensorboardX  import SummaryWriter
from torchvision.datasets import ImageFolder
from torch import nn,optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import torch
import numpy as np
import os
from utils import validate,show_confMat
from mobilenetv2 import MobileNetV2
from Bnnet import Net,NetBN1Nopool
from resnet import resnet50

torch.cuda.set_device(0)

# log 保存日志
result_dir = 'Result'
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
classes_name = ['Discuss', 'Gill', 'Hand_high_put', 'Hand_low_put', 'Listen', 'Read', 'Relax', 'Sleepy', 'Stand']
writer = SummaryWriter(log_dir=log_dir)


"""
train_data_path='D:\\BehaviorAnalysis\\trunk\src\\poseEstimation1.0.1\\data\\temp_image\\img'
valid_data_path="D:\\BehaviorAnalysis\\trunk\src\\poseEstimation1.0.1\\data\\temp_image\\img_test"
train_set=ImageFolder(root=train_data_path,transform=transforms.Compose(
    [transforms.Resize(64),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649), std=(0.15804708, 0.16410254, 0.0643605))
     ]))
valid_set=ImageFolder(root=valid_data_path,transform=transforms.Compose([transforms.Resize(64),transforms.ToTensor(),
transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649),std=(0.15804708, 0.16410254, 0.0643605))
                                                                         ]))
train_loader=DataLoader(dataset=train_set,batch_size=1024,shuffle=True)
valid_loader=DataLoader(dataset=valid_set,batch_size=1024)
"""

#Loading in the dataset
#img_dir='D:\\codyy_data\\all_openpose_data\\temp_image\\img'
img_dir='D:\\codyy_data\\all_openpose_data\\temp_image2\\img'
data = ImageFolder(img_dir,transform=transforms.Compose(
    [transforms.Resize(64), #Bnnet 改成64,mobilenet,resnet改成224
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649), std=(0.15804708, 0.16410254, 0.0643605))
     ]))
# number of subprocesses to use for data loading
# percentage of training set to use as validation
valid_size = 0.2
test_size = 0.1
# obtain training indices that will be used for validation
num_train = len(data)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_split = int(np.floor((valid_size) * num_train))
test_split = int(np.floor((valid_size+test_size) * num_train))
valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

print(len(valid_idx), len(test_idx), len(train_idx))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(data, batch_size=1024,sampler=train_sampler)#Bnet batch_size=1024   mobilenet batch_size=32  resnet  batch_size=8
valid_loader = DataLoader(data, batch_size=1024,sampler=valid_sampler)#Bnet batch_size=1024   mobilenet batch_size=32  resnet  batch_size=8
test_loader = DataLoader(data, batch_size=1024,sampler=test_sampler)#Bnet batch_size=1024     mobilenet batch_size=32  resnet  batch_size=8

if torch.cuda.is_available()==True:
    model=NetBN1Nopool(num_classes=9).cuda()
    #model=MobileNetV2(n_class=9).to('cuda')
    #model=resnet50(num_class=9,pretrained=False).to('cuda')
    print(model)
    print("cuda:0")
else:
    model=NetBN1Nopool(num_classes=9).cuda()
    #model = MobileNetV2(n_class=9).to("cuda")
    #model=resnet50(num_class=9).to("cuda")
    print("cpu")

criterion=nn.CrossEntropyLoss()
optimizer=adabound.AdaBound(model.parameters(),lr=1e-3,final_lr=0.1)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)

epochs=10
for epoch in range(epochs):
    
    model.train()
    loss_sigma=0.0
    correct=0.0
    total=0.0
    scheduler.step()
    for i,(data,label) in enumerate(train_loader):
        data=data.cuda()
        label=label.cuda()
        optimizer.zero_grad()
        outputs=model(data)
        loss=criterion(outputs,label)
        loss.backward()
        optimizer.step()
        _,predicted=torch.max(outputs.data,dim=1)
        total += label.size(0)
        correct += (predicted == label).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()
        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10== 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}"\
                  .format(epoch + 1, epochs, i + 1, len(train_loader), loss_avg, correct / total))

            # 记录训练loss
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # 每个epoch，记录梯度，权值
    for name, layer in model.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
    
        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 1== 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        model.eval()
        for i, (data,label) in enumerate(valid_loader):
            # forward
            data=data.to("cuda")
            label=label.to("cuda")
            outputs = model(data)
            outputs.detach_()
            # 计算loss
            loss = criterion(outputs, label)
            loss_sigma += loss.item()
            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor
            # 统计混淆矩阵
            for j in range(len(label)):
                cate_i = label[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0
        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

torch.save(model, 'net.pkl' + str(epochs))  # 保存整个神经网络的结构和模型参数
torch.save(model.state_dict(), 'net_params_BN1_Nopool.pkl' + str(epoch))  # 只保存神经网络的模型参数

conf_mat_train, train_acc = validate(model.cpu(), train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(model.cpu(), test_loader, 'test', classes_name)
show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'test', log_dir)
