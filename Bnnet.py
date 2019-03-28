
from torch import nn
from torch.nn import functional as F

class NetBN1Nopool(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self,num_classes):
        super(NetBN1Nopool, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(3, 6, 5,stride=2)  # 卷积层
        self.bn1=nn.BatchNorm2d(6)#添加层
        self.conv2 = nn.Conv2d(6, 16, 5,stride=2)  # 卷积
        self.bn2=nn.BatchNorm2d(16)#添加层
        self.fc1 = nn.Linear(16 * 13* 13, 120)  # 全连接层

        self.bn3=nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4=nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)  # 输出

    def forward(self, x):  # 前向传播

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x=x.view(x.size()[0],-1)
        x=F.relu(self.bn3(self.fc1(x)))
        x=F.relu(self.bn4(self.fc2(x)))
        x=self.fc3(x)
        return x

class NetBN1(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self,num_classes):
        super(NetBN1, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.bn1=nn.BatchNorm2d(6)#添加层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.poo2=nn.MaxPool2d(2,2)
        self.bn2=nn.BatchNorm2d(16)#添加层
        self.fc1 = nn.Linear(16 * 13* 13, 120)  # 全连接层
        #self.dropout1=nn.Dropout()#添加层
        self.bn3=nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        #self.dropout2=nn.Dropout()#添加层
        self.bn4=nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)  # 输出


    def forward(self, x):  # 前向传播
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.poo2(F.relu(self.bn2(self.conv2(x))))
        x=x.view(x.size()[0],-1)
        x=F.relu(self.bn3(self.fc1(x)))
        x=F.relu(self.bn4(self.fc2(x)))
        x=self.fc3(x)
        return x

class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self,num_classes):
        super(Net, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.poo2=nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 13* 13, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 输出
    def forward(self, x):  # 前向传播
        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x