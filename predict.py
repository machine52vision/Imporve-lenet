import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from Bnnet import Net,NetBN1Nopool
import torch.nn.functional as F
from PIL import Image
import os
import time

#img_path='D:\\codyy_data\\all_openpose_data\\temp_image\\test'
img_path='D:\\codyy_data\\all_openpose_data\\temp_image\\test\\Gill'
if torch.cuda.is_available()==True:
    model=NetBN1Nopool(num_classes=9).to('cuda')
    #model=MobileNetV2(n_class=9).to('cuda')
    #model=resnet50(num_class=9,pretrained=False).to('cuda')
    print(model)
    print("cuda:0")
else:
    model=NetBN1Nopool(num_classes=9).to('cpu')
    #model = MobileNetV2(n_class=9).to("cuda")
    #model=resnet50(num_class=9).to("cuda")
    print("cpu")
model.load_state_dict(torch.load('net_params_BN1_Nopool.pkl9'))
model.eval()

data_transforms = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649),
                                                    std=(0.15804708, 0.16410254, 0.0643605))])
start = time.time()
for file in os.listdir(img_path):

    imagepath=img_path+"\\"+file
    print(imagepath)
    image = Image.open(imagepath)
    imgblob = data_transforms(image).unsqueeze(0)
    torch.no_grad()
    pre = F.softmax(model(imgblob.cuda()))
    prediction_label=torch.argmax(pre).item()
    prob=torch.max(pre).item()
    print("prob=",prob)
    print("prediction_label=",prediction_label)
    print("*************************next_image**********************")

print("time=",time.time()-start)
"""
data = ImageFolder(img_path,
                   transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.034528155, 0.033598177, 0.009853649),
                                                                      std=(0.15804708, 0.16410254, 0.0643605))]))
test_loader = DataLoader(data, batch_size=2000)
for i,(data,real_label) in enumerate(test_loader):
    torch.no_grad()
    prediction = F.softmax(model(data.cuda()))

    #print("real_label=",real_label.cuda())
    #print('prediction_result=',torch.argmax(prediction,dim=1))
    #print(torch.eq(real_label.cuda(),torch.argmax(prediction, dim=1)))
    sum=torch.sum(torch.eq(real_label.cuda(),torch.argmax(prediction, dim=1)))
    print('right=',sum)
    print( 'probability=',sum.float()/len(real_label))

    print("*****************************next-batch************************")

"""
