import os
import numpy 
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class nettest1(nn.Module):
    def __init__(self):
        super(nettest1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x,count):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        temp1=x
        temp1=temp1.detach().numpy()
        #a = torch.FloatTensor(2,3)
        #print(a.type())
        #print (a.numpy())
        
        #print(temp1)
        print(numpy.shape(temp1))
        print(type(temp1))
        val=0
        k=0
        res=0
        tmp=[]
        img=temp1[0][1]
        img=img/3
        img=img*255
        img=img.astype('int32')
        print(numpy.shape(img))
        print(type(img))
        for i in range(256):  
            tmp.append(0)  
        for i in range(len(img)):  
            for j in range(len(img[i])):  
                val = img[i][j]  
                tmp[val] = float(tmp[val] + 1)  
                k =  float(k + 1)  
        for i in range(len(tmp)):  
            tmp[i] = float(tmp[i] / k)  
        for i in range(len(tmp)):  
            if(tmp[i] == 0):  
                res = res  
            else:  
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))  
        print (res)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #temp1=x
        #temp2=temp2.detach.numpy()
        #print(temp2)
        #print(temp2.size())
        count=count+1
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1),count
        
