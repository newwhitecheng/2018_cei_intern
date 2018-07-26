import os
import numpy as np
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from datasetgenerator import datasetgenerator
import torch.nn.functional as func
from mnistmodels import nettest1

#-------------------------------------------------------------------------------
class mnisttrainer():
    def deploy(self,pathdirdata,pathfiledeploy,pathmodel):
        
        model=nettest1()
        #model=torch.nn.DataParallel(model).cuda()
        #modelcheckpoint=torch.load(pathmodel)
        model.load_state_dict(torch.load('model1.pth'))
        normalize=transforms.Normalize((0.1307,), (0.3081,))
        transformSequence=transforms.Compose([transforms.ToTensor,transforms.Normalize((0.1307,), (0.3081,))])
        datasetTest = datasetgenerator(pathImageDirectory=pathdirdata, pathDatasetFile=pathfiledeploy, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=1, num_workers=1,shuffle=False)
        aa=0.0
        test_loss=0
        correct=0
        model.eval()
        torch.cuda.set_device(1)
        print(123)
        for i, (input, target) in enumerate(dataLoaderTest):
            
            output,aa=model(x=input,count=aa)
            print(aa)
            #test_loss += func.nll_loss(output, target, size_average=False).item() # sum up batch loss
            #pred = output.max(1)[1] # get the index of the max log-probability
            #correct += pred.eq(target).sum().item()

        #test_loss /= len(data_loader.dataset)
        #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #    test_loss, correct, len(dataLoaderTest.dataset),
        #    100. * correct / len(dataLoaderTest.dataset)))

