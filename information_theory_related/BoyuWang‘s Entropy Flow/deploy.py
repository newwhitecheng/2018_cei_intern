import os
import numpy as np
import time
import sys

from mnist_trainer import mnisttrainer


# --------------------------------------------------------------------------------

def main():
#    runTest()
#    runTrain()
    deploy()

def deploy():
    pathDirData = '/home/xg52/boyuwang/data/ultdata/mnistdata'
    pathFileDeploy = '/home/xg52/boyuwang/data/ultdata/mnistdata/mnist/test/newlabels.txt'
    pathModel = 'net1.pkl'
    Trainer = mnisttrainer()
    Trainer.deploy(pathDirData, pathFileDeploy, pathModel)






if __name__ == '__main__':
    
    main()


