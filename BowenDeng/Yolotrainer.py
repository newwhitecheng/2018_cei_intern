import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from . import DatasetGenerator
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------------
class Loss(nn.Module):
    """
    objectness score represents the probability an object is contained inside a bb.
    class confidence represent the probabilities of the dectected object belonging to
    particular class.
    define the confidence as Pr(obj) * IOU, if no obj exists, the confidence score = 0,
    otherwise it should equal the IOU
    C conditional class probabilitie s Pr(Classi|Obj). These probabilities are conditioned on
    the grid cell containing an obj.
    At test time, multiply the conditional class probabilities and the individual box confidence
    class probabilities * confidence = Pr(Classi) * IOU
    Increase bb coordinate loss by a factor of 5 and decrease the confidence loss for boxes that
    don't contain objs by a factor of 0.5
    """

    def __init__(self):
        super(Loss, self).__init__()
        self.loss = 0

    def forward(self, prediction, label, lamda_obj=5, lamda_noobj=0.5):
        indexer = torch.from_numpy(np.argsort(prediction.cpu.numpy()[:, 0]))
        prediction = prediction[indexer]
        row_label = label[0]
        # 假设label_name是class index
        label_name = []
        batch_id = -1
        self.loss = 0
        for row in range(prediction.size(0)):
            if batch_id != prediction[row][0]:
                batch_id = prediction[row][0]
                label_name = [(i, row_label[i]) for i in range(0, len(row_label), 5)]
            flag = 0
            for item in label_name:
                if prediction[row] == item[1]:
                    begin = item[0]
                    x_min, y_min = row_label[begin + 1], row_label[begin + 3]
                    x_max, y_max = row_label[begin + 2], row_label[begin + 4]
                    delta_x = (x_min + x_max) / 2 - (prediction[row][1] + prediction[row][3]) / 2
                    delta_y = (y_min + y_max) / 2 - (prediction[row][2] + prediction[row][4]) / 2
                    delta_h = torch.sqrt(y_max - y_min) - torch.sqrt(prediction[row][4] + prediction[row][2])
                    delta_w = torch.sqrt(x_max - x_min) - torch.sqrt(prediction[row][3] + prediction[row][1])
                    self.loss += torch.pow(delta_x, 2) + torch.pow(delta_y, 2)
                    self.loss += torch.pow(delta_w, 2) + torch.pow(delta_h, 2)
                    self.loss += torch.pow(prediction[row][5] - 1, 2)
                    self.loss += torch.pow(prediction[row][6] - 1, 2)
                    self.loss += torch.pow(prediction[row][1] - row_label)
                    flag += 1
                    break
            if flag != 1:
                self.loss += 2
        return self.loss


class YoloTrainer():

    def train(self, model, pathDirData, pathFileTrain, pathFileVal, pathBBTrain, pathBBVal, trBatchSize, trMaxEpoch):

        model = torch.nn.DataParallel(model).cuda()

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(path_image=pathDirData, path_imagefile=pathFileTrain,
                                        path_bndboxfile=pathBBTrain, transform=True)
        datasetVal = DatasetGenerator(path_image=pathDirData, path_imagefile=pathFileVal,
                                      path_bndboxfile=pathBBVal, transform=False)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=24,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24,
                                   pin_memory=True)
        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        # need to know what is used, to be done
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

        # -------------------- SETTINGS: LOSS
        loss = Loss()

        # ---- TRAIN THE NETWORK
        MaxEpoch = 1000
        lossMIN = 10000
        for epoch in range(0, MaxEpoch):
            scheduler.step()
            self.epochTrain(model, dataLoaderTrain, optimizer, loss)
            lossVal = self.epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, loss)
            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, 'm-' + '.pth.tar')
                print('Epoch [' + str(epoch + 1) + '] [save] [loss={}]'.format(lossVal))
            else:
                print('Epoch [' + str(epoch + 1) + '] [----] [loss={}]'.format(lossVal))

    # --------------------------------------------------------------------------------

    def epochTrain(self, model, dataLoader, optimizer, loss):

        model.train()

        for batchID, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput)

            lossvalue = loss(varOutput, varTarget)

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()

    # --------------------------------------------------------------------------------
    def epochVal(self, model, dataLoader, optimizer, scheduler, epochMax, loss):
        '''
        validation while training the network
        '''

        model.eval()

        lossVal = 0
        lossValNorm = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)
            varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            lossVal += losstensor.data[0]
            lossValNorm += 1

        losstensorMean = lossVal / lossValNorm
        return losstensorMean

