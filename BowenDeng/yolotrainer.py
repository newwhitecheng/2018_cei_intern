

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DatasetGenerator import DatasetGenerator


# --------------------------------------------------------------------------------

class YoloTrainer():

    def train(self, pathDirData, pathFileTrain, pathFileVal, trBatchSize, trMaxEpoch):


        model = torch.nn.DataParallel(model).cuda()

        # -------------------- DATA ARGUMENT??
        transformList = []
        transformSequence = transforms.Compose(transformList)

        # -------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain,
                                        transform=transformSequence)
        datasetVal = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal,
                                      transform=transformSequence)

        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True, num_workers=24,
                                     pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24,
                                   pin_memory=True)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        # need to know what is used, to be done
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, mode='min')

        # -------------------- SETTINGS: LOSS
        # need to know what is used, to be done
        # loss =


        # ---- TRAIN THE NETWORK

        lossMIN = 10000
        for epoch in range(0, MaxEpoch):

            self.epochTrain(model, dataLoaderTrain, optimizer, loss)
            lossVal, losstensor = self.epochVal(model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, loss)

            # scheduler.step(losstensor.data[0])

            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print('Epoch [' + str(epochID + 1) + '] [save] [loss={}]'.format(lossVal))
            else:
                print('Epoch [' + str(epochID + 1) + '] [----] [loss={}]'.format(lossVal))

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

        losstensorMean = 0

        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda(async=True)

            varInput = torch.autograd.Variable(input, volatile=True)
            varTarget = torch.autograd.Variable(target, volatile=True)
            varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor

            lossVal += losstensor.data[0]
            lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm

        return outLoss, losstensorMean
