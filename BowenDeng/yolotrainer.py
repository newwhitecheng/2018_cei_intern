import random
from PIL import Image
import torch.optim as optim

import torchvision.transforms as transforms
from DatasetGenerator import DatasetGenerator


# --------------------------------------------------------------------------------

class YoloTrainer():

    def train(self, pathDirData, pathFileTrain, pathFileVal, trBatchSize, trMaxEpoch):

        model = torch.nn.DataParallel(model).cuda()

        # -------------------- DATA ARGUMENT??
        transformList = []
        transformList.append(transforms.ToTensor())
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
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

        # -------------------- SETTINGS: LOSS
        # need to know what is used, to be done
        # loss =

        # ---- TRAIN THE NETWORK
        #MaxEpoch=?
        lossMIN = 10000
        for epoch in range(0, MaxEpoch):
            scheduler.step()
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


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if (random.randint(1, 10000) % 2):
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    flip = random.randint(1, 10000) % 2
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    return img
