import pickle
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import os
import imageio
import seaborn as sn
import convertGIF

NET_NAME = 'LeNet5'
ARCH = '6-16-120'
DATASET = 'MNIST'
ACTIVATION = 'tanh'

sample_id = 0

SavePic = True
Epoch = "00000005"

cur_dir = "rawdata/%s_%s%s_%s/"%(NET_NAME, DATASET, ACTIVATION, ARCH)

#For LeNet Structure Only

#Correlation
epochfile_List = os.listdir(cur_dir)
epochfile_List.sort(key=lambda x:int(x[-4:]))

for sample_id in range(10):
    print("********* Processing " + str(sample_id) +" Sample ***************" )

    if not os.path.exists("img_out/HM"+str(sample_id)):
        os.makedirs("img_out/HM"+str(sample_id))
        
    corr = []

    for epochfile in epochfile_List:
        with open(cur_dir + epochfile, 'rb') as f:
            d = pickle.load(f)
        print("Processing " + epochfile)
        epochs = int(epochfile[-4:])
        #epoch.append(epochs)

        activity = d['data']['activity_tst']
        num_filters = [activity[0].shape[3], activity[1].shape[3]]
        corr.append([])
        num_samples = activity[0].shape[0]

        for lndx in range(len(num_filters)):
            corr[len(corr)-1].append(np.zeros(shape=(num_filters[lndx], num_filters[lndx])))
            for i in range(num_filters[lndx]):
                for j in range(num_filters[lndx]):
                        corr[len(corr) - 1][lndx][i][j] += np.correlate(activity[lndx][sample_id, :,:, i].reshape(-1), activity[lndx][sample_id, :,:, j].reshape(-1))

            #Save Output Of the Feature Map
            f, ax = plt.subplots(figsize = (5, 1.3) if (lndx==0) else (8, 2.2))
            num_rows = int(np.ceil(num_filters[lndx] / (6 +  2 * lndx)))
            for i in range(num_filters[lndx]):
                plt.subplot(num_rows, 6 +  2 * lndx, i + 1)
                plt.imshow(activity[lndx][sample_id, :, :, i])
                plt.xticks([])
                plt.yticks([])
            plt.suptitle("ConvLayer " + str(lndx) + "   Epoch "+ str(epochs))
            f.savefig("img_out/HM"+str(sample_id) + "/LeNet5_Conv" + str(lndx + 1) + "Act" + epochfile + '.png')
            plt.close('all')
            #corr[lndx] = corr[lndx] / num_samples
            # plt.imsave("img_out/LeNet5_Conv" + str(lndx+1) + epochfile + '.png',corr[len(corr) - 1][lndx],
            #            cmap=plt.cm.plasma, vmin=-max(abs(corr[len(corr) - 1][lndx].reshape(-1))), vmax= max(abs(corr[len(corr) - 1][lndx].reshape(-1))))
            # img = PIL.Image.fromarray(imageio.imread("img_out/LeNet5_Conv" + str(lndx+1) + epochfile + '.png'))
            # img = img.resize((img.width * 20, img.height*20))
            # img = draw_epoch_text(img, epochfile)
            # imageio.imsave("img_out/LeNet5_Conv" + str(lndx+1) + epochfile  + '.png', np.array(img))
   
    print("Drawing Heat Map...")
    for i in range(len(corr)):
        num_filters = [6, 16]
        for lndx in range(len(num_filters)):
            f, ax= plt.subplots(figsize=(10, 8))
            ax.set_title("Correlation of Feature Maps in Conv Layer " + str(lndx) , fontsize=18, position=(0.5,1.05))
            ax.xaxis.tick_top()
            hm = sn.heatmap(corr[i][lndx], vmin=-max(np.absolute(corr[len(corr) - 1][lndx].reshape(-1))),
                            vmax=max(np.absolute(corr[len(corr) - 1][lndx].reshape(-1))), annot=True, annot_kws={'size': 9 + 3*int( 1 - lndx)})
            plt.xlabel(epochfile_List[i],fontsize=12)
            hm.get_figure().savefig("img_out/HM"+str(sample_id) + "/LeNet5_Conv" + str(lndx + 1) + epochfile_List[i] + '.png')
            plt.close('all')

    print("Generating GIF...")
    convertGIF.saveGif("img_out/HM"+str(sample_id), "img_out/HM"+str(sample_id) + "conv1.gif",  0.2 , 1, 'Conv1epoch')
    convertGIF.saveGif("img_out/HM"+str(sample_id), "img_out/HM"+str(sample_id) + "conv2.gif",  0.2 , 1, 'Conv2epoch')
    convertGIF.saveGif("img_out/HM"+str(sample_id), "img_out/HM"+str(sample_id) + "conv1act.gif",  0.2 , 1, 'Conv1Act')
    convertGIF.saveGif("img_out/HM"+str(sample_id), "img_out/HM"+str(sample_id) + "conv2act.gif",  0.2 , 1, 'Conv2Act')
