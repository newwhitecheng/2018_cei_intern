from PIL import Image
import numpy as np
import imageio
import os

def saveGif(srcpath, gifpath, interval, scaling = 1.0,  keyword=''):
    frames = []
    pngFile = os.listdir(srcpath)
    pngFile.sort(key=lambda x: int(x[-9:-4]))
    image_list = [os.path.join(srcpath, f) for f in pngFile]
    for image_name in image_list:
        if image_name.find(keyword) != -1:
            img = Image.fromarray(imageio.imread(image_name))
            img = img.resize((img.width* scaling, img.height* scaling))
            frames.append(np.array(img))

    imageio.mimsave(gifpath + '/', frames, 'GIF', duration = interval)
