import torch
import os
import random
from PIL import Image
from xml.dom.minidom import parse as parse
from torch.utils.data import Dataset


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
    if random.randint(1, 10000) % 2:
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, hue, saturation, exposure):
    flip = random.randint(1, 10000) % 2
    sized = img.resize(shape)
    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    return img


class DatasetGenerator(Dataset):

    def __init__(self, path_image, path_imagefile, path_bndboxfile, transform):
        """
            fetch the data and labels
            :param path_image: path to the dir containing the images
            :param path_imagefile: path to the dir of image files, i.e., *.txt
            :param path_bndboxfile: path to the dir of bb files, i.e, *.xml
            :param transform: flag for using data argumentation or not
        """
        # -------------------- DATA ARGUMENT
        self.shape = 446
        self.hue = 0.1
        self.saturation = 1.5
        self.exposure = 1.5
        self.imagelist = []
        self.labellist = []
        self.transform = transform
        label_dir = os.listdir(path_bndboxfile)
        image_dir = os.listdir(path_imagefile)

        # read imagepath
        for file in image_dir:
            file_name = os.path.join(path_imagefile, file)
            with open(file_name) as f:
                lines = f.readlines()
                for line in lines:
                    image_name = line.split()[0] + '.JPEG'
                    image = os.path.join(path_image, image_name)
                    self.imagelist.append(image)

        # read imagelabel, i.e, (name, xmin, xmax, ymin, ymax)
        for file in label_dir:
            if file.split('.')[1] == 'xml':
                file_name = os.path.join(path_bndboxfile, file)
                with open(file_name) as f:
                    xml_tree = parse(f).documentElement
                    objects = xml_tree.getElementsByTagName('object')
                    for object in objects:
                        label = []
                        name = object.getElementsByTagName('name')[0]
                        label.append(name.childNodes[0].data)
                        bndbox = object.getElementsByTagName('bndbox')[0]
                        for node in bndbox.childNodes:
                            if node.nodeType == node.ELEMENT_NODE:
                                label.append(node.childNodes[0].data)
                        self.labellist.append(label)
            else:
                print('Expect files in xml format. but get {}'.format(file.split('.')[1]))

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        image_path = self.imagelist[index]
        image_data = Image.open(image_path)
        image_label = torch.LongTensor([int(x) for x in self.labellist[index]])
        if self.transform:
            image_data = data_augmentation(image_data, self.shape, self.hue, self.saturation, self.exposure)
        return image_data, image_label

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.imagelist)
