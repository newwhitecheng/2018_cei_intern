import torch
import os
from PIL import Image
from xml.dom.minidom import parse as parse

from torch.utils.data import Dataset


class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, path_image, path_imagefile, path_bndboxfile, transform):

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
        image_label = torch.LongTensor(self.labellist[index])
        if self.transform is not None:
            image_data = self.transform(image_data)

        return image_data, image_label

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.imagelist)
