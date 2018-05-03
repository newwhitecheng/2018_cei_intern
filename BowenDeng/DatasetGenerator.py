import os
from PIL import Image

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------

class DatasetGenerator(Dataset):

    # --------------------------------------------------------------------------------

    def __init__(self, pathImageDirectory, pathDatasetFile, transform):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        # ---- Open file, get image paths and labels

        with open(pathDatasetFile, "r") as f:
        # ---- get into the loop
            line = True

            while line:

                line = f.readline()

                # --- if not empty
                if line:
                    lineItems = line.split()

                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]

                    self.listImagePaths.append(imagePath)
                    self.listImageLabels.append(imageLabel)

    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        pass

    # --------------------------------------------------------------------------------

    def __len__(self):

        return len(self.listImagePaths)

# --------------------------------------------------------------------------------
