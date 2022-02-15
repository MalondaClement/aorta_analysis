#
# main.py
#
# Clément Malonda
#

import time
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.irm_seg import IRM_SEG
from models.utils import get_model

if __name__ == "__main__" :

    epochs = 40

    train = IRM_SEG(images_dir="../RawData/Training/img", labels_dir="../RawData/Training/img")

    train_dataloader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)

    start = time.time()
    for epoch in range(epochs) :
        print("Epoch {}/{}".format(epoch+1, epochs))

        for i, data in enumerate(train_dataloader, 0) :
            print("Batch {}/{}".format(i+1, int(len(train)/4)))
        break

    end = time.time()
    print("Training done in {} s".format(end - start))
