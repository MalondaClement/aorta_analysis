#
# main.py
#
# Clément Malonda
#

import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.irm_seg import IRM_SEG
from models.utils import get_3d_segmentation_model

if __name__ == "__main__" :

    epochs = 1

    model = get_3d_segmentation_model("UNet3D", num_classes=14)

    train = IRM_SEG(images_dir="../RawData/Training/img", labels_dir="../RawData/Training/label")

    train_dataloader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    for epoch in range(epochs) :
        print("Epoch {}/{}".format(epoch+1, epochs))

        for i, data in enumerate(train_dataloader, 0) :
            print("Batch {}/{}".format(i+1, int(len(train)/4)))
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

    end = time.time()
    print("Training done in {} s".format(end - start))
