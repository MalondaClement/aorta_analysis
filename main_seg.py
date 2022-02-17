#
# main_seg.py
#
# Clément Malonda
#

import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.irm_seg import IRM_SEG
from models.utils import get_2d_segmentation_model

if __name__ == "__main__" :

    epochs = 4

    batch_size = 2

    model = get_2d_segmentation_model("DeepLabV3_MobileNetV3", num_classes=14)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    train = IRM_SEG(images_dir="../RawData/Training/img", labels_dir="../RawData/Training/label", transform=ToTensor(), target_transform=ToTensor())

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start = time.time()

    for epoch in range(epochs) :
        print("Epoch {}/{}".format(epoch+1, epochs))

        with torch.set_grad_enabled(True) :
            for i, data in enumerate(train_dataloader, 0) :
                print("Batch {}/{}".format(i+1, int(len(train)/batch_size)))

                inputs, labels = data
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()

                optimizer.zero_grad()

                outputs = model(inputs)

                outputs = outputs["out"]

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

    end = time.time()
    print("Training done in {} s".format(end - start))
