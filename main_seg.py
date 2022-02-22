#
# main_seg.py
#
# Clément Malonda
#

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.irm_seg import IRM_SEG
from models.utils import get_2d_segmentation_model
from utils.metrics import compute_iou

if __name__ == "__main__" :

    epochs = 4
    batch_size = 2
    model_name = "DeepLabV3_MobileNetV3"
    save_path = model_name + date.today().isoformat() + "-" + str(int(time.time()))

    if not os.path.isdir("../saves"):
        os.makedirs("../saves")
    os.makedirs(os.path.join("../saves",save_path))

    model = get_2d_segmentation_model(model_name, num_classes=14)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    train = IRM_SEG(images_dir="../RawData/Training/img", labels_dir="../RawData/Training/label", transform=ToTensor())
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    val = IRM_SEG(images_dir="../RawData/Evaluating/img", labels_dir="../RawData/Evaluating/label", transform=ToTensor())
    val_dataloader = DataLoader(val, batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loss = list()
    val_loss = list()

    train_iou = list()

    start = time.time()

    for epoch in range(epochs) :
        print("Train for epoch {}/{}".format(epoch+1, epochs))

        train_loss_epoch = list()
        val_loss_epoch = list()

        train_iou_epoch = list()

        model.train()
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

                train_loss_epoch.append(loss.item())

                train_iou_epoch.append(compute_iou(torch.argmax(outputs, 1).cpu().numpy(), labels.cpu().numpy()))

                del inputs
                del labels
                torch.cuda.empty_cache()

        train_loss.append(np.mean(train_loss_epoch))
        print("Train loss for epoch {} {}".format(epoch, train_loss[-1]))
        train_iou.append(np.mean(train_iou_epoch))
        print("IoU for epoch {} {}".format(epoch, train_iou[-1]))

        #TEST : Implement validation for each epoch
        print("Eval for epoch {}/{}".format(epoch+1, epochs))
        model.eval()
        with torch.no_grad() :
            for i, data in enumerate(val_dataloader, 0) :
                print("Batch {}/{}".format(i+1, int(len(val)/batch_size)))
                inputs, labels = data
                inputs = inputs.float().cuda()
                labels = labels.long().cuda()

                outputs = model(inputs)
                outputs = outputs["out"]

                loss = criterion(outputs, labels)

                val_loss_epoch.append(loss.item())

        val_loss.append(np.mean(val_loss_epoch))
        print("Val loss for epoch {} {}".format(epoch, val_loss[-1]))

    end = time.time()
    print("Training done in {} s".format(end - start))
