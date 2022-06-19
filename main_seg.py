#
# main_seg.py
#
# Clément Malonda
#

import os
import time
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.irm_seg import IRM_SEG, FEATURE_DICT
from models.utils import get_2d_segmentation_model
from utils.metrics import compute_iou
from utils.plots import plot_curves

if __name__ == "__main__" :

    one_class_mode = ""
    while one_class_mode not in ["yes", "no"]:
        one_class_mode = input("Do you want to use one class mode ? {} : ".format(["yes", "no"]))

    if one_class_mode == "yes":
        is_unique_label_mode = True
        num_classes = 2
    else:
        is_unique_label_mode = False
        num_classes = 14

    if is_unique_label_mode:
        selected_label = ""
        while selected_label not in FEATURE_DICT.keys():
            print(FEATURE_DICT)
            selected_label = input("Select a feature value in the list: ")
        selected_label = int(selected_label)
    else:
        selected_label = -1

    epochs = 40
    batch_size = 16
    model_name = "DeepLabV3_MobileNetV3"
    save_path = model_name + "-" +date.today().isoformat() + "-" + str(int(time.time()))

    if not os.path.isdir("../saves"):
        os.makedirs("../saves")
    os.makedirs(os.path.join("../saves",save_path))

    model = get_2d_segmentation_model(model_name, num_classes=num_classes)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    train = IRM_SEG(images_dir="../RawData/Training/img", labels_dir="../RawData/Training/label", is_unique_label_mode=is_unique_label_mode, label_value=selected_label, transform=ToTensor())
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    val = IRM_SEG(images_dir="../RawData/Evaluating/img", labels_dir="../RawData/Evaluating/label", is_unique_label_mode=is_unique_label_mode, label_value=selected_label, transform=ToTensor())
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

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join("../saves", save_path, "checkpoint.pth.tar"))

    end = time.time()
    print("Training done in {} s".format(end - start))

    plot_curves(title="Learning curves", datas=[train_loss, val_loss, train_iou], legends=["Train loss", "Val loss", "mIoU"], path=os.path.join("../saves", save_path, "learning_curves.png"))
    print("Train loss: {}".format(train_loss))
    print("Val loss: {}".format(val_loss))
    print("IoU: {}".format(train_iou))
