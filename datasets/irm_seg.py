#
# datasets/irm_seg.py
#
# Clément Malonda
#

import os
import cv2
import numpy as np
import nibabel as nib

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset

class IRM_SEG(Dataset) :
    def __init__(self, images_dir, labels_dir, is_unique_label_mode=False, label_value=-1, transform=None, target_transform=None) :
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images_vol_list = os.listdir(self.images_dir)
        self.labels_vol_list = os.listdir(self.labels_dir)

        self.images_list = list()
        self.labels_list = list()

        self.scaler = MinMaxScaler()

        self.is_unique_label_mode = is_unique_label_mode
        self.label_value = label_value

        for img in self.images_vol_list :
            image = nib.load(os.path.join(self.images_dir, img))
            for i in range(int(image.shape[2]/3), int((2*image.shape[2])/3)):
                self.images_list.append((img, i))
                self.labels_list.append(("label"+img[3:], i))
        print("Total slices : {}".format(len(self.images_list)))

    def __len__(self) :
        return len(self.images_list)

    def __getitem__(self, idx) :
        image = nib.load(os.path.join(self.images_dir, self.images_list[idx][0]))
        tmp = image.get_fdata()[:,:,self.images_list[idx][1]]
        tmp = self.scaler.fit_transform(tmp)
        tmp = cv2.resize(tmp, (96, 96))
        image = np.array([tmp, tmp, tmp])
        image = image.transpose(1, 2, 0)

        label = nib.load(os.path.join(self.labels_dir, self.labels_list[idx][0]))
        label = label.get_fdata()[:,:,self.labels_list[idx][1]]
        label = cv2.resize(label, (96, 96))

        if self.is_unique_label_mode:
            label = np.where(label != self.label_value, 0, label)
            label = np.where(label == self.label_value, 1, label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
