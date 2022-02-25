#
# datasets/irm_seg.py
#
# Clément Malonda
#

import os
import numpy as np
import nibabel as nib

from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset

class IRM_SEG(Dataset) :
    def __init__(self, images_dir, labels_dir, transform=None, target_transform=None) :
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images_vol_list = os.listdir(self.images_dir)
        self.labels_vol_list = os.listdir(self.labels_dir)

        self.images_list = list()
        self.labels_list = list()

        self.scaler = scaler = MinMaxScaler()

        for img in self.images_vol_list :
            image = nib.load(os.path.join(self.images_dir, img))
            for i in range(image.shape[2]):
                self.images_list.append((img, i))
                self.labels_list.append(("label"+img[3:], i))
        print("Total slices : {}".format(len(self.images_list)))

    def __len__(self) :
        return len(self.images_list)

    def __getitem__(self, idx) :
        image = nib.load(os.path.join(self.images_dir, self.images_list[idx][0]))
        tmp = image.get_fdata()[:,:,self.images_list[idx][1]]
        print(np.min(tmp))
        print(np.max(tmp))
        tmp = self.scaler.fit_transform(tmp)
        print(np.min(tmp))
        print(np.max(tmp))
        # image = image.get_fdata()[:,:,self.images_list[idx][1]]
        image = np.array([tmp, tmp, tmp])
        image = image.transpose(1, 2, 0)

        label = nib.load(os.path.join(self.labels_dir, self.labels_list[idx][0]))
        label = label.get_fdata()[:,:,self.labels_list[idx][1]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
