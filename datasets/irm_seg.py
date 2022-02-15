#
# datasets/irm_seg.py
#
# Clément Malonda
#

import os
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset

class IRM_SEG(Dataset) :
    def __init__(self, images_dir, labels_dir, transform=None, target_transform=None) :
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images_list = os.listdir(self.images_dir)
        self.labels_list = os.listdir(self.labels_dir)

        # TEMP:
        nb_slices_list = list()
        for img in self.images_list :
            image = nib.load(os.path.join(self.images_dir, img))
            nb_slices_list.append(image.shape[2])

        print(nb_slices_list)
        # print(nb_slices_list/np.min(nb_slices_list))
        # print([int(i) for i in nb_slices_list/np.min(nb_slices_list)])

    def __len__(self) :
        return len(self.images_list)

    # TODO: Get the same number of slice each time
    def __getitem__(self, idx) :
        image = nib.load(os.path.join(self.images_dir, self.images_list[idx]))
        image = image.get_fdata()[:,:,:85]

        label = nib.load(os.path.join(self.labels_dir, self.labels_list[idx]))
        label = label.get_fdata()[:,:,:85]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
