#
# utils/metrics.py
#
# Clément Malonda

import numpy as np
#

def compute_iou(outputs, targets):
    intersection = (outputs & targets).float().sum((1, 2))
    union = (outputs | targets).float().sum(1, 2)

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded
