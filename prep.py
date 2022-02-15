#
# prep.py
#
# Clément Malonda
#

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__" :
    irmPath = "../Data/IRM2/wetransfer-f810e9"
    # irmPath = "../Data/Data_IRM/Projet crosse de l_aorte/CAS 1/I7"

    datasetPath = "../irm"


    if not os.path.isdir(os.path.join(datasetPath, "train")) :
        os.makedirs(os.path.join(datasetPath, "train"))
    if not os.path.isdir(os.path.join(datasetPath, "test")) :
        os.makedirs(os.path.join(datasetPath, "test"))

    irmList = os.listdir(irmPath)

    for irm in irmList :
        print("File : {}".format(irm))
        ds = pydicom.read_file(os.path.join(irmPath, irm))
        savePath = os.path.join(datasetPath, "train", irm+".png")
        plt.imsave(savePath, ds.pixel_array, cmap='gray')
