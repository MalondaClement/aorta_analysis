#
# inference.py
#
# Clément Malonda
#


import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import ToTensor, ToPILImage

from models.utils import get_2d_segmentation_model

if __name__ == "__main__" :
    path = "../DeepLabV3_MobileNetV3_Res1"

    if not os.path.isdir(os.path.join(path, "inference")) :
        os.makedirs(os.path.join(path, "inference"))

    model = get_2d_segmentation_model("DeepLabV3_MobileNetV3", num_classes=14)

    checkpoint = torch.load(os.path.join(path, "checkpoint.pth.tar"), map_location=torch.device('cpu'))
    state = checkpoint["model_state_dict"]
    new_state = {}
    for key in state:
        new_state[key[7:]] = state[key]
    model.load_state_dict(new_state)
    model.eval()

    scaler = MinMaxScaler()

    with torch.no_grad():
        for file in os.listdir("../RawData/Training/img") :
            data = nib.load(os.path.join("../RawData/Training/img", file))
            tmp = data.get_fdata()[:, :, int(data.get_fdata().shape[2]/2)]
            tmp = scaler.fit_transform(tmp)
            tmp = cv2.resize(tmp, (96, 96))
            img = np.array([tmp, tmp, tmp])
            img = img.transpose(1, 2, 0)
            input = ToTensor()(img)
            input = input.unsqueeze(0)
            input = input.float()
            output = model(input)
            output = output["out"]
            preds = torch.argmax(output, 1)
            preds = preds.numpy()
            preds = preds.transpose(1, 2, 0)

            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
            fig.set_size_inches(16.5, 5.5)
            ax0.imshow(img)
            ax0.set_title("Image d'origine")
            ax1.imshow(preds)
            ax1.set_title("Prédiction")
            ax2.imshow(img)
            ax2.imshow(preds, alpha=0.6)
            ax2.set_title("Superposition de l'image avec la prédiction")
            print(os.path.join(path, "inference", file[:-7]+".png"))
            fig.savefig(os.path.join(path, "inference", file[:-7]+".png"))
            plt.close(fig)
