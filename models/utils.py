#
# models/utils.py
#
# Clément Malonda
#

import torch
import torchvision.models as models

def get_3d_segmentation_model(model_name, num_classes) :
    if model_name == "UNet3D" :
        return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=85, out_channels=num_classes, init_features=32, pretrained=False)

def get_2d_segmentation_model(model_name, num_classes):
    if  model_name == "DeepLabV3_Resnet50":
        return models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes = num_classes)

    elif model_name == "DeepLabV3_Resnet101":
        from torchvision.models.segmentation import deeplabv3_resnet101
        return models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes = num_classes)

    elif model_name == "DeepLabV3_MobileNetV3":
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        return models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False, num_classes = num_classes)

    elif model_name == "FCN_Resnet50":
        from torchvision.models.segmentation import fcn_resnet50
        return models.segmentation.fcn_resnet50(pretrained=False, num_classes = num_classes)

    elif model_name == "FCN_Resnet101":
        from torchvision.models.segmentation import fcn_resnet101
        return models.segmentation.fcn_resnet101(pretrained=False, num_classes = num_classes)
    exit(1)

def get_classification_model(model_name, num_classes) :
    if model_name == "AlexNet" :
        return models.alexnet(pretrained = False, progress = True, num_classes = num_classes)
    elif model_name == "VGG" :
        return models.vgg16(pretrained = False, progress = True, num_classes = num_classes)
    elif model_name == "ResNet" :
        return models.resnet18(pretrained = False, progress = True, num_classes = num_classes)
    elif model_name == "SqueezeNet" :
        return models.squeezenet1_0(pretrained = False, progress = True, num_classes = num_classes)
    elif model_name == "DenseNet" :
        return models.densenet161(pretrained = False, progress = True, num_classes = num_classes)
    elif model_name == "Inception v3" :
        return models.inception_v3(pretrained = False, progress = True, num_classes = num_classes)

    else :
        print("Model\'s name not in the list")
        exit(1)
