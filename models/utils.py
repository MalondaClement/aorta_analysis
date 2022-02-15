#
# models/utils.py
#
# Clément Malonda
#

import torchvision.models as models

def get_model(model_name, num_classes) :

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
