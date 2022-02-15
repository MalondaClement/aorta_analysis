#
# main.py
#
# Clément Malonda
#

import time
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from datasets.irm import IRM
from models.utils import get_classification_model

if __name__ == "__main__" :

    epochs = 40

    train = IRM("../irm/train.csv", "../irm/train", transform=ToTensor)
    test = IRM("../irm/test.csv", "../irm/test", transform=ToTensor)

    train_dataloader = DataLoader(train, batch_size=8, shuffle=True)

    model = get_model("AlexNet", 3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start = time.time()

    for epoch in range(epochs) :
        print("Start Epoch {}/{}".format(epoch+1, epochs))
        loss_value = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_value += loss.item()
