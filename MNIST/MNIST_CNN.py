import torch
import torch.nn as nn
import torch.nn.functional as F


# create CNN
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # conv2d
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # maxPool
        self.pool = nn.MaxPool2d(2, 2)

        # linear
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28*28--> 14*14
        x = self.pool(F.relu(self.conv2(x)))  #14*14--> 7*7
        # relu是激活函数
        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


model = MNIST_CNN()

print(model)
