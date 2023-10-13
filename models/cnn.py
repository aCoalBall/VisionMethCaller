from torch import nn
import torch
from PIL import Image
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)           
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        #(3, 112, 112)
        x = self.layer1(x)
        x = self.layer2(x)
        #(64, 7, 7)
        y = self.dense(x)
        #> 64 * 7 x 7 > 64 > 2
        return y
    

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128)           
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 14 * 128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        #(3, 224, 224)
        x = self.layer1(x)
        y = self.layer2(x)
        #(128, 14, 14)
        y = self.dense(y)
        #> 128 * 14 x 14 > 32 > 2
        return y


class k5_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(256)           
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        #(3, 224, 224)
        x = self.layer1(x)
        x = self.layer2(x)
        #(128, 14, 14)
        y = self.dense(x)
        #> 128 * 14 x 14 > 32 > 2
        return y
    
class k17_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=17, stride=1, padding=8),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(256)           
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        #(3, 224, 224)
        x = self.layer1(x)
        y = self.layer2(x)
        #(128, 14, 14)
        y = self.dense(y)
        #> 256 * 7 x 7 > 32 > 2
        return y,x
    