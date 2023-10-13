from torch import nn
import torch

class CNN_numerical(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64)           
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.dense(x)
        return y


class CNN_numerical_options(nn.Module):
    def __init__(self, length=150):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64)           
        )
        fea_out:int
        if length == 50:
            fea_out = 3
        elif length == 100:
            fea_out = 6
        elif length == 150:
            fea_out = 9
        elif length == 200:
            fea_out = 12
        elif length == 300:
            fea_out = 19
        elif length == 400:
            fea_out = 25
        elif length == 500:
            fea_out = 31

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fea_out * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.dense(x)
        return y


