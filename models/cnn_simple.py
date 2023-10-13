from torch import nn
import torch
from PIL import Image
import numpy as np

class CNN_Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(11, 3), stride=1, padding=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128)           
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.dense(x)
        return y


class CNN_Deep_4x(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(11, 3), stride=1, padding=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128)           
        )


        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)           
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, padding=(1,0)),
            nn.BatchNorm2d(256)           
        )


        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 1, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dense(x)
        return x
    
class CNN_base_independent(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(11, 3), stride=1, padding=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(5, 3), stride=1, padding=(2, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.sig_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128)           
        )

        self.seq_layer = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 128, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm1d(128)
        )

        self.merge_layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.BatchNorm2d(128)           
        )

        self.merge_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)           
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, sig, seq):
        sig = self.sig_layer1(sig)
        sig = self.sig_layer2(sig)
        seq = self.seq_layer(seq)
        seq = seq.unsqueeze(-1)
        merge = torch.cat((sig, seq), dim=-1)
        merge = self.merge_layer1(merge)
        merge = self.merge_layer2(merge)
        x = self.dense(merge)
        return x