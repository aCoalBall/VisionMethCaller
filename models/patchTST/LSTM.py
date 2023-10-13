from torch import nn
import torch

def swish(x):
    return x * torch.sigmoid(x)

class LSTM_Model(nn.Module):
    def __init__(self, size=64, num_out=4):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(1, 4, 5)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 16, 5)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, size, 9, 3)
        self.bn3 = nn.BatchNorm1d(size)

        self.lstm1 = nn.LSTM(62, 62, 1)
        self.lstm2 = nn.LSTM(62, 62, 1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(62 * size, num_out)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.lstm1(x)[0]
        x = self.relu(x)
        x = self.lstm2(x)[0]
        x = self.flatten(x)
        x = self.linear(x)
        return x
