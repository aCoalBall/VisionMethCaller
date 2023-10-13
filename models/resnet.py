'''
from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
'''

from torch import nn
import torch
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block_type, layers, linear_input = 256 * 3 * 1, num_classes = 2, last_avg_size=3, maxpool=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(4, 64, kernel_size = (11, 3), stride = 1, padding = (5, 1)),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        if maxpool:
            self.maxpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size = (2, 1), stride = 2)
        self.layer0 = self._make_layer(block_type, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block_type, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block_type, 256, layers[2], stride = 2)
        self.avgpool = nn.AvgPool2d((7, last_avg_size), stride=(7, 1), padding=(2, 0))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input, num_classes),
        )
        
    def _make_layer(self, block_type, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block_type(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        y = x
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x, y
    


