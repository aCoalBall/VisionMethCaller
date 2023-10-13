from typing import List
import torch
from torch import nn
import time
from sklearn.metrics import roc_auc_score
import numpy as np
from copy import deepcopy
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
from models.resnet import ResNet, ResidualBlock
from data_processing.get_dataloader import *
from models.cnn import *
from PIL import Image

fig, ax = plt.subplots(1,2)




path='/home/coalball/projects/methBert2/toys/224_chunks/test'

dataloader = get_chunked_image_dataloaders(path, batch_size=1)
sample = next(iter(dataloader))[0].float()

print(sample.shape)
t = sample.squeeze(0)
t = torch.permute(t, (1,2,0)).numpy().astype(np.uint8)
print(t.shape)

ax[0].matshow(t, interpolation='nearest')
ax[0].axis('off')

device = 'cuda'
sample = sample.to(device)

model = k17_CNN()
model.to(device)
model_path = '/home/coalball/projects/methBert2/toys_1m/saved_models/k17CNN/model_epoch14.pt'
model.load_state_dict(torch.load(model_path))

y, x = model(sample)
print(x.shape)
x = x.cpu().detach().numpy()
x = np.average(x, axis=1)
x = x.squeeze()
print(x.shape)

ax[1].matshow(x, interpolation='nearest')
ax[1].axis('off')
plt.savefig('sample_2.png')
'''
#slot 1
model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=1, maxpool=False)
model = model.to(device)
model_path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/Simple_Res/model_epoch%d.pt'%i
model.load_state_dict(torch.load(model_path))
'''


'''
#slot 20
model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 1)
model = model.to(device)
model_path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/Simple_Res/model_epoch4.pt'
model.load_state_dict(torch.load(model_path))
x, y = model(sample)
print(y.shape)
y = y.cpu().detach().numpy()
y = np.average(y, axis=1)
y = y.squeeze()
print(y.shape)

plt.imshow(y, cmap='gray', interpolation='nearest')
plt.savefig('sample_2.png')
plt.clf()


path='/home/coalball/projects/methBert2/toys/simple_1/test'

dataloader = new_get_chunked_image_dataloaders(path, batch_size=1)
sample = next(iter(dataloader))[0].float()

print(sample.shape)
t = sample.squeeze(0)
t = t[0].numpy()
print(t.shape)

plt.imshow(t, cmap='gray', interpolation='nearest')
plt.savefig('sample_3.png')
plt.clf()


sample = sample.to(device)

#slot 1
model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=1, maxpool=False)
model = model.to(device)
model_path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/Simple_Res_1/model_epoch7.pt'
model.load_state_dict(torch.load(model_path))

x, y = model(sample)
print(y.shape)
y = y.cpu().detach().numpy()
y = np.average(y, axis=1)
y = y.squeeze(0)
print(y.shape)

plt.imshow(y, cmap='gray', interpolation='nearest')
plt.savefig('sample_4.png')
'''