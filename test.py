from torch import nn
import torch
from PIL import Image
import numpy as np
import time

from torch.utils.data import Dataset, IterableDataset, DataLoader, ConcatDataset, random_split, Subset

from models.ViT import Swin
from models.cnn import CNN, k5_CNN, k17_CNN, BigCNN
from models.cnn_simple import CNN_Simple, CNN_base_independent, CNN_Deep_4x
from models.cnn_num import CNN_numerical, CNN_numerical_options
from torchvision.models import resnet18

from models.patchTST.patchTST import PatchTST
from data_processing.datasets import *
from data_processing.get_dataloader import *
from data_processing.fast5_util import *
from transformers import AutoConfig, AutoModel

from models.resnet import ResNet, ResidualBlock


from main import main
import argparse
import h5py

'''
path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/'

simple_a = (np.load(path + 'Simple_Res' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
simple_p = (np.load(path + 'Simple_Res' + '/recorder.npy', allow_pickle=True).item())['test']['precision']
simple_r = (np.load(path + 'Simple_Res' + '/recorder.npy', allow_pickle=True).item())['test']['recall']
simple_f = (np.load(path + 'Simple_Res' + '/recorder.npy', allow_pickle=True).item())['test']['f1']
simple_auc = (np.load(path + 'Simple_Res' + '/recorder.npy', allow_pickle=True).item())['test']['auc']

print('Simple_Res_20')
print('\n')
print('Accuracy')
print(simple_a)
print('\n')

print('Precison')
print(simple_p)
print('\n')

print('Recall')
print(simple_r)
print('\n')

print('F1')
print(simple_f)
print('\n')

print('AUC')
print(simple_auc)
print('\n')

simple_a = (np.load(path + 'Simple_Res_1' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
simple_p = (np.load(path + 'Simple_Res_1' + '/recorder.npy', allow_pickle=True).item())['test']['precision']
simple_r = (np.load(path + 'Simple_Res_1' + '/recorder.npy', allow_pickle=True).item())['test']['recall']
simple_f = (np.load(path + 'Simple_Res_1' + '/recorder.npy', allow_pickle=True).item())['test']['f1']
simple_auc = (np.load(path + 'Simple_Res_1' + '/recorder.npy', allow_pickle=True).item())['test']['auc']

print('Simple_Res_1')
print('\n')
print('Accuracy')
print(simple_a)
print('\n')

print('Precison')
print(simple_p)
print('\n')

print('Recall')
print(simple_r)
print('\n')

print('F1')
print(simple_f)
print('\n')

print('AUC')
print(simple_auc)
print('\n')

simple_a = (np.load(path + 'Simple_Res_10' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
simple_p = (np.load(path + 'Simple_Res_10' + '/recorder.npy', allow_pickle=True).item())['test']['precision']
simple_r = (np.load(path + 'Simple_Res_10' + '/recorder.npy', allow_pickle=True).item())['test']['recall']
simple_f = (np.load(path + 'Simple_Res_10' + '/recorder.npy', allow_pickle=True).item())['test']['f1']
simple_auc = (np.load(path + 'Simple_Res_10' + '/recorder.npy', allow_pickle=True).item())['test']['auc']

print('Simple_Res_10')
print('\n')
print('Accuracy')
print(simple_a)
print('\n')

print('Precison')
print(simple_p)
print('\n')

print('Recall')
print(simple_r)
print('\n')

print('F1')
print(simple_f)
print('\n')

print('AUC')
print(simple_auc)
print('\n')

simple_a = (np.load(path + 'Simple_Res_40' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
simple_p = (np.load(path + 'Simple_Res_40' + '/recorder.npy', allow_pickle=True).item())['test']['precision']
simple_r = (np.load(path + 'Simple_Res_40' + '/recorder.npy', allow_pickle=True).item())['test']['recall']
simple_f = (np.load(path + 'Simple_Res_40' + '/recorder.npy', allow_pickle=True).item())['test']['f1']
simple_auc = (np.load(path + 'Simple_Res_40' + '/recorder.npy', allow_pickle=True).item())['test']['auc']

print('Simple_Res_40')
print('\n')
print('Accuracy')
print(simple_a)
print('\n')

print('Precison')
print(simple_p)
print('\n')

print('Recall')
print(simple_r)
print('\n')

print('F1')
print(simple_f)
print('\n')

print('AUC')
print(simple_auc)
print('\n')


simple_a = (np.load(path + 'Simple_Res_60' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
simple_p = (np.load(path + 'Simple_Res_60' + '/recorder.npy', allow_pickle=True).item())['test']['precision']
simple_r = (np.load(path + 'Simple_Res_60' + '/recorder.npy', allow_pickle=True).item())['test']['recall']
simple_f = (np.load(path + 'Simple_Res_60' + '/recorder.npy', allow_pickle=True).item())['test']['f1']
simple_auc = (np.load(path + 'Simple_Res_60' + '/recorder.npy', allow_pickle=True).item())['test']['auc']

print('Simple_Res_60')
print('\n')
print('Accuracy')
print(simple_a)
print('\n')

print('Precison')
print(simple_p)
print('\n')

print('Recall')
print(simple_r)
print('\n')

print('F1')
print(simple_f)
print('\n')

print('AUC')
print(simple_auc)
print('\n')

path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/'
remora_a = (np.load(path + 'Remora_likerecorder.npy', allow_pickle=True).item())['test']['accuracy']
#fill = (np.load(path + 'fill' + '/recorder.npy', allow_pickle=True).item())['test']['accuracy']
remora_p = (np.load(path + 'Remora_likerecorder.npy', allow_pickle=True).item())['test']['precision']
remora_r = (np.load(path + 'Remora_likerecorder.npy', allow_pickle=True).item())['test']['recall']
remora_f = (np.load(path + 'Remora_likerecorder.npy', allow_pickle=True).item())['test']['f1']
remora_auc = (np.load(path + 'Remora_likerecorder.npy', allow_pickle=True).item())['test']['auc']

print('Accuracy')
print(remora_a)
print('\n')

print('Precison')
print(remora_p)
print('\n')

print('Recall')
print(remora_r)
print('\n')

print('F1')
print(remora_f)
print('\n')

print('AUC')
print(remora_auc)
print('\n')
'''
'''
filename = '/home/coalball/projects/methBert2/toys/M_Sssl_Cg/Datasystem01_20161111_FNFAB46282_MN17250_sequencing_run_20161111_Methylases_Library3_66431_ch143_read2495_strand.fast5'
norm_signal, events, align_info = get_signal_event_from_fast5(file_name=filename)
print('signal : ', type(norm_signal))
print('events : ', type(events))
print(events[10])
print('info : ', align_info)
'''
'''
parser = argparse.ArgumentParser(description='<Test>')
parser.add_argument('--length', default=100, type=int, required=True)
args = parser.parse_args()
main(model_type='Numerical', epochs=15, batch_size=256, rawsignal_num=args.length)
'''

'''
dataset = (path='/home/coalball/projects/methBert2/toys/simple_40/test/process0.npy')
s = time.time()
i = 0
for x in dataset:
    i += 1
    #print(x[0].shape)
    print(x[0].shape)
    if i % 100 == 0:
        break
e = time.time()
t = e - s
print(i)
print(t)
'''
dataloader = get_cached_matrix_dataloaders(path='/home/coalball/projects/methBert2/toys/simple/test')
s = time.time()
for i, (signals, label) in enumerate(dataloader):
    pass
print(i)
e = time.time()
t = e - s
print(t)

dataloader = get_cached_matrix_dataloaders(path='/home/coalball/projects/methBert2/toys/simple/val')
s = time.time()
for i, (signals, label) in enumerate(dataloader):
    pass
print(i)
e = time.time()
t = e - s
print(t)



dataloader = get_cached_matrix_dataloaders(path='/home/coalball/projects/methBert2/toys/simple/train')
s = time.time()
for i, (signals, label) in enumerate(dataloader):
    pass
print(i)
e = time.time()
t = e - s
print(t)

'''
print('slots 1')
print('\n')
model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=1, maxpool=False)
print('training epochs : 15')
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

print('slots 10')
print('\n')
model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=2)
print('training epochs : 15')
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

print('slots 20')
print('\n')
model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 1)
print('training epochs : 15')
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

print('slots 40')
print('\n')
model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 3)
print('training epochs : 15')
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

print('slots 60')
print('\n')
model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 6)
print('training epochs : 15')
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)
'''


'''
device = 'cuda'

y = torch.zeros(2)
y = y.to(device)

sig = torch.rand(2, 4, 150, 1)
sig = sig.to(device)

#print(x.shape)
model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=1, maxpool=False).to(device)
print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
p = model(sig)
print(p.shape)
'''
