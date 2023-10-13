from torch import nn
import torch
from PIL import Image
import numpy as np
import time
import sys
sys.path.append('..')
from trainer import Trainer

from models.ViT import Swin
from models.cnn import CNN, k5_CNN, k17_CNN, BigCNN
from models.cnn_simple import CNN_Simple
from models.cnn_num import CNN_numerical, CNN_numerical_options
from torchvision.models import resnet18

from models.patchTST.patchTST import PatchTST
from data_processing.datasets import create_patch, ChunkedImageDataset, get_datasets
from data_processing.get_dataloader import *
from transformers import AutoConfig, AutoModel


from main import main
import argparse

parser = argparse.ArgumentParser(description='<Test>')
parser.add_argument('--length', default=100, type=int, required=True)
args = parser.parse_args()

def check_num_points(rawsignal_num:int, batch_size=256, device='cuda'):
    meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
    pcr_fold_path = '/home/coalball/projects/sssl/Control'
    train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path, rawsignal_num=rawsignal_num)
    print('loading training data')
    train_loader = get_numerical_dataloaders(train_set, batch_size=batch_size)
    print('loading validation data')
    val_loader = get_numerical_dataloaders(val_set, batch_size=batch_size)
    print('loading test data')
    test_loader = get_numerical_dataloaders(test_set, batch_size=batch_size)

    model = CNN_numerical_options(length=rawsignal_num)
    model.to(device)

    print('number of raw signal points : ', rawsignal_num)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    trainer = Trainer(model=model, dataloaders=[train_loader, val_loader, test_loader], lr = (1e-4) * (batch_size//64))
    subpath = 'numerical_length_%d'%rawsignal_num
    path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/%s'%subpath
    if not os.path.exists(path): os.mkdir(path)
    trainer.train(n_epochs=15, model_saved_dir=path)

check_num_points(rawsignal_num=args.length)