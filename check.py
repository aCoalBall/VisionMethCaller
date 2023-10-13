import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from data_processing.get_dataloader import *
from data_processing.datasets import ImageDataset, ChunkedMatrixDataset, ChunkedImageDataset, NumericalDataset, PatchTSTDataset, get_datasets
from trainer import Trainer
from models.cnn_simple import CNN_Simple
from models.cnn import CNN, BigCNN, k5_CNN, k17_CNN
from models.cnn_num import CNN_numerical
from models.ViT import ViT, ViT_tiny, Swin
from models.patchTST.patchTST import PatchTST
from constants import *

import argparse

def f(data_type:str, epochs:int=10, batch_size:int=256, device='cuda'):
    if data_type == 'simple':
        train_path = '/home/coalball/projects/methBert2/toys/simple/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple/test'
    
    elif data_type == 'fill':
        train_path = '/home/coalball/projects/methBert2/toys/simple_fill/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_fill/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_fill/test'
    
    train_loader = get_cached_matrix_dataloaders(train_path, batch_size=batch_size)
    val_loader = get_cached_matrix_dataloaders(val_path, batch_size=batch_size)
    test_loader = get_cached_matrix_dataloaders(test_path, batch_size=batch_size)

    model = CNN_Simple()
    model.to(device)

    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainer = Trainer(model=model, dataloaders=[train_loader, val_loader, test_loader], lr = (1e-4) * (batch_size//64))
    path = 'saved_models_full/%s/'%data_type
    if not os.path.exists(path): os.mkdir(path)
    trainer.train(n_epochs=epochs, model_saved_dir=path)

if __name__ == '__main__':
    #f(data_type='simple', epochs=15, batch_size=256)
    f(data_type='fill', epochs=15, batch_size=256)

