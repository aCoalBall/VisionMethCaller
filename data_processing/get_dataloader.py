import os
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader, ChainDataset, ConcatDataset

from constants import *
from .datasets import *
def fix_worker_init_fn(worker_id):
	random.seed(RANDOM_SEED)

def get_numerical_dataloaders(read_dataset:Dataset, batch_size:int=256):
    dataset = NumericalDataset(read_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def remora_like_dataloaders(read_dataset:Dataset, batch_size:int=256)-> DataLoader:
    dataset = NumericalDataset_Full(read_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_patchtst_dataloaders(read_dataset:Dataset, batch_size:int=256, patch_len:int=10, stride:int=10):
     dataset = PatchTSTDataset(read_dataset, patch_len, stride)
     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
     return dataloader

def get_image_dataloaders(path:str, upper_bound:int=None, batch_size:int=256, num_workers:int=8) -> DataLoader:
    dataset = ImageDataset(path, upper_bound)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=fix_worker_init_fn)
    return dataloader

def get_chunked_image_dataloaders(path:str, batch_size:int=256) -> DataLoader:
    processes = os.listdir(path)
    dataset_list = []
    for p in processes:
         abs_path = os.path.join(path, p)
         sub_dataset = ChunkedImageDataset(abs_path)
         dataset_list.append(sub_dataset)
    dataset = ChainDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def new_get_chunked_image_dataloaders(path:str, batch_size:int=256) -> DataLoader:
    filenames = os.listdir(path)
    dataset_list = []
    for f in filenames:
         abs_path = os.path.join(path, f)
         sub_dataset = NewChunkedImageDataset(abs_path)
         dataset_list.append(sub_dataset)
    dataset = ChainDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader 

def get_cached_matrix_dataloaders(path:str, batch_size:int=256) -> DataLoader:
    processes = os.listdir(path)
    dataset_list = []
    for p in processes:
         abs_path = os.path.join(path, p)
         sub_dataset = ChunkedMatrixDataset(abs_path)
         dataset_list.append(sub_dataset)
    dataset = ChainDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def get_cached_matrix_dataloaders_full(path:str, batch_size:int=256) -> DataLoader:
    processes = os.listdir(path)
    dataset_list = []
    for p in processes:
         abs_path = os.path.join(path, p)
         sub_dataset = ChunkedMatrixDataset_Full(abs_path)
         dataset_list.append(sub_dataset)
    dataset = ChainDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
