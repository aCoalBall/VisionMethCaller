import os
import numpy as np
import torch
import glob
import random
from torch.utils.data import Dataset, IterableDataset, DataLoader, ConcatDataset, random_split, Subset

from .ref_util import get_fast5s
from .data_loading import MethylDataSet

from constants import RANDOM_SEED 

class ImageDataset(Dataset):
    def __init__(self, path:str, upper_bound:int=None):
        super().__init__()
        self.path = path
        #self.filenames:list[str] = os.listdir(path)
        self.filenames:list[str] = [f for f in glob.glob(pathname=path + '/**/*.npz', recursive=True)]
        random.seed(RANDOM_SEED)
        if upper_bound:
            self.filenames = random.sample(population=self.filenames, k=upper_bound)
    
    def __getitem__(self, index:int):
        filename = self.filenames[index]
        data = np.load(filename)
        signals, ref = data['a'], data['b']
        signals = torch.from_numpy(signals)
        #print(signals.shape)
        ref = torch.from_numpy(ref)
        return signals, ref

    def __len__(self):
        return len(self.filenames)

class ChunkedMatrixDataset(IterableDataset):
    def __init__(self, path:str):
        super().__init__()
        self.path = path
        [chunk_filename,] = os.listdir(path)
        self.chunk_filename = path + '/' + chunk_filename
        #print(self.chunk_filename)

    def __iter__(self):
        chunk = open(self.chunk_filename, 'rb')
        while True:
            try: 
                seq, signals, label = np.load(chunk, allow_pickle=True)
                signals = signals[np.newaxis, ...]
                #if signals.shape == (3, 224, 224):
                yield signals, label
            except:
                break

class ChunkedMatrixDataset_Full(IterableDataset):
    def __init__(self, path:str):
        super().__init__()
        self.path = path
        [chunk_filename,] = os.listdir(path)
        self.chunk_filename = path + '/' + chunk_filename
        #print(self.chunk_filename)

    def __iter__(self):
        chunk = open(self.chunk_filename, 'rb')
        while True:
            try: 
                seq, signals, label = np.load(chunk, allow_pickle=True)
                signals = signals[np.newaxis, ...]
                seq = base_encoder_1d(seq)
                #if signals.shape == (3, 224, 224):
                yield seq, signals, label
            except:
                break


class ChunkedImageDataset(IterableDataset):
    def __init__(self, path:str):
        super().__init__()
        self.path = path
        [chunk_filename,] = os.listdir(path)
        self.chunk_filename = path + '/' + chunk_filename
        #print(self.chunk_filename)

    def __iter__(self):
        chunk = open(self.chunk_filename, 'rb')
        while True:
            try: 
                signals, label = np.load(chunk, allow_pickle=True)
                #if signals.shape == (3, 224, 224):
                yield signals, label
            except:
                break

class NewChunkedImageDataset(IterableDataset):
    def __init__(self, path:str):
        super().__init__()
        self.path = path

    def __iter__(self):
        chunk = open(self.path, 'rb')
        while True:
            try: 
                signals, label = np.load(chunk, allow_pickle=True)
                #if signals.shape == (3, 224, 224):
                yield signals, label
            except:
                break

class NumericalDataset(Dataset):
    def __init__(self, read_dataset:Dataset):
        super().__init__()
        self.signals_list = []
        self.ref_list = []
        self.get_all_data(read_dataset)
    
    def get_all_data(self, read_dataset):
        i = 0
        for read in read_dataset:  
            if read == None:
                continue
            read_feat, read_label, _ = read
            for site, ref in zip(read_feat, read_label):
                signals = site[1]
                self.signals_list.append(signals)
                self.ref_list.append(ref)
                i += 1
        print('Number of samples : ', i)

    def __getitem__(self, index:int):
        signals = self.signals_list[index]
        signals = torch.tensor(signals) #shape (n_points)
        signals = signals.unsqueeze(0) #shape (1, n_points)
        ref = self.ref_list[index]
        ref = torch.tensor(ref)
        return signals, ref
    
    def __len__(self):
        return len(self.signals_list)

class NumericalDataset_Full(Dataset):
    def __init__(self, read_dataset:Dataset):
        super().__init__()
        self.signals_list = []
        self.bases_list = []
        self.ref_list = []
        self.get_all_data(read_dataset)
    
    def get_all_data(self, read_dataset):
        i = 0
        for read in read_dataset:  
            if read == None:
                continue
            read_feat, read_label, _ = read
            for site, ref in zip(read_feat, read_label):
                bases = site[0][4:13]
                signals = site[1]
                self.signals_list.append(signals)
                self.bases_list.append(bases)
                self.ref_list.append(ref)
                i += 1

    def __getitem__(self, index:int):
        signals = self.signals_list[index]
        signals = torch.tensor(signals) #shape (n_points)
        signals = signals.unsqueeze(0) #shape (1, n_points)
        bases = self.bases_list[index] #str
        bases = base_encoder_1d(bases)
        bases = torch.tensor(bases)
        #bases = bases.unsqueeze(0)
        ref = self.ref_list[index]
        ref = torch.tensor(ref)
        return bases, signals, ref
    
    def __len__(self):
        return len(self.signals_list)
    
class PatchTSTDataset(NumericalDataset):
    def __init__(self, read_dataset:Dataset, patch_len:int, stride:int):
        super().__init__(read_dataset=read_dataset)
        self.patch_len = patch_len
        self.stride = stride

    def __getitem__(self, index: int):
        signals = self.signals_list[index]
        signals = torch.tensor(signals) #shape (n_points)
        signals = signals.unsqueeze(1) #shape (n_points, 1)
        signals, _ = create_patch(signals, patch_len=self.patch_len, stride=self.stride)
        ref = self.ref_list[index]
        ref = torch.tensor(ref)
        return signals, ref
    

def create_patch(xb, patch_len, stride):
    """
    xb: [seq_len x n_vars]
    """
    seq_len = xb.shape[0]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[s_begin:, :]                                                    # xb: [tgt_len x nvars]
    xb = xb.unfold(dimension=0, size=patch_len, step=stride)                 # xb: [num_patch x n_vars x patch_len]
    return xb, num_patch

def base_encoder_1d(base:str):
    x = np.array(list(base))
    x[x == 'A'] = 0
    x[x == 'T'] = 1
    x[x == 'C'] = 2
    x[x == 'G'] = 3
    x = x.astype(int)
    x = x[np.newaxis, ...]
    return x


def base_encoder_2d(base:str):
    pass

def get_datasets(meth_fold_path:str, pcr_fold_path:str, split=(0.8, 0.1), rawsignal_num=150):
    torch.manual_seed(RANDOM_SEED)
    train_split, test_split = split
    pcr = get_fast5s(pcr_fold_path)
    meth_reads = get_fast5s(meth_fold_path)

    dataset_meth = MethylDataSet(meth_reads, 1, motif=["CG"], m_shift=0, w_len=17, rawsignal_num=rawsignal_num)
    dataset_pcr = MethylDataSet(pcr, 0, motif=["CG"], m_shift=0, w_len=17, rawsignal_num=rawsignal_num)

    #balance the positives and negatives
    indices = torch.randperm(len(dataset_pcr))[:len(dataset_meth)]
    dataset_pcr= torch.utils.data.Subset(dataset_pcr, indices)

    n_train, n_test = int(len(dataset_meth) * train_split), int(len(dataset_meth) * test_split)
    n_val = len(dataset_meth) - n_train - n_test
    p_train, p_test, p_val = random_split(dataset_meth, [n_train, n_test, n_val])


    n_train, n_test = int(len(dataset_pcr) * train_split), int(len(dataset_pcr) * test_split)
    n_val = len(dataset_pcr) - n_train - n_test
    ns_train, ns_test, ns_val = random_split(dataset_pcr, [n_train, n_test, n_val])

    train_set = ConcatDataset([p_train, ns_train])
    test_set  = ConcatDataset([p_test,  ns_test])
    val_set   = ConcatDataset([p_val,   ns_val])

    return train_set, val_set, test_set



if __name__=='__main__':


    dataset = ImageDataset(path='/home/coalball/projects/methBert2/toys/r9_images_test', upper_bound=10000)
    signals, ref = dataset[0]
    print(type(signals))
 
    print(len(dataset))
    print('im size : ', signals.shape)
    print(type(ref))
