import os
import sys
sys.path.append('..')
#import h5py
import random
import numpy as np
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
import matplotlib as mpl
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from data_processing.ref_util import get_fast5s
from data_processing.data_loading import AlignedMethylDataset, MethylDataSet
from constants import *

from data_processing.datasets import NumericalDataset#, get_datasets



def get_datasets(meth_fold_path:str, pcr_fold_path:str, split=(0.8, 0.1), dataset_type='Align'):
    random_seed = 123
    torch.manual_seed(random_seed)
    train_split, test_split = split
    pcr = get_fast5s(pcr_fold_path)
    meth_reads = get_fast5s(meth_fold_path)

    if dataset_type=='Methyl':
        dataset_meth = MethylDataSet(meth_reads, 1, motif=["CG"], m_shift=0, w_len=21)
        dataset_pcr = MethylDataSet(pcr, 0, motif=["CG"], m_shift=0, w_len=21)
    else:
        dataset_meth = AlignedMethylDataset(meth_reads, 1, motif=["CG"], m_shift=0, w_len=21)
        dataset_pcr = AlignedMethylDataset(pcr, 0, motif=["CG"], m_shift=0, w_len=21)      

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

def plotter(dataset:Dataset):
    k = 0
    for read in dataset:
        if read == None:
            continue
        read_signal, read_label = read
        for aligned_signals, label in zip(read_signal, read_label):
            if aligned_signals == None:
                continue
            k += 1
    print('total_CpG_sites : ', k)



if __name__=='__main__':

    meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
    pcr_fold_path = '/home/coalball/projects/sssl/Control'
    train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path)
    print('check_val_set')
    plotter(val_set)

    train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path, dataset_type='Methyl')
    print('check_val_set')
    plotter(val_set)

    '''
    print('check_train_set_for_num')
    d1 = NumericalDataset(train_set)
    plotter(d1)
    print('check_val_set_for_num')
    d2 = NumericalDataset(val_set)
    plotter(d2)
    print('check_test_set_for_num')
    d3 = NumericalDataset(test_set)
    plotter(d3)
    '''