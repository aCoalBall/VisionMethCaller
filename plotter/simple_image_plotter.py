import os
import sys
sys.path.append('..')
#import h5py
import random
import time
import numpy as np
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
import matplotlib as mpl
mpl.use('agg')
mpl.rc('figure', max_open_warning = 0)
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

from data_processing.ref_util import get_fast5s
from data_processing.data_loading import AlignedMethylDataset
from constants import *

import argparse

def get_datasets(meth_fold_path:str, pcr_fold_path:str, split=(0.8, 0.1)):
    random_seed = 123
    torch.manual_seed(random_seed)
    train_split, test_split = split
    pcr = get_fast5s(pcr_fold_path)
    meth_reads = get_fast5s(meth_fold_path)

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

def reform_aligned_signals(aligned_signals):
	signals = np.concatenate([i[0] for i in aligned_signals])
	base_range = []
	start = 0
	for signals_chunk, base in aligned_signals:
		l = len(signals_chunk)
		base_range.append((base, (start, start + l)))
		start = start + l
	return signals, base_range

def plotter(i, dataset:Dataset, split:str, slot:int):

    def generate_matrix(signals, bases, slot):
        max_sig = max(signals)
        min_sig = min(signals)
        signals = np.array(signals)

        signals_id = signals - min_sig
        max_sig = max_sig - min_sig

        signals_id = signals_id * ((slot - 0.1) / max_sig)
        signals_id = signals_id.astype(int)

        #signals_matrix = np.zeros([len(signals), slot])
        #signals_matrix[np.arange(len(signals)), signals_id] = signals
        signals_matrix = np.tril(np.ones((slot, slot)))
        signals_matrix = signals_matrix[signals_id]
        for i in range(len(signals_id)):
            signals_matrix[i, :signals_id[i] + 1] = signals[i]


        bases_type_matrix = np.zeros((150, slot))
        for base, (start, end) in bases:
            bases_type_matrix[start:end, :] = INT_DICT[base]
        
        bases_number_matrix = np.zeros((150, slot))
        i = 0
        for base, (start, end) in bases:
            bases_number_matrix[start:end, :] = i
            i += 1
        
        c_note_matrix = np.zeros((150, slot))
        for base, (start, end) in bases:
            if base == 'X':
                c_note_matrix[start:end, :] = 1
        
        signals_matrix = signals_matrix[np.newaxis, ...]
        bases_number_matrix = bases_number_matrix[np.newaxis, ...]
        bases_type_matrix = bases_type_matrix[np.newaxis, ...]
        c_note_matrix = c_note_matrix[np.newaxis, ...]
        matrix = np.concatenate([signals_matrix, bases_number_matrix, bases_type_matrix, c_note_matrix])
        return matrix
    
    path = '/home/coalball/projects/methBert2/toys/test_simple_%d/'%slot
    if not os.path.exists(path): os.mkdir(path)

    if split == 'train':
        save_path = path + 'train/'
    elif split == 'val':
        save_path = path + 'val/'
    else:
        save_path = path + 'test/'

    if not os.path.exists(save_path): os.mkdir(save_path)

    k = 0
    j = 0
    all_pairs = []
    for read in dataset:
        if read == None:
            continue
        read_signal, read_label = read
        k += 1
        for aligned_signals, label in zip(read_signal, read_label):
            j += 1
            if aligned_signals == None:
                continue
            signal, base = reform_aligned_signals(aligned_signals)
            pair = [signal, base, label]
            all_pairs.append(pair)
            #signal = generate_matrix(signal, base, slot=20)
            #print(signal.shape)
            #print(signal[1])
    random.shuffle(all_pairs)
    save_file = open(save_path + 'process%d.npy'%i, 'wb')
    for [signal, base, label] in all_pairs:
        signal = generate_matrix(signal, base, slot=slot)
        pair = np.asanyarray([signal, label], dtype=object)
        np.save(save_file, pair, allow_pickle=True)
    save_file.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='<Multiprocess Plotter>')
    parser.add_argument('--height', default=20, type=int, required=False, help="The height of the matrix, default 20.")
    args = parser.parse_args()

    meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
    pcr_fold_path = '/home/coalball/projects/sssl/Control'

    train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path)

    train_subsets = []
    for j in range(22):
        indices = list(range(j, len(train_set), 22))
        random.shuffle(indices)
        train_subset = Subset(train_set, indices)
        train_subsets.append(train_subset)
    

    
    val_subsets = []
    for j in range(3):
        indices = list(range(j, len(val_set), 3))
        random.shuffle(indices)
        val_subset = Subset(val_set, indices)
        val_subsets.append(val_subset)

    test_subsets = []
    for j in range(3):
        indices = list(range(j, len(test_set), 3))
        random.shuffle(indices)
        test_subset = Subset(test_set, indices)
        test_subsets.append(test_subset)

    s = time.time()
    print('Strat time : ', s)
    print('Parent process %s.' % os.getpid())
    p = multiprocessing.Pool(28)
    for i in range(22):
        p.apply_async(plotter, args=(i, train_subsets[i], 'train', args.height))
    for j in range(3):
        p.apply_async(plotter, args=(j, val_subsets[j], 'val', args.height))
    for k in range(3):
        p.apply_async(plotter, args=(k, test_subsets[k], 'test', args.height))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    e = time.time()
    print('End time : ', e)
    print('Time Cost : ', e - s)
