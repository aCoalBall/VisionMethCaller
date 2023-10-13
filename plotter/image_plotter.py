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

def get_datasets(meth_fold_path:str, pcr_fold_path:str, split:tuple[float]=(0.8, 0.1)):
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

def plotter(i, dataset:Dataset, split:str):

    def generate_image(signals, base_range):
        signals = np.array(signals)
        fig, ax = plt.subplots(figsize=(2.24,2.24))
        #fig = plt.figure(figsize=(2.24,2.24))
        for base, (start, end) in base_range:
            ax.axvspan(start, end, color=COLOR_DICT[base])
            ax.axvline(end, linewidth=0.5)
        ax.plot(signals, color='black', linewidth=1.0)
        plt.axis('off')
        plt.margins(0,0)
        plt.margins(0,0)
        image_rgb = mplfig_to_npimage(fig)
        image_rgb = np.transpose(image_rgb, (2, 0, 1))
        plt.close(fig)
        return image_rgb
    
    save_sub_path = 'process%d/'%i

    if split == 'train':
        save_path = '/home/coalball/projects/methBert2/toys/224_full_test/train/'
    elif split == 'val':
        save_path = '/home/coalball/projects/methBert2/toys/224_full_test/val/'
    else:
        save_path = '/home/coalball/projects/methBert2/toys/224_full_test/test/'

    path = save_path + save_sub_path
    if not os.path.exists(path): os.mkdir(path)

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
    random.shuffle(all_pairs)
    save_file = open(path + 'process%d.npy'%i, 'wb')
    for [signal, base, label] in all_pairs:
        signal = generate_image(signal, base)
        pair = np.asanyarray([signal, label], dtype=object)
        np.save(save_file, pair, allow_pickle=True)
    save_file.close()


if __name__=='__main__':

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
        p.apply_async(plotter, args=(i, train_subsets[i], 'train'))
    for j in range(3):
        p.apply_async(plotter, args=(j, val_subsets[j], 'val'))
    for k in range(3):
        p.apply_async(plotter, args=(k, test_subsets[k], 'test'))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    e = time.time()
    print('End time : ', e)
    print('Time Cost : ', e - s)


