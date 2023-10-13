from .feature_extract import single_read_process, single_read_process_aligned
import multiprocessing as mp
import numpy as np
import time, tqdm
import random
from itertools import combinations

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

# 20200503 pytorch data load
class MethylDataSet(Dataset):
	def __init__(self, fast5_files, label, motif, m_shift, w_len, transform=None, test_region=None, rawsignal_num=150):
		self.fast5_files = fast5_files
		self.label = label
		self.transform = transform
		self.motif = motif
		self.m_shift = m_shift
		self.w_len = w_len
		self.test_region = test_region
		self.rawsignal_num = rawsignal_num

	def __getitem__(self,index):
		# process the target read
		feat_list, test_sample_tag = single_read_process(self.fast5_files[index], self.motif, self.m_shift, self.w_len, self.test_region, rawsignal_num=self.rawsignal_num)
		
		# any transormation
		if self.transform:
			feat_list = self.transform(feat_list)

		if feat_list is not None:
			# add the read-level alignment information , the third one is new added
			return feat_list, [self.label]*len(feat_list), [test_sample_tag]*len(feat_list)
		
	def __len__(self):
		return len(self.fast5_files)
	
class AlignedMethylDataset(Dataset):
	def __init__(self, fast5_files, label, motif, m_shift, w_len, transform=None, test_region=None, rawsignal_num=150):
		self.fast5_files = fast5_files
		self.label = label
		self.transform = transform
		self.motif = motif
		self.m_shift = m_shift
		self.w_len = w_len
		self.test_region = test_region
		self.rawsignal_num = rawsignal_num

	def __getitem__(self, index):
		aligned_signals = single_read_process_aligned(self.fast5_files[index], self.motif, self.m_shift, self.w_len, self.test_region, rawsignal_num=self.rawsignal_num)
		if aligned_signals:
			return aligned_signals, [self.label]*len(aligned_signals)

	def __len__(self):
		return len(self.fast5_files)