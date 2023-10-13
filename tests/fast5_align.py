import h5py
import numpy as np
import random
import sys
from itertools import groupby
sys.path.append('..')
from data_processing.fast5_util import get_signal_event_from_fast5
from data_processing.feature_extract import get_methysite_in_ref, read_motif_input_extraction, single_read_process_aligned
from data_processing.data_loading import AlignedMethylDataset
from data_processing.ref_util import get_fast5s
from constants import *

import matplotlib.pyplot as plt

#path = '/home/coalball/projects/methBert2/toys/M_Sssl_Cg/Datasystem01_20161111_FNFAB46282_MN17250_sequencing_run_20161111_Methylases_Library3_66431_ch143_read2495_strand.fast5'

#signal, events, align_info = get_signal_event_from_fast5(path)

#aligned_signals = single_read_process_aligned(file_name=path, motif=["CG"], m_shift=0, w_len=17)
#print(len(signal))
#print(len(events))
#print(align_info)

#print(aligned_signals[0])
#print(len(np.concatenate([i[0] for i in aligned_signals[0]])))


path = '/home/coalball/projects/sssl/M_Sssl_Cg'
meth_reads = get_fast5s(path)
dataset_meth = AlignedMethylDataset(meth_reads, 1, motif=["CG"], m_shift=0, w_len=17)
print(len(dataset_meth))
x = dataset_meth[0]
print(len(x[0]))
print(len(x[1]))


def reform_aligned_signals(aligned_signals):
	signals = np.concatenate([i[0] for i in aligned_signals])
	base_range = []
	start = 0
	for signals_chunk, base in aligned_signals:
		l = len(signals_chunk)
		base_range.append((base, (start, start + l)))
		start = start + l
	return signals, base_range

sig, b = reform_aligned_signals(x[0][10])

print(sig)
print(b)

signals = np.array(sig)
fig, ax = plt.subplots(figsize=(2.24,2.24))
for base, (start, end) in b:
    ax.axvspan(start, end, color=COLOR_DICT[base])
    ax.axvline(end, linewidth=1.0)
ax.plot(signals, color='black', linewidth=0.5)
plt.axis('off')
plt.margins(0,0)
plt.savefig('test.png')

'''

def get_aligned_central_sigs(signals_list:list, rawsignal_num=360):
    signal_len = sum([len(x[0]) for x in signals_list])
    if signal_len < rawsignal_num:
        x = ([0,] * (rawsignal_num - signal_len), 'None')
        signals = signals_list.append(x)
    else:
        mid_loc = int((len(signals_list) - 1) / 2)
        C_len = len(signals_list[mid_loc][0])
        if C_len >= rawsignal_num:
            allcentsignals = signals_list[mid_loc][0]
            cent_signals = [allcentsignals[x] for x in sorted(random.sample(range(len(allcentsignals)),
                                                                            rawsignal_num))]
            signals = [(cent_signals, 'X')]
        else:
            left_len = (rawsignal_num - C_len) // 2
            right_len = rawsignal_num - C_len - left_len
            assert (right_len + left_len + C_len == rawsignal_num)

            left_signals_list = []
            i = mid_loc - 1
            l = left_len
            while l > 0 and i >= 0:
                if len(signals_list[i][0]) <= l:
                    left_signals_list.insert(0, signals_list[i])
                    l = l - len(signals_list[i][0])
                else:
                    segment = (signals_list[i][0][-l:], signals_list[i][1])
                    left_signals_list.insert(0, segment)
                    l = 0
                i -= 1

            right_signals_list = []  
            i = mid_loc + 1
            l = right_len
            while l > 0 and i >= 0:
                if len(signals_list[i][0]) <= l:
                    right_signals_list.append(signals_list[i])
                    l = l - len(signals_list[i][0])
                else:
                    segment = (signals_list[i][0][-l:], signals_list[i][1])
                    right_signals_list.append(segment)
                    l = 0
                i += 1      
            
            current_left_len = len(np.concatenate([i[0] for i in left_signals_list]))
            current_right_len = len(np.concatenate([i[0] for i in right_signals_list]))

            if current_left_len < left_len:
                x = ([0,] * (left_len - current_left_len), 'None')
                left_signals_list.insert(0, x)
            elif current_right_len < right_len:
                x = ([0,] * (right_len - current_right_len), 'None')
                right_signals_list.append(x)

            signals = left_signals_list + [signals_list[mid_loc],] + right_signals_list

    assert(len(np.concatenate([i[0] for i in signals])) == 150)
    return signals


""" input is the extracted re-squggiled sequence and signals"""
## add the position infomraiton here
def extract_aligned_signals_from_CpG(signal, event, align_info, motif, m_shift, r_mer, rawsignal_num=150):

	#1. get candidate region context for the sequence of motif
    seq_event = "".join([e[2] for e in event])
    loc_event = [ (e[0],e[1]) for e in event ]
    #print(event)

    feat_input = []
    motif_local_loci = get_methysite_in_ref(seq_event, motif, m_shift)
	
    shift = (r_mer - 1)//2
	
    for loci in motif_local_loci:
        start = loci - shift
        end = loci + shift + 1

        # skip the out of range ones
        if(start < 0 or end > len(seq_event)):
            continue
			
            ## chrome_name, start, motif_relative_location, align_strand, strand_len, $$ new added read_id, strand_tempalte)
            ## original used
        a_info = (align_info[0], align_info[1], loci, align_info[3], len(seq_event), align_info[4], align_info[2])
			
            # nucleotide sequence
        r_mer_seq =  seq_event[start:end]

            # signal sequence

        r_mer_signal = [ signal[ l[0]:l[0]+l[1] ] for l in loc_event[start:end]]
        #r_mer_signal_aligned = [ (signal[ l[0]:l[0]+l[1] ], l[2]) for l in event[start:end]]
        #print(r_mer_signal_aligned)
			
        r_mer_signal_aligned_left = [ (signal[ l[0]:l[0]+l[1] ], l[2]) for l in event[start:loci]]
        r_mer_signal_aligned_cent = [(signal[ event[loci][0] : event[loci][0] + event[loci][1] ], 'X'),]
        r_mer_signal_aligned_right = [ (signal[ l[0]:l[0]+l[1] ], l[2]) for l in event[loci + 1: end]]
        r_mer_signal_aligned = r_mer_signal_aligned_left + r_mer_signal_aligned_cent + r_mer_signal_aligned_right

            # added the length normalization
        total_seg_len = sum([len(r) for r in r_mer_signal])
        event_stat   = [[np.mean(r), np.std(r), len(r)/total_seg_len] for r in r_mer_signal]

            #event_stat   = [[np.mean(r), np.std(r), len(r)] for r in r_mer_signal]

        c_signal  = get_aligned_central_sigs(r_mer_signal_aligned, rawsignal_num=rawsignal_num)

            #print(len(c_signal))
            # filtering out the extrem signals ...
        mean_s_min = min([s[0] for s in event_stat])
        mean_s_max = max([s[0] for s in event_stat])

			# filtering out the extrem values
        if mean_s_min < -10 or mean_s_max > 10:
				#print("min=%f, max=%f" %(mean_s_min, mean_s_max))
            continue

			# masking the according signals
			# event_stat[shift+m_shift]=[0,0,0]
		
        feat_input.append((r_mer_seq, c_signal, a_info, event_stat))

			
    return feat_input


read_motif_input = extract_aligned_signals_from_CpG(signal, events, align_info, motif=["CG"], m_shift=0, r_mer=17, rawsignal_num=150)
one_site = read_motif_input[-1]
signals = one_site[1]
bases = one_site[0]

print(bases)
print([s[1] for s in signals])
print(signals)


'''