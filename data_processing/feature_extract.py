""" features used for methylation prediction in DeepSignal """

# search for the motifs sequences in a given sequence.
import numpy as np
import random
from .ref_util import _get_central_signals, _get_alignment_info_from_fast5
from .fast5_util import get_signal_event_from_fast5

def get_methysite_in_ref(seqstr, motifs, methy_shift=1, singleton=False):

	# remove duplicate
	motifs = list(set(motifs))
	ref_seq_len = len(seqstr)
	# assume all motifs has the same length, only test CG
	motiflen = len(motifs[0])
	num_non_singleton = 0

	sites = []
	for i in range(0, ref_seq_len - motiflen + 1):
		if seqstr[i:i + motiflen] in motifs:
        
			if singleton:
        		# checking the target motif not in the left and right stream
				left_region = seqstr[max(i-10,0) : i]
				right_region = seqstr[i+motiflen : min(i+motiflen+10, ref_seq_len)]

				if ((motifs[0] not in left_region) and (motifs[0] not in right_region)):
					sites.append(i + methy_shift)
				else:
					num_non_singleton += 1
			else:
				sites.append(i + methy_shift)

    #if singleton:
    #	print(" |-* [singleton mode] filtered non-singleton %d" %(num_non_singleton))

	return sites


""" input is the extracted re-squggiled sequence and signals"""
## add the position infomraiton here
def read_motif_input_extraction(signal, event, align_info, motif, m_shift, r_mer, rawsignal_num=150):

	#1. get candidate region context for the sequence of motif
	seq_event = "".join([e[2] for e in event])
	loc_event = [ (e[0],e[1]) for e in event ]

	feat_input = []
	motif_local_loci = get_methysite_in_ref(seq_event, motif, m_shift)
	
	shift = (r_mer - 1)//2
	for loci in motif_local_loci:

		try:
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

			# added the length normalization
			total_seg_len = sum([len(r) for r in r_mer_signal])
			event_stat   = [[np.mean(r), np.std(r), len(r)/total_seg_len] for r in r_mer_signal]

			#event_stat   = [[np.mean(r), np.std(r), len(r)] for r in r_mer_signal]

			c_signal  = _get_central_signals(r_mer_signal, rawsignal_num=rawsignal_num)

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

		except Exception:
			# when erorr exists, this information will not be enough for debug.
			print("[!] Skip one motif r-mer out of boundary.")
			
	return feat_input
		


# single read input feature extraction
def single_read_process(file_name, motif, m_shift, w_len, test_region = None, rawsignal_num=150):

	try:
		# merge the following two functions 
		signal, events, align_info = get_signal_event_from_fast5(file_name)
		read_motif_input = read_motif_input_extraction(signal, events, align_info, motif, m_shift, w_len, rawsignal_num=rawsignal_num)

		# cheching reads in the test region
		test_tag = 0
		if test_region != None:
			if align_info[0] == test_region[0]:
				if test_region[1] <= align_info[1] and align_info[1] + len(events) <= test_region[2]:
					test_tag = 1
					
		return read_motif_input, test_tag

	except:
		return None, None


def get_aligned_central_sigs(signals_list:list, rawsignal_num=360):
    #########
    Temp = None
    #########
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
            Temp = True
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
            while l > 0 and i < len(signals_list):
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

    #print('Signals : ', len(np.concatenate([i[0] for i in signals])))
    #print('Bases : ', [i[1] for i in signals])
    #print(signals)
    #if signals == None:
        #print('Signal is None')
    assert(len(np.concatenate([i[0] for i in signals])) == 150)
    return signals, Temp



""" input is the extracted re-squggiled sequence and signals"""
## add the position infomraiton here
def extract_aligned_signals_from_CpG(signal, event, align_info, motif, m_shift, r_mer, rawsignal_num=150, filename=''):

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
        #print('each :')
        #for i in r_mer_signal_aligned:
            #print(len(i[0]))
        c_signal, Temp  = get_aligned_central_sigs(r_mer_signal_aligned, rawsignal_num=rawsignal_num)


            #print(len(c_signal))
            # filtering out the extrem signals ...
        mean_s_min = min([s[0] for s in event_stat])
        mean_s_max = max([s[0] for s in event_stat])

			# filtering out the extrem values
        if mean_s_min < -10 or mean_s_max > 10:
				#print("min=%f, max=%f" %(mean_s_min, mean_s_max))
            continue

        #if Temp == True:
            #print('C is Longer Than 150')
            #print('filename : ', filename)
            #print('loci :', loci)
            #print(event[loci])

			# masking the according signals
			# event_stat[shift+m_shift]=[0,0,0]
		
        feat_input.append(c_signal)

			
    return feat_input


# single read input feature extraction
def single_read_process_aligned(file_name, motif, m_shift, w_len, test_region = None, rawsignal_num=150):
	try:
		# merge the following two functions 
		signal, events, align_info = get_signal_event_from_fast5(file_name)
		aligned_signals = extract_aligned_signals_from_CpG(signal, events, align_info, motif, m_shift, w_len, rawsignal_num=rawsignal_num, filename=file_name)

		return aligned_signals

	except:
		return None, None