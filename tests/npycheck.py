import numpy as np
from itertools import groupby

npy_path = '/home/coalball/projects/methBert2/shared/new_data/val.npy'

f = open(npy_path, 'rb')
i = 0
j = 0
while True:
    try:
        matrix, base, base_num, c_denote, label = np.load(f, allow_pickle=True)
        base = [i[0] for i in groupby(base)]
        j += 1
        if len(base) == 1:
            i += 1
            #print(base)
            #print(base_num)
    except:
        break
print(i)
print(j)
print(i / j)
    


