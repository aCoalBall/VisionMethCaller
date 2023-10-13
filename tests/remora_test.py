import sys
sys.path.append('..')

import torch
import numpy as np
from models.remora_like import RemoraNet

device = 'cuda'

sig = torch.rand(2, 1, 150)
sig = sig.to(device)

seq = torch.rand(2, 1, 9)
seq = seq.to(device)

model = RemoraNet().to(device)

z = model(sig, seq)
print(z.shape)