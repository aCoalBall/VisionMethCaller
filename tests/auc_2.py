
from typing import List
import torch
from torch import nn
import time
from sklearn.metrics import roc_auc_score
import numpy as np
from copy import deepcopy
import sys
sys.path.append('..')
from torchvision.models import resnet18
from models.cnn_simple import CNN_Deep_4x
from data_processing.get_dataloader import get_chunked_image_dataloaders

test_path = '/home/coalball/projects/methBert2/toys/224_full/test'
test_loader = get_chunked_image_dataloaders(test_path, batch_size=256)

model = resnet18()
num_feat = model.fc.in_features
model.fc = nn.Linear(num_feat, 2)
model.to('cuda')


model_path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/ResNet/model_epoch3.pt'
model.load_state_dict(torch.load(model_path))
device = 'cuda'
model.to(device)
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for _, [signals, ref] in enumerate(test_loader):
        signals = signals.to(device).float()
        ref = ref.to(device)

        pred = model(signals)
        preds = pred[:, 1].tolist()
        refs = ref.tolist()
        #print(len(preds))
        #print(len(refs))
        all_labels += refs
        all_preds += preds

label = torch.tensor(all_labels)
pred = torch.tensor(all_preds)

auc = roc_auc_score(all_labels, all_preds)

print(auc)

        

