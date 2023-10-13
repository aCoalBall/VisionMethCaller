from torch import nn
import torch
from PIL import Image
import numpy as np
import time
import sys
sys.path.append('..')
from trainer import Trainer
from sklearn.metrics import roc_auc_score

from models.ViT import Swin
from models.cnn import CNN, k5_CNN, k17_CNN, BigCNN
from models.cnn_simple import CNN_Simple, CNN_base_independent
from models.cnn_num import CNN_numerical, CNN_numerical_options
from torchvision.models import resnet18
from models.remora_like import RemoraNet

from models.patchTST.patchTST import PatchTST
from data_processing.datasets import create_patch, ChunkedImageDataset, get_datasets
from data_processing.get_dataloader import *
from transformers import AutoConfig, AutoModel


from main import main
import argparse

class NewTrainer(Trainer):
    def __init__(self, model, dataloaders, device:str='cuda', lr:float=1e-4):
        super().__init__(model, dataloaders, device, lr)

    def train_one_epoch(self):
        self.model.train()
        loss_sum = 0
        num_batches = 0
        for i, [seq, signals, ref] in enumerate(self.train_loader):

            num_batches += 1

            seq = seq.to(self.device).float()
            signals = signals.to(self.device).float()
            ref = ref.to(self.device)

            pred = self.model(signals, seq)
            loss = self.loss_fn(pred, ref.long())
            loss_sum += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 2000 == 0:
                print('trainig progress : %d batches done'%i)

        avg_loss = loss_sum / num_batches
        self.recorder['train']['loss'].append(avg_loss)
        print('train_loss : ', avg_loss)
    
    def validate_one_epoch(self):
        print("start validation")
        self.model.eval()
        loss_sum, correct = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_labels, all_scores = [], []
        size, num_batches = 0, 0
        with torch.no_grad():
            for _, [seq, signals, ref] in enumerate(self.val_loader):

                num_batches += 1
                size += signals.shape[0]

                seq = seq.to(self.device).float()
                signals = signals.to(self.device).float()
                ref = ref.to(self.device)

                pred = self.model(signals, seq)
                loss = self.loss_fn(pred, ref.long())
                loss_sum += loss.item()

                correct += (pred.argmax(1) == ref).type(torch.float).sum().item()
                tn, tp, fn, fp = self.increment_metrics(pred.argmax(1), ref, tn, tp, fn, fp)

                all_labels += ref.tolist()
                all_scores += pred[:, 1].tolist()
                
        avg_loss = loss_sum / num_batches
        accuracy = correct / size
        precision, recall, f1, fallout = self.get_metrics(tn, tp, fn, fp)
        auc = roc_auc_score(all_labels, all_scores)
        self.record('val', avg_loss, accuracy, precision, recall, f1, fallout, auc)
        print('val_loss : ', avg_loss, 'val_accuracy : ', accuracy)

    def test_one_epoch(self):
        print('start test')
        self.model.eval()
        loss_sum, correct = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0
        all_labels, all_scores = [], []
        size, num_batches = 0, 0
        with torch.no_grad():
            for _, [seq, signals, ref] in enumerate(self.test_loader):

                num_batches += 1
                size += signals.shape[0]

                seq = seq.to(self.device).float()
                signals = signals.to(self.device).float()
                ref = ref.to(self.device)

                pred = self.model(signals, seq)
                loss = self.loss_fn(pred, ref.long())
                loss_sum += loss.item()
                correct += (pred.argmax(1) == ref).type(torch.float).sum().item()
                tn, tp, fn, fp = self.increment_metrics(pred.argmax(1), ref, tn, tp, fn, fp)

                all_labels += ref.tolist()
                all_scores += pred[:, 1].tolist()

        avg_loss = loss_sum / num_batches
        accuracy = correct / size
        precision, recall, f1, fallout = self.get_metrics(tn, tp, fn, fp)
        auc = roc_auc_score(all_labels, all_scores)
        self.record('test', avg_loss, accuracy, precision, recall, f1, fallout, auc)
        print('test_loss : ', avg_loss, 'test_accuracy : ', accuracy)


def check_indepedent_model(batch_size=256, device='cuda', rawsignal_num=100):
    meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
    pcr_fold_path = '/home/coalball/projects/sssl/Control'

    print('loading datasets...')
    train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path, rawsignal_num=rawsignal_num)
    print('loading training data')
    train_loader = remora_like_dataloaders(train_set, batch_size=batch_size)
    print('loading validation data')
    val_loader = remora_like_dataloaders(val_set, batch_size=batch_size)
    print('loading test data')
    test_loader = remora_like_dataloaders(test_set, batch_size=batch_size)


    model = RemoraNet()
    model.to(device)

    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer = NewTrainer(model=model, dataloaders=[train_loader, val_loader, test_loader], lr = (1e-4) * (batch_size//64))
    subpath = 'Remora_like'
    path = '/home/coalball/projects/methBert2/toys_1m/saved_models_full/%s'%subpath
    if not os.path.exists(path): os.mkdir(path)
    trainer.train(n_epochs=15, model_saved_dir=path)

check_indepedent_model()