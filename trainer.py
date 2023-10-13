
from typing import List
import torch
from torch import nn
import time
from sklearn.metrics import roc_auc_score
import numpy as np
from copy import deepcopy

class Trainer:
    def __init__(self, model, dataloaders, device:str='cuda', lr:float=1e-4):
        super().__init__()
        self.model = model.to(device)
        self.train_loader, self.val_loader, self.test_loader = dataloaders
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        metircs_train = {'loss':[]}
        metrics_val = {'loss':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'fallout':[], 'auc':[]}
        metrics_test = {'loss':[], 'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'fallout':[], 'auc':[]}

        self.recorder = {'train' : metircs_train, 'val':metrics_val, 'test':metrics_test}
        #self.recorder = {'train_loss':[], 'valid_loss':[], 'valid_accuracy':[], 'test_loss':[], 'test_accuracy':[]}
    
    def train(self, n_epochs:int, model_saved_dir:str):
        for i in range(n_epochs):
            s = time.time()
            print('epoch%d'%i)
            print('#########################')
            self.train_one_epoch()
            self.validate_one_epoch()
            self.test_one_epoch()
            model_saved_path = model_saved_dir + 'model_epoch%d.pt'%i
            recorder_saved_path = model_saved_dir + 'recorder.npy'
            e = time.time()
            self.save_model(path = model_saved_path)
            #auc = self.get_auc(self.recorder['test']['recall'], self.recorder)
            self.save_recorder(path = recorder_saved_path)
            print('total time spent : ', e - s)
    
    def train_one_epoch(self):
        self.model.train()
        loss_sum = 0
        num_batches = 0
        for i, [signals, ref] in enumerate(self.train_loader):

            num_batches += 1

            signals = signals.to(self.device).float()
            ref = ref.to(self.device)

            pred = self.model(signals)
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
            for _, [signals, ref] in enumerate(self.val_loader):

                num_batches += 1
                size += signals.shape[0]

                signals = signals.to(self.device).float()
                ref = ref.to(self.device)

                pred = self.model(signals)
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
            for _, [signals, ref] in enumerate(self.test_loader):

                num_batches += 1
                size += signals.shape[0]
            
                signals = signals.to(self.device).float()
                ref = ref.to(self.device)

                pred = self.model(signals)
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
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def save_recorder(self, path):
        np.save(path, self.recorder)
    '''
    def get_auc(self):
        label = 
        pred = 
        auc = roc_auc_score(label, pred)
        return auc
    '''

    def get_metrics(self, tn, tp, fn, fp):
        if tp + fp: 
            precision = tp / (tp + fp) 
        else: precision = None
        if tp + fn: 
            recall = tp / (tp + fn)
        else: recall = None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else: f1 = None
        if fp + tn:
            fallout = fp / (fp + tn)
        else: fallout = None
        return precision, recall, f1, fallout

    def increment_metrics(self, pred_max, ref, tn, tp, fn, fp):
        tn += torch.logical_and((pred_max == ref), (pred_max == 0)).type(torch.float).sum().item()
        tp += torch.logical_and((pred_max == ref), (pred_max == 1)).type(torch.float).sum().item()
        fn += torch.logical_and((pred_max != ref), (pred_max == 0)).type(torch.float).sum().item()
        fp += torch.logical_and((pred_max != ref), (pred_max == 1)).type(torch.float).sum().item()
        return tn, tp, fn, fp

    def record(self, set_type:str, avg_loss, accuracy, precision, recall, f1, fallout, auc):
        self.recorder[set_type]['loss'].append(avg_loss)
        self.recorder[set_type]['accuracy'].append(accuracy)
        self.recorder[set_type]['precision'].append(precision)
        self.recorder[set_type]['recall'].append(recall)
        self.recorder[set_type]['f1'].append(f1)
        self.recorder[set_type]['fallout'].append(fallout)
        self.recorder[set_type]['auc'].append(auc)
    
