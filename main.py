import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from data_processing.get_dataloader import *
from data_processing.datasets import ImageDataset, ChunkedMatrixDataset, ChunkedImageDataset, NumericalDataset, PatchTSTDataset, get_datasets
from trainer import Trainer
from models.cnn import CNN, BigCNN, k5_CNN, k17_CNN
from models.cnn_num import CNN_numerical
from models.cnn_simple import CNN_Deep_4x
from models.ViT import ViT, ViT_tiny, Swin
from models.patchTST.patchTST import PatchTST
from models.resnet import ResNet, ResidualBlock
from constants import *


import random
import argparse


def main(model_type:str, epochs:int=10, batch_size:int=256, device='cuda', rawsignal_num=150):

    if model_type == 'Numerical':
        meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
        pcr_fold_path = '/home/coalball/projects/sssl/Control'

        print('loading datasets...')
        train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path, rawsignal_num=rawsignal_num)
        print('loading training data')
        train_loader = get_numerical_dataloaders(train_set, batch_size=batch_size)
        print('loading validation data')
        val_loader = get_numerical_dataloaders(val_set, batch_size=batch_size)
        print('loading test data')
        test_loader = get_numerical_dataloaders(test_set, batch_size=batch_size)
    
    elif model_type == 'PatchTST':
        meth_fold_path = '/home/coalball/projects/sssl/M_Sssl_Cg'
        pcr_fold_path = '/home/coalball/projects/sssl/Control'

        #meth_fold_path = '/home/coalball/projects/methBert2/toys/M_Sssl_Cg'
        #pcr_fold_path = '/home/coalball/projects/methBert2/toys/Control2'

        print('loading datasets...')
        train_set, val_set, test_set = get_datasets(meth_fold_path, pcr_fold_path)
        print('loading training data')
        train_loader = get_patchtst_dataloaders(train_set, batch_size=batch_size, patch_len=PATCH_LEN, stride=STRIDE)
        print('loading validation data')
        val_loader = get_patchtst_dataloaders(val_set, batch_size=batch_size, patch_len=PATCH_LEN, stride=STRIDE)
        print('loading test data')
        test_loader = get_patchtst_dataloaders(test_set, batch_size=batch_size, patch_len=PATCH_LEN, stride=STRIDE)      
    
    elif model_type == 'Simple' or model_type == 'Simple_Res':
        train_path = '/home/coalball/projects/methBert2/toys/simple_4x/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_4x/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_4x/test'

        train_loader = get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = get_chunked_image_dataloaders(test_path, batch_size=batch_size) 

    elif model_type == 'Simple_Res_40':
        train_path = '/home/coalball/projects/methBert2/toys/simple_40/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_40/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_40/test'

        train_loader = new_get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = new_get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = new_get_chunked_image_dataloaders(test_path, batch_size=batch_size) 

    elif model_type == 'Simple_Res_60':
        train_path = '/home/coalball/projects/methBert2/toys/simple_60/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_60/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_60/test'

        train_loader = new_get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = new_get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = new_get_chunked_image_dataloaders(test_path, batch_size=batch_size) 

    elif model_type == 'Simple_Res_10':
        train_path = '/home/coalball/projects/methBert2/toys/simple_10/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_10/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_10/test'

        train_loader = new_get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = new_get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = new_get_chunked_image_dataloaders(test_path, batch_size=batch_size) 

    elif model_type == 'Simple_Res_1':
        train_path = '/home/coalball/projects/methBert2/toys/simple_1/train'
        val_path = '/home/coalball/projects/methBert2/toys/simple_1/val'
        test_path = '/home/coalball/projects/methBert2/toys/simple_1/test'

        train_loader = new_get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = new_get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = new_get_chunked_image_dataloaders(test_path, batch_size=batch_size) 
    
    else:
        train_path = '/home/coalball/projects/methBert2/toys/224_full/train'
        val_path = '/home/coalball/projects/methBert2/toys/224_full/val'
        test_path = '/home/coalball/projects/methBert2/toys/224_full/test'

        train_loader = get_chunked_image_dataloaders(train_path, batch_size=batch_size)
        val_loader = get_chunked_image_dataloaders(val_path, batch_size=batch_size)
        test_loader = get_chunked_image_dataloaders(test_path, batch_size=batch_size)

    if model_type == 'ResNet':
        model = resnet18()
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 2)
        model.to(device)
    elif model_type == 'CNN':
        model = CNN()
        model.to(device)
    elif model_type == 'bigCNN':
        model = BigCNN()
        model.to(device)
    elif model_type == 'k5CNN':
        model = k5_CNN()
        model.to(device)
    elif model_type == 'k17CNN':
        model = k17_CNN()
        model.to(device)
    elif model_type == 'ViT':
        model = ViT(n_classes=2, head_dropout=0.2)
        model.to(device)
    elif model_type == 'tinyViT':
        model = ViT_tiny(n_classes=2, head_dropout=0.2)
        model.to(device)
    elif model_type == 'Swin':
        model = Swin(n_classes=2, head_dropout=0.2)
        model.to(device)
    elif model_type == 'Numerical':
        model = CNN_numerical()
        model.to(device)
    elif model_type == 'Simple':
        model = CNN_Deep_4x()
        model.to(device)
    elif model_type == 'Simple_Res':
        model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 1).to(device)
    elif model_type == 'Simple_Res_40':
        model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 3).to(device)
    elif model_type == 'Simple_Res_60':
        model = ResNet(ResidualBlock, [2, 2, 2], linear_input = 256 * 3 * 6).to(device)
    elif model_type == 'Simple_Res_10':
        model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=2).to(device)
    elif model_type == 'Simple_Res_1':
        model = ResNet(block_type=ResidualBlock, layers=[2, 2, 2], linear_input=256 * 3 * 1, last_avg_size=1, maxpool=False).to(device)
    elif model_type == 'PatchTST':
        num_patch = (max(SIGNAL_LENGTH, PATCH_LEN)-PATCH_LEN) // STRIDE + 1
        model = PatchTST(c_in=1,
                target_dim=2,
                patch_len=PATCH_LEN,
                stride=STRIDE,
                num_patch=num_patch,
                n_layers=3,
                n_heads=16,
                d_model=128,
                shared_embedding=True,
                d_ff=128,                        
                dropout=0.2,
                head_dropout=0.2,
                act='relu',
                head_type='classification',
                res_attention=True
            )
        model.to(device)   
    else:
        print('No Such Model Type')
        return
    #model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2).to(device)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    #model = CNN().to('cuda')
    trainer = Trainer(model=model, dataloaders=[train_loader, val_loader, test_loader], lr = (1e-4) * (batch_size//64))
    
    path = 'saved_models_full/%s/'%model_type
    if not os.path.exists(path): os.mkdir(path)
    trainer.train(n_epochs=epochs, model_saved_dir=path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='<MethBERT2 Training>')
    parser.add_argument('--model', default='ViT', type=str, required=True, help="DL models used for the training.")
    #parser.add_argument('--samples', default=None, type=int, required=True, help="Number of epochs.")
    parser.add_argument('--epochs', default=1, type=int, required=True, help="Number of epochs.")
    parser.add_argument('--batchsize', default=256, type=int, required=False, help='set batch size for training')
    args = parser.parse_args()
    main(model_type=args.model, epochs=args.epochs, batch_size=args.batchsize)
    #main(model_type=args.model, samples=args.samples, epochs=args.epochs)

