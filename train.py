"""
e.g.
python train.py --dataset mri --batch_size 128 --depth 3 --patch_size 128 --up_mode upconv
python train.py --dataset mri --batch_size 64 --depth 3 --patch_size 128 --up_mode upsample --interpolation_mode bicubic

"""

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.unet import UNet
from utils.dataset import N2V_dataset
from utils.loss import pixel_mse_loss

    
def train():
    parser = argparse.ArgumentParser(description='argparse argument')

    parser.add_argument('--dataset',
                        type=str,
                        help='your dataset modality',                        
                        default='mri',
                        dest='dataset')

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',                        
                        default='16',
                        dest='batch_size')
    
    parser.add_argument('--depth',
                        type=int,
                        help='U-Nnet depth',   
                        default=5,                        
                        dest='depth')

    parser.add_argument('--patch_size',
                        type=int,
                        default=64,
                        dest='patch_size')

    parser.add_argument('--up_mode',
                        help='upconv or upsample',
                        default='upconv',
                        dest='up_mode')
    
    parser.add_argument('--interpolation_mode',
                        help='upsample interpolation type. nearest, linear, bilinear, bicubic, trilinear ...',
                        default='nearest',
                        dest='interpolation_mode')

    args = parser.parse_args()

    model = UNet(in_channels=1, 
                out_channels=1, 
                depth=args.depth, 
                padding=True, 
                batch_norm=True, 
                up_mode=args.up_mode,
                interpolation_mode=args.interpolation_mode)

    if torch.cuda.is_available():    
        device = torch.device("cuda:0")
        print(device)
        model.to(device)

    # summary(model, (1, 100, 100))    

    epochs = 500    
    batch_size = args.batch_size
    patch_size = (args.patch_size, args.patch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=1) 

    if args.dataset == 'kaggle':
        train_dataset = N2V_dataset('dataset_path', 
                                    patch_size=patch_size,
                                    train_val='train')

        train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=4)

        val_dataset = N2V_dataset('dataset_path', 
                                patch_size=patch_size,
                                train_val='val')
        
        val_loader = DataLoader(val_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=0)


    print(f'train images : {len(train_loader)}')
    print(f'val images : {len(val_loader)}')

    train_loss_history = []
    val_loss_history = []

    best_val_loss = 999999

    for epoch in range(epochs):   
        train_loss = 0.0
        val_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        # train        
        for i, (sources, targets, pixel_pos) in loop:
            sources, targets = sources.to('cuda'), targets.to('cuda')
            
            pred = model(sources)

            loss = pixel_mse_loss(pred, targets, pixel_pos)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()   

            # progress bar
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        

        # validation
        model.eval()
        with torch.no_grad():

            loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)

            for i, (sources, targets, pixel_pos) in loop:            
                sources, targets = sources.to('cuda'), targets.to('cuda')

                pred = model(sources)

                loss = pixel_mse_loss(pred, targets, pixel_pos)
                
                val_loss += loss.item()       
                
                # progress bar
                loop.set_description(f'validation')                    


        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)        

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f'Epoch: {epoch+1}\t train_loss: {train_loss}\t val_loss: {val_loss}')

        # scheduler.step(val_loss)
        scheduler.step()

        if best_val_loss > val_loss:
            print('=' * 100)
            print(f'val_loss is improved from {best_val_loss:.4f} to {val_loss:.4f}\t saved current weight')
            print('=' * 100)
            best_val_loss = val_loss
            
            # save weight
            torch.save(model.state_dict(), './weights/weight.pth')
            

    f = open('./results/train_loss.txt', 'w')
    train_loss_history = list(map(str, train_loss_history))
    for i,v in enumerate(train_loss_history):
        f.write(v+'\n')
    f.close()

    f = open('./results/val_loss.txt', 'w')
    val_loss_history = list(map(str, val_loss_history))
    for i,v in enumerate(val_loss_history):
        f.write(v+'\n')
    f.close()

    print('Finished')


if __name__ == '__main__':
    train()
