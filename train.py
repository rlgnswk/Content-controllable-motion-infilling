from ast import arg
import torch
import torch.nn as nn               # Linear
import torch.nn.functional as F     # relu, softmax
import torch.optim as optim         # Adam Optimizer
from torch.distributions import Categorical # Categorical import from torch.distributions module
import torch.multiprocessing as mp # multi processing
import time 

from matplotlib import pyplot as plt ###for plot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import os

import models
import utils
import data_load
#input sample of size 69 × 240
#latent space 3 × 8 × 256 tensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--model_type', type=str, default='AE') 
parser.add_argument('--datasetPath', type=str, default='C:/Users/VML/Desktop/2022_Spring/Motion_Graphics/Final_project/downloadCode/train_data/')
parser.add_argument('--ValdatasetPath', type=str, default='C:/Users/VML/Desktop/2022_Spring/Motion_Graphics/Final_project/downloadCode/valid_data/')
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=200, help='input batch size for training')
parser.add_argument('--batchSize', type=int, default=80, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()




def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter("runs/"+ saveUtils.save_dir_tensorBoard)
    
    
    if args.model_type == 'VAE':
        model = models.Convolutional_VAE().to(device)
    else:
        model = models.Convolutional_AE().to(device)
    
    train_dataloader = data_load.get_dataloader(args.datasetPath , args.batchSize)
    valid_dataloader = data_load.get_dataloader(args.ValdatasetPath , args.batchSize)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    
    print_interval = 100
    print_num = 0
    for num_epoch in range(args.numEpoch):
        
        total_loss = 0
        total_v_loss = 0
        for iter, item in enumerate(train_dataloader):
            print_num +=1
            
            masked_input, gt_image = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            
            pred = model(masked_input)
            
            train_loss = loss_function(pred, gt_image)
            total_loss += train_loss.item()
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            if iter % print_interval == 0 and iter != 0:
                train_iter_loss =  total_loss*0.01
                log = "Train: [Epoch %d][Iter %d] [Train Loss: %.4f]" % (num_epoch, iter, train_iter_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Train Loss/ iter", train_iter_loss, print_num)
                total_loss = 0
                
        #validation per epoch ############
        for iter, item in enumerate(valid_dataloader):
            model.eval()
            masked_input, gt_image = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            
            with torch.no_grad():
                pred = model(masked_input)
            
            val_loss = loss_function(pred, gt_image)
            total_v_loss += val_loss.item()

            model.train()
            
        saveUtils.save_result(pred, gt_image, masked_input, num_epoch)
        valid_epoch_loss = total_v_loss/len(valid_dataloader)
        log = "Valid: [Epoch %d] [Valid Loss: %.4f]" % (num_epoch, valid_epoch_loss)
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("Valid Loss/ Epoch", valid_epoch_loss, num_epoch)    
        saveUtils.save_model(model, num_epoch) # save model per epoch
        #validation per epoch ############
        
        
        
if __name__ == "__main__":
    main(args)