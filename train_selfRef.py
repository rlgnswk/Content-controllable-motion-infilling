from ast import arg
import torch
import torch.nn as nn               # Linear
import torch.nn.functional as F     # relu, softmax
import torch.optim as optim         # Adam Optimizer
from torch.distributions import Categorical # Categorical import from torch.distributions module
import torch.multiprocessing as mp # multi processing
import time 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt ###for plot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import os

from torchinfo import summary

import models_selfRef as models
import utils4SelfRef as utils
import data_load
#input sample of size 69 × 240
#latent space 3 × 8 × 256 tensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--model_type', type=str, default='AE') 
parser.add_argument('--datasetPath', type=str, default='/input/MotionInfillingData/train_data')
parser.add_argument('--ValdatasetPath', type=str, default='/input/MotionInfillingData/valid_data')
parser.add_argument('--saveDir', type=str, default='./experiment')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=200, help='input batch size for training')
parser.add_argument('--batchSize', type=int, default=80, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_recon', type=float, default=0.1, help='learning rate')
parser.add_argument('--weight_content', type=float, default=10.0, help='learning rate')
parser.add_argument('--weight_style', type=float, default=10.0, help='learning rate')
parser.add_argument('--weight_output_style', type=float, default=1.0, help='learning rate')
args = parser.parse_args()


def calc_mean_std( feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_style_loss(out_feat_list, style_feat_list, style_loss_function):
    assert len(out_feat_list) == len(style_feat_list)
    style_loss= 0
    
    out_mean, out_std = calc_mean_std(out_feat_list)
    style_mean, stlye_std = calc_mean_std(style_feat_list)
    style_loss = (style_loss_function(out_mean, style_mean) + style_loss_function(out_std, stlye_std))
    return style_loss


def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter("runs/"+ saveUtils.save_dir_tensorBoard)
    
    
    if args.model_type == 'Up':
        model = models.Convolutional_AE_SelfRef().to(device)
    else:
        model = models.Convolutional_AE_SelfRef().to(device)
    
    
    #load pretrained Encoder weight only 
    #torch.load_dict(pre_model.Encoder)
    saveUtils.save_log(str(args))
    saveUtils.save_log(str(summary(model, ((1,1,69,240), (1,1,69,240)))))
    
    train_dataloader, train_dataset = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    
    train_style_dataloader, _ = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    
    
    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    valid_style_dataloader, _ = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    style_loss_function = nn.MSELoss()
    
    print_interval = 100
    print_num = 0
    for num_epoch in range(args.numEpoch):
        
        total_loss = 0
        total_train_recon_loss = 0
        total_train_content_loss = 0
        total_train_style_loss = 0
        total_train_output_style_loss = 0

        total_v_loss = 0
        total_v_recon_loss = 0
        total_v_content_loss = 0
        total_v_style_loss = 0
        total_v_output_style_loss = 0

        if train_dataset.masking_length_mean < 120 and num_epoch is not 0 and num_epoch%10 == 0:
            train_dataset.masking_length_mean = train_dataset.masking_length_mean + 10
            valid_dataset.masking_length_mean = train_dataset.masking_length_mean
            train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
            
            log = "Current train_dataset.masking_length_mean: %d" % train_dataset.masking_length_mean
            print(log)
            saveUtils.save_log(log)
            
        for iter, item in enumerate(zip(train_dataloader,train_style_dataloader)):
            print_num +=1
            content_item, style_item = item
            _, motion_a = content_item
            _, motion_b = style_item
            #masked_content_input = masked_content_input.to(device, dtype=torch.float)
            motion_a = motion_a.to(device, dtype=torch.float)
            motion_b = motion_b.to(device, dtype=torch.float)
            
            out_style_B_Content_A, out_style_A_Content_B, content_latent_a, style_latent_b, content_latent_b, style_latent_a  = model(motion_a, motion_b)
            

            out_style_A_Content_A, out_style_B_Content_B, content_latent_out_style_B_Content_A, style_latent_out_style_A_Content_B, content_latent_out_style_A_Content_B, style_latent_out_style_B_Content_A \
                                                                                                                                                                = model(out_style_B_Content_A, out_style_A_Content_B)

            reconstruction_loss_a = loss_function(motion_a, out_style_A_Content_A)
            reconstruction_loss_b = loss_function(motion_b, out_style_B_Content_B)
            
            style_loss_a = calc_style_loss(motion_a, out_style_A_Content_B, style_loss_function)
            style_loss_b = calc_style_loss(motion_b, out_style_B_Content_A, style_loss_function)
            
            loss_content_latent_a = loss_function(content_latent_a, content_latent_out_style_B_Content_A)
            loss_content_latent_b = loss_function(content_latent_b, content_latent_out_style_A_Content_B)
            
            loss_style_latent_a = loss_function(style_latent_a, style_latent_out_style_A_Content_B)
            loss_style_latent_b = loss_function(style_latent_b, style_latent_out_style_B_Content_A)


            recon_total = (reconstruction_loss_a + reconstruction_loss_b) * args.weight_recon
            output_style_losss = (style_loss_a + style_loss_b) * args.weight_output_style
            content_total = (loss_content_latent_a + loss_content_latent_b) *args.weight_content
            style_total = (loss_style_latent_a + loss_style_latent_b) * args.weight_style
            
            total_train_loss =  recon_total + content_total + style_total + output_style_losss
            
            total_loss += total_train_loss.item()
            total_train_recon_loss += recon_total.item()
            total_train_content_loss += content_total.item()
            total_train_style_loss += style_total.item()
            total_train_output_style_loss += output_style_losss.item()
            
            optimizer.zero_grad()
            total_train_loss.backward()
            #train_loss.backward()
            optimizer.step()
            
            if iter % print_interval == 0 and iter != 0:
                train_iter_loss =  total_loss * 0.01
                total_train_recon_loss = total_train_recon_loss * 0.01
                train_iter_content_loss = total_train_content_loss * 0.01
                train_iter_style_loss = total_train_style_loss * 0.01
                train_iter_output_style_loss = total_train_output_style_loss * 0.01

                log = "Train: [Epoch %d][Iter %d] [Valid Loss: %.4f] [Recon Loss: %.4f] [Content Latent Loss: %.4f] [Style Latent Loss: %.4f] [Output Style Latent Loss: %.4f]" %\
                                             (num_epoch, iter, train_iter_loss, total_train_recon_loss, train_iter_content_loss, train_iter_style_loss, train_iter_output_style_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Train Total Loss/ iter", train_iter_loss, print_num)
                writer.add_scalar("Train Recon Loss/ iter", total_train_recon_loss, print_num)
                writer.add_scalar("Train Content Loss/ iter", train_iter_content_loss, print_num)
                writer.add_scalar("Train Style Loss/ iter", train_iter_style_loss, print_num)
                writer.add_scalar("Train Output Style Loss/ iter", train_iter_output_style_loss, print_num)
                total_loss = 0
                total_train_recon_loss = 0
                total_train_content_loss = 0
                total_train_style_loss = 0
                total_train_output_style_loss = 0
                 
                
        #############validation per epoch ############
        for iter, item in enumerate(zip(valid_dataloader,valid_style_dataloader)):
            model.eval()
            content_item, style_item = item
            _, motion_a = content_item
            _, motion_b = style_item
            #masked_content_input = masked_content_input.to(device, dtype=torch.float)
            motion_a = motion_a.to(device, dtype=torch.float)
            motion_b = motion_b.to(device, dtype=torch.float)
            
            with torch.no_grad():
                out_style_B_Content_A, out_style_A_Content_B, content_latent_a, style_latent_b, content_latent_b, style_latent_a  = model(motion_a, motion_b)
            
                out_style_A_Content_A, out_style_B_Content_B, content_latent_out_style_B_Content_A, style_latent_out_style_A_Content_B, content_latent_out_style_A_Content_B, style_latent_out_style_B_Content_A \
                                                                                                                                                                = model(out_style_B_Content_A, out_style_A_Content_B)
            reconstruction_loss_a = loss_function(motion_a, out_style_A_Content_A)
            reconstruction_loss_b = loss_function(motion_b, out_style_B_Content_B)
            
            style_loss_a = calc_style_loss(motion_a, out_style_A_Content_B, style_loss_function)
            style_loss_b = calc_style_loss(motion_b, out_style_B_Content_A, style_loss_function)

            loss_content_latent_a = loss_function(content_latent_a, content_latent_out_style_B_Content_A)
            loss_content_latent_b = loss_function(content_latent_b, content_latent_out_style_A_Content_B)
            
            loss_style_latent_a = loss_function(style_latent_a, style_latent_out_style_A_Content_B)
            loss_style_latent_b = loss_function(style_latent_b, style_latent_out_style_B_Content_A)
            

            recon_total = (reconstruction_loss_a + reconstruction_loss_b) * args.weight_recon
            output_style_losss = (style_loss_a + style_loss_b) * args.weight_output_style
            content_total = (loss_content_latent_a + loss_content_latent_b) *args.weight_content
            style_total = (loss_style_latent_a + loss_style_latent_b) * args.weight_style
            
            total_val_loss =  recon_total + content_total + style_total + output_style_losss
            
            total_v_loss += total_val_loss.item()
            total_v_recon_loss += recon_total.item()
            total_v_content_loss += content_total.item()
            total_v_style_loss += style_total.item()
            total_v_output_style_loss += output_style_losss.item()
            
            model.train()

        #pred, gt, masked_input, style_input,
        saveUtils.save_result(motion_a, motion_b, out_style_B_Content_A, out_style_A_Content_A, num_epoch)
        valid_epoch_loss = total_v_loss/len(valid_dataloader)
        valid_epoch_recon_loss = total_v_recon_loss/len(valid_dataloader)
        valid_epoch_content_loss = total_v_content_loss/len(valid_dataloader)
        valid_epoch_style_loss = total_v_style_loss/len(valid_dataloader)
        valid_epoch_output_style_loss = total_v_output_style_loss/len(valid_dataloader)
        log = "Valid: [Epoch %d] [Valid Loss: %.4f] [Recon Loss: %.4f] [Content Latent Loss: %.4f] [Style Latent Loss: %.4f] [Output Style Latent Loss: %.4f]" %\
                                             (num_epoch, valid_epoch_loss, valid_epoch_recon_loss, valid_epoch_content_loss, valid_epoch_style_loss, valid_epoch_output_style_loss)
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("Valid Total Loss/ Epoch", valid_epoch_loss, num_epoch)
        writer.add_scalar("Valid Recon Loss/ Epoch", valid_epoch_recon_loss, num_epoch) 
        writer.add_scalar("Valid Content Loss/ Epoch", valid_epoch_content_loss, num_epoch) 
        writer.add_scalar("Valid Style Loss/ Epoch", valid_epoch_style_loss, num_epoch) 
        writer.add_scalar("Valid Output Style Loss/ Epoch", valid_epoch_output_style_loss, num_epoch) 

        saveUtils.save_model(model, num_epoch) # save model per epoch
        #validation per epoch ############
        
        
        
if __name__ == "__main__":
    main(args)