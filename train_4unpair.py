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
from torch.autograd import Variable

import models_unpair as models
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
parser.add_argument('--weight_recon', type=float, default=1.0, help='learning rate')
parser.add_argument('--weight_content', type=float, default=1.0, help='learning rate')
parser.add_argument('--weight_style', type=float, default=1.0, help='learning rate')
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



def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter("runs/"+ saveUtils.save_dir_tensorBoard)
    
    

    model = models.Convolutional_unpair().to(device)
    NetD = models.Discriminator().to(device)
    #load pretrained Encoder weight only 
    #torch.load_dict(pre_model.Encoder)
    saveUtils.save_log(str(args))
    saveUtils.save_log(str(summary(model, ((1,1,69,240), (1,1,69,240)))))
    saveUtils.save_log(str(summary(NetD, (1,1,69,240))))
    train_dataloader, train_dataset = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    
    train_style_dataloader, _ = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    
    
    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    valid_style_dataloader, _ = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(NetD.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    
    criterion_D = nn.BCELoss()
    criterion_G = nn.BCELoss()

    print_interval = 100
    print_num = 0
    for num_epoch in range(args.numEpoch):
        
        total_G_loss = 0
        total_recon_loss = 0
        total_G_loss = 0
        total_D_loss = 0

        total_v_loss = 0
        total_v_recon_loss = 0
        total_v_G_loss = 0
        total_v_D_loss = 0

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
            _, content_motion = content_item
            _, style_motion = style_item
            #masked_content_input = masked_content_input.to(device, dtype=torch.float)
            content_motion = content_motion.to(device, dtype=torch.float)
            style_motion = style_motion.to(device, dtype=torch.float)
            
            stylized_motion = model(content_motion, style_motion)
            
            #NetD training
            for p in NetD.parameters():
                p.requires_grad = True
            NetD.zero_grad()

            real = NetD(style_motion)
            true_labels = Variable(torch.ones_like(real))
            loss_D_real = criterion_D(real, true_labels.detach())
            
            fake = NetD(stylized_motion.detach())
            fake_labels = Variable(torch.zeros_like(fake))
            loss_D_fake = criterion_D(fake, fake_labels.detach())
            total_loss_D = loss_D_fake + loss_D_real
            
            total_loss_D.backward()
            optimizer_D.step()

            #Generator training
            for p in NetD.parameters():
                p.requires_grad = False
            NetD.zero_grad()
            
            consistency_motion = model(content_motion, content_motion)
            
            recon_loss = loss_function(consistency_motion, content_motion)
            loss_G = criterion_G(NetD(stylized_motion), true_labels.detach())
            
            total_loss = recon_loss + loss_G
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            ####
             
            total_G_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_G_loss += loss_G.item()

            total_D_loss += total_loss_D.item()
    
            
            if iter % print_interval == 0 and iter != 0:
                train_G_iter_loss =  total_G_loss * 0.01
                train_D_iter_loss = total_D_loss * 0.01
                train_recon_iter_loss =  total_recon_loss * 0.01
                train_G_iter_loss = total_G_loss * 0.01
                log = "Train: [Epoch %d][Iter %d] [total_G_iter_loss: %.4f] [train_D_iter_loss: %.4f] [recon loss: %.4f] [G loss: %.4f] " %\
                                             (num_epoch, iter, train_G_iter_loss, train_D_iter_loss, train_recon_iter_loss, train_G_iter_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("total_G_iter_loss/ iter", train_G_iter_loss, print_num)
                writer.add_scalar("train_D_iter_loss/ iter", train_D_iter_loss, print_num)
                writer.add_scalar("total_recon_loss/ iter", train_recon_iter_loss, print_num)
                writer.add_scalar("total_G_loss/ iter", train_G_iter_loss, print_num)

                total_G_loss = 0
                total_recon_loss = 0
                total_G_loss = 0
                total_D_loss = 0
                 
                
        #############validation per epoch ############
        for iter, item in enumerate(zip(valid_dataloader,valid_style_dataloader)):
            model.eval()
            NetD.eval()
            content_item, style_item = item
            _, content_motion = content_item
            _, style_motion = style_item
            #masked_content_input = masked_content_input.to(device, dtype=torch.float)
            content_motion = content_motion.to(device, dtype=torch.float)
            style_motion = style_motion.to(device, dtype=torch.float)
            
            with torch.no_grad():
                stylized_motion = model(content_motion, style_motion)
                consistency_motion = model(content_motion, content_motion)
                real = NetD(style_motion)
                fake = NetD(stylized_motion)

            loss_D_real = criterion_D(real, true_labels)
            loss_D_fake = criterion_D(fake, fake_labels)
            recon_loss = loss_function(consistency_motion, content_motion)
            loss_G = criterion_G(NetD(stylized_motion), true_labels)
            
            total_val_loss =  recon_loss + loss_G
            total_val_D_loss = loss_D_fake + loss_D_real

            total_v_loss += total_val_loss.item()
            total_v_recon_loss += recon_loss.item()
            total_v_G_loss += loss_G.item()
            total_v_D_loss += total_val_D_loss.item()
            
            model.train()
            NetD.train()
        #pred, gt, masked_input, style_input,
        saveUtils.save_result(content_motion, style_motion, stylized_motion, consistency_motion, num_epoch)
        valid_epoch_loss = total_v_loss/len(valid_dataloader)
        valid_epoch_recon_loss = total_v_recon_loss/len(valid_dataloader)
        valid_epoch_G_loss = total_v_G_loss/len(valid_dataloader)
        valid_epoch_D_loss = total_v_D_loss/len(valid_dataloader)
        
        log = "Valid: [Epoch %d] [valid_epoch_loss(G): %.4f] [valid_epoch_recon_loss: %.4f] [valid_epoch_G_loss: %.4f] [valid_epoch_D_loss: %.4f]" %\
                                             (num_epoch, valid_epoch_loss, valid_epoch_recon_loss, valid_epoch_G_loss, valid_epoch_D_loss)
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("valid_epoch_loss/ Epoch", valid_epoch_loss, num_epoch)
        writer.add_scalar("valid_epoch_recon_loss/ Epoch", valid_epoch_recon_loss, num_epoch) 
        writer.add_scalar("valid_epoch_G_loss/ Epoch", valid_epoch_G_loss, num_epoch) 
        writer.add_scalar("valid_epoch_D_loss/ Epoch", valid_epoch_D_loss, num_epoch) 
        saveUtils.save_model(model, num_epoch) # save model per epoch
        #validation per epoch ############
        
        
        
if __name__ == "__main__":
    main(args)