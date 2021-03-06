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

import models as pretrain_models
import models_blend_controllable as models
import models_unpair as Discriminator_model
import utils4blendtestABC as utils
import data_load_ABC as data_load
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
args = parser.parse_args()


def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter(saveUtils.save_dir_tensorBoard)
    
    
    if args.model_type == 'VAE':
        model = models.Convolutional_blend().to(device)
    else:
        model = models.Convolutional_blend().to(device)
    
    pretrained_path = "/root/Motion_Style_Infilling/pertrained/0530maskDone1CurriculLearning_bn_model_199.pt"
    GT_model = pretrain_models.Convolutional_AE().to(device)
    GT_model.load_state_dict(torch.load(pretrained_path))
    GT_model.eval()


    pretrained_modelpath = "/root/Motion_Style_Infilling/experiment/controllableFirst0609/model/model_310.pt"
    model.load_state_dict(torch.load(pretrained_modelpath))
    model.eval()

    NetD = Discriminator_model.Discriminator().to(device)

    saveUtils.save_log(str(args))
    saveUtils.save_log(str(summary(model, ((1,1,69,240), (1,1,69,240)))))
    saveUtils.save_log(str(summary(NetD, (1,1,69,240))))

    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    valid_dataset.masking_length_mean = 120
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
    log = "valid_dataset.masking_length_mean: %d" % valid_dataset.masking_length_mean
    print(log)
    saveUtils.save_log(log)

    #validation per epoch ############
    for iter, item in enumerate(valid_dataloader):
        model.eval()
        NetD.eval()

        masked_input, gt_image, blend_part, blend_gt, tail_image = item
        masked_input = masked_input.to(device, dtype=torch.float)
        gt_image = gt_image.to(device, dtype=torch.float)
        blend_part = blend_part.to(device, dtype=torch.float)
        blend_gt = blend_gt.to(device, dtype=torch.float)
        tail_image = tail_image.to(device, dtype=torch.float)


        blend_input = masked_input + blend_part
        
        with torch.no_grad():
            
            if iter%100 == 0:
                gt_blended_image= GT_model(blend_input)
                pred_affine, pred_recon = model(masked_input, blend_gt)
                saveUtils.save_result(pred_affine, gt_image, blend_gt, gt_blended_image, blend_input, masked_input, tail_image, iter)
                for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    output = model.test2(masked_input, gt_image, blend_gt, alpha)
                    saveUtils.save_result_test(output, iter, alpha)
            
        
if __name__ == "__main__":
    main(args)