import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

def noise(option = True):
    if option == True:
        return random.randint(-5, 5)
    else:
        return 0

def masking_input(original_input, masking_length = 60, IsNoise = False):
    # input sample of size 69 × 240
    # in paper 10 ~ 120
    # I will set 60
    orig_height = original_input.shape[0]
    orig_width = original_input.shape[1]
    masked_input = original_input.copy() # deep copy
    mask_width = masking_length + noise(IsNoise) #55 ~ 65
    
    masking = np.zeros((orig_height, mask_width))# generate zeros matrix for masking: orig_height x mask_width
    index = random.randint(0, orig_width - mask_width)# sampling the start point of masking 
    masked_input[: , index : index+mask_width] = masking # masking
    
    return masked_input, original_input



class MotionLoader(Dataset):
        def __init__(self, root):
            super(MotionLoader, self).__init__()
            #set the item list
            
        def __getitem__(self, idx):
            #load item
            
            #processing
            
            #return maksed_input and gt
            pass
        
        def __len__(self):
            pass
        

def get_dataloader(dataroot, batch_size):
    dataloader = MotionLoader(dataroot)
    print("# of train_dataset:", len(dataloader))
    
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader