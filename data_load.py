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
    orig_height = original_input.shape[0] #69
    orig_width = original_input.shape[1] #240
    masked_input = original_input.copy() # deep copy
    mask_width = masking_length + noise(IsNoise) #55 ~ 65
    
    masking = np.zeros((orig_height, mask_width))# generate zeros matrix for masking: orig_height x mask_width
    index = random.randint(0, orig_width - mask_width)# sampling the start point of masking 
    masked_input[: , index : index+mask_width] = masking # masking
    
    return masked_input, original_input



class MotionLoader(Dataset):
        def __init__(self, root, IsNoise=False):
            super(MotionLoader, self).__init__()
            #loda all the data as array
            print("#### MotionLoader ####")
            print("####### load data from {} ######".format(root))
            
            self.IsNoise = IsNoise
            data_list = os.listdir(root)
            for idx , name in enumerate(data_list):
                file_path = os.path.join(root, name)
                if idx == 0 :
                    self.data = np.load(file_path)['clips'] #(clip num, 240, 73) (2846, 240, 73)
                else:
                    self.data = np.concatenate((self.data, np.load(file_path)['clips']), axis=0) # concat all the data (# of data , 240 , 73)
            
            print("####### total Lentgh is {} ######".format(self.__len__()))
        
        def __getitem__(self, idx):
            #load item #processing
            gt_image = self.remove_foot_contacts(self.data[idx]) # remove_foot_contacts  (240 , 73) --> (240 , 69)
            #switch (240 , 69) --> (69, 240)
            gt_image = np.transpose(gt_image)
            #get masked input
            masked_input, gt_image = masking_input(gt_image, self.IsNoise)
            
            #return maksed_input and gt CHW #(69, 240) --> (1, 69, 240)
            return np.expand_dims(masked_input, axis=0), np.expand_dims(gt_image, axis=0) # it will (batch, 1, 69, 240)
        
        def __len__(self):
            return len(self.data)
    
        
        def remove_foot_contacts(self, data): # chaneel 73 -> 69, 69 is baseline 
            assert data.shape[1] == 73
            return np.delete(data, obj=list(range(data.shape[1] - 4, data.shape[1])), axis=1)
        
        
def get_dataloader(dataroot, batch_size):
    dataloader = MotionLoader(dataroot)
    print("# of dataset:", len(dataloader))
    
    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return dataloader


if __name__ == "__main__":
    print("START")
    data_root = 'C:/Users/VML/Desktop/2022_Spring/Motion_Graphics/Final_project/downloadCode/valid_data/'
    batch_size = 32
    datalodaer = get_dataloader(data_root , 32)
    
    for iter, item in enumerate(datalodaer): 
        masked_input, gt_image = item
        print(iter)
        print(masked_input.shape)
        print(gt_image.shape)
        if iter == 5 :
            break
    
    print("END")