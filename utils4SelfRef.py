import os
import os.path
import torch
import sys
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

class saveData():
    def __init__(self, args):
        self.args = args
        #Generate Savedir folder
        self.save_dir = os.path.join(args.saveDir, args.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        #Generate Savedir/model
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        #Generate Savedir/validation
        self.save_dir_validation = os.path.join(self.save_dir, 'validation')
        if not os.path.exists(self.save_dir_validation):
            os.makedirs(self.save_dir_validation)

        #Generate Savedir/checkpoint
        self.save_dir_checkpoint = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(self.save_dir_checkpoint):
            os.makedirs(self.save_dir_checkpoint)

        #Generate Savedir/tensorBoard
        self.save_dir_tensorBoard = os.path.join(self.save_dir, 'tensorBoard')
        if not os.path.exists(self.save_dir_tensorBoard):
            os.makedirs(self.save_dir_tensorBoard)

        #Generate Savedir/log.txt
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
    
    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()
        
    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_' + str(epoch) + '.pt')
    
    def save_result(self, motion_a, motion_b, out_style_B_Content_A, out_style_A_Content_A, epoch):
        motion_a = motion_a.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        motion_b = motion_b.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        out_style_B_Content_A = out_style_B_Content_A.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        out_style_A_Content_A = out_style_A_Content_A.detach().squeeze(1).permute(0,2,1).cpu().numpy()

        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "motion_a", motion_a)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "motion_b", motion_b)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "out_style_B_Content_A", out_style_B_Content_A)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "out_style_B_Content_A", out_style_A_Content_A)

        cmap = plt.get_cmap('jet') 
        
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(motion_a[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("prediction", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'motion_a'+str(i)+'.png')
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(motion_b[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("gt", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'motion_b'+str(i)+'.png')  
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(out_style_B_Content_A[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("masked_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'out_style_B_Content_A'+str(i)+'.png')

        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(out_style_A_Content_A[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("style_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'out_style_A_Content_A'+str(i)+'.png')
        
        plt.close('all')