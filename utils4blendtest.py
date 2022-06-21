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
    
    def save_result(self, pred, gt_image, blend_gt, gt_blended_image, blend_input, maksed_input, epoch):
        pred = pred.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        gt_image = gt_image.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        blend_gt = blend_gt.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        gt_blended_image = gt_blended_image.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        blend_input = blend_input.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        maksed_input = maksed_input.detach().squeeze(1).permute(0,2,1).cpu().numpy()

        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_pred", pred)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_gt_image", gt_image)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_gt_blend", blend_gt)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_gt_blended_image", gt_blended_image)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_blend_input", blend_input)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_Masked_input", maksed_input)


        cmap = plt.get_cmap('jet') 
        
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(pred[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("prediction", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_prediction_'+str(i)+'.png')
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(gt_image[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("gt_image", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_gt_image_'+str(i)+'.png')  
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(blend_gt[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("blend_gt", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_blend_gt_'+str(i)+'.png')

        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(gt_blended_image[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("gt_blended_image", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_gt_blended_image_'+str(i)+'.png')
        
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(blend_input[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("blend_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_blend_input_'+str(i)+'.png')

        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(maksed_input[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("maksed_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_masked_input'+str(i)+'.png')

        plt.close('all')
    
    def save_result_test(self, output, iter, alpha):
        output = output.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        np.save(self.save_dir_validation + '/epoch_' + str(iter)+"_alpha_"+str(alpha) + "_TestOutput", output)
