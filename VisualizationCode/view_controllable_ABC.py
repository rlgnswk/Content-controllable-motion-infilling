import os
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from matplotlib.animation import ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
from Quaternions import Quaternions


def animation_plot(animations, interval=33.33):
    footsteps = []

    for ai in range(len(animations)):
        anim = animations[ai][0].copy()[:, 3:]

        joints, root_x, root_z, root_r = anim[:, :-7], anim[:, -7], anim[:, -6], anim[:, -5]
        joints = joints.reshape((len(joints), -1, 3))

        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0, 0, 0]])

        for i in range(len(joints)):
            joints[i, :, :] = rotation * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
            joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
            #rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
            rotation = Quaternions.from_angle_axis(-0.0, np.array([0, 1, 0])) * rotation
            offsets.append(rotation * np.array([0, 0, 1]))
            #translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
            translation = translation + rotation * np.array([0, 0, 0])

        animations[ai] = joints
        footsteps.append(anim[:, -4:])

    footsteps = np.array(footsteps)
    print(footsteps.shape)

    scale = 1.25 * ((len(animations)) / 2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale * 30, scale * 30)
    ax.set_zlim3d(0, scale * 60)
    ax.set_ylim3d(-scale * 30, scale * 30)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    #ax.set_aspect('equal')
    ax.set_aspect('auto')
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    lines = []

    parents = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 11, 17, 18, 19])

    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0, 0], [0, 0], [0, 0], color=acolors[ai],
                               lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in
                      range(anim.shape[1])])

    def animate(i):
        changed = []
        for ai in range(len(animations)):
            offset = 25 * (ai - ((len(animations)) / 2))
            for j in range(len(parents)):
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [animations[ai][i, j, 0] + offset, animations[ai][i, parents[j], 0] + offset],
                        [-animations[ai][i, j, 2], -animations[ai][i, parents[j], 2]])
                    lines[ai][j].set_3d_properties(
                        [animations[ai][i, j, 1], animations[ai][i, parents[j], 1]])
            changed += lines

        return changed

    plt.tight_layout()

    ani = animation.FuncAnimation(fig,
                                  animate, np.arange(len(animations[0])), interval=interval)

    plt.show()
    
    
def add_foot_contacts(data): # chaneel 73 -> 69, 69 is baseline 
    assert data.shape[2] == 69
    new_one = np.zeros(data[:,:,:1].shape)
    con = np.concatenate((data ,new_one),axis= 2)
    con = np.concatenate((con ,new_one),axis= 2)
    con = np.concatenate((con ,new_one),axis= 2)
    result = np.concatenate((con ,new_one),axis= 2)
    return result
            
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str) # experiment name 
parser.add_argument('--epoch', type=int, default=199, help='input batch size for training') #epoch num
args = parser.parse_args()


if __name__ == '__main__':

    data_path = "C:/Users/VML/Documents/GitHub/Motion_Style_Infilling/experiment/"+args.name +"/validation/"
    epoch = args.epoch
    # style 
    '''db_GT = 'epoch_'+str(epoch)+'motion_a.npy'
    db_Results = 'epoch_'+str(epoch)+'motion_b.npy'
    db_Input = 'epoch_'+str(epoch)+'out_style_B_Content_A.npy'
    '''
    #infilling
    #db_GT = 'epoch_'+str(epoch)+'_GT.npy'
    #db_Results = 'epoch_'+str(epoch)+'_Masked_Input.npy'
    #db_Input = 'epoch_'+str(epoch)+'_Results.npy'

    #blend
    db_GT = 'epoch_0_gt_image.npy'
    db_GT_tail = 'epoch_0_tail_input.npy'
    
    db_ResultAp0 = 'epoch_0_alpha_0.0_output.npy'
    
    db_ResultAp4 = 'epoch_0_alpha_0.4_output.npy'
    
    db_ResultAp6 = 'epoch_0_alpha_0.8_output.npy' 
    
    db_ResultAp10 = 'epoch_0_alpha_1.0_output.npy' 
    
    db_Blend_GT = 'epoch_0_gt_blend.npy'
    
    
    database_GT= np.load(os.path.join(data_path, db_GT))
    #print(database.shape)
    database_GT = add_foot_contacts(database_GT)


    database_GT_tail= np.load(os.path.join(data_path, db_GT_tail))
    #print(database.shape)
    database_GT_tail = add_foot_contacts(database_GT_tail)

    database_ResultAp0= np.load(os.path.join(data_path, db_ResultAp0))
    #print(database.shape)
    database_ResultAp0 = add_foot_contacts(database_ResultAp0)

    database_ResultAp4= np.load(os.path.join(data_path, db_ResultAp4))
    #print(database.shape)
    database_ResultAp4 = add_foot_contacts(database_ResultAp4)
    
    database_ResultAp6= np.load(os.path.join(data_path, db_ResultAp6))
    #print(database.shape)
    database_ResultAp6 = add_foot_contacts(database_ResultAp6)

    database_ResultAp10= np.load(os.path.join(data_path, db_ResultAp10))
    #print(database.shape)
    database_ResultAp10 = add_foot_contacts(database_ResultAp10)

    database_GT_blend= np.load(os.path.join(data_path, db_Blend_GT))
    #print(database.shape)
    database_GT_blend = add_foot_contacts(database_GT_blend)

    

    
    for i in range(20):
        index0 = np.random.randint(0, len(database_GT))
        #index1 = np.random.randint(0, len(database))
        #index2 = np.random.randint(0, len(database))
        #print("database[index0:index0 + 1]: ",database[index0:index0 + 1].shape)
        animation_plot([
            database_GT[index0:index0 + 1],
            database_GT_tail[index0:index0 + 1],
            database_ResultAp0[index0:index0 + 1],
            database_ResultAp4[index0:index0 + 1],
            database_ResultAp6[index0:index0 + 1],
            database_ResultAp10[index0:index0 + 1],
            database_GT_blend[index0:index0 + 1],
        ])
        
        
