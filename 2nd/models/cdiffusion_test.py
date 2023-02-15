from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb

from cdiffusion import DDPM, ContextUnet

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

x_dim = 496
y_dim = 1
n_T = 500
n_feat = 256 # 128 ok, 256 better (but slower)
ws_test = [0.0] # strength of generative guidance

ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.load_state_dict(torch.load('wandb/run-20230120_022502-18e02c1h/files/ddpm_20000.pth'))
ddpm.to(device)

def flatten(x_lst, window=1):
    x_flat = np.zeros_like(x_lst)
    for i in range(x_dim//2): 
        # x axis
        x_pos = np.tile(x_lst[:,:,:248],(1,1,3))
        low = 248 + i - window
        high = 248 + i + window
        x_flat[:,:,i] = np.mean(x_pos[:,:,low:high],axis=2)
        # y axis
        y_pos = np.tile(x_lst[:,:,248:],(1,1,3))
        low = 248 + i - window
        high = 248 + i + window
        x_flat[:,:,i + x_dim//2] = np.mean(y_pos[:,:,low:high],axis=2)
    return x_flat

ddpm.eval()
wandb.init(project='B4_thesis', entity='shim0114')
with torch.no_grad():
    n_class = 8
    n_variation = 100
    n_sample = n_variation * n_class
    cl_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    # cdiffusion.pyの方も！
    wandb_log_dict = dict()
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = ddpm.sample(n_sample, (1, x_dim), device, guide_w=w)
        x_gen = x_gen.detach().cpu().numpy()
        for i in range(n_sample):
            x_i = np.zeros((248,2)) 
            x_i[:,0] = x_gen[i,0,:248]
            x_i[:,1] = x_gen[i,0,248:]
            np.savetxt('../../tmp/cdiffusion' + \
                        '_cl_' + str(cl_list[i%n_class]) + \
                        '_g_' + str(w) + \
                        '_w_' + str(1) + \
                        '_' + str(i//n_class) + '.txt', x_i)
        for win in [7]:#[1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
            for i in range(n_sample):
                x_flat = flatten(x_gen, win)
                x_i = np.zeros((248,2)) 
                x_i[:,0] = x_flat[i,0,:248]
                x_i[:,1] = x_flat[i,0,248:]
                np.savetxt('../../tmp/cdiffusion' + \
                        '_cl_' + str(cl_list[i%n_class]) + \
                        '_g_' + str(w) + \
                        '_w_' + str(win*2+1) + \
                        '_' + str(i//n_class) + '.txt', x_i)


        # for i in range(n_sample):
        #     plt.scatter(x_gen[i,0,:248],x_gen[i,0,248:])
        #     plt.savefig('../../tmp/test_cdiffusion'+str(i)+'.png')
        #     plt.clf()
        # index1 = 'sample_img_guidance_' + str(w)
        # index2 = index1 + '_flat'
        # wandb_log_dict[index1] = [wandb.Image(Image.open('../../tmp/test_cdiffusion'+str(i)+'.png')) for i in range(5)]
        # wandb.log(wandb_log_dict)
        # for win in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        #     for i in range(5):
        #         x_flat = flatten(x_gen, win)
        #         plt.scatter(x_flat[i,0,:248],x_flat[i,0,248:])
        #         plt.savefig('../../tmp/test_cdiffusion_flat'+str(i)+'.png')
        #         plt.clf()
        #     wandb_log_dict[index2] = [wandb.Image(Image.open('../../tmp/test_cdiffusion_flat'+str(i)+'.png')) for i in range(5)]
        #     wandb.log(wandb_log_dict)

        # for i in range(5 * n_class):
        #     plt.scatter(x_gen[i,0,:248],x_gen[i,0,248:],label='N=0')
        #     x_flat = flatten(x_gen, 7)
        #     plt.scatter(x_flat[i,0,:248],x_flat[i,0,248:],c='red',label='N=7')
        #     # x_flat = flatten(x_gen, 11)
        #     # plt.scatter(x_flat[i,0,:248],x_flat[i,0,248:],c='orange',label='N=11')
        #     plt.legend()
        #     plt.savefig('../../tmp/test_cdiffusion'+str(i)+'.png')
        #     plt.clf()
        # index1 = 'sample_img_guidance_' + str(w)
        # wandb_log_dict[index1] = [wandb.Image(Image.open('../../tmp/test_cdiffusion'+str(i)+'.png')) for i in range(5)]
        # wandb.log(wandb_log_dict)
