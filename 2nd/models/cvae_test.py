from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
from pixyz.utils import print_latex
from tqdm import tqdm
import wandb

x_dim = 496
y_dim = 1
z_dim = 64

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

coords = np.load("../wing_data/standardized_NandJ_coords.npz")
CLs = np.load("../wing_data/standardized_NandJ_perfs.npz")

# inference model q(z|x,y)
class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(var=["z"], cond_var=["x","y"], name="q")

        self.fc11 = nn.Linear(x_dim, 512)
        self.fc12 = nn.Linear(y_dim, 512)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc31 = nn.Linear(1024, z_dim)
        self.fc32 = nn.Linear(1024, z_dim)

    def forward(self, x, y):
        hx = F.relu(self.fc11(x))
        hy = F.relu(self.fc12(y))
        h = F.relu(self.fc2(torch.cat([hx,hy],dim=-1)))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

# generative model p(x|z,y)    
class Generator(Normal):
    def __init__(self):
        super(Generator, self).__init__(var=["x"], cond_var=["z","y"], name="p")
        self.fc11 = nn.Linear(z_dim,512)
        self.fc12 = nn.Linear(y_dim, 512)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, x_dim)

    def forward(self, z, y):
        hz = F.relu(self.fc11(z))
        hy = F.relu(self.fc12(y))
        h = F.relu(self.fc2(torch.cat([hz,hy],dim=-1)))
        x = self.fc3(h)
        return {"loc": x, "scale": torch.tensor(0.75).to(x.device)}

p = Generator().to(device)
q = Inference().to(device)
prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
               var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

p.load_state_dict(torch.load('wandb/run-20230123_050817-88zmnbcd/files/model_p_20000.pth'))
q.load_state_dict(torch.load('wandb/run-20230123_050817-88zmnbcd/files/model_q_20000.pth'))

wandb.init(project='B4_thesis', entity='shim0114')

num_sample = 100

for cl in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    z = torch.randn([num_sample, z_dim]).to(device)
    y = torch.tensor([[(cl - CLs['arr_1'])/CLs['arr_2']] for _ in range(num_sample)]).float().to(device)
    print(y)
    x_hat = p.sample_mean({'z':z,'y':y}).detach().cpu().numpy()

    for i in range(num_sample):
        x_i = np.zeros((248,2)) 
        x_i[:,0] = x_hat[i,:248]
        x_i[:,1] = x_hat[i,248:]
        np.savetxt('../../tmp/cdiffusion' + \
                    '_cl_' + str(cl) + \
                    '_' + str(i) + '.txt', x_i)

#     for i in range(num_sample):
#         plt.scatter(x_hat[i,:248],x_hat[i,248:])
#         plt.savefig('../../tmp/cvae_test_cl_'+str(cl)+'_'+str(i)+'.png')
#         plt.clf()
#     wandb_log_dict = dict()
#     wandb_log_dict['sample_img'] = [wandb.Image('../../tmp/cvae_test_cl_'+str(cl)+'_'+str(j)+'.png') for j in range(num_sample)]
#     wandb.log(wandb_log_dict)

