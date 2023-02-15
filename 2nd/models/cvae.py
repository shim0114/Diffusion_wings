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

BATCH_SIZE = 256
NUM_TRAIN_DATA, NUM_VALID_DATA = 3767, 0
x_dim = 496
y_dim = 1
z_dim = 64
epochs = 20000

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

coords = np.load("../wing_data/standardized_NandJ_coords.npz")
CLs = np.load("../wing_data/standardized_NandJ_perfs.npz")

# building dataloader
class MyDataset(Dataset):
    def __init__(self, coords, CLs):
        super().__init__()
        
        self.coords = torch.tensor(coords).float()
        self.cls = torch.tensor(CLs).float()
        self.len = len(coords)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        coord = self.coords[index]
        cl = self.cls[index]
        
        return coord, cl

wing_dataset = MyDataset(coords['arr_0'], CLs['arr_0'])

train_dataset, valid_dataset = torch.utils.data.random_split(
    wing_dataset, 
    [NUM_TRAIN_DATA, NUM_VALID_DATA]
)

# 学習用Dataloader
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2, 
    drop_last=True,
    pin_memory=True
)

# 評価用Dataloader
valid_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2, 
    drop_last=True,
    pin_memory=True
)

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

kl = KullbackLeibler(q, prior)

model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr":1e-3})

def train(epoch):
    train_loss = 0
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)        
        loss = model.train({"x": x, "y": y})
        train_loss += loss
 
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def test(epoch):
    test_loss = 0
    for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)
        loss = model.test({"x": x, "y": y})
        test_loss += loss

    test_loss = test_loss / len(valid_loader)
    print('Test loss: {:.4f}'.format(test_loss))

    z_sample = torch.randn([5,64]).to(device)
     
    y_sample = torch.tensor([[-1.5],[-0.75],[0.0],[0.75],[1.5]]).to(device)
    sample = p.sample_mean({'z':z_sample,'y':y_sample}).detach().cpu().numpy()
    for i in range(5):
        plt.scatter(sample[i,:248],sample[i,248:])
        plt.savefig('../../tmp/cvae'+str(i)+'.png')
        plt.clf()

    return test_loss


wandb.init(project='B4_thesis', entity='shim0114')

for epoch in range(1, epochs + 1):
    wandb_log_dict = dict()

    train_loss = train(epoch)
    test_loss = test(epoch)

    wandb_log_dict['train_loss'] = train_loss.item()
    wandb_log_dict['test_loss'] = test_loss.item()
    wandb_log_dict['sample_img'] = [wandb.Image(Image.open('../../tmp/cvae'+str(i)+'.png')) for i in range(5)]


    if (epoch+1) % 25 == 0:
        torch.save(p.state_dict(), os.path.join(wandb.run.dir, 'model_p_'+str(epoch+1)+'.pth'))
        torch.save(q.state_dict(), os.path.join(wandb.run.dir, 'model_q_'+str(epoch+1)+'.pth'))

    wandb.log(wandb_log_dict)