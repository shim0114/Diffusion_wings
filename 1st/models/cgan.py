from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from pixyz.distributions import Deterministic
from pixyz.distributions import Normal
from pixyz.models import GAN
from pixyz.utils import print_latex
from tqdm import tqdm
import wandb

from utils import TwoConvBlock_2D

batch_size = 128
epochs = 100

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

root = '../data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambd=lambda x: x.view(-1))])
kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=True, transform=transform, download=True),
    shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=root, train=False, transform=transform),
    shuffle=False, **kwargs)

x_dim = 784
y_dim = 10
z_dim = 100

# generator model p(x|z,y)    
class Generator(Deterministic):
    def __init__(self):
        super(Generator, self).__init__(var=["x"], cond_var=["z","y"], name="p")
        self.fc = nn.Linear(y_dim+z_dim, 49)
        self.dropout = nn.Dropout(0.2)
        self.tcb1 = TwoConvBlock_2D(1,512)
        self.tcb2 = TwoConvBlock_2D(512,256)
        self.tcb3 = TwoConvBlock_2D(256,128)
        self.convt1 = nn.ConvTranspose2d(512, 512, kernel_size =2, stride = 2)
        self.convt2 = nn.ConvTranspose2d(256, 256, kernel_size =2, stride = 2)
        self.conv1 = nn.Conv2d(128, 1, kernel_size = 2, padding="same")

    def forward(self, z, y):
        y = torch.eye(10)[y].to(device)
        h = torch.cat([z, y], dim=-1)
        h = self.dropout(h)
        h = self.fc(h)
        h = torch.reshape(h, (-1, 1, 7, 7))
        h = self.tcb1(h)
        h = self.convt1(h)
        h = self.tcb2(h)
        h = self.convt2(h)
        h = self.tcb3(h)
        h = self.conv1(h)
        x = torch.sigmoid(h)
        x = x.view(-1, x_dim)
        return {"x": x}

# prior model p(z)
prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
               var=["z"], features_shape=[z_dim], name="p_{prior}").to(device)

# generative model
p_g = Generator()
p = (p_g*prior).marginalize_var("z").to(device)

# discriminator model p(t|x,y)
class Discriminator(Deterministic):
    def __init__(self):
        super(Discriminator, self).__init__(var=["t"], cond_var=["x","y"], name="d")

        self.tcb1 = TwoConvBlock_2D(1,64)
        self.tcb2 = TwoConvBlock_2D(64, 128)
        self.tcb3 = TwoConvBlock_2D(128, 256)

        self.avgpool_2D = nn.AvgPool2d(2, stride = 2)
        self.global_avgpool_2D = nn.AvgPool2d(7)

        self.fc1 = nn.Linear(256, 20)
        self.fc2 = nn.Linear(20, 1)
        self.rl = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.1)
        self.embed = nn.Embedding(10, 256)

    def forward(self, x, y):
        x = x.view(-1, 1, 28, 28)
        x = self.tcb1(x)
        x = self.avgpool_2D(x)
        x = self.tcb2(x)
        x = self.avgpool_2D(x)
        x = self.tcb3(x)
        x = self.global_avgpool_2D(x)
        x = x.view(-1, 256)
        _x = x
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.rl(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        _y = self.embed(y)   #ラベルをembedding層で埋め込む
        xy = (_x*_y).sum(1, keepdim=True)   #出力ベクトルとの内積をとる

        x = x+xy   #内積を加算する
        t = torch.sigmoid(x)
        return {"t": t}
    
d = Discriminator().to(device)

model = GAN(p, d,
            optimizer=optim.Adam, optimizer_params={"lr":0.0002},
            d_optimizer=optim.Adam, d_optimizer_params={"lr":0.0002})

def train(epoch):
    train_loss = 0
    train_d_loss = 0
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y = y.to(device) # torch.eye(10)[y].to(device)     
        loss, d_loss = model.train({"x": x, "y": y})
        train_loss += loss
        train_d_loss += d_loss
 
    train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
    train_d_loss = train_d_loss * train_loader.batch_size / len(train_loader.dataset)
    print('Epoch: {} Train loss: {:.4f}, {:.4f}'.format(epoch, train_loss.item(), train_d_loss.item()))
    return train_loss

def test(epoch):
    test_loss = 0
    test_d_loss = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device) # torch.eye(10)[y].to(device)     
        loss, d_loss = model.test({"x": x, "y": y})
        test_loss += loss
        test_d_loss += d_loss

    test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
    test_d_loss = test_d_loss * test_loader.batch_size / len(test_loader.dataset)
    
    print('Test loss: {:.4f}, {:.4f}'.format(test_loss, test_d_loss.item()))
    return test_loss
    
def plot_image_from_latent(z, y):
    with torch.no_grad():
        sample = p_g.sample_mean({"z": z, "y": y}).view(-1, 1, 28, 28).cpu()
        return sample

wandb.init(project='B4_thesis', entity='shim0114')

# plot_number = 4

z_sample = torch.randn(64, z_dim).to(device)
y_sample = torch.tensor([1]*32+[8]*32).to(device) # torch.eye(10)[[plot_number]*64].to(device)

for epoch in range(1, epochs + 1):
    wandb_log_dict = dict()

    train_loss = train(epoch)
    test_loss = test(epoch)
    
    sample = plot_image_from_latent(z_sample, y_sample)

    wandb_log_dict['train_loss'] = train_loss.item()
    wandb_log_dict['test_loss'] = test_loss.item()

    wandb_log_dict['Image_from_latent'] = [wandb.Image(image) for image in sample]

    wandb.log(wandb_log_dict)