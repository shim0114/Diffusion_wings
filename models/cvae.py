from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE
from pixyz.utils import print_latex
from tqdm import tqdm
import wandb

batch_size = 128
epochs = 10
seed = 1
torch.manual_seed(seed)

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
z_dim = 64

# inference model q(z|x,y)
class Inference(Normal):
    def __init__(self):
        super(Inference, self).__init__(var=["z"], cond_var=["x","y"], name="q")

        self.fc1 = nn.Linear(x_dim+y_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], 1)))
        h = F.relu(self.fc2(h))        
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

# generative model p(x|z,y)    
class Generator(Bernoulli):
    def __init__(self):
        super(Generator, self).__init__(var=["x"], cond_var=["z","y"], name="p")

        self.fc1 = nn.Linear(z_dim+y_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z, y):
        h = F.relu(self.fc1(torch.cat([z, y], 1)))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}

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
        y = torch.eye(10)[y].to(device)        
        loss = model.train({"x": x, "y": y})
        train_loss += loss
 
    train_loss = train_loss * train_loader.batch_size / len(train_loader.dataset)
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def test(epoch):
    test_loss = 0
    for x, y in test_loader:
        x = x.to(device)
        y = torch.eye(10)[y].to(device)
        loss = model.test({"x": x, "y": y})
        test_loss += loss

    test_loss = test_loss * test_loader.batch_size / len(test_loader.dataset)
    print('Test loss: {:.4f}'.format(test_loss))
    return test_loss

def plot_reconstrunction(x, y):
    with torch.no_grad():
        z = q.sample({"x": x, "y": y}, return_all=False)
        z.update({"y": y})
        recon_batch = p.sample_mean(z).view(-1, 1, 28, 28)
    
        recon = torch.cat([x.view(-1, 1, 28, 28), recon_batch]).cpu()
        return recon
    

def plot_image_from_latent(z, y):
    with torch.no_grad():
        sample = p.sample_mean({"z": z, "y": y}).view(-1, 1, 28, 28).cpu()
        return sample


wandb.init(project='B4_thesis', entity='shim0114')

plot_number = 9

z_sample = 0.5 * torch.randn(64, z_dim).to(device)
y_sample = torch.eye(10)[[plot_number]*64].to(device)

_x, _y = iter(test_loader).next()
_x = _x.to(device)
_y = torch.eye(10)[_y].to(device)

for epoch in range(1, epochs + 1):
    wandb_log_dict = dict()

    train_loss = train(epoch)
    test_loss = test(epoch)
    
    recon = plot_reconstrunction(_x[:8], _y[:8])
    sample = plot_image_from_latent(z_sample, y_sample)

    wandb_log_dict['train_loss'] = train_loss.item()
    wandb_log_dict['test_loss'] = test_loss.item()

    wandb_log_dict['Image_from_latent'] = [wandb.Image(image) for image in sample]
    wandb_log_dict['Image_reconstrunction'] = [wandb.Image(image) for image in recon]

    wandb.log(wandb_log_dict)