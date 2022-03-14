from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Directory containing the data.
root = './data/celeba'

seed_value=303

train_log = open('./train_log_{seed_value}.txt', 'w')
test_log = open('./test_log_{seed_value}.txt', 'w')

np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)
if device != 'cpu':
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_celeba(batch_size):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = datasets.ImageFolder(root=root, transform=transform)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)] )
    # Create the dataloader.
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

class AE(nn.Module):
    def __init__(self, nc, ngf, ndf, bottleneck = 128):
        super(AE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, bottleneck)

        # decoder
        self.d1 = nn.Linear(bottleneck, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))


    def forward(self, x):
        z = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        res = self.decode(z)
        return res

def train_epoch(model, optimizer, train_loader, device, recon_function):
    model.train()
    train_loss = 0
    for i, image in tqdm(enumerate(train_loader)):
        image = image[0].to(device)
        optimizer.zero_grad()
        recon_batch = model(image)
        loss = recon_function(recon_batch, image)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if not i % 1000:
            print('Train Batch {}/{}. Per-batch loss: {:.4f}\n'.format(i, len(train_loader), loss.item()))
            train_log.write('Train Batch {}/{}. Per-batch loss: {:.4f}\n'.format(i, len(train_loader), loss.item()))
    return train_loss / (len(train_loader))

def test(model, optimizer, test_loader, device, recon_function):
    model.eval()
    test_loss = 0
    for n, image in enumerate(test_loader):
        image = image[0].to(device)
        recon_batch = model(image)
        loss = recon_function(recon_batch, image)
        test_loss += loss.item()
    test_loss /= (len(test_loader))
    return test_loss, image, recon_batch

def train(model, optimizer, train_loader, test_loader, epochs, device):
    for i in tqdm(range(epochs)):
        loss = train_epoch(model, optimizer, train_loader, device, recon_function)
        train_log.write('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, loss))
        print('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, loss))
        test_loss, image, recon_batch = test(model, optimizer, test_loader, device, recon_function)
        print('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))
        test_log.write('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))
        torchvision.utils.save_image(image[:8].data, f'./imgs/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_data.jpg', nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch[:8].data, f'./imgs/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_recon.jpg', nrow=8, padding=2)
        #plt.imsave('./Seed_{seed_value}_epoch_{i}_recon_plt.jpg', np.transpose(recon_batch[0].detach().cpu().numpy(), (1, 2, 0)))
        torch.save(model.state_dict(), f"./models/model_seed_{seed_value}.pth")

if __name__ == '__main__':
    epochs = 30
    model = AE(nc=3, ngf=128, ndf=128, bottleneck=128)
    model.to(device)
    recon_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader, test_loader = get_celeba(batch_size=64)
    train(model, optimizer, train_loader, test_loader, epochs, device)
    train_log.close()
    test_log.close()
