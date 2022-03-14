#import sys
#sys.path.append('/home/gbobrovskih/sk_fast_dnn_ensembles/')

from comet_ml import Experiment
experiment = Experiment(
    api_key="KfIQxtLQwFBi7wahbWN9aCeav",
    project_name="sk-fast-dnn-ensembles",
    workspace="grebenkovao", log_code = False)


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
from perceptual import VGGPerceptualLoss as perceptual_loss
import argparse




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
            experiment.log_metric(name = 'train_batch_loss', value = loss.item(), epoch=i)
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
        experiment.log_metric(name = 'train_loss', value = loss, epoch=i)
        train_log.write('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, loss))
        print('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, loss))
        test_loss, image, recon_batch = test(model, optimizer, test_loader, device, recon_function)
        print('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))
        experiment.log_metric(name = 'test_loss', value = test_loss, epoch=i)
        test_log.write('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))
        torchvision.utils.save_image(image[:8].data, f'./imgs/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_data.jpg', nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch[:8].data, f'./imgs/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_recon.jpg', nrow=8, padding=2)
        #plt.imsave('./Seed_{seed_value}_epoch_{i}_recon_plt.jpg', np.transpose(recon_batch[0].detach().cpu().numpy(), (1, 2, 0)))
        torch.save(model.state_dict(), f"./models/model_seed_{seed_value}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoches', type=int, help='number of epoches for training and testing')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed to initialize pytorch')
    parser.add_argument('-l', '--loss', type=str, choices=['mse', 'vgg'], help='loss: mse or vgg perceptual loss')    
    parser.add_argument('-lr', type=float, help='learning rate for training')
    parser.add_argument('-d', '--data-dir', type=str, default='./data/celeba', help='directory with CelebA dataset') 
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_value = args.seed
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    epochs = args.epoches
    model = AE(num_channels=3, encoder_features=64, decoder_features=64, bottleneck=128)
    model.to(device)

    if args.loss == 'mse':
        recon_function = nn.MSELoss()
    elif args.loss == 'vgg':
        recon_function = perceptual_loss(nn.MSELoss()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader, test_loader = get_celeba(batch_size=64)
    train(model, optimizer, train_loader, test_loader, epochs, device)
    train_log.close()
    test_log.close()
