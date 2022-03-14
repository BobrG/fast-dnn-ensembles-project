import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from comet_ml import Experiment
experiment = Experiment(
    api_key="KfIQxtLQwFBi7wahbWN9aCeav",
    project_name="sk-fast-dnn-ensembles",
    workspace="grebenkovao", log_code = False)

from losses import VGGPerceptualLoss as perceptual_loss


def main(model, optimizer, loss, train_loader, test_loader, epochs, device, seed_value='42', loss_name='mse'):
    
    for i in tqdm(range(epochs)):
        train_loss = train_epoch(model, optimizer, train_loader, device, loss)
        experiment.log_metric(name = 'train_loss', value = train_loss, epoch=i)
        print('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, train_loss))
 
        #train_log.write('Train Epoch: {}. Average loss: {:.4f}\n'.format(i, loss))
        test_loss, image, recon_batch = test(model, optimizer, test_loader, device, recon_function)
        experiment.log_metric(name = 'test_loss', value = test_loss, epoch=i)
        print('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))

        #test_log.write('Test Epoch: {}. loss: {:.4f}\n'.format(i, test_loss))
        
        torchvision.utils.save_image(image[:8].data, f'./imgs/seed_{seed_value}_loss_{loss_name}/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_data.jpg', nrow=8, padding=2)
        torchvision.utils.save_image(recon_batch[:8].data, f'./imgs/seed_{seed_value}_loss_{loss_name}/seed_{seed_value}/Seed_{seed_value}_epoch_{i}_recon.jpg', nrow=8, padding=2)
        #plt.imsave('./Seed_{seed_value}_epoch_{i}_recon_plt.jpg', np.transpose(recon_batch[0].detach().cpu().numpy(), (1, 2, 0)))
        
        torch.save({
                    'epoch': i,
                    'model': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                    }, f"./models/model_seed_{seed_value}_loss_{loss_name}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoches', type=int, help='number of epoches for training and testing')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed to initialize pytorch')
    parser.add_argument('-l', '--loss', type=str, choices=['mse', 'vgg'], help='loss: mse or vgg perceptual loss')    
    parser.add_argument('-lr', type=float, help='learning rate for training')
    parser.add_argument('-d', '--data-dir', type=str, default='./data/celeba', help='directory with CelebA dataset') 
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./imgs'):
        print('./imgs doesn not exist ==> creating a directory')
        os.makedirs('./imgs')
    if not os.path.exists('./models'):
        print('./models doesn not exist ==> creating a directory')
        os.makedirs('./models')

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
        loss = nn.MSELoss()
    elif args.loss == 'vgg':
        loss = perceptual_loss(nn.MSELoss()).to(device)

    if not os.path.exists(f'./imgs/seed_{seed_value}_loss_{args.loss}'):
        print(f'./imgs/seed_{seed_value}_loss_{args.loss} doesn not exist ==> creating a directory')
        os.makedirs(f'./imgs/seed_{seed_value}_loss_{args.loss}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader, test_loader = get_celeba(batch_size=64)
    main(model, optimizer, loss, train_loader, test_loader, epochs, device, seed_value=seed_value, loss_name=args.loss)


