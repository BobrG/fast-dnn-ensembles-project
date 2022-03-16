import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.datasets import CelebA

def get_data_loader(args, split='train'):
    """
    Loads CelebA dataset from torchvision and applies proproccesing steps to it.
    Input: - args
           - split: 'train'/'test' - running mode
    Output: - TrainDataLoader / TestDataLoader
    """
    # Creating transforms.
    if split == 'train':
        transform = transforms.Compose([transforms.Resize(128),
                                          transforms.CenterCrop(128),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))])

    # Create the dataset and loader.
    
    dataset = CelebA(root='./celeba', split=split, download=True, transform=transform) 
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_sz,
                                         shuffle=(split == 'train'), num_workers=args.num_workers)
    return loader

def get_celeba(root, batch_size):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    print(f'Loading data from {root}')
    print(f'Using batch size = {batch_size} and train/test split 0.8/0.2')

    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = datasets.ImageFolder(root=root, transform=transform)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    # Create the dataloader.

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

