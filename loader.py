import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CelebA

def get_data_loader(args, split='train')
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
        transform = transform.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))])

    # Create the dataset and loader.
    
    dataset = CelebA(root='./celeba', split=split, download=True, transform=transform) 
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_sz,
                                         shuffle=(split == 'train'), num_workers=args.num_workers)
    return loader

