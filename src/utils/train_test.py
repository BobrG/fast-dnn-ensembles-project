import torch
import numpy as np
from tqdm import tqdm 

def train_epoch(model, optimizer, train_loader, device, loss_func, experiment):
    model.train()
    train_loss = 0
    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.to(device)
        print('model input', image.shape)
        optimizer.zero_grad()
        out = model(image)
        print('model out', out.shape)
        loss = loss_func(out, image)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if not i % 10:
            #print('Train Batch {}/{}. Per-batch loss: {:.4f}\n'.format(i, len(train_loader), loss.item()))
            experiment.log_metric(name = 'train_batch_loss', value = loss.item(), epoch=i)
    
    return train_loss / (len(train_loader))

def test_epoch(model, optimizer, test_loader, device, loss_func, experiment):
    model.eval()
    test_loss = 0
    for n, (image, _) in enumerate(test_loader):
        image = image.to(device)
        out = model(image)
        loss = loss_func(out, image)
        test_loss += loss.item()
        if not i % 10:
            #print('Test Batch {}/{}. Per-batch loss: {:.4f}\n'.format(i, len(test_loader), loss.item()))
            experiment.log_metric(name = 'test_batch_loss', value = loss.item(), epoch=i)

    test_loss /= (len(test_loader))
    
    return test_loss, image, out
