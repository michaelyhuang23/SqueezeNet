import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from squeeze_net import SqueezeNet
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = DataLoader(datasets.CIFAR100(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
),batch_size=64)

test_data = DataLoader(datasets.CIFAR100(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
),batch_size=64)


loss_fn = CrossEntropyLoss()
model = SqueezeNet().to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%100==0:
            acc = torch.sum((torch.argmax(pred,dim=-1)==y).float())/y.shape[-1]
            loss, current = loss.item(), batch * len(X)
            print(f"training: {current}/{size};  train loss is: {loss}; acc is {acc}")

def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    lossSum = 0
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y).item()
        lossSum += loss
        if batch%100==0:
            acc = torch.sum((torch.argmax(pred,dim=-1)==y).float())/y.shape[-1]
            current = batch * len(X)
            print(f"evaluating: {current}/{size};  eval loss is: {loss}; acc is {acc}")
    return lossSum/size*dataloader.batch_size

EPOCH = 20
minLoss = 1e9
for epoch in range(EPOCH):
    print(f"training epoch {epoch+1}")
    train(train_data,model,loss_fn,optimizer)
    with torch.no_grad():
        print(f"evaluating epoch {epoch+1}")
        loss = eval(test_data,model,loss_fn)
        print(f"average evaluation loss: {loss}")
        if loss<minLoss:
            torch.save(model.state_dict(), 'squeeze_net_weights.pth')
