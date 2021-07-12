import torch
from torch.nn.modules.loss import BCELoss
from squeeze_net import SqueezeNet
import numpy as np
import time
import gc
from preprocessing import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data, val_data, test_data = load_data()
loss_fn = BCELoss()
model = SqueezeNet().to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = torch.squeeze(model(X))
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = torch.sum(torch.isclose(torch.floor(pred+0.5),y))/y.shape[-1]
            current = batch * y.shape[-1]
            print(f"training: {current}/{size};  train loss is: {loss.item()}; acc is {acc}")
            del pred, loss, acc
            gc.collect() 

def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    lossSum = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = torch.squeeze(model(X))
        loss = loss_fn(pred, y).item()
        lossSum += loss
        acc = torch.sum(torch.isclose(torch.floor(pred+0.5),y))/y.shape[-1]
        current = batch * y.shape[-1]
        print(f"evaluating: {current}/{size};  eval loss is: {loss}; acc is {acc}")
    return lossSum/size*dataloader.batch_size

EPOCH = 50
minLoss = 1e9
for epoch in range(EPOCH):
    print(f"training epoch {epoch+1}")
    train(train_data,model,loss_fn,optimizer)
    model.eval()
    with torch.no_grad():
        print(f"evaluating epoch {epoch+1}")
        loss = eval(val_data,model,loss_fn)
        print(f"average evaluation loss: {loss}")
        if loss<minLoss:
            torch.save(model.state_dict(), 'squeeze_net_weights.pth')
