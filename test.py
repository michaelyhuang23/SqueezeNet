from squeeze_net import SqueezeNet
from preprocessing import load_data
import time
from torch import nn
import torch
import gc
EPOCH=50
train_data, val_data, test_data = load_data()
X, y = next(iter(train_data))
loss_fn = nn.BCELoss()
model = SqueezeNet().to('cpu')
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
for epoch in range(EPOCH):
    pred = torch.squeeze(model(X))
    loss = loss_fn(pred, y)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        acc = torch.sum(torch.isclose(torch.floor(pred+0.5),y))/y.shape[-1]
        print(f"training: {epoch}/{EPOCH};  train loss is: {loss}; acc is {acc}")
        del pred, loss, acc
        gc.collect() 
