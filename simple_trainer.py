from simple_model import SimpleModel
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import numpy as np

class Graying(object):
    
    def __init__(self):
        super(Graying)
        self.grayer = transforms.Grayscale()

    def __call__(self, image):
        if image.shape[0]==3:
            image = self.grayer(image)
        return image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),Graying()])
total_dataset = datasets.Caltech101(
    root='data',
    download=True,
    transform=trans
)

train_size = int(len(total_dataset)*0.7)
val_size = int(len(total_dataset)*0.1)
test_size = len(total_dataset)-train_size-val_size
train_dataset,val_dataset,test_dataset = random_split(total_dataset, [train_size,val_size,test_size])
train_data = DataLoader(train_dataset,batch_size=64)
test_data = DataLoader(test_dataset,batch_size=64)
val_data = DataLoader(val_dataset,batch_size=64)


loss_fn = CrossEntropyLoss()
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.0001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch%5==0:
            acc = torch.sum((torch.argmax(pred,dim=-1)==y).float())/y.shape[-1]
            loss, current = loss.item(), batch * len(X)
            print(f"training: {current}/{size};  train loss is: {loss}; acc is {acc}")

def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    lossSum = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y).item()
        lossSum += loss
        if batch%5==0:
            acc = torch.sum((torch.argmax(pred,dim=-1)==y).float())/y.shape[-1]
            current = batch * len(X)
            print(f"evaluating: {current}/{size};  eval loss is: {loss}; acc is {acc}")
    return lossSum/size*dataloader.batch_size

EPOCH = 50
minLoss = 1e9
for epoch in range(EPOCH):
    print(f"training epoch {epoch+1}")
    train(train_data,model,loss_fn,optimizer)
    with torch.no_grad():
        print(f"evaluating epoch {epoch+1}")
        loss = eval(val_data,model,loss_fn)
        print(f"average evaluation loss: {loss}")
        if loss<minLoss:
            torch.save(model.state_dict(), 'simple_net_weights.pth')
