from typing import Counter
from squeeze_net import SqueezeNet
import torch
from preprocessing import load_data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SqueezeNet()
model.load_state_dict(torch.load('squeeze_net_weights-7.pth',map_location=torch.device('cpu')))
train_data, val_data, test_data = load_data()

model = model.to(device)
model.eval()
with torch.no_grad():
    correct = 0
    for X, y in test_data:
        X = X.to(device)
        y = y.to(device)
        preds = model(X)
        count = torch.sum(torch.isclose(torch.floor(preds+0.5),y))
        correct+=count
        print(count/test_data.batch_size)
    print(correct/len(test_data.dataset))

torch.save(model.state_dict(), 'squeeze_net_weights.pth')