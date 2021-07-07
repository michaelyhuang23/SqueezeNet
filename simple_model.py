import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.sequential_convs = nn.Sequential(
            nn.Conv2d(3,96,(5,5),padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2,padding=0),
            nn.Conv2d(96,48,(1,1)),
            nn.ReLU(),
            nn.Conv2d(48,96,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2,padding=0),
            nn.Conv2d(96,48,(1,1)),
            nn.ReLU(),
            nn.Conv2d(48,96,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2,padding=0),
            nn.Conv2d(96,48,(1,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(768,128),
            nn.ReLU(),
            nn.Linear(128,100)
        )

    def forward(self, x):
        x = self.sequential_convs(x)
        return x
