import torch
from torch import nn

class FireModule(nn.Module):
    def __init__(self, input_s, output_s, output_e_1, output_e_3):
        super(FireModule, self).__init__()
        self.conv_s = nn.Conv2d(input_s, output_s, (1,1), padding=0)
        self.conv_e1 = nn.Conv2d(output_s, output_e_1, (1,1), padding=0)
        self.conv_e3 = nn.Conv2d(output_s, output_e_3, (3,3), padding=1)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.conv_s(s)
        s = self.relu(s)
        e1 = self.conv_e1(s)
        e1 = self.relu(e1)
        e3 = self.conv_e3(s)
        e3 = self.relu(e3)
        return torch.cat([e1,e3],dim=1)

class ForwardCNN(nn.Module):
    def __init__(self):
        super(ForwardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,96,(7,7),stride=2,padding=3)
        self.pool1 = nn.MaxPool2d((3,3),stride=2,padding=0)
        self.fire1 = FireModule(96,16,64,64)
        self.fire2 = FireModule(128,16,64,64)
        self.fire3 = FireModule(128,32,128,128)
        self.pool2 = nn.MaxPool2d((3,3),stride=2,padding=0)
        self.fire4 = FireModule(256,32,128,128)
        self.fire5 = FireModule(256,48,192,192)
        self.fire6 = FireModule(384,48,192,192)
        self.fire7 = FireModule(384,64,256,256)
        self.pool3 = nn.MaxPool2d((3,3),stride=2,padding=0)
        self.fire8 = FireModule(512,64,256,256)
        self.conv2 = nn.Conv2d(512,1000,(1,1),padding=0)
        self.pool4 = nn.AvgPool2d((13,13))
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x1 = self.fire1(x)
        x2 = self.fire2(x1)
        x3 = self.fire3(x2+x1)
        x = self.pool2(x3)
        x4 = self.fire4(x)
        x5 = self.fire5(x+x4)
        x6 = self.fire6(x5)
        x7 = self.fire7(x6+x5)
        x = self.pool3(x7)
        x8 = self.fire8(x)
        x = self.conv2(x+x8)
        x = self.flatten(x)
        return x

model = ForwardCNN().to('cpu')
print(model)