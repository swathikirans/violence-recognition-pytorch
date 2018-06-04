import torch.nn as nn
from torchvision import models
from ConvLSTMCell import *

class ViolenceModel(nn.Module):
    def __init__(self, mem_size):
        super(ViolenceModel, self).__init__()
        self.mem_size = mem_size
        self.alexnet = models.alexnet(pretrained=True)
        self.convNet = nn.Sequential(*list(self.alexnet.features.children()))
        self.alexnet = None
        self.conv_lstm = ConvLSTMCell(256, self.mem_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(3*3*self.mem_size, 1000)
        self.lin2 = nn.Linear(1000, 256)
        self.lin3 = nn.Linear(256, 10)
        self.lin4 = nn.Linear(10, 2)
        self.BN = nn.BatchNorm1d(1000)
        self.classifier = nn.Sequential(self.lin1, self.BN, self.relu, self.lin2, self.relu,
                                        self.lin3, self.relu, self.lin4)

    def forward(self, x):
        state = None
        seqLen = x.size(0) - 1
        for t in range(0, seqLen):
            x1 = x[t] - x[t+1]
            x1 = self.convNet(x1)
            state = self.conv_lstm(x1, state)
        x = self.maxpool(state[0])
        x = self.classifier(x.view(x.size(0), -1))
        return x