import os
import math
import torch
import torch.nn as nn
import torchvision.models
from lib_net.resnet_224 import resnet18, resnet50

        
        
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = resnet50()
        self.fc1 = nn.Linear(512 * 4, 12)    #
        #self.fc2 = nn.Linear(1024, 31)    
        
        
    def forward(self, x):
        x = self.layers(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x
        
