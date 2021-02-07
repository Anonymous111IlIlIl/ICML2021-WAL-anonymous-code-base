import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib_net.resnet_224 import resnet18, resnet50
#from lib_net.vgg_32 import vgg19
import random

        
class MNet(nn.Module):
    def __init__(self, n_class):
        super(MNet, self).__init__()
        self.f0 = resnet50()#vgg19()
        
        self.n_class = n_class
        self.fc1 = nn.Linear(4 * 512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.n_class)
            
        
        self.fc1_1 = nn.Linear(4*512 + self.n_class, 1024)#512
        self.fc1_2 = nn.Linear(1024, self.n_class)
        
    def diff(self, x1, x2):
        x = torch.mean(x1, dim = 0, keepdim=True) - torch.mean(x2, dim = 0, keepdim=True)
        xxT = torch.mean(torch.mm(x, torch.transpose(x, 0, 1)))
        return xxT

    def cmmd_loss(self, x, st_ind_x, c_x):

        loss = 0.0

        for i in range(self.n_class):
            selected_x = x[c_x == i]
            selected_st_ind_x = st_ind_x[c_x == i]
            selected_s_x = selected_x[selected_st_ind_x == 0]
            selected_t_x = selected_x[selected_st_ind_x == 1]
            #print(selected_x.shape)
            #print(selected_st_ind_x.shape)
            if len(selected_s_x)==0 or len(selected_t_x)==0:
                continue
            loss += self.diff(selected_s_x, selected_t_x)
            #print(loss)
            #break
        return loss

    def forward(self, x, w=None, st_ind_x = None, c_x = None):
        x = self.f0(x)

        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        y1 = self.fc3(x1)
        
        if w == None:
            if st_ind_x == None:
                
                return y1, None
            else:
                
                loss2 = self.cmmd_loss(x1, st_ind_x, c_x)
            
                return y1, loss2
        else:
            x2 = F.relu(self.fc1_1(torch.cat((x,w),1)))
            y2 = F.tanh(self.fc1_2(x2))
            return y1, y2
            
