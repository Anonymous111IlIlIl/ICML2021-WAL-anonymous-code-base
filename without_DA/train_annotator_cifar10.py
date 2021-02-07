''' this file aims to train and get the weak annotators'''
from getdata_cifar10 import Cus_Dataset
from torch.utils.data import DataLoader as DataLoader
from annotator_network_cifar10 import Net
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata_cifar10
from utils import get_cifar_data
from models.Vgg import VGG
import numpy as np

dataset_dir = './data/cifar-10-batches-py/'

model_cp = './model/'
workers = 10
batch_size = 64
lr = 0.001
nepoch = 7
device = torch.device("cuda")


def validate(val_loader, model, epoch):
    model.eval()
    
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        images.float()
        
        
        # compute y_pred
        y_pred = model(images)
        
        #y_pred = F.softmax(y_pred,dim=1)
        #print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels[:,0], axis = 1)).sum().item()

    print('   * EPOCH {epoch} | Accuracy: {acc:.3f}'.format(epoch=epoch, acc=(100.0 * correct / total)))
    model.train()

def validate_class(val_loader, model, epoch):
    model.eval()
    
    correct = 0
    total = 0
    c_class = [0 for i in range(10)]
    t_class = [0 for i in range(10)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # compute y_pred
        y_pred= model(images)
        #y_pred = F.softmax(y_pred,dim=1)
        #print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels[:,0], axis = 1)
        correct += (predicted == true_label).sum().item()
        #print(predicted.shape[0])
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1
        
    
    print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=(100.0 * correct / total)))
    for j in range(10):
        print(' class {0}={1}%'.format(j,(100.0*c_class[j]/t_class[j])))
    print('\n')
    model.train()
    
def train():
    print("loading whole cifar10 dataset")
    data_set = get_cifar_data(dataset_dir) #[list_img[ind], list_label[ind], data_size]

    datafile = Cus_Dataset(mode = 'train', data_set = data_set, begin_ind = 0, end_ind = 9000)
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    
    datafile_val = Cus_Dataset(mode = 'val', data_set = data_set, begin_ind = 40000, end_ind = 42000)
    valloader = DataLoader(datafile_val, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    model = Net()
    #model = VGG('VGG19')
    
    model = model.to(device)
    #model = nn.DataParallel(model)
    model.train()

    '''
    model.load_state_dict(torch.load('./model/model.pth'))
    validate_class(valloader, model, 1)

    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), torch.tensor(label).to(device)
            
            #label = torch.argmax(label, dim = 1)
            img.float()
            label = label.float()
            
            out = model(img)
            
            out = F.log_softmax(out,dim=1)
            loss = criterion_KL(out, label)
            #loss = criterion(out, label.squeeze())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            if i % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss.item()))
        
        #test
        validate(valloader, model, epoch)
        
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))
    

if __name__ == '__main__':
    train()