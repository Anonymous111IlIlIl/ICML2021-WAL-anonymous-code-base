''' this file aims to train and get the weak annotators'''
from getdata_vis import Cus_Dataset
from torch.utils.data import DataLoader as DataLoader
from annotator_network_vis import Net
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata_vis
from utils import get_vis_data, rgb_loader
#from models.Vgg import VGG
import numpy as np

dataset_dir1 = './data/train/'
dataset_dir2 = './data/validation/'

model_cp = './model_vis_backup/'
workers = 10
batch_size = 128
lr = 0.01
nepoch = 40
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
    return (100.0 * correct / total)

    
def validate_class(val_loader, model, epoch):
    model.eval()
    
    correct = 0
    total = 0
    c_class = [0 for i in range(12)]
    t_class = [0 for i in range(12)]
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
    for j in range(12):
        print(' class {0}={1}%'.format(j,(100.0*c_class[j]/t_class[j])))
    print('\n')
    model.train()

def train():

    print("loading whole VisDA-C dataset")
    dataset1 = get_vis_data(dataset_dir1)
    dataset2 = get_vis_data(dataset_dir2)#[list_img[ind], list_label[ind], data_size]

    print('length of train data is {0}'.format(len(dataset1[0])))
    print('length of validation is {0}'.format(len(dataset2[0])))

    
    datafile = Cus_Dataset(mode = 'train_annotator', \
                            dataset_1 = dataset1, begin_ind1 = 0, size1 = 10000,\
                            dataset_2 = dataset2, begin_ind2 = 0, size2 = 10000,\
                            dataset_3 = None, begin_ind3 = None, size3 = None)
    
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    
    datafile_val = Cus_Dataset(mode = 'val', dataset_1 = dataset2, begin_ind1 = 10000, size1 = 2000)
    valloader = DataLoader(datafile_val, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    print('validation dataset loaded!')

    model = Net()
    
    model = model.to(device)
    #model = nn.DataParallel(model)
    '''
    model.load_state_dict(torch.load('./model_vis/model.pth'))
    validate_class(valloader, model, 1)


    '''
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            
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
        acc = validate_class(valloader, model, epoch)
        if (epoch +1 )% 5 == 0:
            torch.save(model.state_dict(), '{0}/model-{1}-{2}.pth'.format(model_cp, epoch+1, acc))
    
    


if __name__ == '__main__':
    train()