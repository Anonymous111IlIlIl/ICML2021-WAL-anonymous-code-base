from getdata_cifar10 import Cus_Dataset
from torch.utils.data import DataLoader
from network_cifar10 import MNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata_cifar10
from utils import freeze_by_names, unfreeze_by_names, get_cifar_data

dataset_dir = './data/cifar-10-batches-py/'

model_cp = './model/'
workers = 10
batch_size = 128


device = torch.device("cuda")
f = open('./log/log.txt', mode='w',buffering=1)

def validate(val_loader, model, epoch):
    model.eval()
    
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        
        # compute y_pred
        y_pred, _ = model(images)
        #y_pred = F.softmax(y_pred,dim=1)
        #print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels[:,0], axis = 1)).sum().item()

    print('   * EPOCH {epoch} | Accuracy: {acc:.3f}'.format(epoch=epoch, acc=(100.0 * correct / total)))
    f.write('   * EPOCH {epoch} | Accuracy: {acc:.3f}\n'.format(epoch=epoch, acc=(100.0 * correct / total)))
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
        y_pred, _= model(images)
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
        
    
    f.write('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=(100.0 * correct / total)))
    for j in range(10):
        f.write(' class {0}={1}%'.format(j,(100.0*c_class[j]/t_class[j])))
    f.write('\n')
    model.train()
    
def train_NN_soft(model, train_dataloader, val_dataloader, optimizer, criterion, nepoch):
    model = model.to(device)
    model.train()
    
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, label) in enumerate(train_dataloader):
            img, label = img.float().to(device), torch.tensor(label).to(device)
            label = label.float()

            out, _ = model(img)
            
            loss = criterion(F.log_softmax(out,dim=1), label)# + criterion(F.log_softmax(label,dim=1), F.softmax(out,dim=1)) + 
            #loss = loss/2.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            if i % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))
                f.write('Epoch:{0},Frame:{1}, train_loss {2}\n'.format(epoch, cnt*batch_size, loss/batch_size))
            
        #test
        if epoch % 2 ==1:
            validate_class(val_dataloader, model, epoch)

def train_NN_new(model, train_dataloader, val_dataloader, optimizer, criterion, nepoch):
    model = model.to(device)
    model.train()
    optimizer.zero_grad()
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, label, c_label, st_label) in enumerate(train_dataloader):
            img, label = img.float().to(device), torch.tensor(label).to(device)
            label = label.float()
            c_label, st_label = c_label.to(device).int(), st_label.to(device).int()
            out, loss2 = model(img, st_ind_x = c_label, c_x = st_label)

            #print("================================")
            
            #print(criterion(F.log_softmax(out,dim=1), label), "-----", loss2)

            loss = criterion(F.log_softmax(out,dim=1), label) + loss2 * 1e-4
            # + criterion(F.log_softmax(label,dim=1), F.softmax(out,dim=1)) + 
            #loss = loss/2.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            if i % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))
                f.write('Epoch:{0},Frame:{1}, train_loss {2}\n'.format(epoch, cnt*batch_size, loss/batch_size))
        
        #test
        if epoch % 2 ==1:
            validate_class(val_dataloader, model, epoch)
    
def train_NN_diff(model, train_dataloader, val_dataloader, optimizer, nepoch):
    model = model.to(device)
    model.train()
    
    
    criterion_KL = torch.nn.KLDivLoss()
    criterion = torch.nn.MSELoss()
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, diff_label, w_label) in enumerate(train_dataloader):
            img, diff_label, w_label = img.float().to(device), torch.tensor(diff_label).float().to(device), torch.tensor(w_label).float().to(device)

            out, diff = model(img, w=w_label)
            
            diff_label = F.softmax(diff_label)
            diff = F.log_softmax(diff,dim=1)
            loss = criterion_KL(diff, diff_label)
            #loss = criterion(diff, diff_label)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            if i % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))
                f.write('Epoch:{0},Frame:{1}, train_loss {2}\n'.format(epoch, cnt*batch_size, loss/batch_size))
        

def train_weak(weak_dataloader, val_dataloader, strong_dataloader):

    model = MNet(10)
    freeze_by_names(model, ('fc1_1', 'fc1_2'))

    lr = 0.001
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    print("training NN using weak data")
    f.write("training NN using weak data\n")
    train_NN_soft(model, weak_dataloader, val_dataloader, optimizer1, criterion_KL, 40)
    
    print("finetuning NN using strong data")
    f.write("finetuning NN using strong data\n")
    train_NN_soft(model, strong_dataloader, val_dataloader, optimizer2, criterion_KL, 30)

    unfreeze_by_names(model, ('fc1_1', 'fc1_2'))

    torch.save(model.state_dict(), '{0}/model_new.pth'.format(model_cp))

def train_diff(data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size, diff_dataloader):
    model_file = './model/model_new.pth'
    model = MNet(10).to(device)
    model.load_state_dict(torch.load(model_file))
    model.train()

    freeze_by_names(model, ('f0', 'fc1', 'fc2', 'fc3'))

    lr = 0.001

    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    train_NN_diff(model, diff_dataloader, diff_dataloader, optimizer1, 70)
    
    unfreeze_by_names(model, ('f0', 'fc1', 'fc2', 'fc3'))
    print("train diff finished...")
    f.write("train diff finished...\n")
    dataloader = regenerate_dataset(model, data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size)
    return dataloader
    
def regenerate_dataset(model, data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size):
    model.eval()
    new_data = Cus_Dataset(mode = 'new_with_st_index', data_set = data_set, \
        begin_ind = weak_data_begin_ind, end_ind = weak_data_begin_ind+weak_data_size, \
        begin_ind1 = strong_data_begin_ind, end_ind1 = strong_data_begin_ind+strong_data_size, new_model = model)
    new_dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return new_dataloader

    
def train_on_new(dataloader, val_dataloader, strong_soft_dataloader):
    model_file = './model/model_new.pth'
    model = MNet(10).to(device)
    #model.load_state_dict(torch.load(model_file))
    model.train()
    freeze_by_names(model, ('fc1_1', 'fc1_2'))

    lr = 0.001
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)#, weight_decay=0.00000005
    criterion_KL = torch.nn.KLDivLoss()
    print("train on new dataset")
    f.write("train on new dataset\n")
    train_NN_new(model, dataloader, val_dataloader, optimizer1, criterion_KL, 70)

    '''
    freeze_by_names(model, ('f0'))
    print("try method on finetuning last layer using strong data")
    f.write("try method on finetuning last layer using strong data\n")
    train_NN_soft(model, strong_soft_dataloader, val_dataloader, optimizer2, criterion_KL, 20)
    unfreeze_by_names(model, ('f0'))
    '''
    unfreeze_by_names(model, ('fc1_1', 'fc1_2'))
    
    

if __name__ == '__main__':
    
    
    strong_data_begin_ind = 20000
    strong_data_size = 1000

    weak_data_begin_ind = 0
    weak_data_size = 10000

    val_data_begin_ind = 40000
    val_data_size = 2000

    print("loading whole cifar10 dataset")
    f.write("loading whole cifar10 dataset\n")
    data_set = get_cifar_data(dataset_dir) #[list_img[ind], list_label[ind], data_size]

    #we also need to carefully consider the quantity and proportion of weak/strong data used
    print("loading weak soft dataset...")
    weak_soft_data = Cus_Dataset(mode = 'weak', data_set = data_set, begin_ind = weak_data_begin_ind, end_ind = weak_data_begin_ind+weak_data_size)
    weak_soft_dataloader = DataLoader(weak_soft_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading val dataset...")
    datafile_val = Cus_Dataset(mode = 'val', data_set = data_set, begin_ind = val_data_begin_ind, end_ind = val_data_begin_ind+val_data_size)
    val_dataloader = DataLoader(datafile_val, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading strong soft dataset...")
    strong_soft_data = Cus_Dataset(mode = 'train', data_set = data_set, begin_ind = strong_data_begin_ind, end_ind = strong_data_begin_ind+strong_data_size)
    strong_soft_dataloader = DataLoader(strong_soft_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading diff dataset...")
    diff_data = Cus_Dataset(mode = 'diff', data_set = data_set, begin_ind = strong_data_begin_ind, end_ind = strong_data_begin_ind+strong_data_size)
    diff_dataloader = DataLoader(diff_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    
    #beacuse the model is saved, you skip the next line once you have run it
    train_weak(weak_soft_dataloader, val_dataloader, strong_soft_dataloader)
    
    dataloader = train_diff(data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size, diff_dataloader)
    
    train_on_new(dataloader, val_dataloader, strong_soft_dataloader)