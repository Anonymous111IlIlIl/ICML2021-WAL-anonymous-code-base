from getdata_vis import Cus_Dataset
from torch.utils.data import DataLoader
from network_vis import MNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata_vis
from utils import freeze_by_names, unfreeze_by_names, get_vis_data

dataset_dir1 = './data/train/'
dataset_dir2 = './data/validation/'

model_cp = './model_vis'
workers = 10
batch_size = 64
n_class = 12
lr_default = 0.001
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
    c_class = [0 for i in range(12)]
    t_class = [0 for i in range(12)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # compute y_pred
        y_pred, _ = model(images)
        #y_pred = F.softmax(y_pred,dim=1)
        #print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels[:,0], axis = 1)
        correct += (predicted == true_label).sum().item()
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1
        
    print('   * EPOCH {epoch} | Accuracy: {acc:.3f}'.format(epoch=epoch, acc=(100.0 * correct / total)))
    f.write('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=(100.0 * correct / total)))
    for j in range(12):
        f.write(' class {0}={1}% '.format(j,(100.0*c_class[j]/t_class[j])))
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
        if (epoch % 2) == 1:
            validate_class(val_dataloader, model, epoch)

def train_NN_soft_temp1(model, train_dataloader, val_dataloader, optimizer, criterion, nepoch):
    model = model.to(device)
    model.train()
    
    cnt = 0
    for epoch in range(nepoch):
        for i, (img, label, c_label, st_label) in enumerate(train_dataloader):
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
        if (epoch % 2) == 1:
            validate_class(val_dataloader, model, epoch)

def train_NN_new(model, train_dataloader, val_dataloader, optimizer, criterion, nepoch, p=1e-4):
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
        if (epoch % 2) == 1:
            validate_class(val_dataloader, model, epoch)
    return model
    
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

    model = MNet(n_class)
    freeze_by_names(model, ('fc1_1', 'fc1_2'))

    lr = lr_default
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    print("training NN using weak data")
    f.write("training NN using weak data\n")
    train_NN_new(model, weak_dataloader, val_dataloader, optimizer1, criterion_KL, 60)#40
    
    print("finetuning NN using strong data")
    f.write("finetuning NN using strong data\n")
    train_NN_soft(model, strong_dataloader, val_dataloader, optimizer2, criterion_KL, 60)#40

    unfreeze_by_names(model, ('fc1_1', 'fc1_2'))

    torch.save(model.state_dict(), '{0}/model_new.pth'.format(model_cp))

def train_only_strong(weak_dataloader, val_dataloader, strong_dataloader):

    model = MNet(n_class)
    freeze_by_names(model, ('fc1_1', 'fc1_2'))

    lr = lr_default
    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    print("finetuning NN using strong data")
    f.write("finetuning NN using strong data\n")
    train_NN_soft(model, strong_dataloader, val_dataloader, optimizer2, criterion_KL, 60)#40

    unfreeze_by_names(model, ('fc1_1', 'fc1_2'))

def train_diff(diff_dataloader):
    model_file = './model_vis/model_new.pth'
    model = MNet(n_class).to(device)
    model.load_state_dict(torch.load(model_file))
    model.train()

    #freeze_by_names(model, ('f0', 'fc1', 'fc2', 'fc3'))

    lr = lr_default

    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    train_NN_diff(model, diff_dataloader, diff_dataloader, optimizer1, 60)#40
    
    #unfreeze_by_names(model, ('f0', 'fc1', 'fc2', 'fc3'))
    print("train diff finished...")
    f.write("train diff finished...\n")
    #dataloader = regenerate_dataset(model, data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size)
    return model
'''
def regenerate_dataset(model, data_set, strong_data_begin_ind, strong_data_size, weak_data_begin_ind, weak_data_size):
    model.eval()
    Cus_Dataset(mode = 'new_with_st_index', \
                        dataset_1 = dataset3, begin_ind1 = 0, size1 = 2817,\
                        dataset_2 = None, begin_ind2 = None, size2 = None,\
                        dataset_3 = None, begin_ind3 = None, size3 = None)
    new_dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return new_dataloader
'''
    
def train_on_new(dataloader, val_dataloader):
    model_file = './model_vis/model_new.pth'
    model = MNet(n_class).to(device)
    #model.load_state_dict(torch.load(model_file))
    model.train()
    freeze_by_names(model, ('fc1_1', 'fc1_2'))

    lr = lr_default
    optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)#, weight_decay=0.00000005
    criterion_KL = torch.nn.KLDivLoss()
    print("train on new dataset")
    f.write("train on new dataset\n")
    model = train_NN_new(model, dataloader, val_dataloader, optimizer1, criterion_KL, 160, 1e-4)#150
    #train_NN_new(model, dataloader, val_dataloader, optimizer1, criterion_KL, 50, 0)


    unfreeze_by_names(model, ('fc1_1', 'fc1_2'))

    

if __name__ == '__main__':
    
    

    print("loading whole VisDA-C dataset")
    f.write("loading whole VisDA-C dataset\n")
    dataset1 = get_vis_data(dataset_dir1)
    dataset2 = get_vis_data(dataset_dir2)#[list_img[ind], list_label[ind], data_size]
    print('length of train data is {0}'.format(len(dataset1[0])))#
    print('length of val is {0}'.format(len(dataset2[0])))#
    f.write('length of train data is {0}\n'.format(len(dataset1[0])))#
    f.write('length of val is {0}\n'.format(len(dataset2[0])))#



    print("loading weak soft dataset...")
    weak_soft_data = Cus_Dataset(mode = 'weak', \
                            dataset_1 = dataset1, begin_ind1 = 0, size1 = 20000,\
                            dataset_2 = dataset2, begin_ind2 = 0, size2 = 1000,\
                            dataset_3 = None, begin_ind3 = None, size3 = None)
    weak_soft_dataloader = DataLoader(weak_soft_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading val dataset...")
    datafile_val = Cus_Dataset(mode = 'val', \
                            dataset_1 = dataset2, begin_ind1 = 20000, size1 = 2000,\
                            dataset_2 = None, begin_ind2 = None, size2 = None,\
                            dataset_3 = None, begin_ind3 = None, size3 = None)
    val_dataloader = DataLoader(datafile_val, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading strong soft dataset...")
    strong_soft_data = Cus_Dataset(mode = 'strong', \
                            dataset_1 = dataset2, begin_ind1 = 0, size1 = 1000,\
                            dataset_2 = None, begin_ind2 = None, size2 = None,\
                            dataset_3 = None, begin_ind3 = None, size3 = None)
    strong_soft_dataloader = DataLoader(strong_soft_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    print("loading diff dataset...")
    diff_data = Cus_Dataset(mode = 'diff', \
                            dataset_1 = dataset2, begin_ind1 = 0, size1 = 1000,\
                            dataset_2 = None, begin_ind2 = None, size2 = None,\
                            dataset_3 = None, begin_ind3 = None, size3 = None)
    diff_dataloader = DataLoader(diff_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)



    #beacuse the model is saved, you skip the next line once you have run it
    train_weak(weak_soft_dataloader, val_dataloader, strong_soft_dataloader)
    
    model = train_diff(diff_dataloader)
    model.eval()
    new_data = Cus_Dataset(mode = 'new_with_st_index', \
                        dataset_1 = dataset1, begin_ind1 = 0, size1 = 20000,\
                        dataset_2 = dataset2, begin_ind2 = 0, size2 = 1000,\
                        dataset_3 = None, begin_ind3 = None, size3 = None,\
                        new_model = model)
    new_dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    train_on_new(new_dataloader, val_dataloader)

    f.close()