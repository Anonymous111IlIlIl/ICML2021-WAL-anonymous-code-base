import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from annotator_network_vis import Net
import torch
import torch.nn as nn
import math
from scipy.special import softmax
from utils import AverageMeter, accuracy, get_vis_data, rgb_loader


IMAGE_SIZE = 224 #32

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.RandomGrayscale(),

    #transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), #to [0.0, 1.0], H×W×C -> C×H×W
    #transforms.ToPILImage(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])
    
class Cus_Dataset(data.Dataset):
    def __init__(self, mode = None, \
                            dataset_1 = None, begin_ind1 = 0, size1 = 0,\
                            dataset_2 = None, begin_ind2 = 0, size2 = 0,\
                            dataset_3 = None, begin_ind3 = 0, size3 = 0,\
                            new_model = None):

        self.mode = mode
        self.list_img = []
        
        self.list_diff = []
        self.list_label = []
        self.list_w_label = []
        self.c_label = []
        self.st_label = []
        self.data_size = 0
        self.transform = dataTransform
        self.n_class = 31

        if self.mode == 'train_annotator': #used for training the weak annotator
            
            self.data_size = size1 + size2# + size3

            path_list = np.concatenate((dataset_1[0][begin_ind1: begin_ind1+size1], dataset_2[0][begin_ind2: begin_ind2+size2]), axis = 0)#, dataset_3[0][begin_ind3: begin_ind3+size3]
            for file_path in path_list:
                img = np.array(rgb_loader(file_path))
                self.list_img.append(img)


            self.list_label = np.concatenate((dataset_1[1][begin_ind1: begin_ind1+size1], dataset_2[1][begin_ind2: begin_ind2+size2]), axis = 0)#, dataset_3[1][begin_ind3: begin_ind3+size3]

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img = np.asarray(self.list_img)
            self.list_img = self.list_img[ind]

            self.list_label = np.asarray(self.list_label)
            self.list_label = self.list_label[ind]


        elif self.mode == 'val': #val data

            self.data_size = size1
            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]
            for file_path in path_list:
                img = np.array(rgb_loader(file_path))
                self.list_img.append(img)

            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]

        elif self.mode == 'strong': #used for finetuning
            
            self.data_size = size1
            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]
            for file_path in path_list:
                img = np.array(rgb_loader(file_path))
                self.list_img.append(img)
            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]

        elif self.mode == 'test':
            pass

        elif self.mode == 'weak':
            self.list_label = []

            self.data_size = size1+size2
            path_list = np.concatenate((dataset_1[0][begin_ind1: begin_ind1+size1], dataset_2[0][begin_ind2: begin_ind2+size2]), axis=0)
            for i in range(size1):
                self.st_label.append(0)
            for i in range(size2):
                self.st_label.append(1)

            for file_path in path_list:
                img = np.array(rgb_loader(file_path))
                self.list_img.append(img)

            device = torch.device("cuda")
            model_file = './model_vis/model.pth'
            model = Net().to(device)
            #model = nn.DataParallel(model)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            
            batch_size = 64
            for i in range(int(len(self.list_img)/batch_size)):
                if (i == int(len(self.list_img)/batch_size) -1) and (i * batch_size < len(self.list_img)):
                    imgs = self.list_img[(i)*batch_size:]
                else:
                    imgs = self.list_img[(i)*batch_size:(i+1)*batch_size]

                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model(imgs)
                out = F.softmax(out, dim=1)
                out = out.data.cpu().numpy()
                #self.list_label.append(np.argmax(out))
                self.list_label += [out[j] for j in range(out.shape[0])]
                for j in range(out.shape[0]):
                    self.c_label.append(np.argmax(out[j])) 

        elif self.mode == 'diff': #calulate soft label difference 

            self.data_size = size1
            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]
            for file_path in path_list:
                img = np.array(rgb_loader(file_path))
                self.list_img.append(img)
            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]
            

            device = torch.device("cuda")
            model_file = './model_vis/model.pth'
            model = Net().to(device)
            #model = nn.DataParallel(model)
            model.load_state_dict(torch.load(model_file))
            model.eval()

            batch_size = 64
            for i in range(int(len(self.list_img)/batch_size)):
                if (i == int(len(self.list_img)/batch_size) -1) and (i * batch_size < len(self.list_img)):
                    imgs = self.list_img[(i)*batch_size:]
                else:
                    imgs = self.list_img[(i)*batch_size:(i+1)*batch_size]
                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model(imgs)
                out = F.softmax(out, dim=1)
                out = out.data.cpu().numpy()
                
                #self.list_label.append(np.argmax(out))
                
                #self.list_diff += [np.eye(self.n_class)[np.argmax(self.list_label[batch_size*i+j])] - out[j] for j in range(out.shape[0])]
                self.list_diff += [self.list_label[batch_size*i+j] - out[j] for j in range(out.shape[0])]
                self.list_w_label += [out[j] for j in range(out.shape[0])]

        elif self.mode == 'new_with_st_index':

            self.data_size = size1 + size2
            self.list_img = []
            self.list_label = []
            self.c_label = []
            self.st_label = []
            device = torch.device("cuda")
            model = new_model.to(device)

            #model_file = './model_office/model_new.pth'
            #model.load_state_dict(torch.load(model_file))
            #model = nn.DataParallel(model)
            model.eval()
            
            model_file = './model_vis/model.pth'
            model_annotator = Net().to(device)
            #model = nn.DataParallel(model)
            model_annotator.load_state_dict(torch.load(model_file))
            model_annotator.eval()
            
            #cifar10 does not have domain discrepancy, we pretend it has here by dividing it into the front and back parts.
            path_list_s = dataset_1[0][begin_ind1: begin_ind1+size1]
            list_img_s = []
            for file_path in path_list_s:
                img = np.array(rgb_loader(file_path))
                list_img_s.append(img)
            

            path_list_t = dataset_2[0][begin_ind2: begin_ind2+size2]
            list_img_t = []
            for file_path in path_list_t:
                img = np.array(rgb_loader(file_path))
                list_img_t.append(img)
            



            print("predicting the weak soft label using annotator with weak data")
            batch_size = 32
            self.list_img = list_img_s
            
            for i in range(int(len(list_img_s)/batch_size)):
                if (i == int(len(list_img_s)/batch_size) -1) and (i * batch_size < len(list_img_s)):
                    imgs = list_img_s[(i)*batch_size:]
                else:
                    imgs = list_img_s[(i)*batch_size:(i+1)*batch_size]
                
                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model_annotator(imgs)
                out = F.softmax(out, dim=1)
                
                
                _, diff = model(imgs, out)

                out = out.data.cpu().numpy()
                diff = diff.data.cpu().numpy()
                
                #self.list_label += [softmax(out[j] + diff[j]\
                #    *(0.1/((np.linalg.norm(x=diff[j], ord=2)/math.pow(10.0,0.5)) + 0.1))\
                #     , axis=0) for j in range(out.shape[0])]
                #otherwise
                #self.list_label += [  softmax(out[j] + diff[j], axis=0),  for j in range(out.shape[0])]
                for j in range(out.shape[0]):
                    l1 = softmax(out[j] + diff[j], axis=0)
                    self.list_label.append(l1)
                    self.c_label.append(np.argmax(l1))
                    self.st_label.append(0)
            
            
            
            print("adding strong data")
            #self.list_img = np.concatenate((self.list_img, list_img_t), axis = 0)
            self.list_img += list_img_t
            reconstruct_strong_data = True
            
            if reconstruct_strong_data:
                batch_size = 32
                
                for i in range(int(len(list_img_t)/batch_size)):
                    if (i == int(len(list_img_t)/batch_size) -1) and (i * batch_size < len(list_img_t)):
                        imgs = list_img_t[(i)*batch_size:]
                    else:
                        imgs = list_img_t[(i)*batch_size:(i+1)*batch_size]
                    imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                    imgs = torch.tensor(imgs).to(device)
                    
                    out = model_annotator(imgs)
                    out = F.softmax(out, dim=1)
                    #out = out.data.cpu().numpy()
                    
                    _, diff = model(imgs, out)
                    
                    out = out.data.cpu().numpy()
                    diff = diff.data.cpu().numpy()
                    
                    for j in range(out.shape[0]):
                        l1 = softmax(out[j] + diff[j], axis=0)
                        self.list_label.append(l1)
                        self.c_label.append(np.argmax(l1))
                        self.st_label.append(1)
            else:
                #self.list_img = np.concatenate((self.list_img, list_img_t), axis = 0)
                for j in range(len(list_img_t)):
                    l1 = dataset_2[1][begin_ind2: begin_ind2+size2][j]
                    self.list_label.append(l1)
                    self.c_label.append(np.argmax(l1))
                    self.st_label.append(1)

        else:
            print('Undefined Dataset!')
            
        

    def __getitem__(self, item):
        if self.mode == 'train_annotator':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor(label)
        elif self.mode == 'val':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'strong':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor(label)
        elif self.mode == 'test':
            pass
        elif self.mode == 'weak':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            c_label = self.c_label[item]
            st_label = self.st_label[item]
            #return self.transform(img), torch.tensor(label)
            return self.transform(img), torch.tensor(label), torch.tensor(c_label), torch.tensor(st_label)
        elif self.mode == 'diff':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            diff = self.list_diff[item]
            label = self.list_label[item]
            w_label = self.list_w_label[item]
            return self.transform(img), torch.tensor(diff), torch.tensor(w_label)
        elif self.mode == 'new_with_st_index':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            c_label = self.c_label[item]
            st_label = self.st_label[item]
            return self.transform(img), torch.tensor(label), torch.tensor(c_label), torch.tensor(st_label)
        else:
            print('None')

    def __len__(self):
        return self.data_size