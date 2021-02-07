import numpy as np
from collections.abc import Iterable
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import softmax


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_cifar_data(dir):

    list_img = []
    list_label = []
    data_size = 0

    #load cifar in the format as (32,32,3)
    for filename in ['%s/data_batch_%d' % (dir,j) for j in range(1, 6)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding = 'bytes')
        for i in range(len(cifar10[b"labels"])):
            img = np.reshape(cifar10[b"data"][i], (3,32,32))
            img = np.transpose(img, (1,2,0))
            #img = img.astype(float)
            list_img.append(img)
            
            list_label.append(np.eye(10)[cifar10[b"labels"][data_size%10000]])
            data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_vis_data(dir):

    list_img = []
    list_label = []
    data_size = 0
    n_class = 12

    sub_dir_list = os.listdir(dir)
    for i in range(len(sub_dir_list)):
        path = os.path.join(dir, sub_dir_list[i])
        if os.path.isdir(path):
            label_ind = i#sub_dir_list[i]
            file_list = os.listdir(path)
            for j in range(len(file_list)):
                file_path = os.path.join(path, file_list[j])
                #img = rgb_loader(file_path)
                #list_img.append(np.array(img))
                list_img.append(file_path)
                list_label.append(np.eye(n_class)[label_ind])
                data_size += 1

        if os.path.isfile(path):
            pass
    
    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]


    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    
    return [list_img, list_label, data_size]



def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)