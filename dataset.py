import torch
import joblib
import glob
import os
from PIL import Image
import cv2
import numpy as np
import tqdm
import threading
import queue

class Trainset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path,label_path, patch_size, scale,aug=True):
        super(Trainset, self).__init__()
        self.data=[]
        self.label=[]
        filename=[[os.path.split(i)[-1] for i in glob.glob(os.path.join(path,'*.png'))] for path in data_path]
        for i in range(len(data_path)):
            for name in filename[i]:
                self.data.append(cv2.imread(os.path.join(data_path[i],name)).transpose([2,0,1]))
                self.label.append(cv2.imread(os.path.join(label_path[i],name)).transpose([2,0,1]))

        self.patch_size = patch_size
        self.scale=scale
        self.aug=aug

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        #patch slice
        position_h = np.random.randint(0, data.shape[1]-self.patch_size+1)
        position_w = np.random.randint(0, data.shape[2]-self.patch_size+1)
        data = data[:, position_h:position_h + self.patch_size, position_w:position_w + self.patch_size]
        label = label[:, self.scale * position_h:self.scale * (position_h + self.patch_size),
                self.scale * position_w:self.scale * (position_w + self.patch_size)]
        if self.aug:
            #data augment
            aug=np.random.rand(3)
            if(aug[0]>0.5):
                data=data[:,:,::-1]
                label=label[:,:,::-1]
            if(aug[1]>0.5):
                data=data[:,::-1,:]
                label=label[:,::-1,:]
            if(aug[2]<0.25):
                data=data.transpose([0,2,1])
                data=data[:,::-1,:]
                label=label.transpose([0,2,1])
                label=label[:,::-1,:]
            elif(aug[2]<0.5):
                data=data[:,::-1,::-1]
                label=label[:,::-1,::-1]
            elif(aug[2]<0.75):
                data=data.transpose([0,2,1])
                data=data[:,:,::-1]
                label=label.transpose([0,2,1])
                label=label[:,:,::-1]
            data=np.ascontiguousarray(data)
            label=np.ascontiguousarray(label)

        return data, label

    def __len__(self):
        return len(self.data)

class Valset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path,label_path, scale):
        super(Valset, self).__init__()
        self.data=[]
        self.label=[]
        filename=[[os.path.split(i)[-1] for i in glob.glob(os.path.join(path,'*.png'))] for path in data_path]
        self.name=[]
        for i in range(len(data_path)):
            for name in filename[i]:
                self.data.append(cv2.imread(os.path.join(data_path[i],name)).transpose([2,0,1]))
                self.label.append(cv2.imread(os.path.join(label_path[i],name)).transpose([2,0,1]))
                self.name.append(os.path.join(data_path[i],name))

        self.scale=scale

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        label = label[:, :label.shape[1]//self.scale*self.scale,:label.shape[2]//self.scale*self.scale]
        return data, label,self.name[index]

    def __len__(self):
        return len(self.data)
