# for data loading
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader
# for autoencoder building
import torch.nn as nn
import torch.nn.functional as F
# for loss function defining
import torch.optim as optim
# for visualization
import matplotlib.pyplot as plt
import numpy as np
import sys


class AutoEncoder_convrgb2(nn.Module):
    def __init__(self,LENGTH):
        super(AutoEncoder_convrgb2, self).__init__()
        self.LENGTH = LENGTH 
        length = LENGTH/16
        self.num_chan = 512
        self.fc_len = int(self.num_chan*pow(length,2))
        self.cv_len = int(length)
        ## encoder layers ##
        conv1 = nn.Conv2d(3, 128, 5, padding=2, stride=2)  
        bn1 = nn.BatchNorm2d(128)
        conv2 = nn.Conv2d(128, 256, 5, padding=2, stride=2)  
        bn2 = nn.BatchNorm2d(256)
        conv3 = nn.Conv2d(256, 256, 5, padding=2, stride=2)  
        bn3 = nn.BatchNorm2d(256)
        conv4 = nn.Conv2d(256, self.num_chan, 5, padding=2, stride=2)  
        ## decoder layers ##
        conv5 = nn.Conv2d(self.num_chan, 256, 5, padding=2)  
        bn4 = nn.BatchNorm2d(256)
        conv6 = nn.Conv2d(256, 256, 5, padding=2)  
        bn5 = nn.BatchNorm2d(256)
        conv7 = nn.Conv2d(256, 128, 5, padding=2)  
        bn6 = nn.BatchNorm2d(128)
        conv8 = nn.Conv2d(128, 3, 5, padding=2)  

        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()  
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.encoder = nn.Sequential(
            conv1,bn1,relu,conv2,bn2,relu,conv3,bn3,relu,conv4,relu)
        self.decoder = nn.Sequential(
            upsample,conv5,bn4,relu,upsample,conv6,bn5,relu,upsample,conv7,bn6,relu,upsample,conv8,sigmoid)  
        self.fc1 = nn.Sequential(nn.Linear(self.fc_len, 128))
        self.fc2 = nn.Sequential(nn.Linear(128, self.fc_len)) 

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        encoded = self.fc1(encoded.view(-1,self.fc_len))
        # print(encoded.shape)
        decoded = self.fc2(encoded).reshape((-1, self.num_chan, self.cv_len, self.cv_len ))
        # print(decoded.shape)
        decoded = self.decoder(decoded)
        # print(decoded.shape)
        return encoded, decoded


class AutoEncoder_convrgb1(nn.Module):
    def __init__(self,LENGTH):
        super(AutoEncoder_convrgb1, self).__init__()
        self.LENGTH = LENGTH 
        length = LENGTH/16
        self.num_chan = 512
        self.fc_len = int(self.num_chan*pow(length,2))
        self.cv_len = int(length)
        ## encoder layers ##
        conv1 = nn.Conv2d(3, 128, 3, padding=1)  
        bn1 = nn.BatchNorm2d(128)
        conv2 = nn.Conv2d(128, 256, 3, padding=1)
        bn2 = nn.BatchNorm2d(256)
        conv3 = nn.Conv2d(256, 256, 3, padding=1)
        bn3 = nn.BatchNorm2d(256)
        conv4 = nn.Conv2d(256, self.num_chan, 3, padding=1)
        pool = nn.MaxPool2d(2, 2)      
        ## decoder layers ##
        conv5 = nn.Conv2d(self.num_chan, 256, 3, padding=1)
        bn4 = nn.BatchNorm2d(256)
        conv6 = nn.Conv2d(256, 256, 3, padding=1)  
        bn5 = nn.BatchNorm2d(256)
        conv7 = nn.Conv2d(256, 128, 3, padding=1)  
        bn6 = nn.BatchNorm2d(128)
        conv8 = nn.Conv2d(128, 3, 3, padding=1)  
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()  
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.encoder = nn.Sequential(conv1,relu,pool,conv2,relu,pool,conv3,relu,pool,conv4,relu,pool)
        self.decoder = nn.Sequential(upsample,conv5,relu,upsample,conv6,relu,upsample,conv7,relu,upsample,conv8,sigmoid)  
        self.fc1 = nn.Sequential(nn.Linear(self.fc_len, 128))
        self.fc2 = nn.Sequential(nn.Linear(128, self.fc_len)) 

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.fc1(encoded.view(-1,self.fc_len))
        decoded = self.fc2(encoded).reshape((-1, self.num_chan, self.cv_len, self.cv_len ))
        decoded = self.decoder(decoded)
        return encoded, decoded


class AutoEncoder_convrgb0(nn.Module):
    def __init__(self,LENGTH):
        super(AutoEncoder_convrgb0, self).__init__()
        self.LENGTH = LENGTH 
        ## encoder layers ##
        conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        conv2 = nn.Conv2d(16, 4, 3, padding=1)
        pool = nn.MaxPool2d(2, 2)      
        ## decoder layers ##
        conv3 = nn.Conv2d(4, 16, 3, padding=1)
        conv4 = nn.Conv2d(16, 3, 3, padding=1)  
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()  
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc_len = int(4*pow(self.LENGTH/4,2))
        self.cv_len = int(self.LENGTH/4)

        self.encoder = nn.Sequential(conv1,relu,pool,conv2,relu,pool)
        self.decoder = nn.Sequential(upsample,conv3,relu,upsample,conv4,sigmoid)  
        self.fc1 = nn.Sequential(nn.Linear(self.fc_len, 128))
        self.fc2 = nn.Sequential(nn.Linear(128, self.fc_len)) 

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.fc1(encoded.view(-1,self.fc_len))
        decoded = self.fc2(encoded).reshape((-1, 4, self.cv_len, self.cv_len ))
        decoded = self.decoder(decoded)
        return encoded, decoded


class AutoEncoder_conv(nn.Module):
    def __init__(self,LENGTH):
        super(AutoEncoder_conv, self).__init__()
        self.LENGTH = LENGTH 
        ## encoder layers ##
        conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        conv2 = nn.Conv2d(16, 4, 3, padding=1)
        pool = nn.MaxPool2d(2, 2)      
        ## decoder layers ##
        conv3 = nn.Conv2d(4, 16, 3, padding=1)
        conv4 = nn.Conv2d(16, 1, 3, padding=1)  
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()  
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc_len = int(4*pow(self.LENGTH/4,2))
        self.cv_len = int(self.LENGTH/4)

        self.encoder = nn.Sequential(conv1,relu,pool,conv2,relu,pool)
        self.decoder = nn.Sequential(upsample,conv3,relu,upsample,conv4,sigmoid)  
        self.fc1 = nn.Sequential(nn.Linear(self.fc_len, 128))
        self.fc2 = nn.Sequential(nn.Linear(128, self.fc_len)) 
        
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = self.fc1(encoded.view(-1,self.fc_len))
        decoded = self.fc2(encoded).reshape((-1, 4, self.cv_len, self.cv_len ))
        decoded = self.decoder(decoded)
        return encoded, decoded


class AutoEncoder_linear(nn.Module):
    def __init__(self, LENGTH):
        super(AutoEncoder_linear, self).__init__()
        self.LENGTH = LENGTH 

        self.encoder = nn.Sequential(
            nn.Linear(LENGTH*LENGTH, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 12),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.LENGTH*self.LENGTH),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

