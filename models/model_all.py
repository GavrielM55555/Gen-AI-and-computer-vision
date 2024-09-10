# VAE + UNET :
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import plotly.offline as pyo
pyo.init_notebook_mode()
import xarray as xr
import os
import pickle
import cartopy.crs as ccrs
import cartopy as crs
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from einops import rearrange
from torchinfo import summary
from datetime import datetime, timedelta
import torch.nn.functional as F
# from torchgeo.samplers import RandomGeoSampler


class VAE(nn.Module):
    def __init__(self,in_channels,out__channels): #the size (batch,in_channels,54,81)
        super(VAE, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, 21, kernel_size=4, stride=2, padding=1) 
        self.enc_conv2 = nn.Conv2d(21, 32, kernel_size=4, stride=2, padding=1)  
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(96*3*5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_mu = nn.Linear(128, in_channels)
        self.fc2_logvar = nn.Linear(128, in_channels)
        
        # Decoder
        self.fc3 = nn.Linear(in_channels, 128)
        self.fc4 = nn.Linear(128, 96*3*5)
        self.dec_conv1 = nn.ConvTranspose2d(96*2, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64*2, 32, kernel_size=4, stride=2, padding=1)  
        self.dec_conv3 = nn.ConvTranspose2d(32*2, 21, kernel_size=4, stride=2, padding=1)  
        self.dec_conv4 = nn.ConvTranspose2d(21*2, 3, kernel_size=4, stride=2, padding=1)   
        self.last_conv= nn.ConvTranspose2d(20,out__channels , kernel_size=3, stride=1, padding=1)   # with the channel of the input
        
    def encode(self, x):
        h1 = torch.relu(self.enc_conv1(x))
        h2 = torch.relu(self.enc_conv2(h1))
        h3 = torch.relu(self.enc_conv3(h2))
        h4 = torch.relu(self.enc_conv4(h3))
        h = h4.view(-1,  96*3*5)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.fc2_mu(h), self.fc2_logvar(h) ,[h4,h3,h2,h1]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z,in_x,layers):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        h = h.view(-1, 96, 3, 5)
        h=torch.cat((h,layers[0]),dim=1)
        h = torch.relu(self.dec_conv1(h))
        h=torch.cat((h,layers[1]),dim=1)
        h = torch.relu(self.dec_conv2(h))
        padding = (0,0, (13-h.shape[2])//2,(13-h.shape[2]) - (13-h.shape[2]) // 2)
        padded_h = F.pad(h, padding)      
        h=torch.cat((padded_h,layers[2]),dim=1)
        h = torch.relu(self.dec_conv3(h))
        padding = (0,0, (27-h.shape[2])//2,(27-h.shape[2]) - (27-h.shape[2]) // 2)
        padded_h = F.pad(h, padding)
        h=torch.cat((padded_h,layers[3]),dim=1)
        h = self.dec_conv4(h)
        padding = (0, 81-h.shape[3], (54-h.shape[2])//2,(54-h.shape[2]) - (54-h.shape[2]) // 2)
        padded_h = F.pad(h, padding)
        h_cat=torch.cat((padded_h,in_x),dim=1)
        h=self.last_conv(h_cat)
        return h 
    
    def forward(self, x):
        in_x=x.clone()
        mu, logvar,layers= self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z,in_x,layers), mu, logvar

def loss_function_mse(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD,MSE

#-----------------------------------------------------------------------------------------------------------------------
#UNET:

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))
        self.down3 = (Down(16, 32))
        self.down4 = (Down(32, 64))
        factor = 2 if bilinear else 1
        self.down5 = (Down(64, 128 // factor))
        self.up1 = (Up(128, 64 // factor, bilinear))
        self.up2 = (Up(64, 32 // factor, bilinear))
        self.up3 = (Up(32, 16 // factor, bilinear))
        self.up4 = (Up(16, 8 // factor, bilinear))
        self.up5 = (Up(8, 4, bilinear))
        self.outc = (OutConv(n_channels+4, n_classes))

    def forward(self, x):
        in_x = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        in_x=torch.cat([in_x,x],dim=1)
        logits = self.outc(in_x)

        return logits
    
#-------------------------------------------------------------------------------------------------------------------
# domain adaptation
class cnn_encoder(nn.Module):
    def __init__(self,in_channels,n,output_linear):
        super(cnn_encoder, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=n, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(n)  # BatchNorm after the second convolution
        self.relu0 = nn.ReLU()
        self.resnet_block0 = ResNetBlock(n, n)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2*n)  # BatchNorm after the first convolution
        self.relu = nn.ReLU()# I WIIL APPLY IT AFTER EVERY COV IN THE FORWARD
        self.resnet_block1 = ResNetBlock(2*n, 2*n)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=2*n, out_channels=3*n, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(3*n)  # BatchNorm after the second convolution
        self.resnet_block2 = ResNetBlock(3*n, 3*n)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=3*n, out_channels=4*n, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4*n)  # BatchNorm after the second convolution
        self.resnet_block3 = ResNetBlock(4*n, 4*n)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=5*n, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(5*n)  # BatchNorm after the second convolution
        self.resnet_block4 = ResNetBlock(5*n, 5*n)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=5*n, out_channels=6*n, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(6*n)  # BatchNorm after the second convolution
        self.resnet_block5 = ResNetBlock(6*n, 6*n)
        self.pool5 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.1)

        self.fc_input_size = 6*n * 1 * 2  
        self.fc1 = nn.Linear(self.fc_input_size, output_linear)
        self.relu = nn.ReLU()


        #for the domain classifer:
        self.fc_input_size = 6*n * 1 * 2  
        self.fc3 = nn.Linear(self.fc_input_size ,256 )
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(256, 2)
        
    def forward(self, x, grl_lambda=1.0):
#         x=x.view(-1,153,54,81)
        x = self.pool0(self.resnet_block0(self.bn0(self.relu(self.conv0(x)))))
        x = self.pool1(self.resnet_block1(self.bn1(self.relu(self.conv1(x)))))
        x = self.pool2(self.resnet_block2(self.bn2(self.relu(self.conv2(x)))))
        x = self.pool3(self.resnet_block3(self.bn3(self.relu(self.conv3(x)))))
        x = self.pool4(self.resnet_block4(self.bn4(self.relu(self.conv4(x)))))
        x = self.pool5(self.resnet_block5(self.bn5(self.relu(self.conv5(x)))))
        x=self.dropout2(x)
        x = x.view(-1, self.fc_input_size)

        # Apply gradient reversal
        feat_domain = x #GradientReverse.apply(x, grl_lambda)

        x = self.relu(self.fc1(x))
 

        # Domain classifier part
        feat_domain = self.fc3(feat_domain)
        feat_domain = self.relu4(feat_domain)
        feat_domain = self.fc4(feat_domain)

        return x, feat_domain
    
class dust_pred(nn.Module):
    def __init__(self,linear_input,output_channels):
        super(dust_pred, self).__init__()
        self.fc1 = nn.Linear(linear_input, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_channels)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x