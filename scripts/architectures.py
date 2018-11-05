import torchvision.models as models
import torch.nn as nn
import numpy as np
import torchvision
import PIL.Image
import shutil
import torch
import math

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image

### CAE ###

class AddCoords(nn.Module):
    def __init__(self, radius_channel=False):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        """
        in_tensor: (batch_size, channels, x_dim, y_dim)
        [0,0,0,0]   [0,1,2,3]
        [1,1,1,1]   [0,1,2,3]    << (i,j)th coordinates of pixels added as separate channels
        [2,2,2,2]   [0,1,2,3]
        taken from mkocabas.
        """
        batch_size_tensor = in_tensor.shape[0]

        xx_ones = torch.ones([1, in_tensor.shape[2]], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(in_tensor.shape[2], dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, in_tensor.shape[3]], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(in_tensor.shape[3], dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 1, 2)
        yy_channel = yy_channel.permute(0, 3, 1, 2)

        xx_channel = xx_channel.float() / (in_tensor.shape[2] - 1)
        yy_channel = yy_channel.float() / (in_tensor.shape[3] - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        out = torch.cat([in_tensor.cuda(), xx_channel.cuda(), yy_channel.cuda()], dim=1)

        if self.radius_channel:
            radius_calc = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out.cuda(), radius_calc.cuda()], dim=1)

        return out

###

class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """
    def __init__(self, radius_channel, *args, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(radius_channel=radius_channel)
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

###

class Encoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        # Encoder of ResNet blocks
        self.init_conv = CoordConv(in_channels=6, out_channels=16, kernel_size=4, radius_channel=True, stride=2, padding=1, bias=False) #64
        self.conv1 = nn.Conv2d(16 ,32, 4, 2, bias=False,padding = 1) #32
        self.conv2 = nn.Conv2d(32 ,64, 4, 2, bias=False, padding = 1) #16
        self.conv3 = nn.Conv2d(64 ,128, 4, 2, bias=False, padding = 1) #8
        self.fc = Flatten()
        self.z = torch.nn.Linear(8192, z_dim)


    def forward(self, x):
## Forward pass through encoder
        encode = self.init_conv(x)
        encode = self.conv1(encode)
        encode = self.conv2(encode)
        encode = self.conv3(encode)
        encode = self.fc(encode)
        x = self.z(encode)
        return x

###

class Decoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.project = nn.Linear(z_dim, 8192)
        self.relu1 = nn.ReLU()
        self.reshape = Reshape((128,8,8,))
        self.bilinear0 = nn.Upsample(scale_factor=2) #16
        self.conv6 = torch.nn.Conv2d(128 ,256, 3, 1, bias=False, padding = 1)
        self.bilinear1 = nn.Upsample(scale_factor=2) # 32
        self.conv7 = torch.nn.Conv2d(256 ,128, 3, 1, bias=False, padding = 1)
        self.bilinear2 = nn.Upsample(scale_factor=2) # 64
        self.conv8 = torch.nn.Conv2d(128 ,64, 3, 1, bias=False, padding = 1)
        self.bilinear3 = nn.Upsample(scale_factor=2) # 128
        self.conv9 = torch.nn.Conv2d(64, 3, 3, 1, bias=False, padding = 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        decode = self.relu(x)
        decode = self.project(decode)
        decode = self.relu1(decode)
        decode = self.reshape(decode)
        
        decode = self.bilinear0(decode)
        decode = self.conv6(decode)

        decode = self.bilinear1(decode)
        decode = self.conv7(decode)

        decode = self.bilinear2(decode)
        decode = self.conv8(decode)

        decode = self.bilinear3(decode)
        decode = self.conv9(decode)

        x_reconstructed = self.sig(decode)
        return x_reconstructed

###

class Model(torch.nn.Module):
    def __init__(self, z_dim):
        super(Model, self).__init__()
        
        # Encoder of ResNet blocks
        self.encoder = Encoder(z_dim)
    
        # Decoder of ResNet blocks
        self.decoder = Decoder(z_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        z  = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

### Adversary Neural Network ### 

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features = 2048, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 256)
        self.fc4 = nn.Linear(in_features = 256, out_features = 128)
        self.fc5 = nn.Linear(in_features = 128, out_features = 64)
        self.fc6 = nn.Linear(in_features = 64, out_features = 32)
        self.fc7 = nn.Linear(in_features = 32, out_features = 10)
        ###
	self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.1)
        ###
    def forward(self, x):
        h1 = self.dropout(self.relu(self.fc1(x)))
        h2 = self.dropout(self.relu(self.fc2(h1)))
        h3 = self.dropout(self.relu(self.fc3(h2)))
        h4 = self.dropout(self.relu(self.fc4(h3)))
        h5 = self.dropout(self.relu(self.fc5(h4)))
        h6 = self.dropout(self.relu(self.fc6(h5)))
        return self.fc7(h6)

