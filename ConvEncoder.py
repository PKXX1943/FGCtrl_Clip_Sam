import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


class ConvEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        out_chans: int = 64,
    ) -> None:
    
        super().__init__()
        self.img_size = img_size

        self.pre = nn.Sequential(
                nn.Conv2d(in_chans, 64, 3, 1, 1, bias=False),
                LayerNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                ResBlock(64, 64))
        
        self.layer1 = self.EncoderLayer(64, 128, 3, stride=2)
        self.layer2 = self.EncoderLayer(128, 256, 3, stride=2)

        self.post = nn.Sequential(
                nn.Conv2d(256, out_chans, 1),
                LayerNorm2d(out_chans),
                nn.ReLU(),
                nn.Conv2d(out_chans, out_chans, 3, 1, 1))

    def EncoderLayer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            LayerNorm2d(outchannel))    
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))    
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.post(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            LayerNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            LayerNorm2d(outchannel) )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)